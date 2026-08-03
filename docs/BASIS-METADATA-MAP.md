# Basis Metadata Map: where channel lookup and AO ordering are defined

**Requirement:** CLN-02 -- "Channel lookup and AO ordering are centralized enough for human
troubleshooting."
**Written:** plan 05-07, measured 2026-08-03.
**Machine half:** `tests/test_support_documentation.py`, four consistency tests.

---

## Why this document exists

Basis metadata is **not** centralized. It is defined in eight places across three modules,
using two different indexing conventions, and until this plan no test asserted any relation
between any two of them.

That is not a hypothetical hazard. Plan 05-06 swept 157 hardcoded orbital-count sites in
`src/dftorch`, and the only real gaps it found were three copies of the per-shell AO count
table truncated one entry short of f --

    n_orb_per_shell = torch.tensor([0, 1, 3, 5])

in `_spin.get_h_spin`, `_spin.get_h_spin_diag` and `_forces.forces_spin` -- sitting a few
modules away from `Constants.shell_dim`'s correct `[0, 1, 3, 5, 7]`. Nothing related the two,
so nothing noticed. A developer troubleshooting an AO-ordering or channel-lookup problem needs
one page that names every definition site, states its convention, and says which one to
believe. This is that page.

**Read this alongside `docs/ORBITAL-COUNT-INVENTORY.md`.** That inventory dispositions every
*use* of a hardcoded count; this map covers the *definitions* those uses index into. The
inventory already records `Structure.py:11-12` and `Constants.py:107` as sites; the rows are
not repeated here.

---

## The definition sites

### 1. `_bond_integral._CHANNELS` -- `src/dftorch/_bond_integral.py:101`

| Property | Value |
|---|---|
| Shape | `list[str]`, **40 entries** |
| Convention | 20 named Hamiltonian channels (`Hff0` ... `Hss0`), then the 20 matching overlap channels (`Sff0` ... `Sss0`) in the **same suffix order** |
| Ordering | High angular momentum first: `ff`, `df`, `dd`, `pf`, `pd`, `pp`, `sf`, `sd`, `sp`, `ss`; within a shell pair, increasing bond type (sigma=0, pi=1, delta=2, phi=3) |

This is the canonical internal representation of an SKF electronic table row. Both simple-format
and extended-format files are normalized into it before anything downstream reads them (SKF-03).

**Consumed by:**

- `_bond_integral.py:415-419` -- the simple-to-extended widening, the only place the two layouts meet.
- `_bond_integral.py:734` -- the zero row used for absent pairs.
- `_bond_integral.py:745` -- `channels = {ch: mat[:, j] for j, ch in enumerate(_CHANNELS)}`, which is where a column index becomes a channel name.
- `_bond_integral.py:792` -- the default `order` argument of the channel-to-matrix conversion.
- `_bond_integral.py:1145` -- the `(n_pairs, npts, len(_CHANNELS), 4)` spline coefficient tensor, so the channel axis of `coeffs_tensor` **is** this list.
- `_slater_koster_pair.py:25` -- `SK_CHANNEL_NAMES = tuple(_BOND_INTEGRAL_CHANNELS)`, re-exported and then indexed by name.

Two derived constants live beside it: `N_SK_CHANNELS` (`_bond_integral.py:171`) is
`len(_CHANNELS)`, and `SK_BLOCK_SIZE` (`_bond_integral.py:170`) is 20, the width of the H half.

### 2. `_bond_integral._SIMPLE_CHANNELS` -- `src/dftorch/_bond_integral.py:144`

| Property | Value |
|---|---|
| Shape | `list[str]`, **20 entries** |
| Convention | The s/p/d subset of `_CHANNELS`, in `_CHANNELS`' own relative order |

The channel order of a legacy simple-format `.skf` electronic row.

**Consumed by:** `_bond_integral.py:415-419` only, via site 3.

### 3. `_bond_integral._SIMPLE_TO_EXTENDED` -- `src/dftorch/_bond_integral.py:167`

| Property | Value |
|---|---|
| Shape | `list[int]`, 20 entries |
| Convention | `[_CHANNELS.index(ch) for ch in _SIMPLE_CHANNELS]` -- derived, not written out |

The index map that scatters a 20-value simple row into a 40-slot canonical row, leaving the f
channels zero (SKF-04). Because it is computed rather than transcribed, it cannot drift from
`_CHANNELS` on its own; it can only become wrong if `_SIMPLE_CHANNELS` names something
`_CHANNELS` no longer carries, which is what `test_channel_tables_are_consistent` checks.

### 4. `_bond_integral.MAX_SHELLS` -- `src/dftorch/_bond_integral.py:172`

| Property | Value |
|---|---|
| Shape | `int`, `4` |
| Convention | The number of supported angular-momentum shells (s, p, d, f) |

**Consumed by:** `_bond_integral.py:679-682` (rejects an SKF file declaring a shell count
outside `1..MAX_SHELLS`), `:930-931` and `:941` (per-element shell presence and occupation
arrays), `:1163` (the `(120, MAX_SHELLS)` shell-presence table).

### 5. `Constants.shell_dim` -- `src/dftorch/Constants.py:106-108`

| Property | Value |
|---|---|
| Shape | `torch.nn.Parameter`, int64, **5 entries**: `[0, 1, 3, 5, 7]` |
| Convention | **Indexed by shell type id**, which runs 1..4. Entry 0 is an unused pad that makes `shell_dim[shell_types]` a direct gather |

**Consumed by:** `ESDriver.py:539` and `:611`, `_scf.py:561`/`:612` and `:1237`/`:1289` (both
shell-resolved paths do `shell_dim[shell_types]` to expand a per-shell quantity to AOs),
`MD.py:413`.

### 6. `Structure.SHELL_DIMS` -- `src/dftorch/Structure.py:11`

| Property | Value |
|---|---|
| Shape | `tuple[int, ...]`, **4 entries**: `(1, 3, 5, 7)` |
| Convention | **Indexed by shell position 0..3.** No pad |

The same physical fact as site 5 with a different indexing convention. See "The two
conventions that differ" below.

**Consumed by:** `Structure.py:118` (`_shell_local_end` adds it to the starts),
`Structure.py:174` and `:201` (the single-system and batched reference-density builders,
which iterate `enumerate(SHELL_DIMS)` and divide the shell occupation by the shell width).

### 7. `Structure.SHELL_LOCAL_STARTS` -- `src/dftorch/Structure.py:12`

| Property | Value |
|---|---|
| Shape | `tuple[int, ...]`, 4 entries: `(0, 1, 4, 9)` |
| Convention | The exclusive prefix sum of `SHELL_DIMS`: the AO offset of each shell inside an atom's 16-AO block |

**Consumed by:** `Structure.py:107` (`_shell_local_start`, which returns `-1` for an absent
shell), and through it `_shell_local_end` (`:115-120`), `_global_shell_start` (`:123-130`) and
`_global_shell_end` (`:133-140`), which is how a shell becomes a slice of `H0`/`S`.

### 8. `Structure.SHELL_TYPE_IDS` -- `src/dftorch/Structure.py:13`

| Property | Value |
|---|---|
| Shape | `tuple[int, ...]`, `(1, 2, 3, 4)` -- `1=s, 2=p, 3=d, 4=f` |
| Convention | The ids that `shell_types` carries, and therefore the ids that index `Constants.shell_dim` |

**Consumed by: nothing in `src/`, as code.** This is worth stating rather than glossing:
`_spin.py:15` declares `_F_SHELL_TYPE_ID: int = 4` as a **deliberate literal copy**, with a
comment at `_spin.py:11-14` explaining that `Structure` sits above `_spin` in the import graph
so the value cannot be imported. The only thing linking the copy to the definition is that
comment; `test_shell_dim_and_shell_dims_agree` now also links them mechanically.

### 9. `Structure.AO_LABEL_TEMPLATE` -- `src/dftorch/Structure.py:25-42`

| Property | Value |
|---|---|
| Shape | `tuple[str, ...]`, **16 entries** |
| Convention | Per-atom AO order: `s`, `px py pz`, `dxy dyz dzx dx2_y2 dz2`, `fx3 fy3 fz3 fx_y2_z2 fy_z2_x2 fz_x2_y2 fxyz` |

**This is the authority for AO ordering within an atom.** The f block's specific order is
source-locked -- see the authority section below.

**Consumed by:** `Structure.py:93` (`_ao_mask_from_shell_present` sizes the AO mask from
`len(AO_LABEL_TEMPLATE)`), `Structure.py:148` (`_flatten_ao_labels`, which produces the
human-readable AO list a developer actually reads when debugging). Referenced normatively but
not imported by `_slater_koster_pair.py:93`, `:142`, `:161` and `:353`, where the f and p
angular transform tables state that their row/column order is this tuple's.

### 10. `Structure.AO_SHELL_TEMPLATE` -- `src/dftorch/Structure.py:44-61`

| Property | Value |
|---|---|
| Shape | `tuple[int, ...]`, 16 entries: `(1, 2,2,2, 3,3,3,3,3, 4,4,4,4,4,4,4)` |
| Convention | The shell type id of each AO position of site 9 |

**Consumed by:** `Structure.py:155-156` (`_ao_shell_types_from_mask`), which is how a flattened
AO index recovers its shell.

---

## Which table is authoritative for which question

| If you are troubleshooting... | Start at | Not at |
|---|---|---|
| the order of AOs within one atom (a block of `H0`/`S` looks transposed or permuted) | `Structure.AO_LABEL_TEMPLATE` (`Structure.py:25`) | the angular tables in `_slater_koster_pair.py`, which are written *to* this order |
| where a shell starts inside an atom's AO block | `Structure.SHELL_LOCAL_STARTS` (`Structure.py:12`) | any local offset literal; plan 05-06 found three truncated ones |
| how wide a shell is, when indexing by a `shell_types` value | `Constants.shell_dim` (`Constants.py:106`) | `Structure.SHELL_DIMS`, which is 0-based and will be off by one |
| how wide a shell is, when iterating shells in position order | `Structure.SHELL_DIMS` (`Structure.py:11`) | `Constants.shell_dim`, whose entry 0 is a pad |
| the order of channels within an SKF electronic row, or the channel axis of `coeffs_tensor` | `_bond_integral._CHANNELS` (`_bond_integral.py:101`) | the legacy `channel + SH_shift * 10` arithmetic, which is wrong under the 40-channel layout -- see the warning at `_slater_koster_pair.py:19-24` |
| how a simple-format file maps onto the canonical layout | `_bond_integral._SIMPLE_CHANNELS` and `_SIMPLE_TO_EXTENDED` (`:144`, `:167`) | `_ml_sk._CHANNEL_MAP`, which is a separate legacy 10-channel numbering kept deliberately unchanged (`_slater_koster_pair.py:37-40`) |
| a channel by name rather than index | `_slater_koster_pair.SK_CHANNEL_INDEX` (`:27`) / `sk_channel_index` | hardcoded integers |

### The authority for the f AO order

The seven f labels at `AO_LABEL_TEMPLATE[9:16]` are **cubic harmonics**, and their order is
source-locked. Do not re-derive it and do not restate it from memory:

> `.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md`

That record fixes the source as Takegahara, Aoki and Yanase, *J. Phys. C: Solid State Phys.*
**13** (1980) 583, DOI `10.1088/0022-3719/13/4/016`, gives the paper-label-to-`Structure.py`-index
table, and records that the paper's basis **is** `AO_LABEL_TEMPLATE`'s -- so the adapter is a
pure permutation with all `+1` signs and no basis rotation. It also records why Sharma,
*Phys. Rev. B* **19**, 2813 (1979) was rejected as the primary source. The lock is reproduced
in code at `_slater_koster_pair.py:142` and `:161`.

---

## The two conventions that differ

`Constants.shell_dim` and `Structure.SHELL_DIMS` are the same physical fact -- how many atomic
orbitals an s/p/d/f shell contributes -- written twice with different indexing:

| | `Constants.shell_dim` | `Structure.SHELL_DIMS` |
|---|---|---|
| Site | `Constants.py:106-108` | `Structure.py:11` |
| Type | `torch.nn.Parameter`, int64 | plain `tuple[int, ...]` |
| Entries | 5 | 4 |
| Value | `[0, 1, 3, 5, 7]` | `(1, 3, 5, 7)` |
| Index | shell **type id**, 1..4 | shell **position**, 0..3 |
| Why | entry 0 is a pad so `shell_dim[shell_types]` is a direct gather on a tensor | it is iterated with `enumerate`, so a pad would be an extra empty shell |

Both spellings are defensible on their own terms; carrying both without an assertion is what is
not. The relation is now pinned:

    Constants.shell_dim[1:] == list(Structure.SHELL_DIMS)

by `tests/test_support_documentation.py::test_shell_dim_and_shell_dims_agree`, which builds
`Constants` from the `tests/f_orbital_data` fixture directory rather than reconstructing the
tensor by hand, so the value under test is the one a real calculation receives. The same test
pins `len(SHELL_DIMS) == _bond_integral.MAX_SHELLS`, `Structure.SHELL_TYPE_IDS == (1, 2, 3, 4)`,
and `_spin._F_SHELL_TYPE_ID == SHELL_TYPE_IDS[-1]`.

**The inconsistency is bounded, not latent.** A one-sided edit to either table now fails the
suite and the failure message prints both tables.

---

## The consistency tests

`tests/test_support_documentation.py`:

| Test | What it pins |
|---|---|
| `test_shell_dim_and_shell_dims_agree` | `Constants.shell_dim[1:] == Structure.SHELL_DIMS`; the leading pad is zero; `len(SHELL_DIMS) == _bond_integral.MAX_SHELLS`; `SHELL_TYPE_IDS` is the 1-based id set; `shell_dim[f_id] == 7`; `_spin._F_SHELL_TYPE_ID` still equals `SHELL_TYPE_IDS[-1]` |
| `test_shell_local_starts_are_prefix_sums` | `SHELL_LOCAL_STARTS` is the exclusive prefix sum of `SHELL_DIMS`, and `SHELL_LOCAL_STARTS[-1] + SHELL_DIMS[-1] == len(AO_LABEL_TEMPLATE)` -- the relation that ties the offset table to the label table |
| `test_ao_label_template_matches_shell_dims` | `AO_LABEL_TEMPLATE` has 16 entries; slicing it at `SHELL_LOCAL_STARTS` yields groups of exactly 1, 3, 5 and 7; each group's labels carry the right `s`/`p`/`d`/`f` prefix; `AO_SHELL_TEMPLATE` agrees position by position |
| `test_channel_tables_are_consistent` | `_CHANNELS` has 40 unique entries; `_SIMPLE_CHANNELS` has 20; `_SIMPLE_TO_EXTENDED` has one index per simple channel and still equals the recomputed map; every simple channel exists in `_CHANNELS`; `N_SK_CHANNELS` and `SK_BLOCK_SIZE` still derive correctly; the S half of `_CHANNELS` repeats the H half's suffixes in order; `SK_CHANNEL_NAMES` still equals `_CHANNELS` |

**What `test_channel_tables_are_consistent` deliberately does not duplicate.**
`docs/SCRIPT-PY-PORT-INVENTORY.md` classifies `check_simple_canonical_channels` (formerly
`script.py:505`) as **covered** by
`tests/test_f_orbital_skf.py::test_f_orbital_skf_parser_and_spline_gate`, which asserts the
*data* property -- that a parsed simple file's 20 values land in the right canonical slots with
the rest zero-filled. That coverage stands and is not re-implemented. What is added here is the
*table* invariant the data property silently assumes: the lengths, the subset relation, the
derived-constant relations and the H/S mirror. A drifted table and a mis-scattered row are
different failures with different first suspects.

---

## How CLN-02 was met

**By documentation plus consistency tests, not by centralizing the tables.**
`05-VALIDATION.md`'s Documentation-Deliverable Verification table requires this to be stated
explicitly, so: no production file was modified by this plan, no single-source basis module was
created, and the eight definition sites above are all still where they were.

The reason is recorded, not improvised. `05-CONTEXT.md` -> *Deferred Ideas* lists:

> **Centralizing duplicated basis metadata** (`_CHANNELS`, `MAX_SHELLS`, `shell_dim`,
> `AO_LABEL_TEMPLATE` across `_bond_integral.py`, `Constants.py`, `Structure.py`) -- CLN-02
> may partially cover this; a full single-source basis module is larger and can stand alone.

Phase 5's purpose is regression safety. Moving four tables that ten modules index into, in the
same phase that already changes a radial-lookup interface (D-01), gates ~112 print calls (D-02),
deletes a shipped module (D-03) and audits 157 orbital-count sites (D-04), would add a refactor
whose only safety net is the suite those same four changes are already leaning on.

**What a future single-source module would have to preserve** -- recorded here so the deferred
work starts from the constraints rather than rediscovering them:

1. Both indexing conventions have live consumers. `shell_dim[shell_types]` is a tensor gather in
   the SCF hot path; `enumerate(SHELL_DIMS)` is a Python loop in the density builders. A single
   table serves one of them awkwardly.
2. `Constants.shell_dim` is a `torch.nn.Parameter` on a device with a dtype;
   `Structure.SHELL_DIMS` is a plain tuple usable at import time. Merging them means choosing
   which of those two properties to lose.
3. `_spin` cannot import `Structure` (`_spin.py:11-14`): `Structure` sits above it in the import
   graph. A centralized module must sit *below* `_spin`, `_forces`, `_bond_integral`,
   `Constants` and `Structure` -- i.e. it is a new leaf module, not a new home inside an
   existing one.
4. The f AO order is source-locked (`03-SOURCE-LOCK.md`). Any move must carry the citation with
   it, not just the tuple.

---

## How to check this map is still true

```
uv run pytest tests/test_support_documentation.py -q
```

Four tests, all of which read the live tables. They will not, however, notice a *new* definition
site appearing in a fourth module -- that is the map's honest limit, and it is why the sweep in
`tests/orbital_count_sweep.py` (which does enumerate sites mechanically) is the complementary
gate. If you add a basis table, add it here and add a relation for it there.

The line numbers above were measured on 2026-08-03. To re-derive them:

```
grep -n "SHELL_DIMS\|SHELL_LOCAL_STARTS\|SHELL_TYPE_IDS\|AO_LABEL_TEMPLATE\|AO_SHELL_TEMPLATE" src/dftorch/Structure.py
grep -n "shell_dim" src/dftorch/Constants.py
grep -n "_CHANNELS\|_SIMPLE_CHANNELS\|_SIMPLE_TO_EXTENDED\|MAX_SHELLS\|SK_BLOCK_SIZE\|N_SK_CHANNELS" src/dftorch/_bond_integral.py
```
