# Orbital Count Inventory

Every place in `src/dftorch/` where the code makes a decision from a hardcoded orbital
count, a hardcoded shell count, or a hardcoded per-shell basis-layout table, with a
disposition for each. Produced for phase 5 decision **D-04**, requirements **REG-04**
and **CLN-03**.

**Total: 141 sites.** Every one carries a disposition and **none is left as
`needs-action`**.

The audit closed at 157 sites in phase 5. Phase 6 plan 06-01 added the 158th,
`ESDriver._krylov_params_for_f_interim`, which is the first site in this table whose f
branch selects a *numerical setting* rather than a matrix block or a refusal. It is
listed the same way as every other: what it decides, what disposition it carries, and
which test proves it.

Phase 6 plan 06-02 then **removed 17**, the whole of `_coulomb_matrix.py`. Requirement
SCC-02 replaced that file's eight hand-enumerated `max_ang` pair masks with a single rule
that compares `max_ang` against a parameter rather than a literal, and deleted the guard
those masks sat behind. A site that no longer hardcodes a shell count is not a site. The
section below keeps the account of what was there in prose, because a resolved site is
still part of the audit's record even when it can no longer be a row.

## Why this is an audit with actions and not a document

A silently unhandled orbital-count site does not raise, does not produce a NaN, and does
not change the shape of anything. It returns a well-formed matrix with zeros where the f
values belong, and it survives every finiteness, shape and symmetry assertion in the
suite.

The worked example is not hypothetical. Commit `4dbffaa` fixed `read_skf_table`, which
inferred shell presence from `E_l != 0.0`. `mio-1-1`'s `H-H.skf` carries a rounding-noise
placeholder `Ep = 0.000039` Hartree, so **hydrogen was assigned four orbitals instead of
one**. The phantom p functions then received real Slater-Koster overlap from the C-H sp
channel rather than zeros, which made the overlap matrix `S` indefinite (eigenvalue
-0.159). `S` is a Gram matrix and must be positive definite, so Loewdin orthogonalisation
`S^(-1/2)` was undefined, one Hamiltonian eigenvalue reached 6.1e16 eV, the density matrix
came out non-symmetric, and `2*Tr(D S)` no longer equalled the electron count. CH4 could
never converge in any configuration. Two independent correct signals were sitting in the
file and both were ignored: the p occupation `fp = 0.0`, and the declared shell count `1`
in the grid line.

That site is **already dispositioned** and is not re-investigated here. It is cited
because it is the concrete proof that a document-only audit would have been insufficient:
nothing in the suite went red, and the defect was found only by asking what a specific
orbital count actually decided.

Regression cover: `tests/test_shell_count_parsing.py`.

## How this table was produced

```bash
uv run python tests/orbital_count_sweep.py
```

or, for the records themselves:

```bash
uv run python -c "import sys; sys.path.insert(0, 'tests'); \
from orbital_count_sweep import sweep_orbital_count_sites; \
[print(s.path, s.line, s.snippet, s.family) for s in sweep_orbital_count_sites('src/dftorch')]"
```

`tests/orbital_count_sweep.py` walks `src/dftorch/**/*.py`, blanks every comment and
string literal, collapses each newline-plus-indent run to a single space, and matches
three families of pattern. It returns one record per match. **This table has one row per
record**, and `tests/test_orbital_count_guards.py::test_inventory_covers_every_site`
matches the two in both directions, so a site present in the code but absent here fails
the suite, and a row here with no matching site fails it too. The audit cannot rot as
later phases add code.

### Why the sweep flattens newlines first

Measured on this tree, flattening and not flattening return the same 141 records, and
that is stated rather than hidden. The flattening is kept for two reasons that are not
cosmetic.

1. It makes the sweep invariant to formatting. `const.n_orb[TYPE[neighbor_I]] == 16` is
   40 characters of a 79-character budget, and any rename, added subscript or `black`
   reflow can split it across lines. A line-oriented sweep would then stop counting it
   with no signal. `test_flattening_is_load_bearing` proves this on a real reflowed
   snippet from `_h0ands.py`.
2. A **pair-level** sweep  --  one matching the whole `(a == x) & (b == y)` mask rather than
   each half  --  undercounts `_h0ands.py` by roughly half without flattening, because the
   single-system pair masks at `_h0ands.py:248-303` wrap after the `& (`. That is the
   measurement recorded at plan time: 22 flattened versus 12 line-oriented.

### What is excluded, and why

**Comments and string literals.** A docstring that says `n_orb == 16` and an exception
message that says `n_orb == 16` decide nothing. The sweep tokenises each file and blanks
every comment and string token before matching, preserving byte offsets so line numbers
stay exact. **24 prose mentions** are excluded on this tree
(`orbital_count_sweep.count_prose_mentions`). Excluding them is what lets every row below
carry an honest disposition instead of one that reads "this is a sentence".

Getting this wrong was measured, not assumed: on Python 3.12+ f-strings are no longer a
single `STRING` token, so before `FSTRING_START`/`MIDDLE`/`END` were added to the blanked
set, the three `_require_*` guards' own error messages were swept as routing decisions
while their real `counts == 16` tests went missing.

## The three families

| Family | What it matches | Values |
| --- | --- | --- |
| `orbital-count` | an orbital-count identifier compared against a basis size | 1, 4, 9, 16 |
| `shell-count` | a shell-count / angular-momentum identifier compared against a shell index | 1, 2, 3, 4 |
| `basis-layout-literal` | a bracketed literal spelling one of the canonical per-shell dimension or AO-offset tables |  --  |

`shell-count` is **not** in the plan's identifier list and is included deliberately. The
Phase 4 shell-resolved Coulomb defect lived in
the pair masks of `_coulomb_matrix.ewald_real_space_vectorized_sr`, which tested `max_ang`
and never mentioned `n_orb`. A sweep restricted to orbital-count names would have missed
the exact defect this project already had to fix. (Those masks are gone as of phase 6
SCC-02, and with them the file's rows; the family stays because the reason it exists does
not.) `basis-layout-literal` is included because CLN-03 names the hardcoded shell
offsets by hand, and because it is the family that found the only two real gaps in this audit --
the truncated `[0, 1, 3, 5]` tables in `_spin.py` and `_forces.py`, neither of which
mentions `n_orb` anywhere.

**Stated limitation.** `basis-layout-literal` matches an enumerated set of literal
spellings (`orbital_count_sweep.BASIS_LAYOUT_SEQUENCES`). A future phase writing a new
layout table with an unlisted spelling will not be swept. The enumeration is deliberately
narrow: the obvious generalisation  --  any bracketed run of small integers  --  matches
`permute(1, 0, 2, 3)` and dozens of unrelated expressions, and a row that cannot carry an
honest disposition is worse than no row. Add the spelling to that tuple when such a table
appears.

## The four dispositions

| Disposition | Meaning |
| --- | --- |
| `extended` | the site already handles 16 orbitals / the f shell. Evidence names the mask, branch or table entry that covers it. |
| `guarded` | reaching this site with a 16-orbital atom raises a named exception. Evidence names the exception class and the raising line. |
| `unreachable` | an f system cannot arrive here. Evidence states **how** that was established  --  an upstream guard and its line, or the import search that found no importer. Never asserted from reading alone. |
| `needs-action` | none of the above. **No row below carries this.** |

### What the audit actually found open

Six of the 157 rows were `needs-action` when the sweep was first run, and all six were
the same finding in two places. `_spin.get_h_spin`, `_spin.get_h_spin_diag` and
`_forces.forces_spin` each declare a local
`n_orb_per_shell = torch.tensor([0, 1, 3, 5])` -- the per-shell AO count table truncated
one entry short of f -- while `Constants.shell_dim` holds the correct
`[0, 1, 3, 5, 7]` a few modules away. The table is indexed by `shell_types`, which is
`4` for an f shell, so an f system indexes one past its end. `forces_spin` additionally
reconstructs 1/2/3-shell masks with no four-shell class, so a four-shell atom's spin
potential would stay exactly zero inside a correctly shaped force.

These were reachable by paths the existing guards do not cover:
`ESDriver.forward`'s `_require_closed_shell_f_system` does not run for `MD.py:745`,
`MD.py:794`, `MD.py:1103` or `_xl_tools.py:852`, each of which reaches `get_h_spin`,
`get_h_spin_diag` or `forces_spin` directly.

Both were **guarded, not widened**. Widening is a one-character edit in each case and
would have been wrong twice over: Phase 4 decision D-12 defers spin-polarized f
entirely, and `tests/f_orbital_data/` ships no `spinw.txt`, so the f spin coupling
constants the widened table would then multiply do not exist. Sizing the block correctly
and filling it from absent parameters is precisely the well-shaped-and-wrong outcome
D-04 exists to prevent.

Two new module-level guards were added, both following the
`ESDriver._require_f_derivatives` template exactly:
`_spin._require_no_f_spin_shells` (raising `FSpinPolarizationUnsupportedError`) and
`_forces._require_no_f_spin_forces` (raising `FDerivativeUnsupportedError`). **No new
exception class was defined**; both reuse the existing four. `forces_spin` takes the
derivative class rather than the spin one because it fails for a reason independent of
spin: it consumes `dS`, which is exactly zero in every f block.

One sweep blind spot is worth stating. `_require_no_f_spin_shells` tests
`shell_types == _F_SHELL_TYPE_ID`, a named constant rather than a literal, so the guard's
own comparison is not itself a swept row -- unlike the other four guards, whose
`counts == 16` tests are rows below. The two `_spin.py` rows are the truncated tables the
guard protects, which is what the audit is for; the guard's reachability is proven by
test rather than by a row.

`extended` was applied conservatively. D-04's rule is that a site needing capability this
phase does not have gets a refusal, not a widening  --  and if the reasoning is that a site
"could probably" handle 16, that is itself the signal to guard, because this phase has no
way to validate the extension. Every `guarded` row below is a site that could have been
widened by editing one literal and was not.

## Summary

| File | Sites | `extended` | `guarded` | `unreachable` |
| --- | ---: | ---: | ---: | ---: |
| `src/dftorch/_legacy/H0andS.py` | 54 | 0 | 0 | 54 |
| `src/dftorch/_h0ands.py` | 52 | 32 | 20 | 0 |
| `src/dftorch/_stress.py` | 20 | 0 | 2 | 18 |
| `src/dftorch/_forces.py` | 5 | 0 | 5 | 0 |
| `src/dftorch/Structure.py` | 3 | 3 | 0 | 0 |
| `src/dftorch/ESDriver.py` | 3 | 1 | 2 | 0 |
| `src/dftorch/_spin.py` | 2 | 0 | 2 | 0 |
| `src/dftorch/Constants.py` | 1 | 1 | 0 | 0 |
| `src/dftorch/_atomic_density_matrix.py` | 1 | 0 | 0 | 1 |
| **total** | **141** | **37** | **31** | **73** |

`src/dftorch/_coulomb_matrix.py` held 17 of these (1 `guarded`, 16 `unreachable`) until
phase 6 requirement SCC-02 removed every one; it is now absent from this table rather than
listed with a zero, because the sweep returns no record for it. The section below says
what was there.

### `src/dftorch/sedacs/`  --  checked, no sites

The sweep covers `src/dftorch/` as a whole, so `sedacs/` is included, and it returns
**zero** records there. This is recorded as a checked-and-empty result rather than left
unstated. `sedacs_interface.py` handles orbital counts only through
`ch_structure.n_orbitals_per_atom`, which it sums (`:471`), uses as
`repeat_interleave` counts (`:533-534`), and reads as a matrix dimension
(`:675`)  --  all basis-size-agnostic operations that carry no hardcoded 1/4/9/16. The
SEDACS interface's real f exposure is the shared-radial-grid read at
`sedacs_interface.py:523`, which is already dispositioned as `deferred` (PHY-05) in
`docs/RADIAL-GRID-CONSUMERS.md`.

### `src/dftorch/_slater_koster_pair.py`  --  checked, no code sites

All of its matches are prose: the two `F*Error` class docstrings, their message
constants, and the `pair_mask_*` parameter documentation. The f angular formulas
themselves are unconditional, so the module has no orbital-count decision of its own.

## Sites

### `src/dftorch/Constants.py` -- 1 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 107 | `[0, 1, 3, 5, 7]` | basis-layout-literal | `Constants.__init__` | AO count contributed by a shell of type id 0/1/2/3/4 (unused/s/p/d/f) | `extended` | Index 4 holds 7, so the table spans the f shell. This is the canonical copy; the truncated `[0, 1, 3, 5]` in `_spin.py` and `_forces.py` is the same table with that entry missing. | `test_canonical_shell_dim_table_spans_f` |

### `src/dftorch/ESDriver.py` -- 3 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 61 | `counts == 16` | orbital-count | `_require_f_derivatives` | whether the system holds an f atom and every derivative-consuming path must refuse | `guarded` | Raises `FDerivativeUnsupportedError` at `ESDriver.py:62`; called as the first statement of `ESDriver.calc_forces` (:1058) and `ESDriverBatch.calc_forces` (:2043). | `test_every_guarded_site_raises_for_f[ESDriver.calc_forces]` |
| 96 | `counts == 16` | orbital-count | `_require_closed_shell_f_system` | whether an f system was asked for a spin-polarized (UNRESTRICTED) calculation | `guarded` | Raises `FSpinPolarizationUnsupportedError` at `ESDriver.py:102`; called as the first statement of `ESDriver.forward` (:224), so it fires for `do_scf=False` too. | `test_every_guarded_site_raises_for_f[ESDriver.forward-unrestricted]` |
| 165 | `counts == 16` | orbital-count | `_krylov_params_for_f_interim` | whether the Krylov convergence accelerator must be switched off for this system, the f shell being what makes it diverge | `extended` | Added by phase 6 plan 06-01. Handles the 16-orbital case rather than refusing it: the branch below this comparison returns a shallow copy of the parameter dict with `KRYLOV_START` raised to `10**6`, above the `SCF_MAX_ITER` cap of 100, so the accelerator is never reached and the loop stays on Anderson/DIIS mixing. The f-free path returns the caller's dict unchanged, so no f-free number can move. Interim only -- the accelerator repair is deferred by the human ruling of 2026-08-04 in `06-RESEARCH.md`, and this function is the single place to delete when it lands. | `test_krylov_accelerator_is_disabled_for_the_f_system` |

### `src/dftorch/Structure.py` -- 3 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 11 | `(1, 3, 5, 7)` | basis-layout-literal | `SHELL_DIMS` | how many AOs each shell contributes (s, p, d, f) | `extended` | The tuple ends in 7, the f-shell dimension, so the f shell is sized correctly wherever this table is consulted. | `test_structure_layout_tables_span_f` |
| 12 | `(0, 1, 4, 9)` | basis-layout-literal | `SHELL_LOCAL_STARTS` | the local AO offset at which each shell begins inside an atom's block | `extended` | The tuple ends in 9, so the f block occupies local AOs 9-15 and the total is 16, matching the 16-entry `AO_LABEL_TEMPLATE` below it. | `test_structure_layout_tables_span_f` |
| 13 | `(1, 2, 3, 4)` | basis-layout-literal | `SHELL_TYPE_IDS` | the angular-momentum id assigned to each shell | `extended` | Carries the f id 4, which is what `Constants.shell_dim` and `shell_types` are indexed by. | `test_structure_layout_tables_span_f` |

### `src/dftorch/_atomic_density_matrix.py` -- 1 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 161 | `(4, 5, 6, 7, 8)` | basis-layout-literal | `atomic_density_matrix` | which local AO offsets receive the d occupation; there is no matching f run (9-15) | `unreachable` | Import search `grep -rn "_atomic_density_matrix\|atomic_density_matrix(" src tests` finds only `_legacy/H0andS.py:9` and `:899` (itself unreachable, below) plus a non-calling smoke import in `tests/test_import.py`. The live `D0` is built by `Structure._atomic_density_matrix_from_shells` (`Structure.py:160`, called at :443), which is driven by the shell tables rather than by fixed offsets. The legacy call also passes five positional arguments to a six-parameter signature, so it could not execute. | `test_atomic_density_matrix_has_no_production_importer` |

### `src/dftorch/_coulomb_matrix.py`  --  0 sites (17 resolved by SCC-02)

This file carried 17 rows through phase 5 and carries none now. The rows are gone rather
than re-dispositioned because
`tests/test_orbital_count_guards.py::test_inventory_covers_every_site` matches sites and
rows as multisets in **both** directions: a row with no matching site fails the suite just
as a site with no row does. Keeping a resolved row would have turned a green audit red.
What the rows recorded is kept here as prose instead, because the point of accounting for
a site does not end when the site does.

What was there:

* one `guarded` row, `counts == 16` in `_require_no_f_shell_resolved_coulomb`, which
  raised the shell-resolved refusal as the first statement of
  `ewald_real_space_vectorized_sr`;
* sixteen `unreachable` rows, the eight `max_ang_I` / `max_ang_J` pair masks that selected
  which shell-pair block a neighbour pair contributed to. They tested `max_ang` against 1,
  2 and 3 only  --  there was deliberately no `max_ang == 4` class  --  and they were marked
  `unreachable` on the strength of the guard above firing before them.

What replaced them. Requirement **SCC-02** in phase 6 built the seven missing f blocks
(s-f, f-s, p-f, f-p, d-f, f-d, f-f) and replaced the eight hand-enumerated masks with one
uniform rule, `_shell_pair_mask(max_ang_I, max_ang_J, shell_i, shell_j)`, returning
`(max_ang_I > shell_i) & (max_ang_J > shell_j)`. That rule compares `max_ang` against a
*parameter*, not against a literal 1/2/3/4, so it is correctly not a swept site: there is
no hardcoded shell count left in the file to disposition. The refusal is retired and the
exception class is kept, with a retirement note, so the f taxonomy stays at four names.

Cover: `tests/test_shell_resolved_coulomb_f.py` (seven tests, including the derived
identity against the per-atom builder and the watch that no production module raises the
retired refusal).

### `src/dftorch/_forces.py` -- 5 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 48 | `counts == 16` | orbital-count | `_require_no_f_spin_forces` | whether the system holds an f atom and the spin force path must refuse | `guarded` | Raises `FDerivativeUnsupportedError` and is called as the first statement of `forces_spin`, so it covers the atom-resolved branch as well as the shell-resolved one. Separate from `ESDriver._require_f_derivatives` because `MD.py:1103` reaches `forces_spin` without passing through `ESDriver.calc_forces`. | `test_every_guarded_site_raises_for_f[forces_spin]` |
| 336 | `[0, 1, 3, 5]` | basis-layout-literal | `forces_spin` | AO count per shell when expanding the shell-resolved spin potential to AOs | `guarded` | Truncated at d: `shell_types` is 4 for an f shell, which indexes one past the end of this four-entry table. `_require_no_f_spin_forces` is now the first statement of `forces_spin` and raises `FDerivativeUnsupportedError` before it is reached. | `test_every_guarded_site_raises_for_f[forces_spin]` |
| 343 | `n_shells_per_atom == 1` | shell-count | `forces_spin` | which shell-resolved W sub-block each atom uses; 1, 2 and 3 shells only | `guarded` | A four-shell (f) atom matches none of these masks, so its `mu_sr` entries would stay exactly zero inside a correctly shaped spin force. `_require_no_f_spin_forces` is the first statement of `forces_spin` and raises `FDerivativeUnsupportedError` before any of these masks is built. | `test_every_guarded_site_raises_for_f[forces_spin]` |
| 351 | `n_shells_per_atom == 2` | shell-count | `forces_spin` | which shell-resolved W sub-block each atom uses; 1, 2 and 3 shells only | `guarded` | A four-shell (f) atom matches none of these masks, so its `mu_sr` entries would stay exactly zero inside a correctly shaped spin force. `_require_no_f_spin_forces` is the first statement of `forces_spin` and raises `FDerivativeUnsupportedError` before any of these masks is built. | `test_every_guarded_site_raises_for_f[forces_spin]` |
| 359 | `n_shells_per_atom == 3` | shell-count | `forces_spin` | which shell-resolved W sub-block each atom uses; 1, 2 and 3 shells only | `guarded` | A four-shell (f) atom matches none of these masks, so its `mu_sr` entries would stay exactly zero inside a correctly shaped spin force. `_require_no_f_spin_forces` is the first statement of `forces_spin` and raises `FDerivativeUnsupportedError` before any of these masks is built. | `test_every_guarded_site_raises_for_f[forces_spin]` |

### `src/dftorch/_h0ands.py` -- 52 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 247 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 248 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 250 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 251 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 253 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 254 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 256 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 257 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 260 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 261 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 263 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 264 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 266 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 267 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 269 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 270 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 272 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 273 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the nine 1/4/9 pair classes a neighbour pair belongs to | `extended` | The same function defines seven 16-orbital classes at :283-303 (HZ, ZH, XZ, ZX, YZ, ZY, ZZ) and passes them to `Slater_Koster_Pair_SKF_vectorized`, so an f pair is routed rather than dropped by these nine. This is the Phase 3 fix; before it existed every f pair fell through all nine. | `test_f_pair_reaches_single_system_sk_assembly` |
| 282 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 283 | `const.n_orb[TYPE[neighbor_J]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 285 | `const.n_orb[TYPE[neighbor_I]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 286 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 288 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 289 | `const.n_orb[TYPE[neighbor_J]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 291 | `const.n_orb[TYPE[neighbor_I]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 292 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 294 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 295 | `const.n_orb[TYPE[neighbor_J]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 297 | `const.n_orb[TYPE[neighbor_I]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 298 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 300 | `const.n_orb[TYPE[neighbor_I]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 301 | `const.n_orb[TYPE[neighbor_J]] == 16` | orbital-count | `H0_and_S_vectorized` | which of the seven 16-orbital (spdf) pair classes a neighbour pair belongs to | `extended` | These masks are the f routing itself, and the f angular formulas they route into are implemented in `_slater_koster_pair`. | `test_f_pair_reaches_single_system_sk_assembly` |
| 621 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 621 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 622 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 622 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 623 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 623 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 624 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 624 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 625 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 625 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 626 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 626 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 627 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 627 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 628 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 628 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 629 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 629 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to | `guarded` | Built at :621-629 but never consumed for an f system: a `NotImplementedError` is raised at `_h0ands.py:636`, before the masks reach `Slater_Koster_Pair_SKF_vectorized_batch` at :674. Batched f H0/S is PHY-04, Phase 8.1. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 635 | `norb_I == 16` | orbital-count | `H0_and_S_vectorized_batch` | whether any batched pair contains a 16-orbital atom and the batched path must refuse | `guarded` | This test is the refusal: it raises `FAngularFormulaSourceError` at `_h0ands.py:637` rather than returning H0/S with silently omitted f blocks. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |
| 635 | `norb_J == 16` | orbital-count | `H0_and_S_vectorized_batch` | whether any batched pair contains a 16-orbital atom and the batched path must refuse | `guarded` | This test is the refusal: it raises `FAngularFormulaSourceError` at `_h0ands.py:637` rather than returning H0/S with silently omitted f blocks. | `test_every_guarded_site_raises_for_f[H0_and_S_vectorized_batch]` |

### `src/dftorch/_legacy/H0andS.py` -- 54 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 151 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 152 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 154 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 155 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 157 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 158 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 160 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 161 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 164 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 165 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 167 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 168 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 170 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 171 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 173 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 174 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 176 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 177 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | `grep -rn "H0andS\|_legacy" src tests pyproject.toml` outside the directory returns exactly one hit, `pyproject.toml:61`, which *excludes* the directory from ruff. `src/dftorch/_legacy/` has no `__init__.py`, so it is not a package and nothing can import from it. | `test_legacy_h0ands_has_no_importer` |
| 434 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 434 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 435 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 435 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 436 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 436 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 437 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 437 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 438 | `norb_I == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 438 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 439 | `norb_I == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 439 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 440 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 440 | `norb_J == 1` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 441 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 441 | `norb_J == 4` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 442 | `norb_I == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 442 | `norb_J == 9` | orbital-count | `H0_and_S_vectorized_batch (legacy)` | which of the nine 1/4/9 batched pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory, and no `__init__.py` to import through. | `test_legacy_h0ands_has_no_importer` |
| 765 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 766 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 768 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 769 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 771 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 772 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 774 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 775 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 778 | `const.n_orb[TYPE[neighbor_I]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 779 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 781 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 782 | `const.n_orb[TYPE[neighbor_J]] == 1` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 784 | `const.n_orb[TYPE[neighbor_I]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 785 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 787 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 788 | `const.n_orb[TYPE[neighbor_J]] == 4` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 790 | `const.n_orb[TYPE[neighbor_I]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |
| 791 | `const.n_orb[TYPE[neighbor_J]] == 9` | orbital-count | `H0_and_S_vectorized_OLD_FOR_POLY (legacy)` | which of the nine 1/4/9 pair classes a neighbour pair belongs to; no 16-orbital class exists | `unreachable` | Same import search as above: no importer outside the directory and no `__init__.py` to import through. This function is additionally unreferenced inside its own module. | `test_legacy_h0ands_has_no_importer` |

### `src/dftorch/_spin.py` -- 2 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 186 | `[0, 1, 3, 5]` | basis-layout-literal | `get_h_spin` | AO count per shell when expanding the shell spin potential to the full AO basis | `guarded` | Truncated at d while `Constants.shell_dim` carries the f entry. `_require_no_f_spin_shells` is now the first statement of `get_h_spin` and raises `FSpinPolarizationUnsupportedError` for any `shell_types` entry equal to 4. | `test_every_guarded_site_raises_for_f[get_h_spin]` |
| 229 | `[0, 1, 3, 5]` | basis-layout-literal | `get_h_spin_diag` | AO count per shell when expanding the shell spin potential to the AO diagonal | `guarded` | Same truncated table as `get_h_spin`; `_require_no_f_spin_shells` is now the first statement of `get_h_spin_diag` too. This entry point is reached from `_xl_tools.py:852`, which the `ESDriver.forward` guard does not cover. | `test_every_guarded_site_raises_for_f[get_h_spin_diag]` |

### `src/dftorch/_stress.py` -- 20 sites

| Line | Site | Family | Owner | What it decides | Disposition | Evidence | Test |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 197 | `nI == 16` | orbital-count | `_pair_grad_from_sk` | whether any pair contains a 16-orbital atom and the analytical stress must refuse | `guarded` | This test is the refusal: it raises `FDerivativeUnsupportedError` at `_stress.py:198` rather than returning an f-incomplete stress tensor. f stress is PHY-02, Phase 8. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 197 | `nJ == 16` | orbital-count | `_pair_grad_from_sk` | whether any pair contains a 16-orbital atom and the analytical stress must refuse | `guarded` | This test is the refusal: it raises `FDerivativeUnsupportedError` at `_stress.py:198` rather than returning an f-incomplete stress tensor. f stress is PHY-02, Phase 8. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 205 | `nI == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 205 | `nJ == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 206 | `nI == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 206 | `nJ == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 207 | `nI == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 207 | `nJ == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 208 | `nI == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 208 | `nJ == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 209 | `nI == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 209 | `nJ == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 210 | `nI == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 210 | `nJ == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 211 | `nI == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 211 | `nJ == 1` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 212 | `nI == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 212 | `nJ == 4` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 213 | `nI == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |
| 213 | `nJ == 9` | orbital-count | `_pair_grad_from_sk` | which of the nine 1/4/9 pair classes a pair belongs to when reconstructing stress masks | `unreachable` | The refusal at `_stress.py:197-202` is a few lines earlier in the same straight-line function body with no branch between, so an f atom raises before these masks are built. | `test_every_guarded_site_raises_for_f[_pair_grad_from_sk]` |

---

## Out of D-04's row set, resolved anyway

`_coulomb_matrix_batch.ewald_k_space_vectorized`'s `do_vec=True` branch was handed over
by plan 05-05 as a printed apology: it printed "vectorized k-space is not implemented for
batched data" and then `return`ed. The function is annotated
`-> tuple[torch.Tensor, torch.Tensor]`, and its only in-package caller
(`ewald_real_space_vectorized_batch`) unpacks two values, so the bare `return` surfaced
as `cannot unpack non-iterable NoneType object` one frame away with the explanation
already scrolled past. That is not a silently degraded result; it is a loud failure with
the diagnosis detached from it.

**The determination: this is not an orbital-count site.** Nothing in that branch depends
on `n_orb`, `max_ang` or the basis layout. It is a batched k-space Ewald gap, requirement
**PHY-04**, Phase 8.1. It is recorded here and classified out of D-04's row set rather
than forced into it, and it carries no row above because the sweep finds nothing there.

It was resolved anyway, because the printed-apology pattern is exactly what D-04 exists
to remove: the branch now raises `NotImplementedError` naming the flag, the working
alternative (`do_vec=False`, which is the default and what every caller uses) and the
deferring requirement. **No new exception class was defined** -- the four
`F*UnsupportedError` classes are the f support policy and none of them covers a batched
Ewald sum, so the built-in is the honest choice. No caller in this package passes
`do_vec=True`, so the change cannot alter any result the suite covers.

Covered by `tests/test_orbital_count_guards.py::test_batched_kspace_site_has_a_disposition`.

## Related inventories

- `docs/RADIAL-GRID-CONSUMERS.md` -- every reader of the shared `R_orb` grid (D-01, REG-05/06).
- `docs/LIBRARY-OUTPUT-INVENTORY.md` -- every `print` in the package (D-02).
- `docs/SCRIPT-PY-PORT-INVENTORY.md` -- what was preserved before `script.py` was deleted (D-03).

*Phase 5, plan 05-06. Decision D-04; requirements REG-04, CLN-03.*
