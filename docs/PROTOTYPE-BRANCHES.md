# Prototype Branches and Temporary Code Paths

**Requirement:** CLN-01 -- "Temporary prototype branches are documented where they are
introduced."
**Written:** plan 05-07, measured 2026-08-03.

**This is a prototype branch.** Everything recorded below is deliberate temporary structure --
a fallback kept alive so an out-of-scope caller does not break, a table left truncated behind a
refusal rather than widened past the parameters that would fill it, an interface widened
additively rather than replaced. None of it is an oversight, and none of it should be tidied
away by someone who does not first read what it is holding up.

`05-VALIDATION.md` is blunt about what CLN-01 can be checked by: "section-presence check plus a
reviewer reading it. No automation beyond 'the section exists and names files that exist.'"
That is honest and this document does not pretend otherwise. What it can do is make every claim
falsifiable by hand: every path below names a file and a line, and every deferral names the
requirement id and the phase.

---

## 1. The git branch

| | |
|---|---|
| Prototype branch | **`f_orbital_initial`** (checked out; `origin/f_orbital_initial` tracks it) |
| Relative to `main` | **51 ahead, 0 behind.** Fork point `36da20e` ("fix: improved backprop stability for param fit"). All f-orbital work, Phases 1-5, lives in those 51 commits |
| Relative to `origin/dev` | 232 ahead, 1 behind, common ancestor `088377a` ("Initial commit"). `origin/dev` is a separate line of development, not an ancestor of this work |
| Local vs remote | local is 15 commits ahead of `origin/f_orbital_initial` at the time of writing |

Re-derive:

```
git rev-list --count main..f_orbital_initial
git rev-list --count f_orbital_initial..main
git log --oneline -1 $(git merge-base main f_orbital_initial)
```

`main` is strictly behind, so the branch has never been merged back. Nothing in this document
assumes it will be merged as one commit or as 51; that decision belongs to whoever ships it.

---

## 2. Temporary code paths, with what each is waiting on

### 2.1 `R_tensor` / `n_grid` as `None`-defaulted keyword parameters

| | |
|---|---|
| Where | `_h0ands.H0_and_S_vectorized`, the two parameters appended at the **end** of the signature; the branch itself at `_h0ands.py:350-355` |
| What it is | When either argument is `None`, the function falls back to the pre-D-01 global expression `searchsorted(R_orb, dR_mskd)` -- every pair measured against the longest grid in the directory |
| Why it exists | `sedacs/sedacs_interface.py:507` calls this function without the new arguments and is out of scope for Phase 5. Appending the parameters at the end rather than inserting them means no existing positional call site shifts (REG-03). The comment at `_h0ands.py:346-349` says so in the source |
| Removed by | **PHY-04** (Phase 8.1) and **PHY-05** (v2) -- the last two callers still on the global path. When `docs/RADIAL-GRID-CONSUMERS.md` rows 3-7 are all converted, the fallback branch and both `None` defaults can go |

**Read the correction in `docs/RADIAL-GRID-CONSUMERS.md` before touching this.** A
`None`-defaulted fallback that silently produces a plausible answer is exactly what let REG-06
go inert for eleven commits: commit `56091af` deleted `ESDriver`'s two keyword arguments as an
unrelated side effect, the driver silently took the fallback, and **all thirteen of plan 05-01's
tests stayed green** because every one of them calls the callee directly. Plan 05-04 restored
the wiring (`2d508b1`) and added
`tests/test_radial_grid.py::test_esdriver_supplies_the_per_pair_grid_arguments`, which watches
the **wiring** rather than the lookup. That test is the thing keeping this fallback honest; do
not delete it while the fallback exists.

### 2.2 `R_orb` kept exported rather than replaced

| | |
|---|---|
| Where | `Constants.py:202-207` (`self.R_orb`, with the comment stating it is the longest grid, kept unchanged); produced in `_bond_integral.get_skf_tensors` |
| What it is | The single global radial grid the per-pair lookup was supposed to replace, retained **additively** so that four consumers outside Phase 5's scope keep working |
| Why it exists | Decision D-01 left the choice to planning and plan 05-01 chose additive: replacing it would have changed an interface the ML-SK, stress, batch and SEDACS paths all read, inside the phase whose purpose is regression safety |
| Removed by | its four remaining readers converting: **PHY-04** (Phase 8.1, `ESDriver.py:1529` batch call), **PHY-02** (Phase 8, `_stress.py:183`), **PHY-06** (v2, `_ml_sk.py:489-500`), **PHY-05** (v2, `sedacs_interface.py:523`) |

### 2.3 The five `deferred` rows in `docs/RADIAL-GRID-CONSUMERS.md`

Each is a consumer still reading the global grid, with its own disposition and reason. Not
repeated here -- the table is the record:

| Row | Site | Waiting on |
|---|---|---|
| 3 | `ESDriver.py:1529`, the batched H0/S call | PHY-04, Phase 8.1 |
| 4 | `_h0ands.py:666-668`, the batched knot lookup | PHY-04, Phase 8.1 |
| 5 | `_stress.py:183` | PHY-02, Phase 8 |
| 6 | `_ml_sk.py:489-500`, `build_pair_type_rcut` | PHY-06, v2 |
| 7 | `sedacs_interface.py:523` | PHY-05, v2 |

REG-05's mixed-step guard (`_bond_integral.py:1088`) bounds the consequence of all five to the
benign same-step case, which is what makes deferring them defensible rather than merely
convenient. A mixed-*length* directory is legitimate, loads, and still leaves these five
clamping against the longest grid.

### 2.4 The 32 `guarded` rows in `docs/ORBITAL-COUNT-INVENTORY.md`

Plan 05-06 swept 157 hardcoded orbital-count, shell-count and basis-layout sites and
dispositioned every one: 36 `extended`, 32 `guarded`, 89 `unreachable`. **Each `guarded` row is
a temporary path** -- a site that could be widened to 16 the day the capability behind it
exists, and refuses until then. Eight distinct guards cover them, and each is driven by a real
Eu-N f system in `tests/test_orbital_count_guards.py::test_every_guarded_site_raises_for_f`:

| Guard case | Waiting on |
|---|---|
| `H0_and_S_vectorized_batch` | PHY-04, Phase 8.1 |
| `ESDriver.calc_forces` | DRV-01 then DRV-02, Phase 7 |
| `_pair_grad_from_sk` (stress) | PHY-02, Phase 8 |
| `forces_spin` | DRV-01, Phase 7 (it fails on `dS`, independently of spin) |
| `get_h_spin`, `get_h_spin_diag` | SPN-01, v2 |
| `ESDriver.forward-unrestricted` | SPN-01, v2 |
| `ewald_real_space_vectorized_sr` | SCC-02, Phase 6 |

The inventory's completeness is enforced bidirectionally by
`test_inventory_covers_every_site`, so a site added by a later phase without a row fails the
suite. That is the one documentation gate in Phase 5 with a real automatable check
(`05-VALIDATION.md`), and it is what makes this section trustworthy rather than a snapshot.

### 2.5 Three truncated copies of the per-shell AO count table, guarded rather than widened

| | |
|---|---|
| Where | `_spin.py:186`, `_spin.py:229`, `_forces.py:336` -- each declares `n_orb_per_shell = torch.tensor([0, 1, 3, 5])` |
| What it is | `Constants.shell_dim` is `[0, 1, 3, 5, 7]`. These three local copies stop one entry short of f, and the table is indexed by `shell_types`, which is `4` for an f shell -- so an f system indexes one past the end |
| Why it is still truncated | Widening it is a **one-character change** and would be wrong twice over: decision **D-12** defers spin-polarized f entirely, and `tests/f_orbital_data/` ships no `spinw.txt`, so the f spin coupling constants the widened table would multiply **do not exist**. A correctly sized f block filled from absent parameters is exactly the well-shaped-and-wrong outcome D-04 exists to prevent |
| Guarded by | `_spin._require_no_f_spin_shells` (raises at `_spin.py:45`) and `_forces._require_no_f_spin_forces` (raises at `_forces.py:49`), both added by plan 05-06 (`1aade8c`) |
| Removed by | **SPN-01** (v2) for the two `_spin` copies; **DRV-01** (Phase 7) for `_forces.forces_spin`, which fails for a reason independent of spin -- it consumes `dS`, exactly zero in every f block |

These sites are reachable past `ESDriver.forward`'s guard: `MD.py:745`, `MD.py:794`,
`MD.py:1103` and `_xl_tools.py:852` each reach `get_h_spin`, `get_h_spin_diag` or `forces_spin`
directly.

### 2.6 The two `F_ANGULAR_*_AVAILABLE` capability flags

| | |
|---|---|
| Where | `_slater_koster_pair.py:170` (`F_ANGULAR_FORMULAS_AVAILABLE = False`), reassigned `True` at `:785`; `_slater_koster_pair.py:256` (`F_ANGULAR_DERIVATIVES_AVAILABLE = False`) |
| What they are | Module-level booleans whose `False` value is the entire reason the corresponding guards fire. `_require_f_formula_source` (`:209`) and `_require_f_derivatives` (`ESDriver.py:48`) both return early when their flag is `True` |
| Why the declare-then-reassign | `F_ANGULAR_FORMULAS_AVAILABLE` is declared `False` beside its exception class, then set `True` at `:785` once the source-locked tables above it are transcribed. The guard has to be defined before the tables it protects; the flag is what lets the file be read top to bottom |
| Removed by | **DRV-01** (Phase 7) flips `F_ANGULAR_DERIVATIVES_AVAILABLE`, which is the single switch that disarms every derivative guard at once. `tests/test_f_orbital_skf.py:1210-1211` pins both flags today (`True` and `False` respectively), so flipping one without intent fails |

### 2.7 The noisy default of `VERBOSE_LIBRARY_OUTPUT`

| | |
|---|---|
| Where | `Constants.py:101`, `self.verbose_output = library_output_enabled(dftorch_params)`, defaulting to `True` |
| What it is | Decision **D-02** gated the library's status chatter behind a flag but deliberately left the default at today's noisy behaviour. Measured on CH4 + mio-1-1: 518 characters over 30 lines by default, 13 characters over 1 line with the flag off, with `e_tot` identical |
| Why it exists | REG-02 and REG-03 require existing simple-format callers and the tutorial notebook to be unaffected. A quiet default would itself be the behaviour change the phase exists to prevent. The accepted consequence, stated plainly in the 05-05 summary: this phase makes the noise **suppressible**, it does not remove it |
| Removed by | **No requirement id.** This one is governed by a *Deferred Idea* in `05-CONTEXT.md` -- "Flipping the verbose default to quiet ... Reconsider once the tutorial-notebook regression baseline is locked" -- not by a numbered requirement, and it is recorded that way rather than given an invented id. The baseline it waits on is `tests/test_simple_format_regression.py` (plan 05-02), which now exists |

A second Deferred Idea sits behind it: migrating to a logging module with levels and per-module
handlers, which D-02 considered and rejected for this phase.

### 2.8 `_spin._F_SHELL_TYPE_ID`, a literal copy of a `Structure` constant

| | |
|---|---|
| Where | `_spin.py:15`, `_F_SHELL_TYPE_ID: int = 4`, with the reason in the comment at `_spin.py:11-14` |
| What it is | A hand-copied duplicate of `Structure.SHELL_TYPE_IDS[-1]`. It is a literal because `Structure` sits **above** `_spin` in the import graph, so the value cannot be imported without a cycle |
| Removed by | **No requirement id.** The CLN-02 *Deferred Idea* -- a single-source basis module -- would fix it, because such a module would sit below `_spin` in the import graph. `docs/BASIS-METADATA-MAP.md` records that import-order constraint as one of the four things the deferred refactor must preserve. Until then, `tests/test_support_documentation.py::test_shell_dim_and_shell_dims_agree` asserts the copy still matches its source |

---

## 3. Dead code retained, not a deferral

`src/dftorch/_legacy/H0andS.py` is **54 of the 157 swept orbital-count sites**, all
dispositioned `unreachable`. The property was established, not assumed: no import statement
anywhere in `src/` references it, and the directory has no `__init__.py`, so it is not an
importable package. `tests/test_orbital_count_guards.py::test_legacy_h0ands_has_no_importer`
holds that claim, matching import statements only -- a substring search would flag the audit
that established the property.

`src/dftorch/_atomic_density_matrix.py` is the same shape: its only production importer is
`_legacy/H0andS.py`, itself unreachable, and it is additionally called there with five
positional arguments against a six-parameter signature. The live `D0` comes from
`Structure._atomic_density_matrix_from_shells`.

**No requirement removes either.** They are dead code, not deferred capability, and are listed
here so nobody mistakes an `unreachable` row for a path something is waiting on.

Two prototype-era files that no longer exist, recorded so a reader of an older document does
not go looking:

- `src/dftorch/script.py` -- 1656 lines of pre-pytest validation that shipped inside the
  runtime package. Its one genuine asset, an independently re-parsing SKF header oracle, was
  ported to `tests/skf_header_oracle.py` / `tests/test_skf_metadata_oracle.py` before deletion
  (decision D-03, plan 05-03). `docs/SCRIPT-PY-PORT-INVENTORY.md` dispositions every function.
- `Constants.ConstantsTest` -- a 911-line dead parallel constants table that bypassed
  `get_skf_tensors()` while claiming to be a test oracle. Removed by `ea3915a` (CLN-05).

---

## 4. Temporary structure that is not code

### 4.1 REG-02 is satisfied in substance, not in form

`docs/REG-02-NOTEBOOK-BASELINE.md` records the developer's decision at a blocking checkpoint:
the tutorial notebook's runnable simple-format calculations were **extracted into pytest**
rather than the notebook being executed. Three verified obstacles made execution impossible --
`experiments/COORD.pdb` is missing from the repository, one cell hardcodes `device = "cuda"`
with no guard, and no notebook execution tooling is installed.

The temporary structure this leaves: **the notebook is never executed**, so a change breaking
only its own call spelling would not be caught, and cell 5's PBC + PME + MD path is entirely
uncovered. `experiments/COORD.pdb`'s absence is a **real repository defect**, not merely a
testing inconvenience -- the tutorial is broken for any new user from a clean checkout.
Repairing it means inventing a geometry, which changes what the tutorial demonstrates, so it
was left for whoever owns the tutorial.

### 4.2 Two open numeric items in `deferred-items.md`

Environment-level drift against literals recorded on an earlier machine state, both inside
their own tolerance bands, neither caused by this milestone's changes: `water8_mio_full`'s
energy (5e-13 eV) and `CH4_DH0_ABS_SUM` (3.4e-13). Owners are named in that file. **Do not
widen a band to make either go away** -- both bands are already correct and already pass.

---

## 5. How to check this document has not rotted

Every claim above is a file, a line or a git command. The ones with a machine gate:

```
uv run pytest tests/test_orbital_count_guards.py -q     # sections 2.4, 2.5, 3
uv run pytest tests/test_radial_grid.py -q              # sections 2.1, 2.2, 2.3
uv run pytest tests/test_support_documentation.py -q    # section 2.8
uv run pytest tests/test_f_orbital_skf.py -q            # section 2.6
```

The ones with no gate and therefore worth re-reading by hand: section 1 (branch topology moves
with every commit), section 2.7 (a default value), and section 4.
