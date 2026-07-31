---
phase: 05-regression-safety-and-support-policy-cleanup
plan: 01
subsystem: slater-koster-radial-lookup
tags: [radial-grid, slater-koster, searchsorted, reg-06, reg-01, d-01, characterization-test, bit-identity]

# Dependency graph
requires:
  - phase: 05-regression-safety-and-support-policy-cleanup
    plan: 02
    provides: the 83-test green baseline and the three pinned simple-format energies that gate this refactor
  - phase: 04-scf-and-reference-simulation-validation
    provides: the Eu-N validation case, the run_with_float64 harness convention, and the ASCII-only test-output rule
provides:
  - "const.n_grid: per-pair tabulated radial grid length, shape (n_pairs,)"
  - "const.R_tensor rows strictly increasing across their full 1301-entry width, so torch.searchsorted against a row is defined behaviour"
  - "_h0ands._pair_knot_lookup(R_tensor, n_grid, pair_type, dR): the per-pair knot lookup, reusable by the batch path and by the stress path when those come into scope"
  - "tests/test_radial_grid.py: 13 characterization and per-pair tests with measured literals and stated provenance"
  - "write_mixed_grid_skf_pair: a synthetic mixed-grid SKF generator that plan 05-04 consumes for the REG-05 guard"
  - "tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md: the fixture-inventory record, including the correction that mio-1-1 already contains mixed grid LENGTHS"
affects: [05-04, 05-03, 07-derivatives, 08-forces-stress]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A characterization test is written and committed BEFORE the refactor it gates, with its post-refactor assertion as a strict xfail naming the task that flips it"
    - "Bit-identity is asserted two ways: new path vs old path with torch.equal in one process, AND new path vs literals recorded before the change existed"
    - "A per-row lookup loops over torch.unique(selector) with 1-D searches on masked subsets rather than batching a 2-D searchsorted, because the batched form materialises the full boundary tensor per query"
    - "An interface is widened additively with None-defaulted keyword parameters appended at the end of the signature, so no existing positional call site shifts"

key-files:
  created:
    - tests/test_radial_grid.py
    - tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md
    - .planning/phases/05-regression-safety-and-support-policy-cleanup/deferred-items.md
  modified:
    - src/dftorch/_bond_integral.py
    - src/dftorch/Constants.py
    - src/dftorch/_h0ands.py
    - src/dftorch/ESDriver.py
    - src/dftorch/script.py

key-decisions:
  - "R_orb is kept exported and unchanged rather than replaced. Four out-of-scope consumers read it (ESDriver batch path, _stress, _ml_sk, sedacs_interface) and two existing tests reproduce the searchsorted(const.R_orb, ...) idiom, so REG-03 is protected additively."
  - "R_tensor rows are made genuinely monotonic by continuing each pair's own arithmetic progression, rather than masking around the zero tail at the call site. Correct by construction beats correct by caller discipline."
  - "The per-pair lookup loops over torch.unique(pair_type). A batched 2-D searchsorted would materialise (n_neighbour_pairs, 1301) and is forbidden by threat T-05-04."
  - "R_tensor and n_grid are appended at the END of H0_and_S_vectorized's signature with None defaults, not inserted after coeffs_tensor, so no caller's positional arguments shift and sedacs_interface keeps working on the global path unmodified."
  - "script.py's unpacking IS updated, contradicting the plan. The plan's premise that nothing executes script.py is false: tests/test_f_orbital_skf.py loads it via load_validation_script() and two tests failed on the stale unpack."
  - "ESDriver.py:1527 is NOT updated. The plan names it as a caller to update but it calls H0_and_S_vectorized_batch, which the same plan says to leave alone."
  - "idx.max() for CH4 is pinned at the MEASURED 177, not the plan's predicted 178, per the plan's own instruction to record the measurement."

patterns-established:
  - "Pre-change checksum oracle: before a bit-identity refactor, record reductions over the affected tensors as exact float literals in a test, so the post-change assertion has an oracle that predates the change instead of comparing the new path against itself."
  - "Pristine-source A/B: to attribute an unexpected numeric delta, reconstruct src/ from the pre-change commit into a scratch tree and import it via PYTHONPATH, confirming the swap with a hasattr probe on a newly added attribute."

requirements-completed: [REG-06, REG-01]

metrics:
  duration: ~55 min
  tasks: 2
  files-created: 3
  files-modified: 5
  tests-added: 13
  suite-before: 83 passed / 0 failed
  suite-after: 96 passed / 0 failed / 0 errors / 0 skipped
  completed: 2026-07-31

status: complete
---

# Phase 5 Plan 01: Per-Pair Slater-Koster Radial Lookup Summary

Every element pair now finds its spline knot on its own row of `R_tensor` instead of on a
single global `R_orb` chosen as the longest grid in the directory, and the change is proven
bit-identical on both real parameter sets by literals recorded before it landed.

## What was built

### Task 1: characterization pins, committed before anything moved (`742f8cc`)

`tests/test_radial_grid.py` records today's grid geometry, knot arithmetic, effective cutoffs
and H0/S checksums as literals with stated provenance. Every number was measured by loading
the real fixtures, not copied from prose. Provenance runs through two chains: grid step and
length come from the SKF grid line parsed at `_bond_integral.py:597` and scaled at
`:713-718`, while `idx` and `dx` come from re-running the production expression at
`_h0ands.py:243-245` on the same inputs.

The measured values:

| Quantity | mio-1-1 (CH4) | f_orbital_data (Eu-N at 2.655 A) |
| --- | --- | --- |
| grid length | 550 | 483 |
| step (Angstrom) | `0.0105835442` | `0.0211670884` |
| last knot | `5.82094931` | `10.2237036972` |
| neighbour pairs | 20 | 2 |
| `idx` range | 98 .. 177 | 124, 124 |
| `dx` range | `0.002540771531197583` .. `0.009085704018195973` | `0.009113950000000148` |
| effective cutoff | `5.291772099999999` | `9.165349277199999` |

`test_r_tensor_rows_are_strictly_increasing` was written as a `strict=True` xfail naming Task
2 as its fix, so Task 2 flipped an existing assertion rather than inventing one.

`write_mixed_grid_skf_pair(path, elem_a, elem_b, *, step_bohr, npts)` synthesises a
mixed-grid SKF file at test time, and `tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md`
records why, including the honest statement that a synthetic fixture validates the
implementation's model of the defect rather than an observed failure.

### Task 2: the per-pair lookup, end to end (`1d5acce`)

`_bond_integral.get_skf_tensors` now fills each `R_tensor` row's tail by continuing that
pair's own arithmetic progression (`R[k] == (k+1) * step`, and the grid starts one step in, so
`R_orb_i[0]` IS the step), and returns a companion `n_grid` vector of per-pair tabulated
lengths. `Constants` registers `self.n_grid`. `_h0ands._pair_knot_lookup` loops over
`torch.unique(pair_type)`, running a 1-D `searchsorted` on each masked subset with the same
`right=True`, the same `- 1`, and the same clamp, now bounded by that pair's own `n_grid`.
`H0_and_S_vectorized` gained `R_tensor=None, n_grid=None` keyword parameters appended at the
end of its signature; supplying both selects the per-pair path, omitting either runs the
verbatim old expression. `ESDriver`'s single-system caller supplies both.

## Verification

`uv run pytest -q`: **96 tests, 0 failures, 0 errors, 0 skipped** (JUnit-XML confirmed).
Entering baseline was 83; 13 tests were added and none were lost. `tests/test_radial_grid.py`
reports zero `xfail` and zero `xpass`.

`tests/test_scf.py`, `tests/test_f_orbital_skf.py`, `tests/test_single_shot_energy.py` and
`tests/test_eu_n_scan.py` all pass unchanged, including the Eu-N scan's recorded 21-point
curve, which is an independent end-to-end check that the knot lookup did not move.

### The three pinned 05-02 energies, measured

| Case | Pinned | Measured after this plan | Bit-identical |
| --- | --- | --- | --- |
| `o2_mio_unrestricted` | `-9.031346598527405` | `-9.031346598527405` | yes |
| `o2_3ob_dftb3` | `-8.270081470047817` | `-8.270081470047817` | yes |
| `water8_mio_full` | `-110.65745489032126` | `-110.65745489032182` | no, delta `-5.542233338928781e-13` eV |

The `water8_mio_full` delta is **pre-existing and not caused by this plan**, established three
ways and written up in full in `deferred-items.md`:

1. With the per-pair lookup on and with it forced off (`const.n_grid = None`, which selects
   the verbatim pre-D-01 expression), the energy is the same to the last bit.
2. Both paths repeat exactly across runs.
3. A pristine `src/` reconstructed from `742f8cc` and imported via `PYTHONPATH` measures the
   same `-110.65745489032182`, with `hasattr(const, "n_grid") == False` confirming the swap.

`water8_mio_full` is the only one of the three that is 24 atoms with a periodic cell and
`COUL_METHOD = "FULL"`, so it is the only one whose energy passes through a large symmetric
eigensolve and a full Coulomb sum. The delta is 5e-15 relative, inside 05-02's own `1e-8` eV
band, and the tolerance was NOT loosened.

### Direct evidence the defect is closed

`test_mixed_step_grids_read_their_own_radius` builds a synthetic directory where H-H uses
0.1 Bohr and N-N uses 0.2 Bohr. The old global expression put N-N's knot at H-H's index,
roughly twice as far along N-N's own grid as it belongs; the per-pair lookup puts it at N-N's
own. The test asserts this as a radius, not only as an index: the per-pair knot sits below
the 2.0 A probe while the borrowed one sits above it, at more than 1.9x the correct radius.
That is the factor-of-two error decision D-01 describes, demonstrated rather than asserted.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 3 - Blocking] `script.py`'s tuple unpacking is NOT inert**

- **Found during:** Task 2, full-suite run after adding `n_grid` to the `get_skf_tensors`
  return tuple.
- **Issue:** The plan states "Do NOT edit `src/dftorch/script.py` ... nothing executes
  `script.py`, so a stale unpacking there is inert until it is removed." That premise is
  false. `tests/test_f_orbital_skf.py:26-35` defines `load_validation_script()`, which loads
  `script.py` as a module and calls into it. Two existing tests failed with
  `ValueError('too many values to unpack (expected 23)')`:
  `test_f_orbital_skf_parser_and_spline_gate` and
  `test_compact_only_f_orbital_skf_directory`.
- **Fix:** Added `_n_grid,` to the unpacking at `script.py:756`, a single-line insertion with
  a comment explaining why the file is not inert. Plan 05-03 deletes this file, and a deletion
  cleanly supersedes a one-line insertion, so the conflict the plan was avoiding does not
  arise.
- **Files modified:** `src/dftorch/script.py`
- **Commit:** `1d5acce`

**2. [Rule 3 - Blocking] `ESDriver.py:1527` is the BATCH caller and was left alone**

- **Found during:** Task 2, reading the two `ESDriver` call sites.
- **Issue:** The plan's action says "Update the callers at `ESDriver.py:266` and
  `ESDriver.py:1527` to pass `const.R_tensor` and `const.n_grid`", but the same action says
  "Leave `H0_and_S_vectorized_batch` and its lookup at lines 556-558 alone". Line 1527 is
  inside the `H0_and_S_vectorized_batch(...)` call, not the single-system one. Passing the new
  arguments there would require changing the batch signature, which the plan forbids.
- **Fix:** Only `ESDriver.py:266` was updated. The batched path keeps the global `R_orb`
  lookup, which is consistent with the plan's own scope statement and with the fact that the
  batch path already refuses f systems at `_h0ands.py:526`.
- **Files modified:** `src/dftorch/ESDriver.py`
- **Commit:** `1d5acce`

**3. [Rule 1 - Measurement] `idx.max()` for CH4 is 177, not the plan's predicted 178**

- **Found during:** Task 1.
- **Issue:** The plan predicted `idx.max() == 178`.
- **Fix:** Recorded the measured 177 with the arithmetic shown in the test docstring
  (`1.8875517915355329 / 0.0105835442 = 178.34`, floor 178, minus 1 gives 177). The plan
  explicitly instructs recording the measurement rather than forcing the prediction.
- **Files modified:** `tests/test_radial_grid.py`
- **Commit:** `742f8cc`

**4. [Rule 3 - Blocking] The Task 1 test harness does not use `script.py`'s machinery**

- **Found during:** Task 1.
- **Issue:** The plan says to copy the `Constants`/`Structure` construction helpers from
  `tests/test_f_orbital_skf.py`. Those helpers route through `load_validation_script()` and
  `validation.load_dftorch_module()`, which are the fake-package dynamic-import machinery that
  decision D-03 explicitly says not to carry over, and which plan 05-03 deletes.
- **Fix:** Copied `run_with_float64` verbatim (which is what the plan actually names) and
  built `Constants`/`Structure` through normal imports, following `tests/test_eu_n_scan.py`,
  the newer of the two templates. The new module has zero dependency on `script.py`.
- **Files modified:** `tests/test_radial_grid.py`
- **Commit:** `742f8cc`

**5. [Rule 2 - Missing critical coverage] The `f_orbital_data` grid-length assertion had to
change its source of truth**

- **Found during:** Task 2.
- **Issue:** `test_f_fixture_grids_are_all_identical` derived each pair's grid length by
  counting nonzero entries in its `R_tensor` row. Once Task 2 fills the tail, that count
  returns the full 1301-column width for every pair, so the assertion was measuring the wrong
  thing.
- **Fix:** Read the length from `const.n_grid`, which is the direct and now-authoritative
  source, with a comment explaining why the nonzero count is no longer valid.
- **Files modified:** `tests/test_radial_grid.py`
- **Commit:** `1d5acce`

### Additions beyond the plan

**6. [Rule 2 - Missing oracle] `test_ch4_h0_s_checksums_are_unchanged` (Task 1)**

The plan's Task 2 acceptance criterion requires H0/S "equal to the values recorded as a
fixture in Task 1's run", but Task 1's behavior block never records any. Without a value
recorded before the change existed, a post-change test can only compare the new path against
itself in one process, which proves nothing about REG-01. Six exact float64 reductions over
CH4's H0, S, dH0 and dS were recorded on the pre-D-01 path in Task 1, and Task 2's
bit-identity test asserts against them with `==`.

**7. [Rule 2 - Missing coverage] `test_mixed_step_grids_read_their_own_radius`**

The plan's only mixed-grid test covers same-step/different-length, which decision D-01 itself
calls BENIGN: the knots coincide, so that test can never demonstrate that the fix changes any
radius. A second synthetic case with differing STEP was added, which is the hazard D-01 exists
to close, and it asserts the divergence as a radius rather than only as an index.

**8. [Rule 2 - Missing coverage] `test_mio_c_h_p_has_genuinely_mixed_grid_lengths`**

The plan and its README task both state that no mixed-grid fixture exists in the repository.
Measured 2026-07-31, that is true of a mixed STEP but **false of a mixed LENGTH**:
`tests/data_skf_mio-1-1` grid lines are `0.02, 500` for most pairs, `0.02 600` for every Zn
pair, and `0.02, 619` for every P pair. A C/H/P system loads nine pairs carrying two real
lengths, 550 and 669, at one common step. Exercising the per-pair machinery against real data
rather than only against a generator closes the weakest part of the synthetic tests. The
correction is recorded in `README-MIXED-GRID-FIXTURE.md`.

### Not deviations

The `_ml_sk.py:437-450` docstring's Angstrom-versus-Bohr claim was read as the plan required
and left untouched: it is wrong, it is contradicted by `_ml_sk.py:522` eleven lines later, and
correcting it belongs to plan 05-04, which owns that file. No unit conversion was inserted
into the lookup.

No package was installed. No physics, spline, cutoff policy or re-fit was changed.

## Threat mitigations

| Threat | Disposition | Evidence |
| --- | --- | --- |
| T-05-01 one pair evaluated against another pair's grid | mitigated | `_pair_knot_lookup` indexes `R_tensor[pair_type]` per entry; `test_mixed_length_grids_use_their_own_rows` and `test_mixed_step_grids_read_their_own_radius` |
| T-05-02 the rewrite silently shifting knots on existing sets | mitigated | Task 1 literals recorded before the change; `test_ch4_h0_s_bit_identical_after_per_pair_lookup` asserts `torch.equal` against the old path AND `==` against the pre-change checksums |
| T-05-03 searchsorted against a non-monotonic row | mitigated | rows filled in `get_skf_tensors`; `test_r_tensor_rows_are_strictly_increasing` asserts across the full 1301-entry width for both real sets, and `test_mio_c_h_p_has_genuinely_mixed_grid_lengths` re-asserts it on a real mixed-length load |
| T-05-04 batched 2-D searchsorted allocating gigabytes | mitigated | `_pair_knot_lookup` loops over `torch.unique(pair_type)`; the reason is stated in its docstring so the loop is not "optimised" away later |
| T-05-05 path leakage in an exception | accepted, no action | no new exception is raised and no path is interpolated |
| T-05-SC package installs | not applicable | nothing was installed |

## Known Stubs

None. No stub, placeholder, skipped test or unrun `<verify>` was left behind. Every
`<automated>` verification in the plan was executed.

## Threat Flags

None. This plan adds no network I/O, no deserialisation of untrusted data, no subprocess, no
dynamic import and no privilege transition, and it introduces no new file-access or schema
surface at a trust boundary.

## Follow-ups for later plans

- **Plan 05-04** owns `_ml_sk.py` and should correct the `:437-450` docstring, and owns the
  REG-05 mixed-step guard. `write_mixed_grid_skf_pair` and `_write_mixed_skf_dir` in
  `tests/test_radial_grid.py` are ready to build the fixture it needs.
  `test_mio_c_h_p_has_genuinely_mixed_grid_lengths` doubles as a guard that the REG-05 refusal
  does not start rejecting `mio-1-1`, which is the benign case D-01 says must keep loading.
- **Plan 05-03** deletes `script.py`; the one-line unpacking added here disappears with it.
- The batched path (`H0_and_S_vectorized_batch`) and the stress path (`_stress.py:179`) still
  use the global `R_orb`. `_pair_knot_lookup` is a module-level function and is directly
  reusable when those come into scope.
- `deferred-items.md` item 1 records the `water8_mio_full` bit-reproducibility question for
  whoever owns the REG-02 baseline.

## Self-Check: PASSED

Files verified present on disk:

- `FOUND: tests/test_radial_grid.py`
- `FOUND: tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md`
- `FOUND: .planning/phases/05-regression-safety-and-support-policy-cleanup/deferred-items.md`
- `FOUND: src/dftorch/_bond_integral.py` (modified)
- `FOUND: src/dftorch/Constants.py` (modified)
- `FOUND: src/dftorch/_h0ands.py` (modified)
- `FOUND: src/dftorch/ESDriver.py` (modified)
- `FOUND: src/dftorch/script.py` (modified)

Commits verified in `git log`:

- `FOUND: 742f8cc` test(05-01): pin today's radial knot arithmetic before the per-pair rewrite
- `FOUND: 1d5acce` feat(05-01): route each element pair through its own radial grid row
