---
phase: 05-regression-safety-and-support-policy-cleanup
plan: 04
subsystem: skf-loader-guard-and-grid-consumer-audit
tags: [radial-grid, reg-05, reg-06, reg-03, d-01, guard, public-api, silent-revert]

# Dependency graph
requires:
  - phase: 05-regression-safety-and-support-policy-cleanup
    plan: 01
    provides: the per-pair knot lookup, write_mixed_grid_skf_pair, _write_mixed_skf_dir, and test_mio_c_h_p_has_genuinely_mixed_grid_lengths as the anti-over-reach guard
  - phase: 04-scf-and-reference-simulation-validation
    provides: threat T-04-07's policy that an exception message never interpolates a path, and the run_with_float64 harness convention
provides:
  - "_bond_integral.SKFRadialGridStepMismatchError: named ValueError subclass for a mixed-step SKFPATH"
  - "_bond_integral._require_uniform_grid_step(steps_by_file): the REG-05 guard, comparing STEP and never LENGTH"
  - "_bond_integral.SKF_GRID_STEP_MISMATCH_MESSAGE: the explanatory paragraph, including the do-not-pad-your-files instruction"
  - "docs/RADIAL-GRID-CONSUMERS.md: all 69 R_orb grep hits classified, 9 read sites dispositioned"
  - "tests/test_radial_grid.py::test_esdriver_supplies_the_per_pair_grid_arguments: a wiring test, reusable shape for any None-defaulted fallback"
  - "tests/test_public_api_contract.py::PINNED_PUBLIC_API: the 24-name literal, independent of __all__"
affects: [05-05, 06-scf, 08-forces-stress, 08.1-batched-f]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A guard whose obvious wrong implementation would pass the case it exists to catch states the correct comparison three times: in the action, in the function docstring, and in a test named for the benign case"
    - "A None-defaulted fallback needs a test that watches the WIRING (what the caller passes), because tests that call the callee directly stay green when the caller stops supplying the arguments"
    - "A docstring corrected against measurement keeps the superseded claim inside it, and a test asserts on both the corrected text and the absence of the old sentence"
    - "A public-surface pin is written as a literal in the test rather than read from the module, so the comparison is a second source and not a tautology"
    - "An exception message groups offenders by the value they declared and reports basenames only, never paths"

key-files:
  created:
    - docs/RADIAL-GRID-CONSUMERS.md
  modified:
    - src/dftorch/_bond_integral.py
    - src/dftorch/ESDriver.py
    - src/dftorch/_h0ands.py
    - src/dftorch/_ml_sk.py
    - src/dftorch/_stress.py
    - tests/test_radial_grid.py
    - tests/test_public_api_contract.py

key-decisions:
  - "The guard compares grid STEP and never grid LENGTH, with a 1e-6 RELATIVE tolerance whose only job is surviving a float round trip; the hazard it separates is a factor of two, five orders of magnitude outside the band."
  - "The step is recovered from R_orb_i[0] / BOHR_TO_ANGSTROM and reported in BOHR, because Bohr is what the user's SKF grid line actually says."
  - "The message groups files by declared step, capped at 6 basenames per group, so a large directory still produces a readable error."
  - "SKFRadialGridStepMismatchError subclasses ValueError, not NotImplementedError: the four F*UnsupportedError classes mark capability that could land later, whereas a mixed-step directory is not coherent as a parameter set at all."
  - "test_mixed_step_grids_read_their_own_radius from 05-01 is kept and made to bypass the guard deliberately, so the project keeps a measured radius showing WHAT the guard prevents rather than only an assertion that it fires."
  - "ESDriver's per-pair wiring, silently reverted by commit 56091af, is restored. REG-06 was inert in every real calculation for eleven commits and no test could see it."
  - "_ml_sk.py:548's '(mixed units, same as R_orb)' comment was corrected too, one line outside the plan's fenced diff, because leaving it propagates the exact falsehood being corrected."

patterns-established:
  - "Wiring test: patch the callee at the CALLER's module attribute, record the kwargs, raise a sentinel to abort early. Costs an SKF load rather than a full calculation."
  - "AST-diff proof that a change is docstring-only: parse before and after, strip every docstring node, compare ast.dump. Stronger than reading a diff."
  - "Real-file hazard backstop: assemble a fixture by COPYING shipped files from two real parameter sets rather than synthesising both sides, so the guard is shown firing on data the project actually ships."

requirements-completed: [REG-05, REG-03]

metrics:
  duration: ~85 min
  tasks: 3
  files-created: 1
  files-modified: 7
  tests-added: 10
  suite-before: 159 passed / 0 failed
  suite-after: 169 passed / 0 failed / 0 errors / 0 skipped
  completed: 2026-08-02

status: complete
---

# Phase 5 Plan 04: Mixed-Step Refusal, Grid Consumer Dispositions and Public API Pin Summary

An `SKFPATH` mixing radial grid steps now refuses to load with an error naming the offending
files and both steps; every remaining reader of the shared grid has a written disposition;
and the audit that produced those dispositions found that REG-06's fix had been silently
reverted eleven commits earlier and was inert in every real calculation.

## What was built

### Task 1: the REG-05 refusal (`87c24fd`)

`_bond_integral` gained three module-level artifacts: `SKFRadialGridStepMismatchError`
(a `ValueError` subclass), `SKF_GRID_STEP_MISMATCH_MESSAGE`, and
`_require_uniform_grid_step(steps_by_file)`. The guard is called exactly once in
`get_skf_tensors`, immediately after the read loop and ahead of the `R_orb_master is None`
check, so a mixed-step directory fails on the grid mismatch rather than on a downstream
symptom.

The step is collected inside the existing loop as `R_orb_i[0] / BOHR_TO_ANGSTROM`, keyed by
`os.path.basename`. `read_skf_table`'s signature is unchanged: the grid is built as
`arange(1, npts_pad + 1) * step * BOHR_TO_ANGSTROM` and starts one step in, so `R_orb_i[0]`
*is* the step, and dividing the conversion back out recovers the Bohr number the file's own
grid line carries. Keying by basename is what makes the no-path-leak property structural
rather than a formatting discipline: no absolute path is ever in the dictionary the guard
sees.

The emitted message groups files by declared step:

```
SKF radial grid step mismatch within a single SKFPATH:
  step 0.02 Bohr: N-N.skf
  step 0.04 Bohr: Eu-Eu.skf, Eu-N.skf, N-Eu.skf
Every SKF file in one SKFPATH must be tabulated on the same radial grid STEP.
...
Differing NUMBERS OF POINTS at the same step are FINE and are NOT what this error
reports. 3ob-3-1 ships C-C at 650 points and Br-Br at 850, both at 0.02 Bohr, ...
Do NOT pad SKF files to a common length in response to this message.
```

That last instruction is deliberate. A user who sees "grid" and "500" in an error has an
obvious wrong repair available, and the message closes it off explicitly.

**The comparison is on step, never on length.** This is stated in the guard's docstring, in
the exception message, and in a test named for the benign case, because the obvious
implementation -- compare grid lengths -- passes the exact case the guard exists to catch
while rejecting two real shipped parameter sets.

**Tolerance.** `_GRID_STEP_RELATIVE_TOLERANCE = 1e-6`, relative rather than absolute so one
band covers a 0.02 Bohr set and a much finer one. Its only job is surviving the float round
trip through `* BOHR_TO_ANGSTROM` and back, which moves a value by a few ULP (order 1e-16
relative). The condition being caught is 100 percent relative. The reasoning is written into
the function docstring, per the acceptance criteria.

Six tests, not the five the plan listed:

| Test | What it holds |
| --- | --- |
| `test_mixed_step_skfpath_is_refused` | the refusal fires, and the class subclasses `ValueError` |
| `test_mixed_step_error_names_both_files_and_steps` | the exact grouped lines, plus the do-not-pad sentence |
| `test_same_step_different_length_is_accepted` | the benign case loads AND really carries two lengths |
| `test_real_parameter_sets_still_load` | `mio-1-1` and `f_orbital_data` both construct `Constants` |
| `test_real_f_and_mio_files_mixed_are_refused` | **added**: real shipped files from both datasets |
| `test_guard_message_leaks_no_path` | six path forms absent from the message |

### Task 2: dispositions, the reverted wiring, and the unit correction (`2d508b1`)

`docs/RADIAL-GRID-CONSUMERS.md` classifies all **69** hits of
`grep -rn "R_orb" src/dftorch --include="*.py"`, quotes the command so completeness is
re-checkable, and gives each of the **9 read sites** a verdict:

| Disposition | Sites |
| --- | --- |
| `converted` | `ESDriver.py:266-286`, `_h0ands.py:353-355` |
| `deferred` | `ESDriver.py:1527` and `_h0ands.py:666-668` (PHY-04, Phase 8.1); `_stress.py:183` (PHY-02, Phase 8); `_ml_sk.py:489-500` (PHY-06, v2); `sedacs_interface.py:523` (PHY-05, v2) |
| `unreachable` | `_legacy/H0andS.py:200-202` and `:464-466` |

`unreachable` is established rather than asserted: `grep -rn "H0andS\|_legacy" src tests`
outside the directory itself returns only `pyproject.toml:61`, which *excludes* the
directory from ruff, and a generated `SOURCES.txt` entry. The directory has no
`__init__.py`.

`build_pair_type_rcut`'s docstring was rewritten. The measured convention -- grid and pair
distances both in Angstrom, conversion happening once at
`arange(...) * step * BOHR_TO_ANGSTROM` -- replaces the claim that the comparison was
deliberately mixed-unit. The superseded claim is recorded inside the new docstring, with the
measurement that settled it, so a reader who remembers the old text can see it was corrected
rather than lost. The example was also re-derived from measurement: the old text's "3ob,
550 points, 11.0 Bohr" does not match what mio-1-1 actually produces, which is last non-zero
interval 498 and a 5.2917721 A cutoff, exactly 10.0 Bohr.

### Task 3: the public surface (`c0d0685`)

`PINNED_PUBLIC_API` is a 24-name literal in `tests/test_public_api_contract.py`.
`test_public_api_all_is_pinned` computes removals and additions separately and reports each
with its own message, because a removal is the API break REG-03 forbids while an addition is
a deliberate commitment. `test_public_api_exports_every_declared_name` separately catches an
`__all__` entry the module does not define, which is a distinct bug: `from dftorch import *`
would raise on it.

## Verification

`uv run pytest -q`: **169 passed / 0 failed / 0 errors / 0 skipped**. Entering baseline was
159; 10 tests were added and none were lost.

`uv run pytest tests/test_f_orbital_skf.py tests/test_eu_n_scan.py tests/test_single_shot_energy.py tests/test_scf.py -q`
passes, proving the guard admits every parameter set the repository already uses.

### Mutation checks actually performed

Three claims are gated by tests whose failure was forced rather than assumed:

1. **The wiring test sees the revert.** Deleting `R_tensor=` and `n_grid=` from
   `ESDriver.py` again makes `test_esdriver_supplies_the_per_pair_grid_arguments` fail with
   `Keyword arguments seen: ['ml_model_data', 'store_stress_metadata', 'verbose']`. Restored
   afterwards; the temporary edit was not committed.
2. **The API pin sees a removal.** Deleting `"get_coulomb_stress_real"` from
   `src/dftorch/__init__.py`'s `__all__` fails `test_public_api_all_is_pinned` with
   `public export(s) REMOVED from dftorch.__all__: ['get_coulomb_stress_real']`.
   `src/dftorch/__init__.py` was restored via `git checkout --` and is byte-unmodified.
3. **The API pin sees an addition, on both branches.** Adding `"NotARealExport"` fails
   `test_public_api_exports_every_declared_name` ("the module does not define them") *and*
   `test_public_api_all_is_pinned` ("ADDED"). Restored.

### The `_ml_sk.py` diff is provably prose-only

Rather than reading the diff, both revisions were parsed with `ast`, every docstring node
stripped, and `ast.dump` compared: **identical**. Comments never enter the AST, so this
proves the change is docstring and comment only, with no executable line touched.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] REG-06's fix had been silently reverted and was inert in production**

- **Found during:** Task 2, enumerating the `R_orb` read sites for the disposition table.
- **Issue:** The plan and plan 05-01's summary both state that `ESDriver`'s single-system
  caller supplies `R_tensor` and `n_grid`, which is what selects `_pair_knot_lookup`.
  `grep -n "R_tensor\|n_grid" src/dftorch/ESDriver.py` returned **nothing**. Commit
  `56091af` ("H0andS working with a test example on EU_N bond", whose stated purpose was
  plan 05-03's independent SKF oracle) deleted both keyword arguments and their comment as
  an unrelated side effect. `H0_and_S_vectorized` falls back to the verbatim global `R_orb`
  expression whenever either argument is `None`, so **REG-06 was implemented, tested, marked
  complete, and then inert in every real calculation for eleven commits.**
- **Why nothing failed:** all thirteen of plan 05-01's tests call `_pair_knot_lookup` or
  `H0_and_S_vectorized` directly. Not one goes through the driver. A `None`-defaulted
  fallback that produces a plausible answer cannot be protected by tests that call the
  callee.
- **Fix:** restored both arguments with a comment naming the reverting commit and the test
  that now guards them. Added `test_esdriver_supplies_the_per_pair_grid_arguments`, which
  patches `H0_and_S_vectorized` at `ESDriver`'s module attribute, records the kwargs the
  driver passes, and raises a sentinel to abort `forward` early -- costing an SKF load and a
  neighbour list rather than a full single-shot run.
- **Numeric impact: none.** The suite is bit-stable across the restore, including the CH4
  H0/S checksums, the Eu-N 21-point scan and the three pinned 05-02 energies. That is the
  outcome plan 05-01 predicted for a uniform-step directory, and it is now measured with the
  per-pair path actually live.
- **Files modified:** `src/dftorch/ESDriver.py`, `tests/test_radial_grid.py`
- **Commit:** `2d508b1`

**2. [Rule 3 - Blocking] The new guard breaks 05-01's `test_mixed_step_grids_read_their_own_radius`**

- **Found during:** Task 1.
- **Issue:** That test builds a mixed-step directory (H-H at 0.1 Bohr, N-N at 0.2) and
  constructs `Constants` from it. The REG-05 guard refuses exactly that. Its docstring even
  says "this directory currently LOADS without complaint... the explicit refusal is
  requirement REG-05 and belongs to plan 05-04."
- **Fix:** the test now replaces `_require_uniform_grid_step` with a no-op for that single
  load, restoring it in a `finally` so a failure cannot disarm the guard for later tests. It
  was kept rather than deleted on purpose: it is the project's only *measurement* of what
  the guard prevents -- the borrowed knot sits at more than 1.9x the correct radius -- and
  deleting it would leave the refusal justified only by assertion. Its docstring now says
  the bypass is deliberate and why.
- **Files modified:** `tests/test_radial_grid.py`
- **Commit:** `87c24fd`

**3. [Rule 1 - Bug] `_h0ands.py:61` pointed at the wrong claim by line number**

- **Found during:** Task 2.
- **Issue:** `_pair_knot_lookup`'s `dR` parameter doc said "see the note on
  `_ml_sk.py:437-450` in the `R_orb` parameter doc of `H0_and_S_vectorized`". The `R_orb`
  parameter doc contains no such note, and after Task 2's rewrite those line numbers point
  at the *corrected* text -- so a reference intended to warn about a wrong claim would have
  silently become a citation of the right one, by line number, with no way to tell.
- **Fix:** replaced with the unit statement itself plus a reference by symbol name to
  `build_pair_type_rcut`'s Units note. Line-number references into a file being edited are
  the failure mode.
- **Files modified:** `src/dftorch/_h0ands.py`
- **Commit:** `2d508b1`

### Deliberate departures from the plan text

**4. [Rule 2 - Missing correctness] `_ml_sk.py:548`'s comment was corrected too**

The plan's acceptance criteria fence the `_ml_sk.py` diff to "comment and docstring lines
*within* `build_pair_type_rcut`". Line 548 sits outside it and read
`dR_mskd = ml_ctx["dR_mskd"]  # A (mixed units, same as R_orb)`. "Mixed units" is the exact
falsehood Task 2 exists to correct, so leaving it would have shipped the corrected docstring
alongside a surviving copy of the claim it corrects. Changed to
`# A, the same units as R_orb (not mixed)`.

The criterion's intent -- no executable change -- is satisfied and was proved by AST diff.
Recorded here because it is a departure from the letter of an acceptance criterion, decided
against its purpose.

**5. [Rule 2 - Missing coverage] `test_real_f_and_mio_files_mixed_are_refused`**

The plan's five Task 1 tests all build their SKFPATH with `write_mixed_grid_skf_pair`, so on
their own the guard is only ever shown firing on a construction of this suite's own making
-- which is precisely what the plan's own backstop truth says is not enough. This test
assembles the directory by **copying shipped files**: `Eu-Eu.skf`, `Eu-N.skf` and `N-Eu.skf`
from `tests/f_orbital_data` at 0.04 Bohr, and `N-N.skf` from `tests/data_skf_mio-1-1` at
0.02 Bohr. That is exactly the case decision D-01 describes -- an Eu system whose
light-element ligand parameters come from a mainstream set -- built from data the project
ships. Verified grid lines: `0.04, 433` and `0.02, 500,2`.

**6. [Rule 2 - Missing oracle] `test_mixed_step_error_names_both_files_and_steps` asserts on structure, not substrings**

The plan asks the test to assert "the numeric substrings". `SKF_GRID_STEP_MISMATCH_MESSAGE`
mentions both `0.02` and `0.04` in its explanatory prose, so a bare substring check for
those numbers would pass even if the guard reported no measured value whatsoever. The test
asserts on the exact grouped lines (`step 0.04 Bohr: N-N.skf`) instead, which cannot be
satisfied by the boilerplate.

**7. [Rule 2 - Missing gate] `test_rcut_values_unchanged_after_docstring_fix` also asserts the docstring**

As the plan specifies it, this test duplicates `test_effective_cutoffs_are_unchanged` and
gates nothing about the correction it is named for. It now additionally asserts that the
docstring states the shared-unit convention, records that the prior text was corrected, and
names `BOHR_TO_ANGSTROM`, and that the superseded sentence `compares these mixed-unit
arrays` is absent. That is the only mechanical gate that the correction survives a later
edit. ASCII-only substrings were chosen because `tests/test_radial_grid.py` is ASCII-only by
Phase 4 rule while the docstring itself spells Angstrom with accents.

## Threat mitigations

| Threat | Disposition | Evidence |
| --- | --- | --- |
| T-05-15 mixed-step SKFPATH silently producing a factor-of-two error | mitigated | `_require_uniform_grid_step` raises before `get_skf_tensors` returns; `test_mixed_step_skfpath_is_refused` and `test_real_f_and_mio_files_mixed_are_refused` |
| T-05-16 the guard rejecting a valid parameter set | mitigated | comparison is on step only, stated in docstring, message and test names; `test_same_step_different_length_is_accepted`, `test_real_parameter_sets_still_load`, and 05-01's `test_mio_c_h_p_has_genuinely_mixed_grid_lengths` all green |
| T-05-17 path leaking into the guard's message | mitigated | the guard's input dictionary is keyed by basename, so no path exists to leak; `test_guard_message_leaks_no_path` asserts six path forms absent |
| T-05-18 a shared-grid reader left unconverted with no record | mitigated | `docs/RADIAL-GRID-CONSUMERS.md`, 9 read sites, every deferral naming a requirement id and owning phase, grep command quoted |
| T-05-19 a public export silently dropped | mitigated | `test_public_api_all_is_pinned` against a literal; mutation-confirmed in both directions |
| T-05-SC package installs | not applicable | nothing was installed |

## Known Stubs

None. No stub, placeholder, skipped test or unrun `<verify>` was left behind; every
`<automated>` verification in the plan was executed.

Two temporary source mutations were made to force test failures, both restored and neither
committed: `src/dftorch/ESDriver.py` (restored by targeted edit, diff verified) and
`src/dftorch/__init__.py` (restored by `git checkout --`, `git status` clean).

## Threat Flags

None. This plan adds no network I/O, no deserialisation of untrusted data, no subprocess, no
dynamic import and no privilege transition. The one new exception message is bounded to file
basenames and numbers by construction.

## Follow-ups for later plans

- **The revert lesson generalises.** `H0_and_S_vectorized`'s `R_tensor=None, n_grid=None`
  is not the only `None`-defaulted fallback in this codebase. Any of them can be silently
  disconnected by a caller-side edit without a test noticing. Whoever owns CLN work should
  consider whether the remaining ones need wiring tests.
- **Rows 3-7 of `docs/RADIAL-GRID-CONSUMERS.md`** are the live work list for Phase 8
  (PHY-02), Phase 8.1 (PHY-04) and v2 (PHY-05, PHY-06). `_pair_knot_lookup` is module-level
  and directly reusable.
- **`sedacs_interface.py:523`** keeps working only because both new parameters default to
  `None`. If either default is ever removed, SEDACS breaks silently in the other direction.
- **REG-05's guard now bounds the deferred rows' consequence** to same-step directories,
  which is what makes deferring them defensible rather than merely convenient. If the guard
  is ever relaxed, those deferrals need revisiting.

## Self-Check: PASSED

Files verified present on disk:

- `FOUND: docs/RADIAL-GRID-CONSUMERS.md`
- `FOUND: src/dftorch/_bond_integral.py` (modified)
- `FOUND: src/dftorch/ESDriver.py` (modified)
- `FOUND: src/dftorch/_h0ands.py` (modified)
- `FOUND: src/dftorch/_ml_sk.py` (modified)
- `FOUND: src/dftorch/_stress.py` (modified)
- `FOUND: tests/test_radial_grid.py` (modified)
- `FOUND: tests/test_public_api_contract.py` (modified)

Commits verified in `git log`:

- `FOUND: 87c24fd` feat(05-04): refuse an SKFPATH whose files disagree on radial grid step
- `FOUND: 2d508b1` fix(05-04): restore the reverted REG-06 wiring and disposition every grid reader
- `FOUND: c0d0685` test(05-04): pin the whole 24-name public import surface (REG-03)
