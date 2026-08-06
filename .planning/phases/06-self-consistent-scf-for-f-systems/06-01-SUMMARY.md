---
phase: 06-self-consistent-scf-for-f-systems
plan: 01
subsystem: testing
tags: [dftb, scf, convergence, krylov, anderson-mixing, pytest, torch]

# Dependency graph
requires:
  - phase: 04-scf-and-reference-simulation-validation
    provides: "the Eu-N diatomic fixture, the pinned single-shot energy EU_N_REFERENCE_E_TOT, the 2.655 A target separation (D-24), and decision D-13 (non-convergence warns, returns the last iterate, never raises)"
  - phase: 05-regression-safety-and-support-policy-cleanup
    provides: "the f-free regression baselines this plan must not move, and the orbital-count sweep in tests/test_orbital_count_guards.py that pins every `== 16` site"
provides:
  - "A self-consistent charge loop that actually settles for Eu-N at 2.60, 2.655 and 2.70 A"
  - "scf_iter_count: a machine-readable convergence result on all four charge loops - a positive pass count on success, the literal -1 when the loop gave up"
  - "structure.scf_iter_count, assigned at all four ESDriver call sites, which plan 06-05's binding-curve graph reads to mark failed separations"
  - "structure.krylov_disabled_for_f, the flag saying which path a result came from"
  - "_krylov_params_for_f_interim: the single deletable place where the Krylov accelerator is switched off for f systems"
  - "tests/test_scf_convergence_f.py, a 12-test module proving all of the above"
affects: [06-02, 06-03, 06-04, 06-05, shell-resolved-charge, binding-curve]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "scipy-style info result: a positive count on success, a sentinel (-1) on failure, returned beside the answer rather than printed"
    - "Interim workarounds live in one named helper with a docstring naming the ruling that justifies them and the condition for deleting them"
    - "Parameter overrides are returned as a shallow copy, never mutated in place, so a reused driver cannot leak a setting between molecules"
    - "Wiring watches by source inspection (inspect.getsource) for connections that have no cheap end-to-end fixture"

key-files:
  created:
    - tests/test_scf_convergence_f.py
  modified:
    - src/dftorch/_scf.py
    - src/dftorch/ESDriver.py
    - tests/test_orbital_count_guards.py

key-decisions:
  - "All four charge loops report, not just SCFx - each has exactly one production call site, so the positional-tuple churn is four small edits, and leaving three printing while one returns would teach a future reader that a printed line is a trustworthy signal"
  - "The result is one extra int on the end of the existing return tuple, unpacked into structure.scf_iter_count - the tuple grows once and structure becomes the extensible surface for any later diagnostic"
  - "The Krylov switch-off lives in the driver, for f systems only, and only when the caller has not set KRYLOV_START themselves - by construction it cannot move a single f-free number"
  - "Non-convergence is forced in tests by lowering SCF_MAX_ITER to 2, never by relying on the accelerator bug, so the test stays independent of the behaviour this phase is changing"

patterns-established:
  - "Convergence result as a number: read structure.scf_iter_count, never scrape stdout"
  - "Runaway detectors, not reference values: a bound with a factor of three of headroom, documented as such in the test docstring (D-6.08)"

requirements-completed: [SCC-01]

coverage:
  - id: D1
    description: "Eu-N settles its charge loop at 2.60, 2.655 and 2.70 A by reaching tolerance rather than by running out of passes"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_eu_n_scf_converges_at_target_separation"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_eu_n_scf_converges_across_the_target_neighbourhood"
        status: pass
    human_judgment: false
  - id: D2
    description: "The settled answer is physical: charge is conserved, no charge has run away, and electrons move from the metal Eu to the non-metal N"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_eu_n_converged_charges_are_not_a_runaway"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_eu_n_charge_transfer_runs_the_chemical_direction"
        status: pass
    human_judgment: false
  - id: D3
    description: "A caller can read the loop's outcome as a number - a positive pass count, or -1 for gave-up - without scraping printed text"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_exhausting_the_iteration_cap_returns_minus_one"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_a_caller_that_ignores_the_convergence_result_still_works"
        status: pass
    human_judgment: false
  - id: D4
    description: "A loop that gave up still hands back its last energy and charges and raises nothing (Phase 4 decision D-13 held intact)"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_a_loop_that_gave_up_still_hands_back_its_last_answer"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_a_loop_that_gave_up_raises_nothing"
        status: pass
    human_judgment: false
  - id: D5
    description: "All four charge loops report the same way, and the element has not silently vanished from any of their returns"
    requirement: SCC-01
    verification:
      - kind: unit
        ref: "tests/test_scf_convergence_f.py#test_all_four_charge_loops_report_a_convergence_result"
        status: pass
    human_judgment: false
  - id: D6
    description: "The Krylov switch-off is scoped to f systems only, stands down when the caller sets KRYLOV_START, and moves no f-free number"
    requirement: SCC-01
    verification:
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_krylov_accelerator_is_disabled_for_the_f_system"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_krylov_accelerator_is_untouched_for_an_f_free_system"
        status: pass
      - kind: integration
        ref: "tests/test_scf_convergence_f.py#test_caller_supplied_krylov_start_is_never_overridden"
        status: pass
      - kind: integration
        ref: "tests/test_single_shot_energy.py"
        status: pass
      - kind: integration
        ref: "tests/test_scf.py"
        status: pass
    human_judgment: false

# Metrics
duration: ~115min
completed: 2026-08-05
status: complete
---

# Phase 6 Plan 01: Self-Consistent Charge Loop Settles for Eu-N Summary

**The Eu-N charge loop now settles at 2.60/2.655/2.70 A because the charges stopped moving, and all four charge loops hand back a readable pass count or -1 instead of only printing "Did not converge".**

## Performance

- **Duration:** ~115 min across two sessions (Tasks 1-2 on 2026-08-04, Task 3 on 2026-08-05 after a session-limit cutoff)
- **Started:** 2026-08-04T~20:30:00Z
- **Completed:** 2026-08-05T20:01:08Z
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- **Eu-N converges.** The loop that previously ran away - piling electrons onto one atom and returning +198 eV against a true scale near -17 eV - now reaches tolerance at all three separations in the neighbourhood of 2.655 A. The cause was the Krylov accelerator (`kernel_update_lr`), not the f orbitals, exactly as `06-RESEARCH.md` rated HIGH confidence.
- **Convergence is now a number, not printed text.** `SCFx`, `scf_x_os`, `SCFx_batch` and `delta_scf_x_os` each gained a fifteenth (thirteenth, for the batch loop's shorter tuple) return element `scf_iter_count`: `int(it)` when the loop's own two tolerance conditions both hold, the literal `-1` otherwise. All four ESDriver call sites unpack it into `structure.scf_iter_count`.
- **The switch-off cannot touch an f-free calculation.** `_krylov_params_for_f_interim` only engages when some atom has `const.n_orb[TYPE] == 16` *and* the caller has not set `KRYLOV_START` themselves, and it returns a shallow copy rather than mutating the driver's dict (threat T-06-02). Methane on mio-1-1 is the live control.
- **Giving up is reported honestly.** Forced deterministically with `SCF_MAX_ITER = 2` - not by re-enabling the accelerator bug - the loop returns exactly `-1`, still hands back a finite `e_tot` and two finite charges, and raises nothing.
- **Full suite: 228 passed, 0 failed**, up from the 216-passed baseline this phase started from (+12, the new module).

## Task Commits

1. **Task 1: End-to-end - Eu-N settles at 2.655 A and reports how it went** (TDD tracer)
   - `e14e8c4` (test) - the seven failing convergence-report tests, written before any source change
   - `e8b6625` (feat) - `SCFx` reports, and Eu-N settles
2. **Task 2: The other three charge loops report the same way** - `bb250df` (feat)
3. **Task 3: Giving up is reported honestly and never raises** - `96dfc1f` (test)

## Files Created/Modified

- `tests/test_scf_convergence_f.py` (created, 557 lines) - 12 tests. Pure ASCII, verified by byte scan, per the Phase 4 rule that a failure be diagnosable from a cp1252 console.
- `src/dftorch/_scf.py` (+131 lines) - `scf_iter_count` computed and returned by all four charge loops; return annotations and docstrings extended. The `print("Did not converge")` warning is untouched in all four, as D-13 requires.
- `src/dftorch/ESDriver.py` (+89 lines) - `_krylov_params_for_f_interim` at line 108; four unpack sites gained `structure.scf_iter_count`.
- `tests/test_orbital_count_guards.py` (+8/-3 lines) - Phase 5's orbital-count sweep extended to admit the new helper (see Deviations).

## Verification Evidence

| Gate | Result |
|---|---|
| `uv run pytest` (whole suite) | **228 passed, 0 failed** in 385 s - at/above the 216 baseline |
| `uv run pytest tests/test_scf_convergence_f.py -x` | 12 passed |
| `uv run pytest tests/test_single_shot_energy.py -x` | passed - the Phase 4 pin `EU_N_REFERENCE_E_TOT` is untouched |
| `uv run pytest tests/test_scf.py -x` | passed - f-free methane self-consistency unchanged |
| ASCII byte scan of the new module | exit 0 |
| Float literals with >3 decimal digits on an assignment RHS (D-6.08) | none found |
| `grep -c scf_iter_count src/dftorch/_scf.py` | 17 (criterion: >= 8) |
| `grep -c structure.scf_iter_count src/dftorch/ESDriver.py` | 4 (criterion: exactly 4) |
| `KRYLOV_START` defaults in `_scf.py` | all five executable sites still `.get("KRYLOV_START", 10)` - the library-wide default is unchanged |
| Post-commit deletion check on `96dfc1f` | no files deleted |

## Line-number and inventory corrections

The plan asked (in "A note on line numbers", and in Task 2's acceptance criteria) that observed sites be recorded rather than predicted ones.

**Call-site inventory, observed after the edits.** The plan's structural claim held exactly - each of the four loops has **exactly one** production call site, so threat T-06-01 (an unfound unpack site silently shifting every attribute) had no extra site to hide in. Only the line numbers moved, and mostly because this plan itself added ~89 lines to `ESDriver.py`:

| Loop | Plan predicted | Actually observed |
|---|---|---|
| `scf_x_os` | ESDriver.py:535 | ESDriver.py:615 |
| `delta_scf_x_os` | ESDriver.py:607 | ESDriver.py:688 |
| `SCFx` | ESDriver.py:678 | ESDriver.py:760 |
| `SCFx_batch` | ESDriver.py:1757 | ESDriver.py:1844 |

**`KRYLOV_START` occurrence count.** The plan's acceptance criterion said "the same four occurrences it showed before this task (lines 454, 839, 1152, and the pair at 1483/1490)" - which names five lines while calling them four. The observed truth: **five executable `.get("KRYLOV_START", 10)` sites** (now at 469, 895, 1225, 1593, 1600 - `delta_scf_x_os` uses it twice), plus two prose mentions in a docstring and a comment (281, 1594). Every default is still `10`. The substance of the criterion - the library-wide default is unchanged - holds; the plan's own count was internally inconsistent.

## Decisions Made

The three discretionary choices were resolved in the plan and implemented as written; nothing was re-decided during execution. Recorded here because they are the ones a later phase will need:

1. **All four loops report, not just `SCFx`.** Each has exactly one production call site (verified above), so the blast radius was four small edits rather than open-ended. Leaving three loops printing while one returns would have taught a future reader that a printed line is a trustworthy signal - which `06-CONTEXT.md` names as the one place this codebase's "fail loudly, never silently zero" pattern is currently violated.
2. **The result is a plain `int` appended to the existing tuple**, after scipy's iterative solvers. A boolean was explicitly rejected: it throws the pass count away. The tuple grows once; `structure` is the extensible surface for any later diagnostic (final residual, final energy change), so the return arity never has to change again.
3. **The Krylov switch-off is driver-scoped and f-only.** Changing the default in `_scf.py` was rejected because it moves the numerical path for every molecule this library computes, against the project's core value. Requiring every caller to pass the key was rejected because then SCC-01 would only be true for a caller who knows a magic key - a workaround, not a supported system.

**Reversibility, as promised in the plan:** the fifteenth return element is costly-but-mechanical to undo (four returns, four unpacks, every site named above). `_krylov_params_for_f_interim` is easy - one helper, one call site, one attribute - and is the single place to delete when the accelerator repair lands.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Phase 5's orbital-count sweep rejected the new Krylov helper**

- **Found during:** Task 1 (adding `_krylov_params_for_f_interim` to `ESDriver.py`)
- **Issue:** `tests/test_orbital_count_guards.py::test_prose_mentions_are_excluded_and_counted` asserts that `ESDriver.py`'s executable `== 16` orbital-count comparisons are exactly `{60, 95}`. The new helper's own `const.n_orb[TYPE] == 16` check is a third such site, so the test failed the moment the helper landed. This is Phase 5's REG-04/D-04 guard doing precisely its job - it is designed to notice a new orbital-count site - so the correct response was to admit the new site, not to weaken the sweep.
- **Fix:** Extended the expected set to `{60, 95, 164}` and rewrote the failure message to name the new site. The helper's docstring also spells out "n_orb == 16" and "16 orbitals" in prose, which made it a third independent chance for the COMMENT/FSTRING blanking to go wrong; the test now confirms that only the *executable* comparison is listed, which is the property it has always held.
- **Files modified:** `tests/test_orbital_count_guards.py`
- **Verification:** `uv run pytest tests/test_orbital_count_guards.py` green; full suite 228 passed.
- **Committed in:** `e8b6625` (part of the Task 1 commit)
- **Note:** `tests/test_orbital_count_guards.py` is not in this plan's `files_modified` frontmatter. It is the only file touched outside the declared set.

**2. [Rule 2 - Missing Critical] Two gave-up failure messages omitted the parameter that produced the failure**

- **Found during:** Task 3 (review of the draft tests carried over from the interrupted session)
- **Issue:** Task 3's `<action>` requires that "every failure message must carry the observed value **and the parameter that produced it**". Two assertions in `test_a_loop_that_gave_up_still_hands_back_its_last_answer` (the charge shape and the charge finiteness checks) reported the observed value but never named `SCF_MAX_ITER = 2` as the cause, and `test_a_loop_that_gave_up_raises_nothing` had a bare `assert structure is not None` with no message at all. A developer hitting these would have seen a shape or a `None` with no clue which setting produced it.
- **Fix:** Interpolated `CAP_TOO_LOW_TO_CONVERGE` into all three messages.
- **Files modified:** `tests/test_scf_convergence_f.py`
- **Verification:** module re-run green (12 passed).
- **Committed in:** `96dfc1f` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing-critical). **Impact:** neither expanded scope. The first is a Phase 5 guard correctly firing on a Phase 6 addition; the second restores a diagnostic requirement the plan stated explicitly.

## Issues Encountered

- **Session-limit cutoff mid-Task-3.** The previous executor was interrupted after committing Tasks 1 and 2, leaving ~100 uncommitted lines of Task 3 tests that had **never been run**. On resume all four commits were verified against `git log`/`git show` before the table in the handoff was trusted; the commits matched. The draft tests were then reviewed against Task 3's `<action>`, `<acceptance_criteria>` and the D-6.08 prohibition, run, and corrected (deviation 2 above) rather than accepted because they existed.
- **Prohibition audit (all four `must_haves` prohibitions were `flagged-unverified` at plan time; all four now check out):** no energy/charge/binding-curve value is frozen as a reference (checked mechanically - no assignment RHS carries a float literal with more than three decimal digits; the only numeric literals are separations, tolerances, the 2.0 runaway bound and the sentinel `-1`). A cap-exhausted loop can never present as converged, because `scf_iter_count` is the counter only when both of the loop's own tolerance clauses hold and `-1` otherwise, and `test_exhausting_the_iteration_cap_returns_minus_one` asserts the exact integer so a boolean substitution fails there. The accelerator is not switched off globally and `_scf.py`'s default is still `10` at all five sites. All four `print("Did not converge")` warnings survive.

## Next Phase Readiness

**Ready.** `structure.scf_iter_count` is populated on every driver path, which is what plan 06-05's binding-curve graph reads to decide which separations to mark as failed; plan 06-05 can consume it immediately.

Carried forward, deliberately and per plan:

- **The wide-range scan is not this plan's job.** `must_haves` carries a backstop truth - "a separation elsewhere in the 1.60-3.60 A range that still fails to settle is reported through the `-1` result and does not turn the test suite red" - and no test here asserts convergence away from the neighbourhood of 2.655 A, per D-6.05. The `-1` reporting path is proven (D3/D4 above); the full 21-point sweep that would exercise it across the range belongs to plan 06-05. `06-RESEARCH.md` measured 21/21 settling with the accelerator off, so the expectation is that nothing reports `-1`, but that is measurement, not a gate this plan installed.
- **The Krylov accelerator is still broken and is only switched off, not repaired.** Deferred by the recorded human ruling of 2026-08-04. `_krylov_params_for_f_interim`'s docstring names it as the single place to delete when the repair lands, and `structure.krylov_disabled_for_f` tells a caller which path any given result came from.
- **The assumption delta for plan 06-03 stands:** per-atom charge is about to stop being the only kind of charge. Once shell-resolved charge lands, `structure.q` becomes one case among two.
- **Per-batch semantics.** `SCFx_batch`'s `scf_iter_count` describes the batch as a whole, never one structure; its docstring says so. A future consumer wanting per-structure counts will need a different shape.

## Self-Check: PASSED

All five files claimed above exist on disk; all four task commits (`e14e8c4`, `e8b6625`, `bb250df`, `96dfc1f`) are present in `git log`.

---
*Phase: 06-self-consistent-scf-for-f-systems*
*Plan: 01*
*Completed: 2026-08-05*
