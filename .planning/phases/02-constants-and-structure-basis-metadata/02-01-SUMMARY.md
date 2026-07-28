---
phase: 02-constants-and-structure-basis-metadata
plan: 01
subsystem: testing
tags: [pytest, dftorch, constants, structure, basis-metadata, f-orbitals]

requires:
  - phase: 01-skf-canonicalization-and-spline-validation
    provides: normalized simple and extended SKF parser output plus f-orbital fixture validation
provides:
  - pytest gates for f-containing Constants and Structure metadata validation
  - explicit StructureBatch global-offset and padded-row validation
  - simple-format f-free metadata regression coverage for s-only, sp, and spd systems
affects: [phase-03-h0s-routing, constants, structure, structurebatch, regression-safety]

tech-stack:
  added: []
  patterns:
    - pytest wrappers call the standalone validation harness so script and CI gates stay aligned
    - validation imports restore dftorch modules to avoid polluting later public API tests

key-files:
  created:
    - tests/test_f_orbital_skf.py
  modified:
    - src/dftorch/script.py
    - tests/test_f_orbital_skf.py

key-decisions:
  - "No Constants.py or Structure.py production metadata drift was exposed by the Phase 2 gate."
  - "Simple-format f-free metadata coverage uses a synthetic s-only SKF fixture and parsed-shell selection from tests/data_skf_mio-1-1 for sp and spd cases."
  - "The tracer checkpoint was not paused because this run was explicitly requested as a full sequential execution."

patterns-established:
  - "Metadata tests call reusable validation helpers instead of duplicating standalone script logic."
  - "Validation helpers that install a fake dftorch package must restore sys.modules before unrelated public API tests run."

requirements-completed: [BAS-01, BAS-02, BAS-03, BAS-04, BAS-05, BAS-06]

coverage:
  - id: D1
    description: "Pytest executes f-containing Constants and Structure metadata gates through the standalone validation harness."
    requirement: BAS-01
    verification:
      - kind: integration
        ref: "uv run python -m pytest tests/test_f_orbital_skf.py -q"
        status: pass
    human_judgment: false
  - id: D2
    description: "StructureBatch validation asserts flattened/global AO offsets, shell AO offsets, shell-index offsets, per-structure HDIM, and padded rows."
    requirement: BAS-05
    verification:
      - kind: integration
        ref: "uv run python src/dftorch/script.py tests/f_orbital_data"
        status: pass
    human_judgment: false
  - id: D3
    description: "Simple-format f-free metadata regression covers s-only, sp, and spd systems without H0/S, SCF, or tutorial parity execution."
    requirement: BAS-03
    verification:
      - kind: integration
        ref: "uv run python -m pytest tests/test_f_orbital_skf.py -q"
        status: pass
    human_judgment: false

duration: 6min
completed: 2026-07-27
status: complete
---

# Phase 02 Plan 01: Constants and Structure Basis Metadata Summary

**Pytest-backed Constants, Structure, and StructureBatch metadata validation for spdf and f-free simple-format basis layouts**

## Performance

- **Duration:** 6 min
- **Started:** 2026-07-27T16:25:15Z
- **Completed:** 2026-07-27T16:30:36Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- Promoted the f-containing `run_constants_tests()` and `run_structure_tests()` validation helpers into pytest via `tests/test_f_orbital_skf.py`.
- Extended StructureBatch validation in `src/dftorch/script.py` to assert `H_INDEX_START_GLOBAL`, `H_INDEX_END_GLOBAL`, `shell_ao_start_global`, `shell_ao_end_global`, `H_INDEX_START_U_GLOBAL`, `H_INDEX_END_U_GLOBAL`, `HDIM_struct`, and padded `diagonal`/`D0` rows.
- Added metadata-only f-free simple-format regression coverage for synthetic s-only, parsed sp, and parsed spd systems.
- Confirmed no Phase 2 metadata drift required production changes in `src/dftorch/Constants.py` or `src/dftorch/Structure.py`.

## Task Commits

1. **Task 1: Pytest tracer for f-containing Constants and Structure metadata** - `4e4cebd` (test)
2. **Task 2: Simple-format f-free metadata regressions for s-only, sp, and spd systems** - `2ba5a6e` (test)
3. **Task 3: Patch exposed metadata drift and run the Phase 2 gate** - `14ea562` (test)

## Files Created/Modified

- `tests/test_f_orbital_skf.py` - Adds pytest gates for parser/spline, Constants, Structure, and f-free simple-format metadata regressions.
- `src/dftorch/script.py` - Adds explicit StructureBatch global-offset and shell-index assertions to the existing validation harness.

## Decisions Made

- No production metadata patches were needed; the failing broad gate was caused by pytest module pollution from the validation loader, not by `Constants` or `Structure` drift.
- The s-only regression uses a temporary synthetic simple-format SKF fixture because repository fixtures do not provide a clean s-only oracle under current shell-presence rules.
- The sp and spd regression cases are selected by independently parsed homonuclear metadata from `tests/data_skf_mio-1-1`, avoiding hardcoded element assumptions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Isolated fake validation package imports from public API tests**
- **Found during:** Task 3
- **Issue:** `load_dftorch_module()` intentionally installs a fake `dftorch` package for validation, and pytest retained that module for later public API smoke tests.
- **Fix:** `run_with_float64()` now saves and restores all `dftorch` entries in `sys.modules` after each validation call.
- **Files modified:** `tests/test_f_orbital_skf.py`
- **Verification:** `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q`
- **Committed in:** `14ea562`

**2. [Process Adjustment] Continued after tracer commit for explicit sequential execution**
- **Found during:** Task 1
- **Issue:** The GSD tracer pattern normally pauses in interactive mode after the tracer commit, but the user explicitly requested complete sequential execution of plan 01.
- **Fix:** Continued to Tasks 2 and 3 after re-running and passing the focused tracer gate.
- **Files modified:** None
- **Verification:** All Phase 2 verification commands passed.
- **Committed in:** N/A

**Total deviations:** 1 auto-fixed issue, 1 process adjustment
**Impact on plan:** No scope creep; all changes stayed inside metadata validation and pytest harness behavior.

## Issues Encountered

- The broad pytest gate initially failed because the validation harness left a fake `dftorch` package in `sys.modules`. This was fixed in `14ea562`.
- Existing unrelated local edits were present before execution in `.planning/codebase/*`, `src/dftorch/_bond_integral.py`, and broader portions of `src/dftorch/script.py`; they were left unstaged except for the explicit StructureBatch assertion hunks needed by this plan.

## Verification

- `uv run python -m pytest tests/test_f_orbital_skf.py -q` — pass (`5 passed`)
- `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` — pass (`13 passed`)
- `uv run python src/dftorch/script.py tests/f_orbital_data` — pass

## Known Stubs

None.

## Authentication Gates

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 3 can consume the locked Constants, Structure, and StructureBatch metadata contract for H0/S routing and f angular block work. H0/S assembly, f angular Slater-Koster formulas, f-containing SCF, and tutorial parity remain intentionally deferred.

## Self-Check: PASSED

- Found summary file at `.planning/phases/02-constants-and-structure-basis-metadata/02-01-SUMMARY.md`.
- Found task commits `4e4cebd`, `2ba5a6e`, and `14ea562`.
- Verified required Phase 2 commands passed.

---
*Phase: 02-constants-and-structure-basis-metadata*
*Completed: 2026-07-27*
