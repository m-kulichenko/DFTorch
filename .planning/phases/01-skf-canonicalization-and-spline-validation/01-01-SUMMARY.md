---
phase: 01-skf-canonicalization-and-spline-validation
plan: 01
subsystem: testing
tags: [skf, f-orbitals, spline, parser, validation]

requires: []
provides:
  - Strengthened `src/dftorch/script.py` parser and spline validation for Phase 1.
  - Added pytest-collected Phase 1 SKF parser/spline coverage.
  - Verified simple-format SKF parsing through `read_skf_table()`.
  - Verified compact/dashed pair helper behavior and skipped-shell errors.
affects: [phase-2-constants-and-structure-basis-metadata, skf-parser]

tech-stack:
  added: []
  patterns:
    - Temporary parser fixtures created inside the validation harness.
    - Parser-boundary assertions against production `_bond_integral.py` helpers.

key-files:
  created:
    - tests/test_f_orbital_skf.py
  modified:
    - src/dftorch/script.py
    - src/dftorch/_bond_integral.py

key-decisions:
  - "`src/dftorch/script.py` remains the f-orbital validation harness, and pytest now wraps the parser/spline gate."
  - "Simple-format coverage now parses a temporary minimal `.skf` through `read_skf_table()` instead of only testing row normalization."
  - "Dashed and compact pair-name parsing now return the same tuple contract."

patterns-established:
  - "Temporary minimal `.skf` files can be used to validate parser dialect behavior without expanding permanent fixture scope."
  - "Validation output reports source-grid spline maxima with file, row, and channel context."

requirements-completed:
  - SKF-01
  - SKF-02
  - SKF-03
  - SKF-04
  - SKF-05
  - SKF-06
  - SKF-07
  - SPL-01
  - SPL-02
  - SPL-03
  - SPL-04

coverage:
  - id: D1
    description: "Simple and extended SKF rows are verified through the normalized 40-channel parser contract."
    requirement: SKF-03
    verification:
      - kind: integration
        ref: "uv run python src/dftorch/script.py tests/f_orbital_data"
        status: pass
    human_judgment: false
  - id: D2
    description: "Temporary simple-format `.skf` parsing validates mapped legacy channels, f-channel zero-fill, and spline reconstruction."
    requirement: SPL-02
    verification:
      - kind: integration
        ref: "uv run python src/dftorch/script.py tests/f_orbital_data"
        status: pass
    human_judgment: false
  - id: D3
    description: "Compact/dashed pair helpers and skipped-shell error checks are covered by the validation harness."
    requirement: SKF-05
    verification:
      - kind: integration
        ref: "uv run python src/dftorch/script.py tests/f_orbital_data"
        status: pass
    human_judgment: false
  - id: D4
    description: "Pytest-collected parser/spline coverage runs on CPU without optional GPU, Triton, DFTB+, or ASE dependencies."
    requirement: SPL-04
    verification:
      - kind: unit
        ref: "uv run python -m pytest tests/test_f_orbital_skf.py -q"
        status: pass
      - kind: unit
        ref: "uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_scf.py -q"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-07-21
status: complete
---

# Phase 1: SKF Canonicalization and Spline Validation Summary

**Parser-boundary validation now proves simple and extended SKF inputs reach the same 40-channel representation.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-07-20T23:03:00Z
- **Completed:** 2026-07-21T19:32:00Z
- **Tasks:** 3 completed
- **Files modified:** 2 source files plus 1 pytest file

## Accomplishments

- Added temporary simple-format `.skf` validation through `read_skf_table()`, proving 20-column input maps into the 40-channel internal representation.
- Added compact/dashed filename and pair-name helper checks, including two-letter compact names like `EuGa`.
- Added skipped-shell negative checks for p-without-s, d-without-p, and f-without-s/p/d layouts.
- Strengthened spline reconstruction output for all 9 `tests/f_orbital_data/*.skf` files plus the temporary simple-format file.
- Added pytest coverage for the Phase 1 parser/spline gate, including a compact-only fixture-directory check.

## Task Commits

No commits were created during execution because the executor subagent stalled before close-out and `.git` writes require explicit local approval in this sandbox. Source changes are present in the working tree.

## Files Created/Modified

- `src/dftorch/script.py` - Adds Phase 1 parser/spline assertions and clearer source-grid reconstruction output.
- `src/dftorch/_bond_integral.py` - Returns a tuple for dashed SKF pair names, matching the compact-name helper contract.
- `tests/test_f_orbital_skf.py` - Pytest wrapper for the Phase 1 parser/spline gate and compact-only SKF directories.
- `.planning/phases/01-skf-canonicalization-and-spline-validation/01-01-SUMMARY.md` - Records this execution summary.

## Decisions Made

- Kept validation in `src/dftorch/script.py` per the locked Phase 1 context, with pytest wrapping the parser/spline gate for SPL-04.
- Used temporary files for simple-format and resolver coverage instead of adding permanent fixtures.
- Left pytest conversion deferred; existing pytest is a secondary regression check only.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Dashed pair helper returned a list instead of a tuple**
- **Found during:** Task 2 verification
- **Issue:** `bond._split_skf_pair_name("Eu-Ga")` returned `["Eu", "Ga"]`, while compact parsing returned `("Eu", "Ga")`.
- **Fix:** Changed the dashed branch to unpack and return `(elem_a, elem_b)`.
- **Files modified:** `src/dftorch/_bond_integral.py`
- **Verification:** `uv run python src/dftorch/script.py tests/f_orbital_data`
- **Committed in:** Not committed; local working tree only.

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix is within Phase 1 parser-boundary scope and makes helper return types consistent.

**2. [Review Finding] Script helpers still assumed dashed filenames**
- **Found during:** Code review
- **Issue:** Compact-only SKF directories failed because independent validation helpers used dashed-only pair/path logic.
- **Fix:** Routed script pair parsing and homonuclear path lookup through the production pair splitter/resolver.
- **Files modified:** `src/dftorch/script.py`
- **Verification:** Compact-only temporary copy of `tests/f_orbital_data` passed.

**3. [Verifier Gap] SPL-04 required pytest-collected parser/spline coverage**
- **Found during:** Phase verification
- **Issue:** Parser/spline validation existed only in `src/dftorch/script.py`.
- **Fix:** Added `tests/test_f_orbital_skf.py` to execute the Phase 1 parser/spline gate through pytest.
- **Files created:** `tests/test_f_orbital_skf.py`
- **Verification:** `uv run python -m pytest tests/test_f_orbital_skf.py -q`

## Issues Encountered

- The spawned executor agent stalled after applying source edits but before writing the summary. The orchestrator shut it down, verified the partial edits, fixed the failing helper assertion, reran both verification commands, and completed this summary inline.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run python src/dftorch/script.py tests/f_orbital_data` — passed.
- Compact-only temporary copy of `tests/f_orbital_data` through `src/dftorch/script.py` — passed.
- `uv run python -m pytest tests/test_f_orbital_skf.py -q` — passed (`2 passed`).
- `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_scf.py -q` — passed (`8 passed`).

## Next Phase Readiness

Phase 1 parser/spline validation is ready to support Phase 2 planning for `Constants` and `Structure` basis metadata. The next phase can rely on the canonical 40-channel parser contract and the f metadata checks already present in the validation harness.

---
*Phase: 01-skf-canonicalization-and-spline-validation*
*Completed: 2026-07-21*
