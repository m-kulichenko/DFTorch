---
phase: 05-regression-safety-and-support-policy-cleanup
plan: 03
subsystem: testing-and-package-cleanup
tags: [skf, independent-oracle, pytest, structure-batch, cleanup, cln-05, d-03]

requires:
  - phase: 01-skf-canonicalization-and-spline-validation
    provides: the parser, spline, Constants, and Structure checks formerly hosted by script.py
  - phase: 03-h0-s-routing-and-f-angular-blocks
    provides: the source-locked spdf AO order checked by the independent expectation tables
provides:
  - Raw SKF header metadata expectations built without any production parser helper
  - A per-function disposition record for every check and derivation in the deleted script
  - Pytest coverage for single and batched Structure AO layout
  - Removal of script.py and its fake-package filesystem import machinery from the runtime package
affects: [05-05, 05-07, package-surface, skf-regression-tests]

tech-stack:
  added: []
  patterns:
    - "Independent oracle: parse trusted fixture bytes with a standard-library-only helper that cannot import the production package"
    - "Deletion audit: mechanically enumerate source functions, assign exactly one disposition, then delete in a separate later commit"
    - "Second written source: duplicate basis expectation tables intentionally so a one-sided production edit fails"

key-files:
  created:
    - docs/SCRIPT-PY-PORT-INVENTORY.md
  modified: []
  deleted:
    - src/dftorch/script.py
  verified-existing:
    - tests/skf_header_oracle.py
    - tests/skf_validation_support.py
    - tests/test_skf_metadata_oracle.py

key-decisions:
  - "Keep the raw-header oracle in tests/skf_header_oracle.py, a standard-library-only module, and drive production Constants/Structure objects from a separate pytest module. This is stronger than colocating both paths."
  - "Use the current source as the inventory authority: it contains 16 check_/run_ functions and 28 check/run/expected/parse functions, not the stale plan-time count of 30 check_/run_ functions."
  - "Do not alter the two inherited CH4 exact-float failures or unrelated print sites; their evidence is recorded for plans 05-01 and 05-05 respectively."

patterns-established:
  - "Oracle independence is executable: inspect helper source and the module AST for forbidden imports and parser symbols."
  - "A cleanup deletion follows, never precedes, a committed inventory and focused green test gate."

requirements-completed: [CLN-05]

coverage:
  - id: D1
    description: "SKF onsite energies, Hubbard values, occupations, and shell metadata are checked against an independent raw-header re-parse for simple and extended fixtures."
    requirement: CLN-05
    verification:
      - kind: integration
        ref: "uv run pytest tests/test_skf_metadata_oracle.py -q"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every check/derivation in the deleted runtime script has one mechanically auditable disposition with named pytest evidence or a concrete drop reason."
    requirement: CLN-05
    verification:
      - kind: other
        ref: "inventory AST audit: 28 source names == 28 inventory rows; all covered test names exist"
        status: pass
    human_judgment: false
  - id: D3
    description: "src/dftorch/script.py and its fake-package filesystem loader are absent from the runtime package while the ported oracle remains green."
    requirement: CLN-05
    verification:
      - kind: integration
        ref: "uv run python -c importlib.util.find_spec('dftorch.script') is None"
        status: pass
      - kind: integration
        ref: "uv run pytest tests/test_skf_metadata_oracle.py -q after deletion"
        status: pass
    human_judgment: false
  - id: D4
    description: "The repository-wide regression suite preserves the working prototype after cleanup."
    requirement: CLN-05
    verification:
      - kind: integration
        ref: "uv run pytest -q"
        status: fail
    human_judgment: true
    rationale: "Two exact CH4 dH0 checksum assertions fail by 3.4e-13 on the unchanged pre-05-03 Hamiltonian path; the focused 05-03 gates pass, and the inherited failure is deferred to its 05-01 owner."

metrics:
  duration: 10 min
  tasks: 2
  files-created: 1
  files-deleted: 1
  focused-suite: 56 passed / 0 failed
  full-suite: 150 passed / 2 pre-existing failed
  completed: 2026-07-31

status: complete
---

# Phase 5 Plan 03: Independent SKF Oracle and Runtime Script Removal Summary

**Raw SKF headers now have a standard-library-only oracle under pytest, every legacy check has a written disposition, and the 1661-line runtime validation script is gone.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-07-31T19:33:56Z
- **Completed:** 2026-07-31T19:43:34Z
- **Tasks:** 2
- **Task files changed:** 2

## Accomplishments

- Audited the runtime script from an authoritative 28-name grep and recorded every check, derivation, expectation table, and support function with exactly one disposition and evidence.
- Verified 56 parametrized oracle tests across simple and extended SKF headers, including production-independent onsite, Hubbard, occupation, shell, AO-layout, and StructureBatch checks.
- Deleted `src/dftorch/script.py` only after the audit and focused gate were committed; editable reinstall confirms `dftorch.script` is not importable.
- Removed exactly 57 `print(` calls and the synthetic-package/filesystem-import machinery from the shipped package.

## Task Commits

1. **Task 1: Inventory every check and preserve the independent oracle** - `4d158dc` (`docs`)
2. **Task 2: Delete script.py from the runtime package** - `bc8c983` (`refactor`)

## Files Created/Modified

- `docs/SCRIPT-PY-PORT-INVENTORY.md` - Mechanical 28-row disposition table plus expectation-table and support-function audit.
- `src/dftorch/script.py` - Deleted after its unique checks were verified under pytest.
- `tests/skf_header_oracle.py` - Verified existing standard-library-only raw-header parser and basis expectation source.
- `tests/test_skf_metadata_oracle.py` - Verified existing 56-case independent-oracle and StructureBatch gate.
- `.planning/phases/05-regression-safety-and-support-policy-cleanup/deferred-items.md` - Recorded inherited numeric drift and the updated print-count baseline.

## Decisions Made

- The independent oracle remains isolated from production imports in `tests/skf_header_oracle.py`; `tests/test_skf_metadata_oracle.py` is the comparison boundary that imports production objects.
- The source file, not stale plan prose, determines inventory completeness. The audited script contained 16 `check_`/`run_` functions and 28 functions in the expanded mechanical sweep.
- The plan does not rewrite exact-float radial-grid assertions or classify unrelated library output. Those concerns remain with their owning plans.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale plan fact] Corrected the function-count premise**

- **Found during:** Task 1 mechanical sweep.
- **Issue:** The plan stated that `grep -c '^def \(check\|run\)_'` returned 30. The audited 1661-line file returns 16; the expanded command required by the action returns 28.
- **Fix:** Used the current file as the authority, created exactly 28 primary inventory rows, and added separate tables for five expectation constants and all 24 remaining top-level support functions.
- **Files modified:** `docs/SCRIPT-PY-PORT-INVENTORY.md`
- **Verification:** AST audit reports `primary rows: 28; source names: 28`, and every named covered test exists.
- **Committed in:** `4d158dc`

---

**Total deviations:** 1 auto-fixed stale plan fact. **Impact:** The audit is stricter than the stale count and no implementation scope changed.

## Issues Encountered

### Pre-existing full-suite exact-float failures

`uv run pytest -q` executes 152 tests: 150 pass and two assertions in `tests/test_radial_grid.py` fail because the measured CH4 derivative absolute sum is `777.8475256952825` while `CH4_DH0_ABS_SUM` is `777.8475256952822` (delta `3.410605131648481e-13`). Both failures reproduce in isolation and predate every 05-03 change. The new inventory is prose-only and script deletion cannot affect H0/dH0 arithmetic. Per the executor scope rule, neither the literal nor production Hamiltonian code was altered; full evidence is in `deferred-items.md` for plan 05-01's owner.

### D-02 print-count drift

The deleted script contains exactly 57 `print(` calls, but the current top-level runtime baseline after deletion is 124 across the same 15 modules, not the plan-time 112. Including tracked subpackages yields 182 across 20 modules. Plan 05-05 owns D-02, so 05-03 changed no unrelated print site and recorded both baselines for that plan to regenerate its inventory.

### Pre-existing port artifacts

The execution baseline already contained `tests/skf_header_oracle.py`, `tests/skf_validation_support.py`, and `tests/test_skf_metadata_oracle.py` in commit `56091af`. Task 1 verified those artifacts and supplied the missing auditable disposition document rather than rewriting working user-owned tests.

## Authentication Gates

None.

## Known Stubs

None. The created inventory contains no placeholder implementation, and the deleted module introduced no replacement runtime surface.

## User Setup Required

None - no external service configuration required. `uv pip install -e .` reinstalled only the local package.

## Verification

| Check | Result |
|---|---|
| `uv run pytest tests/test_skf_metadata_oracle.py -q` | **56 passed / 0 failed** before and after deletion |
| Inventory/source equality audit | **28 rows == 28 source functions; all covered tests exist** |
| Oracle independence source/AST gate | **passed**; no production parser import or symbol is reachable |
| ASCII read of `tests/test_skf_metadata_oracle.py` | **passed** |
| `importlib.util.find_spec('dftorch.script') is None` after editable reinstall | **passed** |
| `git log --oneline -2` | deletion `bc8c983` follows inventory `4d158dc` |
| Runtime script print removals | **57 removed** |
| `uv run pytest -q` | **150 passed / 2 inherited exact-float failures**; deferred with evidence |

## Next Phase Readiness

- Plan 05-04 can proceed; no parser, package import, or scientific behavior was changed by this cleanup.
- Plan 05-05 must use the measured 124-call top-level print baseline (or explicitly include subpackages and use 182), not the stale 112 literal.
- The two CH4 derivative checksum failures remain assigned to plan 05-01's regression owner and do not depend on `script.py`.

## Self-Check: PASSED

Created artifacts exist, the deleted runtime module is absent, both task
commits are reachable, the 28 inventory rows match the source recovered from
git history in order, the oracle test reads as ASCII, and coverage metadata
classifies without schema errors.

---
*Phase: 05-regression-safety-and-support-policy-cleanup*
*Completed: 2026-07-31*
