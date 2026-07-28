---
phase: 02-constants-and-structure-basis-metadata
status: passed
verified: 2026-07-27
requirements:
  - BAS-01
  - BAS-02
  - BAS-03
  - BAS-04
  - BAS-05
  - BAS-06
human_verification: []
gaps: []
---

# Phase 02 Verification

## Verdict

PASSED.

Phase 02 delivers the intended metadata validation gate for `Constants`, `Structure`, and `StructureBatch` without adding H0/S assembly, f angular Slater-Koster formulas, f-containing SCF, or tutorial parity work.

## Requirement Results

| Requirement | Status | Evidence |
| --- | --- | --- |
| BAS-01 | PASS | `test_f_orbital_constants_metadata_gate` calls `run_constants_tests()` and validates f-shell counts, onsite values, Hubbard values, and occupations. |
| BAS-02 | PASS | Constants validation confirms spdf elements expose `n_orb == 16`, `max_ang == 4`, and `max_ang_occ == 4`. |
| BAS-03 | PASS | `test_simple_format_f_free_metadata_regression` covers simple-format s-only, sp, and spd metadata. |
| BAS-04 | PASS | Constants and Structure validation assert shared s/p/d/f shell dimensions. |
| BAS-05 | PASS | `run_structure_tests()` validates 16-AO spdf Structure metadata and StructureBatch global-offset fields. |
| BAS-06 | PASS | f-free simple-format metadata regression keeps `HDIM`, AO ranges, shell indexing, onsite diagonal, and `D0` behavior metadata-only. |

## Automated Checks

- `uv run python -m pytest tests/test_f_orbital_skf.py -q` — PASS, `5 passed`.
- `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py tests/test_scf.py -q` — PASS, `13 passed`.
- `uv run python src/dftorch/script.py tests/f_orbital_data` — PASS.

## Must-Haves

- Constants exposes f-shell metadata and preserves f-free metadata: PASS.
- Structure and StructureBatch expose the locked AO/shell metadata contract: PASS.
- StructureBatch explicitly validates `H_INDEX_START_GLOBAL`, `H_INDEX_END_GLOBAL`, `shell_ao_start_global`, `shell_ao_end_global`, `H_INDEX_START_U_GLOBAL`, and `H_INDEX_END_U_GLOBAL`: PASS.
- No Phase 2 scope creep into H0/S, f angular formulas, f-containing SCF, or tutorial parity: PASS.

## Human Verification

None required. All Phase 02 deliverables are covered by automated checks.
