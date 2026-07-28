# Phase 02: Constants and Structure Basis Metadata - Research

**Researched:** 2026-07-24  
**Scope:** BAS-01 through BAS-06 only  
**Confidence:** HIGH for local code/test findings; no external package or formula research performed.

## Key Findings

- Phase 2 should plan a metadata/test hardening pass, not an H0/S implementation pass. The locked context excludes H0/S assembly and f angular Slater-Koster formulas from this phase. [VERIFIED: `.planning/phases/02-constants-and-structure-basis-metadata/02-CONTEXT.md`]
- `Constants` already registers the Phase 2 f-shell metadata surface: `shell_dim`, `n_orb`, `max_ang`, `max_ang_occ`, `n_s`, `n_p`, `n_d`, `n_f`, `Es`, `Ep`, `Ed`, `Ef`, `U`, `Up`, `Ud`, `Uf`, `shell_present`, `coeffs_tensor`, and `pair_lookup`. [VERIFIED: `src/dftorch/Constants.py:65`, `src/dftorch/Constants.py:101`, `src/dftorch/Constants.py:148`]
- `Structure` and `StructureBatch` already build explicit shell and AO metadata for s/p/d/f layouts, including 16-position AO templates, shell starts/ends, onsite diagonals, `D0`, shell-resolved Hubbard arrays, and shell index ranges. [VERIFIED: `src/dftorch/Structure.py:11`, `src/dftorch/Structure.py:90`, `src/dftorch/Structure.py:333`, `src/dftorch/Structure.py:461`]
- `src/dftorch/script.py` already has standalone Phase 2 validation helpers for `Constants`, `Structure`, and `StructureBatch`, and `uv run python src/dftorch/script.py tests/f_orbital_data` passed on 2026-07-24. [VERIFIED: command run 2026-07-24]
- Pytest currently exposes only `run_bond_integral_tests()` from the validation script; `run_constants_tests()` and `run_structure_tests()` are not yet called from `tests/test_f_orbital_skf.py`. [VERIFIED: `tests/test_f_orbital_skf.py:29`, `tests/test_f_orbital_skf.py:46`]

## Existing Implementation State

| Area | Current state | Planner implication |
| --- | --- | --- |
| `Constants.shell_dim` | Set to `[0, 1, 3, 5, 7]` as a non-trainable parameter. [VERIFIED: `src/dftorch/Constants.py:65`] | Keep this exact representation for BAS-04; add pytest assertion if not already covered by promoted validation. |
| `Constants` f metadata | Consumes `N_F`, `EF`, `UF`, `SHELL_PRESENT` from `get_skf_tensors()` and registers `n_f`, `Ef`, `Uf`, `shell_present`. [VERIFIED: `src/dftorch/Constants.py:101`, `src/dftorch/Constants.py:171`] | BAS-01/BAS-02 likely need validation exposure more than implementation. |
| `Constants` ordered pairs | Builds `pair_lookup` from `ordered_pairs_from_TYPE(TYPE)` and stores 40-channel `coeffs_tensor`. [VERIFIED: `src/dftorch/Constants.py:88`, `src/dftorch/Constants.py:148`] | Validation should assert all Eu/Ga/N ordered pairs are present; existing script does this. |
| `Structure` AO layout | Defines local shell starts `(0,1,4,9)`, shell dims `(1,3,5,7)`, and 16 AO labels including f labels. [VERIFIED: `src/dftorch/Structure.py:11`, `src/dftorch/Structure.py:25`] | BAS-05 should assert exact AO labels, shell types, and starts/ends. |
| `Structure` single metadata | Computes `n_orbitals_per_atom`, `H_INDEX_START`, `H_INDEX_END`, `shell_present`, `has_f`, shell AO starts/ends, diagonal, `D0`, `Hubbard_U_sr`, `shell_types`, `el_per_shell`, `H_INDEX_START_U`, and `H_INDEX_END_U`. [VERIFIED: `src/dftorch/Structure.py:333`, `src/dftorch/Structure.py:363`, `src/dftorch/Structure.py:408`, `src/dftorch/Structure.py:443`] | BAS-05/BAS-06 should lock these as the downstream metadata contract. |
| `StructureBatch` metadata | Mirrors single-structure metadata with padded `diagonal`/`D0`, `HDIM_struct`, global AO offsets, and global shell offsets. [VERIFIED: `src/dftorch/Structure.py:532`, `src/dftorch/Structure.py:594`, `src/dftorch/Structure.py:620`, `src/dftorch/Structure.py:655`] | The plan should include batch validation because Phase 2 context explicitly names `StructureBatch`. |
| Standalone validation | `main()` runs bond-integral, constants, and structure checks in sequence. [VERIFIED: `src/dftorch/script.py:1604`] | Keep as human-readable harness; add or update pytest wrappers for CI-style gate. |
| Existing simple-format tests | Smoke tests use `tests/data_skf_mio-1-1` and `tests/ch4.xyz` for import, neighbor list, and SCF/forces. [VERIFIED: `tests/test_import.py:42`, `tests/test_nearestneighborlist.py:19`, `tests/test_scf.py:22`] | BAS-03/BAS-06 should add metadata-specific simple-format assertions; do not require tutorial numerical parity. |

## Validation Architecture

Required validation tasks for the planner:

| Requirement | Validation task | Suggested path/command |
| --- | --- | --- |
| BAS-01 | Assert `Constants` stores `n_f`, `Ef`, `Uf`, and f reference occupation from Eu homonuclear metadata; assert f-free Ga/N values remain zero. [VERIFIED: `src/dftorch/script.py:988`] | Add pytest wrapper calling `run_constants_tests(project_root, skf_dir, bond)` in `tests/test_f_orbital_skf.py`; run `uv run python -m pytest tests/test_f_orbital_skf.py -q`. |
| BAS-02 | Assert Eu has `n_orb == 16`, `max_ang == 4`, and `max_ang_occ == 4`. [VERIFIED: script output 2026-07-24] | Same promoted constants pytest; keep standalone `uv run python src/dftorch/script.py tests/f_orbital_data`. |
| BAS-03 | Add simple-format metadata regression coverage. `tests/data_skf_mio-1-1` gives sp examples such as C and spd examples such as P/S/Zn under current parser rules; add a tiny synthetic s-only homonuclear simple fixture because H-H has nonzero `Ep` and parses as sp. [VERIFIED: `tests/data_skf_mio-1-1/C-C.skf`, `tests/data_skf_mio-1-1/P-P.skf`, `tests/data_skf_mio-1-1/H-H.skf`] | New focused pytest in `tests/test_f_orbital_skf.py` or a new `tests/test_basis_metadata.py`; do not run H0/S assembly. |
| BAS-04 | Assert `Constants.shell_dim == [0, 1, 3, 5, 7]` and `Structure.SHELL_DIMS == (1, 3, 5, 7)`. [VERIFIED: `src/dftorch/Constants.py:65`, `src/dftorch/Structure.py:11`] | Include in constants/structure metadata pytest. |
| BAS-05 | Assert a minimal f-containing structure has `[4, 4, 16]` orbitals for `['N', 'Ga', 'Eu']`, `HDIM == 24`, correct AO/shell starts, labels, onsite diagonal, `D0`, shell Hubbard values, and shell index ranges. [VERIFIED: script output 2026-07-24] | Add pytest wrapper calling `run_structure_tests(project_root, skf_dir, bond)`; no H0/S call. |
| BAS-06 | Assert existing f-free simple-format metadata keeps expected `HDIM`, onsite diagonal, shell indexing, and `D0`. [VERIFIED: requirement in `.planning/REQUIREMENTS.md:34`] | Add metadata-only test constructing `Constants` and `Structure` for `tests/ch4.xyz` with `tests/data_skf_mio-1-1`; compare current expected values from parsed homonuclear headers. |

Commands verified during research:

```bash
uv run python src/dftorch/script.py tests/f_orbital_data
uv run python -m pytest tests/test_f_orbital_skf.py -q
```

Both commands passed on 2026-07-24. [VERIFIED: command run 2026-07-24]

Phase gate recommendation:

```bash
uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py -q
uv run python src/dftorch/script.py tests/f_orbital_data
```

Do not require `experiments/1_tutorial.ipynb` numerical parity, H0/S assembly, SCF over f-containing systems, or f angular Slater-Koster formulas in this phase. [VERIFIED: `.planning/phases/02-constants-and-structure-basis-metadata/02-CONTEXT.md`]

## Risks

- Pytest false confidence: the existing pytest file passes while skipping the `Constants` and `Structure` portions of the standalone harness. [VERIFIED: `tests/test_f_orbital_skf.py:36`, `src/dftorch/script.py:1606`]
- Simple-format regression gap: current Phase 2 standalone f fixture check proves Ga/N f-free metadata in the f fixture directory, but BAS-03/BAS-06 also require preserving existing simple-format behavior; the repository's `mio-1-1` H-H fixture is not a clean s-only oracle under current presence rules because `Ep` is nonzero. [VERIFIED: `.planning/REQUIREMENTS.md:31`, `.planning/REQUIREMENTS.md:34`, `tests/data_skf_mio-1-1/H-H.skf`]
- Metadata duplication: shell dimensions and AO ordering are encoded in multiple places (`Constants.shell_dim`, `Structure.SHELL_DIMS`, validation expected templates), so a one-file edit can drift unless tests assert all copies together. [VERIFIED: `src/dftorch/Constants.py:65`, `src/dftorch/Structure.py:11`, `src/dftorch/script.py:1139`]
- Batch padding semantics can hide errors if tests only inspect flattened values. `StructureBatch.diagonal` and `StructureBatch.D0` are padded to max per-batch `HDIM`, while true per-structure sizes live in `HDIM_struct`. [VERIFIED: `src/dftorch/Structure.py:594`, `src/dftorch/Structure.py:667`]
- `Constants` prints optional SOC/DFTB3 status during construction; tests should tolerate stdout or avoid brittle output matching. [VERIFIED: command output 2026-07-24]
- `ConstantsTest` still contains a parallel/stale constants table that infers `shell_present` from `max_ang`; avoid using it as the Phase 2 oracle. [VERIFIED: `src/dftorch/Constants.py:205`, `src/dftorch/Constants.py:1077`]

## Recommended Plan Shape

1. Promote existing Phase 2 harness checks into pytest.
   - Add tests that call `run_constants_tests()` and `run_structure_tests()` from `src/dftorch/script.py`.
   - Keep CPU `torch.float64` setup consistent with `tests/test_f_orbital_skf.py`.

2. Add simple-format metadata regression coverage.
   - Use `tests/data_skf_mio-1-1` for sp/spd coverage and add a synthetic s-only simple SKF fixture for the s-only case.
   - Assert metadata only: `Constants` shell dimensions/counts and `Structure` `HDIM`, diagonal, shell indices, and `D0`.
   - Do not invoke `ESDriver`, H0/S assembly, SCF, or notebook parity.

3. Patch production code only if the new pytest assertions expose drift.
   - Likely target paths: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, and `src/dftorch/script.py`.
   - Avoid broad cleanup of remaining H0/S `1/4/9` assumptions; Phase 2 should only guard metadata assumptions that directly affect BAS-01 through BAS-06.

4. Run the Phase 2 gate.
   - `uv run python -m pytest tests/test_f_orbital_skf.py tests/test_import.py tests/test_io.py tests/test_nearestneighborlist.py tests/test_public_api_contract.py -q`
   - `uv run python src/dftorch/script.py tests/f_orbital_data`

5. Leave explicit deferrals in the plan.
   - f angular Slater-Koster formulas: Phase 3.
   - f-containing H0/S assembly execution: Phase 3.
   - tutorial numerical parity: later regression/end-to-end validation phase.
