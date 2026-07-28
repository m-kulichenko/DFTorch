# Coding Conventions

**Analysis Date:** 2026-07-20
**Last Mapped Commit:** `e824543a0b411dcf52462ee55db5362c360e7780`
**Scope:** `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`, `tests/f_orbital_data/`

## Naming Patterns

**Files:**
- Use the existing mixed module naming in the scoped code. Public, class-oriented modules keep PascalCase filenames: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, and `src/dftorch/ESDriver.py`.
- Internal implementation modules use leading-underscore snake_case filenames: `src/dftorch/_bond_integral.py`.
- Standalone validation utilities in this scope use a plain lowercase script filename: `src/dftorch/script.py`.
- f-orbital SKF fixtures use element-pair filenames under `tests/f_orbital_data/`, such as `tests/f_orbital_data/Eu-Eu.skf`, `tests/f_orbital_data/Eu-Ga.skf`, and `tests/f_orbital_data/N-N.skf`. Prefer dashed `Element-Element.skf` names for new scoped fixtures because `src/dftorch/script.py` validates dashed names with `split_dashed_pair()`.

**Functions:**
- Use snake_case for new functions and helpers: `_ao_mask_from_shell_present()`, `_shell_local_start()`, `_global_shell_start()`, and `_atomic_density_matrix_from_shells()` in `src/dftorch/Structure.py`; `_normalize_skf_row()`, `_resolve_skf_path()`, `_validate_nested_shells()`, and `read_skf_table()` in `src/dftorch/_bond_integral.py`; `check_one_skf()` and `run_structure_tests()` in `src/dftorch/script.py`.
- Use a leading underscore for private helpers that should not become package API: `_split_skf_pair_name()` in `src/dftorch/_bond_integral.py`, `_flatten_ao_labels()` in `src/dftorch/Structure.py`, and `_StructProxy` inside `src/dftorch/ESDriver.py`.
- Preserve domain-specific uppercase names where the surrounding scientific code uses them as data model fields or tensors: `TYPE`, `RX`, `RY`, `RZ`, `HDIM`, `H_INDEX_START`, `H_INDEX_END`, `N_ORB`, `MAX_ANG`, `N_F`, `EF`, `UF`, and `SHELL_PRESENT` in `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, and `src/dftorch/_bond_integral.py`.

**Variables:**
- Use lowercase snake_case for ordinary locals: `shell_present`, `atom_ao_start`, `pair_lookup`, `skf_path`, `project_root`, `metadata_dict`, and `batch_elements`.
- Use uppercase names for tensor metadata that mirrors SKF/header concepts: `R_tensor`, `R_orb`, `MAX_ANG_OCC`, `TORE`, `N_S`, `N_P`, `N_D`, `N_F`, `ES`, `EP`, `ED`, `EF`, `US`, `UP`, `UD`, `UF`, and `SHELL_PRESENT` in `src/dftorch/Constants.py` and `src/dftorch/_bond_integral.py`.
- Keep shell and AO templates module-level and uppercase in `src/dftorch/Structure.py`: `SHELL_DIMS`, `SHELL_LOCAL_STARTS`, `SHELL_TYPE_IDS`, `AO_LABEL_TEMPLATE`, and `AO_SHELL_TEMPLATE`.
- Keep parser channel tables module-level and uppercase in `src/dftorch/_bond_integral.py`: `_CHANNELS`, `_SIMPLE_CHANNELS`, `_SIMPLE_TO_EXTENDED`, `SK_BLOCK_SIZE`, `N_SK_CHANNELS`, `MAX_SHELLS`, `EV_PER_HARTREE`, and `BOHR_TO_ANGSTROM`.

**Types:**
- Public classes use PascalCase and inherit from `torch.nn.Module`: `Constants` and `ConstantsTest` in `src/dftorch/Constants.py`, `Structure` and `StructureBatch` in `src/dftorch/Structure.py`, and `ESDriver` and `ESDriverBatch` in `src/dftorch/ESDriver.py`.
- Use `torch.Tensor` type annotations for tensor helpers in newer scoped code, especially `src/dftorch/Structure.py`, `src/dftorch/_bond_integral.py`, and `src/dftorch/script.py`.
- Use `typing.Any` only at boundaries where the concrete constants/structure type is broader than the scoped module can express, such as `const: Any` in `src/dftorch/Structure.py`.

## Code Style

**Formatting:**
- Use Python 3 style with four-space indentation.
- Use `from __future__ import annotations` in modules that use modern annotation syntax: `src/dftorch/Structure.py`, `src/dftorch/_bond_integral.py`, and `src/dftorch/script.py`.
- Keep f-orbital tensor layout expressed through templates and masks instead of repeated ad hoc offsets. Use `AO_LABEL_TEMPLATE`, `AO_SHELL_TEMPLATE`, `_ao_mask_from_shell_present()`, `_shell_local_start()`, and `_global_shell_start()` in `src/dftorch/Structure.py`.
- Prefer line breaks around long tuple returns and tensor stacks, matching `get_skf_tensors()` in `src/dftorch/_bond_integral.py` and the 16-entry onsite templates in `src/dftorch/Structure.py`.
- Formatting tool configuration is not detected in the scoped paths. Preserve the style already present in the touched file when editing.

**Linting:**
- Lint configuration is not detected in the scoped paths.
- Avoid introducing new broad lint suppressions. Scoped code currently uses local suppression only where a historical variable name conflicts with lint rules, such as `for l in f:  # noqa: E741` in `src/dftorch/_bond_integral.py`.
- Do not add new unconditional side-effect imports. `src/dftorch/script.py` intentionally uses `importlib.util` and `types.ModuleType` to load scoped modules without importing `dftorch/__init__.py`.

## Import Organization

**Order:**
1. `from __future__ import annotations` when present, as in `src/dftorch/Structure.py`, `src/dftorch/_bond_integral.py`, and `src/dftorch/script.py`.
2. Standard library imports: `math`, `os`, `re`, `sys`, `time`, `types`, `pathlib.Path`, and `typing`.
3. Third-party imports: `numpy` and `torch`.
4. Local package imports: relative imports in package modules such as `from ._io import read_pdb, read_xyz` in `src/dftorch/Structure.py`; package imports in legacy scoped code such as `from dftorch._coulomb_matrix_batch import coulomb_matrix_vectorized_batch` in `src/dftorch/ESDriver.py`.

**Path Aliases:**
- No scoped custom path aliases are detected.
- Use relative imports for new package-internal code in `src/dftorch/Structure.py` and `src/dftorch/_bond_integral.py`.
- Keep `src/dftorch/script.py` independent from package root import side effects by loading modules with `load_dftorch_module()` and `ensure_fake_dftorch_package()`.

## Error Handling

**Patterns:**
- Raise `ValueError` for malformed SKF content, unsupported shell layouts, invalid file names, and invalid scientific configuration. Examples: `_normalize_skf_row()` and `_validate_nested_shells()` in `src/dftorch/_bond_integral.py`, `split_dashed_pair()` in `src/dftorch/script.py`, and charge/spin validation in `src/dftorch/Structure.py`.
- Raise `FileNotFoundError` for missing fixture directories or empty SKF sets, as in `run_bond_integral_tests()` and `main()` in `src/dftorch/script.py`.
- Raise `RuntimeError` for loader failures that are environmental rather than data-validation errors, such as `find_project_root()` and `load_dftorch_module()` in `src/dftorch/script.py`.
- Raise `AssertionError` with exact expected/actual values in validation checks. `src/dftorch/script.py` uses helper functions such as `assert_int_metadata()`, `assert_float_metadata()`, `assert_bool_metadata()`, and `assert_tensor_float_list()` for actionable failures.
- Use `NotImplementedError` for unsupported algorithm branches in runtime drivers, such as off-diagonal DFTB3 with PME in `src/dftorch/ESDriver.py`.
- Catch only expected optional-file failures. `src/dftorch/Constants.py` catches `(FileNotFoundError, OSError, ValueError)` when optional `spinw.txt` cannot be loaded.

## Logging

**Framework:** `print()` in scoped paths

**Patterns:**
- `src/dftorch/script.py` is a command-line validation script and uses `print()` for progress, per-file pass/fail summaries, and final status.
- `src/dftorch/Constants.py` prints optional SOC and DFTB3 status messages during initialization.
- `src/dftorch/ESDriver.py` uses `verbose` flags for timing/progress output on expensive runtime paths. New scoped runtime output should be guarded by an existing `verbose` argument when possible.
- Do not add unconditional print output to reusable parser or structure helpers in `src/dftorch/_bond_integral.py` or `src/dftorch/Structure.py`; return data or raise exceptions instead.

## Comments

**When to Comment:**
- Comment f-orbital ordering and scientific assumptions where they define external compatibility. `src/dftorch/Structure.py` documents the cubic-harmonic f-label order and the atom-major AO layout.
- Keep parser comments close to layout decisions, such as simple-to-extended 20-to-40 channel normalization in `src/dftorch/_bond_integral.py`.
- Use comments in `src/dftorch/script.py` to divide validation stages and clarify independently computed expectations.
- Avoid adding personal-note comments or stale TODOs. Existing scoped comments such as `# Aryan NOTE`, `# old function`, and broad TODOs should not be copied into new code.

**JSDoc/TSDoc:**
- Not applicable; this is Python.
- Use Python docstrings. Prefer concise NumPy-style sections for parser and scientific helpers, as in `read_skf_table()` and `_normalize_skf_row()` in `src/dftorch/_bond_integral.py`.
- For validation helpers in `src/dftorch/script.py`, use short docstrings that state the invariant being independently checked.

## Function Design

**Size:** Keep new f-orbital helpers focused on one transformation: shell presence to AO mask, local shell starts, global shell starts, row normalization, SKF path resolution, or metadata assertion. Long scientific driver methods exist in `src/dftorch/ESDriver.py`; new f-orbital changes should avoid expanding them unless integration with `H0_and_S_vectorized()` requires it.

**Parameters:** Pass tensors and metadata explicitly. `read_skf_table()` in `src/dftorch/_bond_integral.py` receives mutable metadata tensors (`N_ORB`, `N_F`, `EF`, `UF`, `SHELL_PRESENT`) rather than reaching into globals. `Structure` helpers in `src/dftorch/Structure.py` receive `shell_present`, `shell_local_start`, and `atom_ao_start` directly.

**Return Values:** Return tensors, tuples, or dictionaries that preserve existing call contracts. `get_skf_tensors()` in `src/dftorch/_bond_integral.py` returns a long tuple consumed by `Constants.__init__()` in `src/dftorch/Constants.py`; when adding metadata, update both sides together and add validation in `src/dftorch/script.py`.

## Module Design

**Exports:** `src/dftorch/Structure.py` exports `Structure` and `StructureBatch` through `__all__`. Scoped internal helpers remain unexported. `src/dftorch/_bond_integral.py` has no scoped `__all__`; treat leading-underscore names as private.

**Barrel Files:** Not detected in scoped paths. Do not add a new barrel file for f-orbital helpers; keep parser logic in `src/dftorch/_bond_integral.py`, structure indexing in `src/dftorch/Structure.py`, and runtime integration in `src/dftorch/Constants.py` or `src/dftorch/ESDriver.py`.

---

*Convention analysis: 2026-07-20*
