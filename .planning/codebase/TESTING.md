# Testing Patterns

**Analysis Date:** 2026-07-20
**Last Mapped Commit:** `e824543a0b411dcf52462ee55db5362c360e7780`
**Scope:** `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`, `tests/f_orbital_data/`

## Test Framework

**Runner:**
- Formal pytest configuration is not detected in the scoped paths.
- The scoped f-orbital validation runner is the standalone script `src/dftorch/script.py`.
- `src/dftorch/script.py` validates the f-orbital SKF parser, constants registration, structure AO bookkeeping, and batch structure padding against real SKF fixtures in `tests/f_orbital_data/`.

**Assertion Library:**
- Native Python `assert` is not used in the scoped validation script.
- Use explicit `raise AssertionError(...)` with expected and actual values, following helpers in `src/dftorch/script.py`: `assert_int_metadata()`, `assert_float_metadata()`, `assert_bool_metadata()`, `assert_tensor_int_list()`, `assert_tensor_bool_list()`, and `assert_tensor_float_list()`.
- Tensor comparisons convert to Python scalars or lists before comparing, then include index-specific failure text.

**Run Commands:**
```bash
python src/dftorch/script.py tests/f_orbital_data # Run scoped f-orbital validation against explicit fixture directory
python src/dftorch/script.py                      # Run scoped f-orbital validation against default tests/f_orbital_data
```

## Test File Organization

**Location:**
- The scoped validation harness lives in `src/dftorch/script.py`.
- The scoped fixture directory is `tests/f_orbital_data/`.
- Fixture files include `tests/f_orbital_data/Eu-Eu.skf`, `tests/f_orbital_data/Eu-Ga.skf`, `tests/f_orbital_data/Eu-N.skf`, `tests/f_orbital_data/Ga-Eu.skf`, `tests/f_orbital_data/Ga-Ga.skf`, `tests/f_orbital_data/Ga-N.skf`, `tests/f_orbital_data/N-Eu.skf`, `tests/f_orbital_data/N-Ga.skf`, and `tests/f_orbital_data/N-N.skf`.
- `.DS_Store` exists in `tests/f_orbital_data/`; validation code only reads `*.skf` files via `skf_dir.glob("*.skf")` in `src/dftorch/script.py`.

**Naming:**
- Scoped executable validation is not named `test_*.py`; it is `src/dftorch/script.py`.
- Scoped SKF fixtures use `Element-Element.skf` dashed pair names.
- Validation functions in `src/dftorch/script.py` use verb phrases that name the unit under check: `check_one_skf()`, `check_get_skf_tensors_metadata()`, `run_bond_integral_tests()`, `check_constants_against_expected()`, `run_constants_tests()`, `check_single_structure_layout()`, `check_batch_structure_layout()`, and `run_structure_tests()`.

**Structure:**
```text
src/dftorch/script.py
├── Project/module loading helpers
├── Shared SKF parsing helpers used by the tests
├── Metadata expectations independently parsed from homonuclear SKF headers
├── Assertion helpers
├── _bond_integral.py tests
├── Constants.py tests
├── Structure.py tests
└── main()

tests/f_orbital_data/
├── Eu-Eu.skf
├── Eu-Ga.skf
├── Eu-N.skf
├── Ga-Eu.skf
├── Ga-Ga.skf
├── Ga-N.skf
├── N-Eu.skf
├── N-Ga.skf
└── N-N.skf
```

## Test Structure

**Suite Organization:**
```python
def main() -> None:
    project_root = find_project_root()
    skf_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else project_root / "tests" / "f_orbital_data"
    if not skf_dir.is_dir():
        raise FileNotFoundError(f"SKF directory not found: {skf_dir}")

    bond = load_dftorch_module(project_root, "_bond_integral")
    device = torch.device("cpu")
    dtype = torch.float64

    failures = []
    failures.extend(run_bond_integral_tests(skf_dir, bond, device, dtype))
    failures.extend(run_constants_tests(project_root, skf_dir, bond))
    failures.extend(run_structure_tests(project_root, skf_dir, bond))

    if failures:
        raise AssertionError(f"{len(failures)} test check(s) failed")
```

**Patterns:**
- Run scoped validation on CPU with `torch.float64`, as set in `main()` in `src/dftorch/script.py`.
- Use `find_project_root()` in `src/dftorch/script.py` so the script works from either the project root or the module path.
- Use `ensure_fake_dftorch_package()` and `load_dftorch_module()` in `src/dftorch/script.py` to load `src/dftorch/_bond_integral.py`, `src/dftorch/Constants.py`, and `src/dftorch/Structure.py` without importing package root side effects.
- Parse expected f-orbital metadata independently in `parse_expected_homonuclear_metadata()` in `src/dftorch/script.py`; do not reuse production parser output as its own oracle.
- Collect failures into lists of dictionaries in `run_bond_integral_tests()`, `run_constants_tests()`, and `run_structure_tests()` so one failed section does not hide later checks.
- Use temporary synthetic XYZ files under `.tmp_f_orbital_validation` from `run_constants_tests()` and `run_structure_tests()` in `src/dftorch/script.py`; the synthetic coordinates exist only to drive `Constants`, `Structure`, and `StructureBatch` construction.

## Mocking

**Framework:** Not detected in scoped paths

**Patterns:**
```python
fake_pkg = types.ModuleType("dftorch")
fake_pkg.__path__ = [str(package_dir)]
sys.modules["dftorch"] = fake_pkg
```

**What to Mock:**
- Use the fake package pattern in `ensure_fake_dftorch_package()` only to isolate scoped module loading from unrelated `dftorch/__init__.py` imports.
- Synthetic XYZ files are acceptable fixtures for `src/dftorch/Constants.py` and `src/dftorch/Structure.py` because the validation target is species, metadata, AO labels, shell starts/ends, and density initialization rather than chemistry of the coordinates.

**What NOT to Mock:**
- Do not mock `tests/f_orbital_data/*.skf`; the scoped validation must exercise real SKF parsing, 20-to-40 or 40-channel table handling, homonuclear metadata, and pair lookup behavior.
- Do not mock `src/dftorch/_bond_integral.py` when validating `src/dftorch/Constants.py`; `Constants.__init__()` is expected to consume the real `get_skf_tensors()` tuple and expose `n_f`, `Ef`, `Uf`, and `shell_present`.
- Do not mock `src/dftorch/Structure.py` when validating AO layout; `check_single_structure_layout()` and `check_batch_structure_layout()` compare real `Structure` and `StructureBatch` attributes.

## Fixtures and Factories

**Test Data:**
```python
skf_dir = project_root / "tests" / "f_orbital_data"
elements = collect_elements_from_skf_dir(skf_dir, bond)
TYPE = torch.tensor([bond.symbol_to_number[sym] for sym in elements], dtype=torch.long, device=device)
```

**Location:**
- f-orbital SKF fixtures: `tests/f_orbital_data/`
- Runtime synthetic XYZ directory: `.tmp_f_orbital_validation` created by `src/dftorch/script.py`
- Fixture factory for synthetic coordinate input: `write_test_xyz()` in `src/dftorch/script.py`
- Constants factory for structure checks: `build_test_constants()` in `src/dftorch/script.py`
- Expected metadata factory: `expected_metadata_by_element()` in `src/dftorch/script.py`

## Coverage

**Requirements:** Not detected in scoped paths.

**View Coverage:**
```bash
# Not detected in scoped paths.
```

## Test Types

**Unit Tests:**
- Parser row/channel unit checks live in `check_one_skf()` in `src/dftorch/script.py`. They call `read_skf_table()` from `src/dftorch/_bond_integral.py`, normalize channels with `channels_to_matrix()`, build splines with `cubic_spline_coeffs()`, and verify reconstructed left/right grid values.
- Metadata helper checks live in `check_metadata_from_single_read()` and `check_get_skf_tensors_metadata()` in `src/dftorch/script.py`. They validate `N_ORB`, `MAX_ANG`, `MAX_ANG_OCC`, `TORE`, `N_S`, `N_P`, `N_D`, `N_F`, `ES`, `EP`, `ED`, `EF`, `US`, `UP`, `UD`, `UF`, and `SHELL_PRESENT`.

**Integration Tests:**
- `run_constants_tests()` in `src/dftorch/script.py` instantiates `Constants` from `src/dftorch/Constants.py` against real `tests/f_orbital_data/*.skf` files and checks attributes exposed from `get_skf_tensors()`.
- `run_structure_tests()` in `src/dftorch/script.py` instantiates `Structure` and `StructureBatch` from `src/dftorch/Structure.py`, then checks atom-major AO starts/ends, global batch offsets, onsite diagonal values, `D0`, shell-resolved Hubbard data, shell type IDs, electron counts per shell, and AO labels.
- `src/dftorch/ESDriver.py` is covered only indirectly by scoped conventions. No scoped validation path runs `ESDriver.forward()` against f-orbital fixtures.

**E2E Tests:**
- Not detected in scoped paths. `src/dftorch/script.py` is a scoped validation script, not a full user workflow or CLI E2E test.

## Common Patterns

**Async Testing:**
```python
# Not used in scoped paths. Validation functions are synchronous.
```

**Error Testing:**
```python
if coeffs_tensor.shape[2] != 40:
    raise AssertionError(f"get_skf_tensors coeffs_tensor expected 40 channels, got {coeffs_tensor.shape[2]}")
```

- Prefer direct exception raising with exact source context. `src/dftorch/script.py` includes the source SKF file name in failures from `assert_int_metadata()`, `assert_float_metadata()`, and `assert_bool_metadata()`.
- For malformed SKF parser coverage in future scoped tests, assert `ValueError` messages from `_normalize_skf_row()`, `_split_skf_pair_name()`, `_validate_nested_shells()`, and `read_skf_table()` in `src/dftorch/_bond_integral.py`.
- For runtime unsupported-branch coverage, assert `NotImplementedError` from the PME/off-diagonal DFTB3 branch in `src/dftorch/ESDriver.py`.

---

*Testing analysis: 2026-07-20*
