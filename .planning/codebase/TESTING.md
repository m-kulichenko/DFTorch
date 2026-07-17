# Testing Patterns

**Analysis Date:** 2026-07-17

## Test Framework

**Runner:**
- Pytest `>=7.4`
- Config: `pyproject.toml`
- Test paths: `tests`
- Default addopts: `-ra -q`
- Markers:
  - `slow`: marks tests as slow and supports deselection with `-m 'not slow'`
  - `gpu`: marks tests that require a GPU

**Assertion Library:**
- Native pytest `assert` statements.
- Tensor assertions currently use PyTorch checks such as `torch.isfinite(...).all()`, `torch.is_tensor(...)`, and shape equality in `tests/test_scf.py`, `tests/test_io.py`, and `tests/test_nearestneighborlist.py`.

**Run Commands:**
```bash
uv run pytest              # Run all tests with pyproject addopts
uv run pytest -q           # Run quiet mode explicitly, as CI does
uv run pytest -m "not slow" # Exclude slow tests
```

## Test File Organization

**Location:**
- Tests are stored in top-level `tests/`.
- Test fixtures and scientific data live beside tests: `tests/ch4.xyz`, `tests/data_skf_mio-1-1/`, and `tests/f_orbital_data/`.
- There is no separate `tests/unit/` or `tests/integration/` split; test type is implied by filename and behavior.

**Naming:**
- Test files use `test_*.py`: `tests/test_import.py`, `tests/test_io.py`, `tests/test_public_api.py`, `tests/test_public_api_contract.py`, `tests/test_scf.py`, `tests/test_nearestneighborlist.py`.
- Test functions use `test_*` names, e.g. `test_energy_smoke_import_and_call()` in `tests/test_scf.py`.

**Structure:**
```
tests/
├── test_import.py                 # package and key module import smoke tests
├── test_public_api.py             # root import smoke test
├── test_public_api_contract.py    # expected public symbols
├── test_io.py                     # XYZ read smoke test
├── test_nearestneighborlist.py    # neighbor-list CPU smoke test
├── test_scf.py                    # small CPU SCF/forces smoke test
├── ch4.xyz                        # small molecule fixture
├── data_skf_mio-1-1/              # SKF fixture set for CHON/Zn/P/S/H tests
└── f_orbital_data/                # f-orbital SKF fixture set
```

## Test Structure

**Suite Organization:**
```python
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pathlib

import pytest


@pytest.mark.parametrize("device", ["cpu"])
def test_nearestneighborlist_small_xyz(device):
    root = pathlib.Path(__file__).resolve().parents[1]
    xyz_path = root / "tests" / "ch4.xyz"
    skf_dir = root / "tests" / "data_skf_mio-1-1"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"
```

**Patterns:**
- Disable TorchDynamo/Inductor at module import time for deterministic tests that avoid C++ toolchain requirements. This pattern appears in `tests/test_scf.py`, `tests/test_import.py`, and `tests/test_nearestneighborlist.py`.
- Use `pathlib.Path(__file__).resolve().parents[1]` to locate repo-root fixtures.
- Parameterize device even when only CPU is active: `@pytest.mark.parametrize("device", ["cpu"])` in `tests/test_scf.py` and `tests/test_nearestneighborlist.py`.
- Import heavy dependencies such as `torch` and core DFTorch modules inside tests after fixture preconditions. This keeps import smoke tests distinct and makes missing fixture failures clearer.
- Set `torch.set_default_dtype(torch.float64)` before scientific calculations that require double precision.
- Assert finite tensor outputs and expected shapes rather than exact floating-point values in smoke tests.

## Mocking

**Framework:** Not detected

**Patterns:**
```python
# No monkeypatch/unittest.mock pattern is used in current tests.
# Existing tests exercise small real fixtures from tests/ch4.xyz and tests/data_skf_mio-1-1/.
```

**What to Mock:**
- Mocking is not established in this test suite. Prefer real minimal fixtures for scientific paths when feasible, matching `tests/test_scf.py` and `tests/test_nearestneighborlist.py`.
- If optional external backends must be isolated, use pytest monkeypatching around environment variables or import boundaries, but keep the pattern local to the test file until it repeats.

**What NOT to Mock:**
- Do not mock the public import surface in `tests/test_public_api.py` or `tests/test_public_api_contract.py`; those tests guard actual package exports in `src/dftorch/__init__.py`.
- Do not mock `tests/ch4.xyz` or `tests/data_skf_mio-1-1/` for smoke tests that verify real parsing and SKF-backed execution.

## Fixtures and Factories

**Test Data:**
```python
root = pathlib.Path(__file__).resolve().parents[1]
xyz_path = root / "tests" / "ch4.xyz"
skf_dir = root / "tests" / "data_skf_mio-1-1"

dftorch_params = {
    "FILENAME": str(xyz_path),
    "CELL": [25.0, 25.0, 25.0],
    "SKFPATH": str(skf_dir) + "/",
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 8.0,
    "RCUT_REPULSIVE": 4.0,
    "COUL_METHOD": "PME",
    "SCF_MAX_ITER": 25,
    "KRYLOV_START": 5,
}
```

**Location:**
- Molecule fixture: `tests/ch4.xyz`
- Main SKF fixture directory: `tests/data_skf_mio-1-1/`
- f-orbital SKF fixture directory: `tests/f_orbital_data/`
- Inline minimal tensor fixtures are built in test functions, e.g. `Rx`, `Ry`, `Rz`, `TYPE`, and `cell` in `tests/test_nearestneighborlist.py`.

## Coverage

**Requirements:** `CONTRIBUTING.md` asks contributors to aim for `>80 %` test coverage on any new module. No coverage tool or enforced threshold is configured in `pyproject.toml` or CI.

**View Coverage:**
```bash
# Not configured. Add pytest-cov before using a coverage command.
```

## Test Types

**Unit Tests:**
- `tests/test_public_api.py` and `tests/test_public_api_contract.py` validate import-level behavior and public symbol contracts.
- `tests/test_io.py` validates `src/dftorch/_io.py` XYZ parsing with a small bundled fixture.
- `tests/test_nearestneighborlist.py` validates `src/dftorch/_nearestneighborlist.py` against a minimal CPU tensor setup and real constants.

**Integration Tests:**
- `tests/test_scf.py` runs a small CPU SCF plus force calculation through `Constants`, `Structure`, and `ESDriver`, using `tests/ch4.xyz` and `tests/data_skf_mio-1-1/`.
- `tests/test_import.py` combines import smoke checks with a neighbor-list smoke path.

**E2E Tests:**
- Not used. There is no browser, CLI, or full workflow E2E framework configured.

## Common Patterns

**Async Testing:**
```python
# Not used. Current tests are synchronous pytest functions.
```

**Error Testing:**
```python
assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"
```

- Explicit negative-path tests using `pytest.raises` are not currently present.
- When adding validation coverage for modules such as `src/dftorch/_cell.py`, prefer direct `pytest.raises(ValueError, match=...)` tests for invalid shapes and options.

---

*Testing analysis: 2026-07-17*
