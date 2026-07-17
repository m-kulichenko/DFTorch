# Coding Conventions

**Analysis Date:** 2026-07-17

## Naming Patterns

**Files:**
- Use package modules under `src/dftorch/`.
- Internal implementation modules use leading-underscore snake_case filenames, e.g. `src/dftorch/_cell.py`, `src/dftorch/_tools.py`, `src/dftorch/_io.py`, `src/dftorch/_nearestneighborlist.py`.
- Public class-oriented modules keep historical PascalCase filenames, e.g. `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`.
- Tests use `test_*.py` under `tests/`, e.g. `tests/test_scf.py`, `tests/test_io.py`, `tests/test_public_api_contract.py`.
- Do not add new implementation code under `src/dftorch/_legacy/`; Ruff excludes that subtree in `pyproject.toml`.

**Functions:**
- Use snake_case for functions and helpers: `normalize_cell()` in `src/dftorch/_cell.py`, `ordered_pairs_from_TYPE()` in `src/dftorch/_tools.py`, `read_xyz()` in `src/dftorch/_io.py`.
- Use a leading underscore for private helpers and implementation details: `_maybe_compile()` and `_degen_symeig` in `src/dftorch/_tools.py`, `_ensure_parent_dir()` in `src/dftorch/_io.py`, `_pair_lookup_for_const()` in `src/dftorch/_nearestneighborlist.py`.
- Preserve domain abbreviations used by the package APIs when extending existing call sites, e.g. `TYPE`, `RX`, `RY`, `RZ`, `Nats`, `H_INDEX_START`, and `H_INDEX_END` in `src/dftorch/Structure.py`.
- Factory helpers use verb phrases such as `create_gbsa()` in `src/dftorch/_gbsa.py`, `create_thirdorder()` in `src/dftorch/_thirdorder.py`, and `create_dftd3()` in `src/dftorch/_dftd3.py`.

**Variables:**
- Use snake_case for local Python variables: `root`, `xyz_path`, `skf_dir`, `dftorch_params` in `tests/test_scf.py`.
- Tensor and chemistry state often uses domain-specific uppercase names: `TYPE`, `COORDS`, `Rcut`, `N`, `RX`, `RY`, `RZ`, `Ftot`. Match the surrounding module style when modifying scientific kernels in `src/dftorch/`.
- Configuration dictionaries use uppercase string keys such as `"FILENAME"`, `"CELL"`, `"SKFPATH"`, `"T_ELECTRONIC"`, and `"SCF_MAX_ITER"` in `tests/test_scf.py` and `src/dftorch/Structure.py`.
- Module-level constants use uppercase names: `_COMPILE_ENABLED` and `DEGEN_THRESHOLD` in `src/dftorch/_tools.py`, `SYMBOL_TO_NUMBER` and `NUMBER_TO_SYMBOL` in `src/dftorch/_io.py`.

**Types:**
- Public classes use PascalCase: `Structure`, `StructureBatch`, `Constants`, `ESDriver`, `ESDriverBatch`, `MDXL`, `MDXLBatch`, `MDXLOS`, and `GeoOpt`.
- Internal classes may use a leading underscore when they are not part of the public API, e.g. `_degen_symeig` in `src/dftorch/_tools.py`.
- Type annotations are used in newer modules with `from __future__ import annotations`, e.g. `src/dftorch/_cell.py`, `src/dftorch/_tools.py`, `src/dftorch/_io.py`, and `src/dftorch/Structure.py`.

## Code Style

**Formatting:**
- Use Ruff formatting. The pre-commit hook `ruff-format` is configured in `.pre-commit-config.yaml`; CI checks `uv run ruff format --check .` in `.github/workflows/tests.yml`.
- Run formatting locally with:

```bash
uv run ruff format .
```

- Line length rule `E501` is ignored in `pyproject.toml`; do not force awkward scientific formulas or long parameter dictionaries into unreadable shapes only to satisfy line width.
- Keep the source compatible with Python `>=3.11` per `pyproject.toml`.

**Linting:**
- Use Ruff linting. The pre-commit hook runs `ruff --fix`; CI runs `uv run ruff check .`.
- Ruff selects `E`, `F`, `S`, `I`, and `PERF` in `pyproject.toml`.
- Ruff ignores `E501` and `S311` globally in `pyproject.toml`.
- Per-file ignores in `pyproject.toml`:
  - `__init__.py`: `F401` for re-export imports.
  - `tests/*.py`: `S101` for pytest `assert` usage.
  - `experiments/*.py`: `E402` for notebook/script import placement.
- Use local `# noqa` only for intentionally preserved exceptions, e.g. import smoke tests in `tests/test_import.py` and `tests/test_public_api_contract.py`. Avoid broad file-level disables like `# ruff: noqa` unless isolating legacy or parser-heavy code such as `src/dftorch/_io.py`.

## Import Organization

**Order:**
1. `from __future__ import annotations` when using modern annotations in source modules.
2. Standard library imports: `os`, `re`, `pathlib`, `typing`.
3. Third-party imports: `numpy`, `torch`, `pytest`.
4. Local package imports: relative imports inside `src/dftorch/`, absolute `dftorch` imports in tests.

**Path Aliases:**
- No custom Python path aliases are configured. Package code imports with relative paths such as `from ._cell import normalize_cell` in `src/dftorch/Structure.py`.
- Tests import the installed package via `dftorch`, e.g. `from dftorch.Constants import Constants` in `tests/test_scf.py`.
- Public user-facing imports must come from package root `dftorch`, and `src/dftorch/__init__.py` defines `__all__`.

## Error Handling

**Patterns:**
- Validate inputs early and raise specific built-in exceptions. Use `ValueError` for invalid user/configuration values such as invalid cell shapes in `src/dftorch/_cell.py` and charge/spin combinations in `src/dftorch/Structure.py`.
- Use `ImportError` for optional backend requirements, e.g. ALCHEMI neighbor-list availability in `src/dftorch/_nearestneighborlist.py` and ML model dependencies in `src/dftorch/_ml_sk.py`.
- Use `NotImplementedError` for unsupported algorithm branches, e.g. unsupported electronic driver modes in `src/dftorch/ESDriver.py`.
- Include actionable exception text that names the bad shape, option, or missing capability: `src/dftorch/_cell.py` reports accepted `LBox` shapes; `src/dftorch/ESDriver.py` reports unsupported batched PME.
- Tests should assert preconditions for bundled fixtures before running expensive scientific paths, as in `tests/test_scf.py`, `tests/test_io.py`, and `tests/test_nearestneighborlist.py`.

## Logging

**Framework:** `console` plus limited `logging`

**Patterns:**
- Most runtime status output uses guarded `print()` calls behind `verbose` parameters in scientific routines, e.g. `src/dftorch/_nearestneighborlist.py`, `src/dftorch/_h0ands.py`, `src/dftorch/_scf.py`, and `src/dftorch/Optimizer.py`.
- `src/dftorch/ewald_pme/__init__.py` uses the standard `logging` module to report Triton/PyTorch backend selection.
- New reusable library code should prefer returning values and raising exceptions over unconditional `print()` output. If progress output is needed, expose a `verbose` flag consistent with `src/dftorch/_nearestneighborlist.py` and `src/dftorch/ESDriver.py`.

## Comments

**When to Comment:**
- Use comments to explain numerical stability, scientific assumptions, backend constraints, or non-obvious tensor transformations. Good examples are the degenerate eigensolver notes in `src/dftorch/_tools.py` and periodic wrapping notes in `src/dftorch/Structure.py`.
- Keep comments near domain-heavy parameter dictionaries when they clarify units or algorithm meaning, as in `tests/test_scf.py`.
- Avoid adding comments that restate simple assignments. Several existing modules include historical TODO-style notes; new code should be more precise and actionable.

**JSDoc/TSDoc:**
- Not applicable; this is a Python codebase.
- Use Python docstrings, preferably NumPy-style sections (`Parameters`, `Returns`, `Notes`, `Examples`) as in `src/dftorch/_cell.py`, `src/dftorch/_tools.py`, `src/dftorch/_io.py`, and `src/dftorch/Structure.py`.

## Function Design

**Size:** Keep new functions focused and testable. Scientific kernels can be longer when vectorized tensor operations require staged calculations, but prefer extracting reusable helpers such as `_maybe_compile()` in `src/dftorch/_tools.py`, `_ensure_parent_dir()` in `src/dftorch/_io.py`, and `normalize_cell()` in `src/dftorch/_cell.py`.

**Parameters:** Pass explicit tensors, constants containers, and configuration dictionaries. Preserve established parameter names for domain APIs (`TYPE`, `Rx`, `Ry`, `Rz`, `cell`, `Rcut`, `const`) when extending functions such as `vectorized_nearestneighborlist()` in `src/dftorch/_nearestneighborlist.py`.

**Return Values:** Return tensors, tuples of tensors, or mutate established structure objects according to the module pattern. For public constructors and drivers, preserve object attributes expected by tests, e.g. `structure1.e_tot` and `structure1.f_tot` in `tests/test_scf.py`.

## Module Design

**Exports:** Public exports live in `src/dftorch/__init__.py`. When adding supported public API symbols, import them there, add them to `__all__`, and update `tests/test_public_api_contract.py`.

**Barrel Files:** `src/dftorch/__init__.py` is the package barrel. Do not create additional broad barrel modules unless a subpackage needs a stable public surface similar to `src/dftorch/ewald_pme/__init__.py`.

---

*Convention analysis: 2026-07-17*
