# Phase 1: SKF Canonicalization and Spline Validation - Pattern Map

**Mapped:** 2026-07-20
**Files analyzed:** 2
**Analogs found:** 2 / 2

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/dftorch/_bond_integral.py` | parser / utility | file-I/O + transform | `src/dftorch/_bond_integral.py` | exact-existing |
| `src/dftorch/script.py` | validation harness | file-I/O + batch validation | `src/dftorch/script.py` | exact-existing |

Phase 1 should modify only these existing files unless the implementation needs transient temp files inside the existing script. Do not broaden this phase into `Constants.py`, `Structure.py`, H0/S formula validation, or pytest migration.

## Pattern Assignments

### `src/dftorch/_bond_integral.py` (parser / utility, file-I/O + transform)

**Analog:** `src/dftorch/_bond_integral.py`

**Imports pattern** (lines 1-10):
```python
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import torch

from ._tools import ordered_pairs_from_TYPE
```

Use local helpers in this module for parser-boundary behavior. Keep PyTorch tensor allocation inside parser APIs and keep ordered-pair discovery delegated to `_tools.ordered_pairs_from_TYPE`.

**Canonical channel contract** (lines 101-167):
```python
_CHANNELS: Final[list[str]] = [
    "Hff0",
    "Hff1",
    "Hff2",
    "Hff3",
    ...
    "Ssd0",
    "Ssp0",
    "Sss0",
]

_SIMPLE_CHANNELS: Final[list[str]] = [
    "Hdd0",
    "Hdd1",
    "Hdd2",
    ...
    "Ssd0",
    "Ssp0",
    "Sss0",
]

_SIMPLE_TO_EXTENDED: Final[list[int]] = [_CHANNELS.index(ch) for ch in _SIMPLE_CHANNELS]
```

Copy this pattern when adding assertions or documentation: `_CHANNELS` is the canonical 40-channel internal order; `_SIMPLE_CHANNELS` is only a parser-input compatibility map.

**Simple-to-extended normalization pattern** (lines 397-430):
```python
def _normalize_skf_row(tokens: list[str], path: str, line: str) -> list[float]:
    values = [float(x) for x in tokens]
    if len(values) == len(_CHANNELS):
        return values
    if len(values) == len(_SIMPLE_CHANNELS):
        row = [0.0] * len(_CHANNELS)
        for old_idx, new_idx in enumerate(_SIMPLE_TO_EXTENDED):
            row[new_idx] = values[old_idx]
        return row
    raise ValueError(
        f"Expected 20 or 40 electronic values in {path}, got {len(values)} in line: {line}"
    )
```

Planner should preserve this boundary: raw 20-column and 40-column row differences end here. New validation should assert that f-only indices are zero for simple rows and that mapped s/p/d values survive at `_SIMPLE_TO_EXTENDED`.

**Filename compatibility pattern** (lines 433-462):
```python
def _resolve_skf_path(skfpath: str, label_name: str) -> str:
    dashed = os.path.join(skfpath, f"{label_name}.skf")
    if os.path.isfile(dashed):
        return dashed

    undashed = os.path.join(skfpath, f"{label_name.replace('-', '')}.skf")
    if os.path.isfile(undashed):
        return undashed

    return dashed
```

Use this for compact/dashed coverage. Validation should call production resolution directly; do not duplicate resolver logic in the script.

**Pair-name parsing pattern** (lines 465-493):
```python
def _split_skf_pair_name(name: str) -> tuple[str, str]:
    if "-" in name:
        return name.split("-", 1)

    symbols = sorted(symbol_to_number, key=len, reverse=True)
    for elem_a in symbols:
        if not name.startswith(elem_a):
            continue
        elem_b = name[len(elem_a) :]
        if elem_b in symbol_to_number:
            return elem_a, elem_b

    raise ValueError(f"Could not parse SKF pair name: {name}")
```

When adding parser checks for compact names, assert two-letter symbols are parsed greedily, for example `EuGa -> ("Eu", "Ga")`.

**Unsupported-layout guard pattern** (lines 496-527):
```python
def _validate_nested_shells(
    elem: str,
    has_s: bool,
    has_p: bool,
    has_d: bool,
    has_f: bool,
    path: str,
) -> None:
    if has_f and not (has_s and has_p and has_d):
        raise ValueError(f"{path}: {elem} f-shell basis requires nested s/p/d/f shells")
    if has_d and not (has_s and has_p):
        raise ValueError(f"{path}: {elem} d-shell basis requires nested s/p/d shells")
    if has_p and not has_s:
        raise ValueError(f"{path}: {elem} p-shell basis requires an s shell")
```

New negative validation should reuse this helper directly or trigger it through a minimal parser path. Expected failures should check `ValueError` and include the shell requirement text.

**Parser table flow** (lines 547-712):
```python
def read_skf_table(...):
    lines = Path(path).read_text(errors="ignore").splitlines()
    data_lines = [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]
    ...
    elemA, elemB = _split_skf_pair_name(name)
    homonuclear = elemA == elemB
    ...
    if homonuclear:
        ...
        _validate_nested_shells(elemA, has_s, has_p, has_d, has_f, path)
        ...
        SHELL_PRESENT[el_num] = torch.tensor(
            shell_presence,
            dtype=torch.bool,
            device=SHELL_PRESENT.device,
        )
    ...
    for ln in data_lines[start_idx : start_idx + npts_read - 1]:
        tokens = _expand_tokens(ln.replace(",", " ").split())
        rows.append(_normalize_skf_row(tokens, path, ln))
```

Preserve this data flow: read text, strip comments, parse pair name, update homonuclear metadata in-place, normalize each electronic row, then build tensors.

**Spline coefficient pattern** (lines 779-829):
```python
def cubic_spline_coeffs(R: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    n, m = M.shape
    h = (R[1:] - R[:-1]).unsqueeze(1)

    A = torch.zeros((n, n), dtype=R.dtype, device=R.device)
    rhs = torch.zeros((n, m), dtype=R.dtype, device=R.device)
    ...
    c = torch.linalg.solve(A, rhs)
    ...
    coeffs = torch.stack([a, b, c[:-1], d], dim=2)
    return coeffs
```

Do not introduce a new spline engine in Phase 1. Validation should prove this existing coefficient path reconstructs source knots for the accepted row formats.

**Tensor assembly pattern** (lines 947-1083):
```python
def get_skf_tensors(TYPE: torch.Tensor, skfpath: str) -> tuple[...]:
    _, _, label_list = ordered_pairs_from_TYPE(TYPE)
    ...
    coeffs_tensor = torch.zeros(
        (n_pairs, npts, len(_CHANNELS), 4),
        dtype=dtype,
        device=device,
    )
    ...
    for i, label in enumerate(label_list):
        R_orb_i, channels, R_rep, rep_splines, close_exp = read_skf_table(
            _resolve_skf_path(skfpath, label),
            ...
        )

        channels_matrix = channels_to_matrix(channels)
        coeffs = cubic_spline_coeffs(R_orb_i, channels_matrix)
        ...
    ...
    return (
        R_tensor,
        R_orb,
        coeffs_tensor,
        ...
        N_F,
        ...
        EF,
        ...
        UF,
        SHELL_PRESENT,
    )
```

Planner should keep downstream-facing tensors 40-channel. Any Phase 1 assertion against `get_skf_tensors()` should check `coeffs_tensor.shape[2] == 40` and selected metadata only; broader constants/structure behavior is later-phase scope.

**Error handling pattern** (lines 592-603, 702-738, 751-759):
```python
if not data_lines:
    raise ValueError(f"Empty or comment-only SKF file: {path}")
...
if len(first) < 2:
    raise ValueError(f"Malformed SKF grid line in {path}: {data_lines[grid_idx]}")
...
if start_idx + npts_read - 1 > len(data_lines):
    raise ValueError(
        f"Electronic table in {path} is shorter than expected: "
        f"need {npts_read - 1} rows from index {start_idx}, "
        f"have {max(len(data_lines) - start_idx, 0)} candidate rows"
    )
...
if spline_start is None:
    raise ValueError(f"No Spline block found in {path}")
```

Use direct `ValueError` with file path and malformed detail. Do not silently coerce unsupported parser states.

---

### `src/dftorch/script.py` (validation harness, file-I/O + batch validation)

**Analog:** `src/dftorch/script.py`

**Imports and runner constants** (lines 39-50):
```python
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ATOL = 1.0e-9
EV_PER_HARTREE = 27.21138625
```

Keep Phase 1 validation dependency-free beyond `torch` and standard-library modules already used here. Use CPU `torch.float64`.

**Direct module-loading pattern** (lines 58-99):
```python
def find_project_root() -> Path:
    here = Path(__file__).resolve()

    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "dftorch" / "_bond_integral.py").is_file():
            return candidate
    ...

def ensure_fake_dftorch_package(project_root: Path) -> Path:
    package_dir = project_root / "src" / "dftorch"
    fake_pkg = types.ModuleType("dftorch")
    fake_pkg.__path__ = [str(package_dir)]
    sys.modules["dftorch"] = fake_pkg
    return package_dir

def load_dftorch_module(project_root: Path, module_basename: str):
    package_dir = ensure_fake_dftorch_package(project_root)
    module_path = package_dir / f"{module_basename}.py"
    module_name = f"dftorch.{module_basename}"
    ...
    spec.loader.exec_module(module)
    return module
```

Use this pattern for any new script checks so the harness remains runnable without importing unrelated public package side effects.

**Independent file parsing pattern** (lines 107-144):
```python
def read_data_lines(skf_path: Path) -> list[str]:
    lines = skf_path.read_text(errors="ignore").splitlines()
    return [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]

def split_dashed_pair(skf_path: Path) -> tuple[str, str]:
    name = skf_path.stem
    parts = name.split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected dashed SKF name like Eu-N.skf, got {skf_path.name}")
    return parts[0], parts[1]

def get_original_electronic_row_count(skf_path: Path) -> int:
    data_lines = read_data_lines(skf_path)
    ...
    npts_read = int(first[1])
    return npts_read - 1
```

Keep fixture-oracle parsing independent from production `read_skf_table()`. For compact-name parser checks, do not force all independent helpers to support compact names; add a targeted production-helper check instead.

**Expected homonuclear metadata pattern** (lines 174-259):
```python
def parse_expected_homonuclear_metadata(skf_path: Path, bond) -> dict[str, object] | None:
    elem_a, elem_b = split_dashed_pair(skf_path)
    if elem_a != elem_b:
        return None
    ...
    if extended:
        ...
        ff,
        fd,
        fp,
        fs,
    ) = (float(x) for x in header_tokens[:13])
    else:
        ...
        Ef = 0.0
        Uf = 0.0
        ff = 0.0
    ...
    return {
        "N_F": ff,
        "EF": Ef * EV_PER_HARTREE,
        "UF": Uf * EV_PER_HARTREE,
        "SHELL_PRESENT": shell_present,
    }
```

Use this style when strengthening homonuclear f metadata assertions: expected values come from independent header parsing, not from production parser output.

**Assertion helper pattern** (lines 267-300):
```python
def assert_int_metadata(actual: torch.Tensor, index: int, expected: int, name: str, source: str) -> None:
    got = int(actual[index].item())
    if got != expected:
        raise AssertionError(f"{source}: {name}[{index}] expected {expected}, got {got}")

def assert_float_metadata(...):
    got = float(actual[index].item())
    err = abs(got - expected)
    if err > atol:
        raise AssertionError(
            f"{source}: {name}[{index}] expected {expected:.16e}, got {got:.16e}, err={err:.3e}"
        )

def max_error_location(err: torch.Tensor, channel_names: list[str]) -> tuple[float, int, str]:
    flat_idx = int(torch.argmax(err).item())
    n_channels = err.shape[1]
    row_idx = flat_idx // n_channels
    channel_idx = flat_idx % n_channels
    return float(err[row_idx, channel_idx].item()), row_idx, channel_names[channel_idx]
```

New validations should raise `AssertionError` with source filename, field/channel name, expected value, actual value, and numeric error where relevant.

**Metadata tensor factory pattern** (lines 308-353):
```python
def make_metadata_tensors(device: torch.device, dtype: torch.dtype):
    n_elements = 120
    max_shells = 4

    N_ORB = torch.zeros(n_elements, dtype=torch.int64, device=device)
    MAX_ANG = torch.zeros(n_elements, dtype=torch.int64, device=device)
    MAX_ANG_OCC = torch.zeros(n_elements, dtype=torch.int64, device=device)
    ...
    N_F = torch.zeros(n_elements, dtype=dtype, device=device)
    ...
    EF = torch.zeros(n_elements, dtype=dtype, device=device)
    ...
    UF = torch.zeros(n_elements, dtype=dtype, device=device)

    SHELL_PRESENT = torch.zeros((n_elements, max_shells), dtype=torch.bool, device=device)
```

Use this factory when testing `read_skf_table()` directly. Do not construct partial metadata tuples by hand.

**Spline reconstruction check pattern** (lines 398-455):
```python
def check_one_skf(skf_path: Path, bond, device: torch.device, dtype: torch.dtype) -> dict:
    metadata = make_metadata_tensors(device, dtype)
    metadata_dict = metadata_tuple_to_dict(metadata)

    R, channels, _R_rep, _rep_splines, _close_exp = bond.read_skf_table(
        str(skf_path),
        *metadata,
        device=device,
        dtype=dtype,
    )

    M = bond.channels_to_matrix(channels)
    coeffs = bond.cubic_spline_coeffs(R, M)
    original_rows = get_original_electronic_row_count(skf_path)
    ...
    left_reconstructed = a[:original_rows]
    left_target = M[:original_rows]
    left_err = (left_reconstructed - left_target).abs()
    ...
    right_reconstructed = a + b * h + c * h**2 + d * h**3
```

This is the core SPL-01/SPL-02 pattern. Add simple-format parser/spline coverage by reusing this shape of check, but avoid requiring a complete simple-format ordered-pair directory.

**Get-SKF-tensors channel assertion** (lines 471-502):
```python
def check_get_skf_tensors_metadata(skf_dir: Path, bond, device: torch.device) -> None:
    elements = collect_elements_from_skf_dir(skf_dir, bond)
    TYPE = torch.tensor([bond.symbol_to_number[sym] for sym in elements], dtype=torch.long, device=device)

    (
        _R_tensor,
        _R_orb,
        coeffs_tensor,
        ...
        N_F,
        ...
        EF,
        ...
        UF,
        SHELL_PRESENT,
    ) = bond.get_skf_tensors(TYPE, str(skf_dir))

    if coeffs_tensor.shape[2] != 40:
        raise AssertionError(f"get_skf_tensors coeffs_tensor expected 40 channels, got {coeffs_tensor.shape[2]}")
```

Keep `get_skf_tensors()` coverage parser-focused: channel width, representative f metadata, pair loading via fixture directory.

**Failure aggregation pattern** (lines 556-569, 1232-1259):
```python
def run_bond_integral_tests(skf_dir: Path, bond, device: torch.device, dtype: torch.dtype) -> list[dict]:
    skf_files = sorted(skf_dir.glob("*.skf"))
    print(f"Testing _bond_integral.py for {len(skf_files)} SKF files")
    ...
    for skf_path in skf_files:
        ...

def main() -> None:
    torch.set_default_dtype(torch.float64)
    ...
    failures = []
    failures.extend(run_bond_integral_tests(skf_dir, bond, device, dtype))
    failures.extend(run_constants_tests(project_root, skf_dir, bond))
    failures.extend(run_structure_tests(project_root, skf_dir, bond))
    ...
    if failures:
        ...
        raise AssertionError(f"{len(failures)} test check(s) failed")
```

For Phase 1 additions, prefer adding checks under the existing `_bond_integral.py` section. If later sections remain in the script, do not expand them for Phase 1 acceptance.

## Shared Patterns

### Parser Boundary Normalization

**Source:** `src/dftorch/_bond_integral.py:397`
**Apply to:** `src/dftorch/_bond_integral.py`, parser-focused checks in `src/dftorch/script.py`

All electronic rows must be normalized before channel matrix or spline construction. Downstream code should see 40 channels only.

### Explicit Parser Errors

**Source:** `src/dftorch/_bond_integral.py:428`, `src/dftorch/_bond_integral.py:522`, `src/dftorch/_bond_integral.py:733`
**Apply to:** malformed row width, skipped shell layouts, missing spline blocks

Raise direct `ValueError` with the path and actionable reason. Do not tolerate ambiguous basis layouts.

### Independent Validation Oracle

**Source:** `src/dftorch/script.py:107`, `src/dftorch/script.py:174`, `src/dftorch/script.py:398`
**Apply to:** metadata and spline validation

The validation script should parse expected values independently where practical, then compare production parser output. Avoid using production parser output as its own oracle.

### CPU Float64 Validation

**Source:** `src/dftorch/script.py:1232`
**Apply to:** all Phase 1 validation harness additions

Use `torch.set_default_dtype(torch.float64)`, `torch.device("cpu")`, and `dtype = torch.float64`. Do not introduce GPU, Triton, DFTB+, ASE, or pytest requirements for this phase.

## No Analog Found

None. Both Phase 1 files have exact existing analogs in the codebase.

## Out Of Scope For This Pattern Map

| File / Area | Reason |
|-------------|--------|
| `src/dftorch/Constants.py` | Phase 1 acceptance should stay at parser/spline boundary; Constants behavior belongs to later phase scope. |
| `src/dftorch/Structure.py` | Structure AO bookkeeping is downstream of parser canonicalization and belongs to later phase scope. |
| H0/S or Slater-Koster formula files | User explicitly requested no broadening into H0/S. |
| New pytest files | User decision D-05 defers pytest conversion. |

## Metadata

**Analog search scope:** `src/dftorch/*.py`, `tests/*.py`, `tests/f_orbital_data/*.skf`, `tests/data_skf_mio-1-1/*.skf`
**Files scanned:** required phase docs, codebase maps, target source files, scoped source/test file list, f-orbital fixture list
**Pattern extraction date:** 2026-07-20
