# Phase 02: Constants and Structure Basis Metadata - Pattern Map

**Mapped:** 2026-07-24
**Files analyzed:** 5
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/dftorch/Constants.py` | model/config | file-I/O -> transform | `src/dftorch/Constants.py` | exact |
| `src/dftorch/Structure.py` | model | transform | `src/dftorch/Structure.py` | exact |
| `src/dftorch/script.py` | utility/validation harness | batch transform | `src/dftorch/script.py` | exact |
| `tests/test_f_orbital_skf.py` | test | request-response pytest wrapper | `tests/test_f_orbital_skf.py` | exact |
| `tests/test_basis_metadata.py` or metadata tests added to `tests/test_f_orbital_skf.py` | test | file-I/O -> transform | `tests/test_f_orbital_skf.py`; `tests/test_import.py`; `tests/test_scf.py` | role-match |

Scope note: Phase 2 validates metadata only. Do not copy patterns from `src/dftorch/_h0ands.py`, `src/dftorch/_slater_koster_pair.py`, `src/dftorch/ESDriver.py`, or tutorial notebooks for this phase.

## Pattern Assignments

### `src/dftorch/Constants.py` (model/config, file-I/O -> transform)

**Analog:** `src/dftorch/Constants.py`

**Imports pattern** (lines 1-11):
```python
from typing import Any

import os

import numpy as np
import torch

from ._elements import atomic_num, label, mass, symbol_to_number
from ._io import read_pdb, read_xyz
from ._tools import load_hubbard_derivs, load_spinw_to_matrix, ordered_pairs_from_TYPE
from ._bond_integral import get_skf_tensors
```

**Shell-dimension contract** (lines 65-67):
```python
self.shell_dim = torch.nn.Parameter(
    torch.tensor([0, 1, 3, 5, 7], dtype=torch.int64), requires_grad=False
)
```

**SKF metadata loading pattern** (lines 101-125):
```python
(
    R_tensor,
    R_orb,
    coeffs_tensor,
    R_rep_tensor,
    rep_splines_tensor,
    close_exp_tensor,
    N_ORB,
    MAX_ANG,
    MAX_ANG_OCC,
    TORE,
    N_S,
    N_P,
    N_D,
    N_F,
    ES,
    EP,
    ED,
    EF,
    US,
    UP,
    UD,
    UF,
    SHELL_PRESENT,
) = get_skf_tensors(TYPE, self.skfpath)
```

**Pair lookup and tensor registration pattern** (lines 88-99, 148-181):
```python
TYPE = torch.tensor(species.flatten())
pairs_tensor, _, _ = ordered_pairs_from_TYPE(TYPE)
pair_lookup = torch.full(
    (len(self.label), len(self.label)),
    -1,
    dtype=torch.long,
    device=TYPE.device,
)
if pairs_tensor.numel() > 0:
    pair_lookup[pairs_tensor[:, 0], pairs_tensor[:, 1]] = torch.arange(
        pairs_tensor.shape[0], dtype=torch.long, device=TYPE.device
    )
```

```python
self.coeffs_tensor = torch.nn.Parameter(coeffs_tensor, requires_grad=False)
self.pair_lookup = torch.nn.Parameter(pair_lookup, requires_grad=False)

self.n_orb = torch.nn.Parameter(N_ORB, requires_grad=False)
self.max_ang = torch.nn.Parameter(MAX_ANG, requires_grad=False)
self.max_ang_occ = torch.nn.Parameter(MAX_ANG_OCC, requires_grad=False)
self.n_s = torch.nn.Parameter(N_S, requires_grad=False)
self.n_p = torch.nn.Parameter(N_P, requires_grad=False)
self.n_d = torch.nn.Parameter(N_D, requires_grad=False)
self.n_f = torch.nn.Parameter(N_F, requires_grad=False)
self.shell_present = torch.nn.Parameter(SHELL_PRESENT, requires_grad=False)

self.U = torch.nn.Parameter(US, requires_grad=self.grad_param)
self.Up = torch.nn.Parameter(UP, requires_grad=self.grad_param)
self.Ud = torch.nn.Parameter(UD, requires_grad=self.grad_param)
self.Uf = torch.nn.Parameter(UF, requires_grad=self.grad_param)
self.Es = torch.nn.Parameter(ES, requires_grad=self.grad_param)
self.Ep = torch.nn.Parameter(EP, requires_grad=self.grad_param)
self.Ed = torch.nn.Parameter(ED, requires_grad=self.grad_param)
self.Ef = torch.nn.Parameter(EF, requires_grad=self.grad_param)
```

**Error handling pattern** (lines 127-146, 184-195):
```python
try:
    w_shell = load_spinw_to_matrix(
        os.path.join(self.skfpath, "spinw.txt"), device=TYPE.device
    )
    self.w_shell = torch.nn.Parameter(w_shell, requires_grad=False)
except (FileNotFoundError, OSError, ValueError):
    print(
        "Warning: could not load spinw.txt file for spin-orbit coupling. Proceeding without SOC."
    )
    self.w = None
```

```python
if self.dftb3:
    _hubbard_path = os.path.join(self.skfpath, "hubbard_derivative.txt")
    try:
        dU_dq = load_hubbard_derivs(_hubbard_path, device=TYPE.device)
        self.dU_dq = torch.nn.Parameter(dU_dq, requires_grad=False)
        self.dftb3 = True
    except (FileNotFoundError, OSError):
        self.dU_dq = None
        self.dftb3 = False
```

### `src/dftorch/Structure.py` (model, transform)

**Analog:** `src/dftorch/Structure.py`

**Imports and module constants pattern** (lines 1-13, 25-61):
```python
from __future__ import annotations

from typing import Any

import torch

from ._cell import normalize_cell, normalize_cell_batch, wrap_positions
from ._io import read_pdb, read_xyz


SHELL_DIMS = (1, 3, 5, 7)
SHELL_LOCAL_STARTS = (0, 1, 4, 9)
SHELL_TYPE_IDS = (1, 2, 3, 4)  # 1=s, 2=p, 3=d, 4=f
```

```python
AO_LABEL_TEMPLATE = (
    "s",
    "px",
    "py",
    "pz",
    "dxy",
    "dyz",
    "dzx",
    "dx2_y2",
    "dz2",
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
)

AO_SHELL_TEMPLATE = (
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
)
```

**Shared shell helper pattern** (lines 90-140, 143-157):
```python
def _ao_mask_from_shell_present(shell_present: torch.Tensor) -> torch.Tensor:
    """Expand ``(...,4)`` shell flags into a ``(...,16)`` AO-position mask."""
    mask = torch.zeros(
        (*shell_present.shape[:-1], len(AO_LABEL_TEMPLATE)),
        dtype=torch.bool,
        device=shell_present.device,
    )
    mask[..., 0] = shell_present[..., 0]
    mask[..., 1:4] = shell_present[..., 1].unsqueeze(-1)
    mask[..., 4:9] = shell_present[..., 2].unsqueeze(-1)
    mask[..., 9:16] = shell_present[..., 3].unsqueeze(-1)
    return mask
```

```python
def _shell_local_start(shell_present: torch.Tensor) -> torch.Tensor:
    """Return local AO starts for s/p/d/f shells, using -1 for absent shells."""
    starts = torch.tensor(
        SHELL_LOCAL_STARTS,
        dtype=torch.int64,
        device=shell_present.device,
    )
    starts = starts.expand(*shell_present.shape[:-1], 4)
    return torch.where(shell_present, starts, torch.full_like(starts, -1))
```

```python
def _flatten_ao_labels(shell_present: torch.Tensor) -> list[str]:
    """Return flattened AO labels for a single structure in atom-major order."""
    mask = _ao_mask_from_shell_present(shell_present).detach().cpu()
    labels: list[str] = []
    for atom_mask in mask:
        labels.extend(label for label, present in zip(AO_LABEL_TEMPLATE, atom_mask) if present)
    return labels
```

**Single-structure metadata pattern** (lines 333-451):
```python
self.n_orbitals_per_atom = const.n_orb[self.TYPE]
self.H_INDEX_START = torch.zeros(self.Nats, dtype=torch.int64, device=device)
self.H_INDEX_START[1:] = torch.cumsum(self.n_orbitals_per_atom, dim=0)[:-1]
self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1
```

```python
self.shell_present = const.shell_present[self.TYPE].to(dtype=torch.bool)
self.has_s = self.shell_present[:, 0]
self.has_p = self.shell_present[:, 1]
self.has_d = self.shell_present[:, 2]
self.has_f = self.shell_present[:, 3]

self.shell_local_start = _shell_local_start(self.shell_present)
self.shell_local_end = _shell_local_end(self.shell_present)
self.shell_ao_start = _global_shell_start(
    self.shell_present, self.shell_local_start, self.H_INDEX_START
)
self.shell_ao_end = _global_shell_end(
    self.shell_present, self.shell_local_end, self.H_INDEX_START
)
```

```python
template = torch.stack(
    (
        EsA,
        EpA,
        EpA,
        EpA,
        EdA,
        EdA,
        EdA,
        EdA,
        EdA,
        EfA,
        EfA,
        EfA,
        EfA,
        EfA,
        EfA,
        EfA,
    ),
    dim=1,
)
ao_mask = _ao_mask_from_shell_present(self.shell_present)

self.diagonal = template[ao_mask]
self.HDIM = self.diagonal.shape[-1]
self.ao_shell_types = _ao_shell_types_from_mask(ao_mask)
self.ao_labels = _flatten_ao_labels(self.shell_present)
```

```python
self.Hubbard_U_sr = template_U[shell_mask]
self.shell_types = template_ang[shell_mask]
self.el_per_shell = template_el_per_shell[shell_mask]
self.n_shells_per_atom = self.shell_present.sum(dim=1).to(torch.int64)
self.H_INDEX_START_U = torch.zeros(self.Nats, dtype=torch.int64, device=device)
self.H_INDEX_START_U[1:] = torch.cumsum(self.n_shells_per_atom, dim=0)[:-1]
self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1

self.D0 = _atomic_density_matrix_from_shells(
    self.H_INDEX_START,
    self.HDIM,
    self.TYPE,
    const,
    self.shell_present,
    self.shell_ao_start,
)
self.D0 = 0.5 * self.D0
```

**Batched metadata pattern** (lines 532-677):
```python
self.n_orbitals_per_atom = const.n_orb[self.TYPE]
self.H_INDEX_START = torch.zeros(
    self.batch_size, self.Nats, dtype=torch.int64, device=device
)
self.H_INDEX_START[:, 1:] = torch.cumsum(self.n_orbitals_per_atom, dim=1)[
    :, :-1
]
self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1
```

```python
self.shell_present = const.shell_present[self.TYPE].to(dtype=torch.bool)
self.has_s = self.shell_present[:, :, 0]
self.has_p = self.shell_present[:, :, 1]
self.has_d = self.shell_present[:, :, 2]
self.has_f = self.shell_present[:, :, 3]
```

```python
self.diagonal_flat = template[ao_mask]
self.HDIM_struct = self.n_orbitals_per_atom.sum(dim=1)
self.HDIM_total = int(self.diagonal_flat.shape[0])
max_HDIM = int(self.HDIM_struct.max().item())
self.diagonal = torch.zeros(
    self.batch_size, max_HDIM, dtype=template.dtype, device=self.device
)
```

```python
struct_offsets = torch.cumsum(self.HDIM_struct, dim=0) - self.HDIM_struct
self.H_INDEX_START_GLOBAL = self.H_INDEX_START + struct_offsets.unsqueeze(-1)
self.H_INDEX_END_GLOBAL = self.H_INDEX_END + struct_offsets.unsqueeze(-1)
self.shell_ao_start_global = _global_shell_start(
    self.shell_present, self.shell_local_start, self.H_INDEX_START_GLOBAL
)
self.shell_ao_end_global = _global_shell_end(
    self.shell_present, self.shell_local_end, self.H_INDEX_START_GLOBAL
)
```

### `src/dftorch/script.py` (utility/validation harness, batch transform)

**Analog:** `src/dftorch/script.py`

**Script imports and module loading pattern** (lines 39-100):
```python
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch


ATOL = 1.0e-9
EV_PER_HARTREE = 27.21138625
```

```python
def ensure_fake_dftorch_package(project_root: Path) -> Path:
    """
    Make src/dftorch importable as a package without importing dftorch/__init__.py.
    """
    package_dir = project_root / "src" / "dftorch"
    fake_pkg = types.ModuleType("dftorch")
    fake_pkg.__path__ = [str(package_dir)]
    sys.modules["dftorch"] = fake_pkg
    return package_dir


def load_dftorch_module(project_root: Path, module_basename: str):
    """Load src/dftorch/{module_basename}.py as dftorch.{module_basename}."""
    package_dir = ensure_fake_dftorch_package(project_root)
    module_path = package_dir / f"{module_basename}.py"
    module_name = f"dftorch.{module_basename}"
```

**Assertion helper pattern** (lines 269-294, 1279-1308):
```python
def assert_int_metadata(actual: torch.Tensor, index: int, expected: int, name: str, source: str) -> None:
    got = int(actual[index].item())
    if got != expected:
        raise AssertionError(f"{source}: {name}[{index}] expected {expected}, got {got}")
```

```python
def assert_tensor_float_list(
    tensor: torch.Tensor,
    expected: list[float],
    name: str,
    atol: float = ATOL,
) -> None:
    got = [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    if len(got) != len(expected):
        raise AssertionError(f"{name} expected length {len(expected)}, got {len(got)}")
    for i, (g, e) in enumerate(zip(got, expected)):
        err = abs(g - e)
        if err > atol:
            raise AssertionError(f"{name}[{i}] expected {e:.16e}, got {g:.16e}, err={err:.3e}")
```

**Constants validation pattern** (lines 1030-1107):
```python
for sym in elements:
    homonuclear = resolve_homonuclear_skf(skf_dir, sym, bond)
    if not homonuclear.is_file():
        continue

    expected = parse_expected_homonuclear_metadata(homonuclear, bond)
    if expected is None:
        continue

    Z = int(expected["Z"])
    source = f"Constants/{homonuclear.name}"

    assert_int_metadata(const.n_orb, Z, int(expected["N_ORB"]), "n_orb", source)
    assert_int_metadata(const.max_ang, Z, int(expected["MAX_ANG"]), "max_ang", source)
    assert_int_metadata(const.max_ang_occ, Z, int(expected["MAX_ANG_OCC"]), "max_ang_occ", source)
```

```python
float_attr_to_expected = {
    "tore": "TORE",
    "n_s": "N_S",
    "n_p": "N_P",
    "n_d": "N_D",
    "n_f": "N_F",
    "Es": "ES",
    "Ep": "EP",
    "Ed": "ED",
    "Ef": "EF",
    "U": "US",
    "Up": "UP",
    "Ud": "UD",
    "Uf": "UF",
}
for attr, expected_name in float_attr_to_expected.items():
    assert_float_metadata(getattr(const, attr), Z, float(expected[expected_name]), attr, source)

assert_bool_metadata(const.shell_present, Z, list(expected["SHELL_PRESENT"]), "shell_present", source)
```

**Structure oracle helpers pattern** (lines 1139-1276):
```python
EXPECTED_SHELL_DIMS = [1, 3, 5, 7]
EXPECTED_SHELL_LOCAL_STARTS = [0, 1, 4, 9]
EXPECTED_SHELL_TYPE_IDS = [1, 2, 3, 4]
EXPECTED_AO_LABEL_TEMPLATE = [
    "s",
    "px",
    "py",
    "pz",
    "dxy",
    "dyz",
    "dzx",
    "dx2_y2",
    "dz2",
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
]
```

```python
def expected_diagonal_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    energies = [float(md["ES"]), float(md["EP"]), float(md["ED"]), float(md["EF"])]
    vals: list[float] = []
    for present, energy, dim in zip(shell_present, energies, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([energy] * dim)
    return vals


def expected_d0_values(md: dict[str, object]) -> list[float]:
    """Expected Structure.D0 values after the 0.5 closed-shell factor."""
    shell_present = list(md["SHELL_PRESENT"])
    occs = [float(md["N_S"]), float(md["N_P"]), float(md["N_D"]), float(md["N_F"])]
    vals: list[float] = []
    for present, occ, dim in zip(shell_present, occs, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([0.5 * occ / float(dim)] * dim)
    return vals
```

**Single Structure validation pattern** (lines 1311-1403):
```python
def check_single_structure_layout(struct, elements: list[str], expected_by_element: dict[str, dict[str, object]]) -> None:
    """Check Structure's atom-major AO and shell metadata."""
    expected_types = [int(expected_by_element[sym]["Z"]) for sym in elements]
    expected_n_orb = [int(expected_by_element[sym]["N_ORB"]) for sym in elements]
    expected_h_start = [0]
    for n_orb in expected_n_orb[:-1]:
        expected_h_start.append(expected_h_start[-1] + n_orb)
    expected_h_end = [start + n_orb - 1 for start, n_orb in zip(expected_h_start, expected_n_orb)]
    expected_hdim = sum(expected_n_orb)

    assert_tensor_int_list(struct.TYPE, expected_types, "Structure.TYPE")
    assert_tensor_int_list(struct.n_orbitals_per_atom, expected_n_orb, "Structure.n_orbitals_per_atom")
    assert_tensor_int_list(struct.H_INDEX_START, expected_h_start, "Structure.H_INDEX_START")
    assert_tensor_int_list(struct.H_INDEX_END, expected_h_end, "Structure.H_INDEX_END")
```

```python
assert_tensor_bool_list(struct.shell_present, [x for row in expected_shell_present for x in row], "Structure.shell_present")
assert_tensor_bool_list(struct.has_s, [row[0] for row in expected_shell_present], "Structure.has_s")
assert_tensor_bool_list(struct.has_p, [row[1] for row in expected_shell_present], "Structure.has_p")
assert_tensor_bool_list(struct.has_d, [row[2] for row in expected_shell_present], "Structure.has_d")
assert_tensor_bool_list(struct.has_f, [row[3] for row in expected_shell_present], "Structure.has_f")

assert_tensor_int_list(struct.shell_local_start, [x for row in expected_shell_local_start for x in row], "Structure.shell_local_start")
assert_tensor_int_list(struct.shell_local_end, [x for row in expected_shell_local_end for x in row], "Structure.shell_local_end")
assert_tensor_int_list(struct.shell_ao_start, [x for row in expected_shell_ao_start for x in row], "Structure.shell_ao_start")
assert_tensor_int_list(struct.shell_ao_end, [x for row in expected_shell_ao_end for x in row], "Structure.shell_ao_end")

assert_list_equal(struct.ao_labels, expected_labels, "Structure.ao_labels")
assert_tensor_int_list(struct.ao_shell_types, expected_ao_types, "Structure.ao_shell_types")
assert_tensor_float_list(struct.diagonal, expected_diagonal, "Structure.diagonal")
assert_tensor_float_list(struct.D0, expected_d0, "Structure.D0")
```

**StructureBatch validation pattern** (lines 1414-1489):
```python
"""Check StructureBatch for two structures with different atom orders.

StructureBatch stores each molecule in a padded row.  This test checks that
the real AO region of each row is correct and that any padding after the
molecule is zero.
"""
```

```python
assert_tensor_int_list(batch_struct.HDIM_struct, expected_hdim_struct, "StructureBatch.HDIM_struct")

assert_tensor_bool_list(batch_struct.shell_present, expected_shell_present_flat, "StructureBatch.shell_present")
assert_tensor_int_list(batch_struct.shell_ao_start, expected_shell_ao_start_flat, "StructureBatch.shell_ao_start")
assert_tensor_int_list(batch_struct.shell_ao_end, expected_shell_ao_end_flat, "StructureBatch.shell_ao_end")
assert_tensor_int_list(batch_struct.n_shells_per_atom, expected_n_shells_flat, "StructureBatch.n_shells_per_atom")

for batch_idx, labels in enumerate(expected_labels):
    assert_list_equal(batch_struct.ao_labels[batch_idx], labels, f"StructureBatch.ao_labels[{batch_idx}]")
    hdim = expected_hdim_struct[batch_idx]
    assert_tensor_float_list(batch_struct.diagonal[batch_idx, :hdim], expected_diagonal_rows[batch_idx], f"StructureBatch.diagonal[{batch_idx}]")
    assert_tensor_float_list(batch_struct.D0[batch_idx, :hdim], expected_d0_rows[batch_idx], f"StructureBatch.D0[{batch_idx}]")
    if hdim < max_hdim:
        assert_tensor_float_list(batch_struct.diagonal[batch_idx, hdim:], [0.0] * (max_hdim - hdim), f"StructureBatch.diagonal padding[{batch_idx}]")
        assert_tensor_float_list(batch_struct.D0[batch_idx, hdim:], [0.0] * (max_hdim - hdim), f"StructureBatch.D0 padding[{batch_idx}]")
```

**Run-all validation pattern** (lines 1587-1616):
```python
def main() -> None:
    torch.set_default_dtype(torch.float64)

    project_root = find_project_root()
    bond = load_dftorch_module(project_root, "_bond_integral")

    if len(sys.argv) > 1:
        skf_dir = Path(sys.argv[1]).resolve()
    else:
        skf_dir = project_root / "tests" / "f_orbital_data"
```

```python
failures = []
failures.extend(run_bond_integral_tests(skf_dir, bond, device, dtype))
failures.extend(run_constants_tests(project_root, skf_dir, bond))
failures.extend(run_structure_tests(project_root, skf_dir, bond))

print()
if failures:
    print("FAILED CHECKS:")
    for result in failures:
        print(f"  {result}")
    raise AssertionError(f"{len(failures)} test check(s) failed")
```

### `tests/test_f_orbital_skf.py` (test, request-response pytest wrapper)

**Analog:** `tests/test_f_orbital_skf.py`

**Imports and dtype fixture pattern** (lines 1-14):
```python
import importlib.util
import shutil
from pathlib import Path

import torch


def run_with_float64(fn):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
```

**Validation-script loading pattern** (lines 17-26):
```python
def load_validation_script():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "dftorch" / "script.py"
    spec = importlib.util.spec_from_file_location("dftorch_phase1_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation script from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

**Pytest wrapper pattern** (lines 29-43):
```python
def test_f_orbital_skf_parser_and_spline_gate():
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        bond = validation.load_dftorch_module(project_root, "_bond_integral")
        skf_dir = project_root / "tests" / "f_orbital_data"

        return validation.run_bond_integral_tests(
            skf_dir,
            bond,
            torch.device("cpu"),
            torch.float64,
        )

    assert run_with_float64(check) == []
```

Planner should mirror this pattern for `run_constants_tests(project_root, skf_dir, bond)` and `run_structure_tests(project_root, skf_dir, bond)`. Those calls need no H0/S assembly.

### `tests/test_basis_metadata.py` or metadata tests in `tests/test_f_orbital_skf.py` (test, file-I/O -> transform)

**Analogs:** `tests/test_f_orbital_skf.py`, `tests/test_import.py`, `tests/test_scf.py`

**Simple fixture path/setup pattern** from `tests/test_import.py` (lines 42-52):
```python
root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

xyz_path = root / "tests" / "ch4.xyz"
skf_dir = root / "tests" / "data_skf_mio-1-1"

assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

import torch

torch.set_default_dtype(torch.float64)
```

**Constants/Structure construction pattern** from `tests/test_scf.py` (lines 38-56):
```python
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

const = Constants(
    dftorch_params,
).to(device)

structure1 = Structure(dftorch_params, const, device=device)
```

For Phase 2 simple-format metadata regression, copy the fixture setup and `Constants`/`Structure` construction shape, but assert metadata only. Do not instantiate `ESDriver`, call SCF, or compare tutorial outputs.

## Shared Patterns

### Device-aware non-trainable metadata
**Source:** `src/dftorch/Constants.py`
**Apply to:** `Constants` metadata fields
```python
self.n_orb = torch.nn.Parameter(N_ORB, requires_grad=False)
self.max_ang = torch.nn.Parameter(MAX_ANG, requires_grad=False)
self.max_ang_occ = torch.nn.Parameter(MAX_ANG_OCC, requires_grad=False)
self.n_f = torch.nn.Parameter(N_F, requires_grad=False)
self.shell_present = torch.nn.Parameter(SHELL_PRESENT, requires_grad=False)
```

### Explicit shell presence, not max-angular inference
**Source:** `src/dftorch/Structure.py`
**Apply to:** `Structure`, `StructureBatch`, validation expectations
```python
self.shell_present = const.shell_present[self.TYPE].to(dtype=torch.bool)
self.has_s = self.shell_present[:, 0]
self.has_p = self.shell_present[:, 1]
self.has_d = self.shell_present[:, 2]
self.has_f = self.shell_present[:, 3]
```

### Atom-major AO indexing
**Source:** `src/dftorch/Structure.py`
**Apply to:** all single and batch structure metadata assertions
```python
self.H_INDEX_START = torch.zeros(self.Nats, dtype=torch.int64, device=device)
self.H_INDEX_START[1:] = torch.cumsum(self.n_orbitals_per_atom, dim=0)[:-1]
self.H_INDEX_END = self.H_INDEX_START + self.n_orbitals_per_atom - 1
```

### Independent metadata oracle
**Source:** `src/dftorch/script.py`
**Apply to:** f-containing and simple-format metadata tests
```python
expected = parse_expected_homonuclear_metadata(homonuclear, bond)
Z = int(expected["Z"])
source = f"Constants/{homonuclear.name}"

assert_int_metadata(const.n_orb, Z, int(expected["N_ORB"]), "n_orb", source)
assert_bool_metadata(const.shell_present, Z, list(expected["SHELL_PRESENT"]), "shell_present", source)
```

### Pytest validation promotion
**Source:** `tests/test_f_orbital_skf.py`
**Apply to:** CI exposure for `run_constants_tests()` and `run_structure_tests()`
```python
def check():
    validation = load_validation_script()
    project_root = validation.find_project_root()
    bond = validation.load_dftorch_module(project_root, "_bond_integral")
    skf_dir = project_root / "tests" / "f_orbital_data"

    return validation.run_bond_integral_tests(
        skf_dir,
        bond,
        torch.device("cpu"),
        torch.float64,
    )

assert run_with_float64(check) == []
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Synthetic s-only simple SKF fixture, if persisted | test fixture | file-I/O | Existing `tests/data_skf_mio-1-1/H-H.skf` is not a clean s-only oracle under current parser rules because `Ep` is nonzero. Prefer generating minimal simple SKF data with `tmp_path` using the helper style in `src/dftorch/script.py:309`. |

## Metadata

**Analog search scope:** `src/dftorch/*.py`, `tests/*.py`, `tests/data_skf_mio-1-1`, `tests/f_orbital_data`
**Files scanned:** 6 primary files plus repository file index
**Pattern extraction date:** 2026-07-24
