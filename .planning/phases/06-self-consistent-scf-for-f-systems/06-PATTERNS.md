# Phase 6: Self-Consistent SCF for f Systems - Pattern Map

**Mapped:** 2026-08-04  
**Files analyzed:** 8 files (5 source modifications, 2 test modules, 1 deliverable)  
**Analogs found:** 8 / 8 (100% coverage)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/dftorch/_scf.py` (SCFx) | service (solver) | iterative/convergence | self (existing SCFx at :158) | exact—same function signature |
| `src/dftorch/ESDriver.py` (forward unpack) | controller | request-response | self (existing unpack at :663-678) | exact—same tuple unpacking |
| `src/dftorch/_coulomb_matrix.py` (ewald_real_space_vectorized_sr) | utility/service | matrix construction | self (existing blocks at :836-1076) | exact—nine existing blocks, seven missing for f |
| `src/dftorch/Structure.py` (scf_iter_count) | model | data structure | self (existing attributes) | exact—add one attribute |
| `tests/test_scf_convergence_f.py` | test | integration/convergence | `tests/test_single_shot_energy.py` + `tests/test_eu_n_scan.py` | role-match—same harness shape |
| `tests/test_shell_resolved_coulomb_f.py` | test | unit/validation | `tests/test_shell_resolved_u.py` | role-match—shell-resolved matrix tests |
| Plot/script deliverable | utility | batch/analysis | `experiments/diatomic_scans/plot_*.py` | role-match—matplotlib + style |
| `docs/F-SUPPORT-STATUS.md` (conditional) | documentation | reference | self (existing) | conditional—updated only if D-6.06 fires |

---

## Pattern Assignments

### `src/dftorch/_scf.py` (service, iterative/convergence)

**Analog:** Self; existing `SCFx` function at line 158 and return statement at line 554

**Function signature & docstring** (lines 158-194):
```python
def SCFx(
    dftorch_params: Dict[str, Any],
    RX, RY, RZ,
    cell: torch.Tensor,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    TYPE: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    dU_dq: Optional[torch.Tensor],
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    req_grad_xyz: bool,
    q_init: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # Q
    torch.Tensor,  # e
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """Self-consistent field (_scf) cycle with finite electronic temperature..."""
```

**Convergence criterion** (lines 393-395):
```python
while (
    (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
    or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
) and it < dftorch_params.get("SCF_MAX_ITER", 100):
    # ... loop body ...
    it += 1
```

**Non-convergence check & print** (lines 520-521):
```python
if it == dftorch_params.get("SCF_MAX_ITER", 100):
    print("Did not converge")
```

**Charge definition** (lines 526-528):
```python
DS = 2 * (D * S.T).sum(dim=1)
q = -1.0 * Znuc
q.scatter_add_(0, atom_ids, DS)
```

**Current return statement** (line 554):
```python
return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul
```

**D-6.04 modification required:** Add convergence flag as 15th element:
```python
# At loop exit, compute convergence flag
converged_iter = it if (
    ResNorm <= dftorch_params.get("SCF_TOL", 1e-6)
    and dEc <= dftorch_params.get("SCF_TOL", 1e-6) * 100
) else -1

return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul, converged_iter
```

**Note on other loops:** `scf_x_os` (line 557), `SCFx_batch` (line 977), and `delta_scf_x_os` (line 1233) carry identical convergence prints at lines 911-912, 1216-1217, and 1551-1552 respectively. Per Claude's Discretion in CONTEXT.md, the plan must decide whether all four loops return the flag or only SCFx.

---

### `src/dftorch/ESDriver.py` (controller, request-response)

**Analog:** Self; existing `forward` method unpack at lines 663-678

**ESDriver.forward signature** (lines 159-232):
```python
def forward(
    self,
    structure,
    const,
    do_scf: bool = True,
    req_grad_xyz: bool = False,
    verbose: bool = False,
    save_intermediate_D: bool = False,
) -> Structure:
    """Compute electronic structure and energy for the input structure..."""
```

**Current 14-element tuple unpacking** (lines 663-678):
```python
else:  # closed-shell
    (
        structure.H,
        structure.Hcoul,
        structure.Hdipole,
        structure.KK,
        structure.D,
        structure.Q,
        structure.e,
        structure.q,
        structure.f,
        structure.mu0,
        structure.e_coul_tmp,
        structure.f_coul,
        structure.dq_p1,
        structure.stress_coulomb,
    ) = SCFx(
        self.dftorch_params,
        structure.RX,
        structure.RY,
        structure.RZ,
        structure.cell,
        structure.Nats,
        structure.Nocc,
        structure.n_orbitals_per_atom,
        structure.Znuc,
        structure.TYPE,
        # ... more args ...
    )
```

**D-6.04 modification required:** Add 15th element:
```python
else:  # closed-shell
    (
        structure.H,
        structure.Hcoul,
        structure.Hdipole,
        structure.KK,
        structure.D,
        structure.Q,
        structure.e,
        structure.q,
        structure.f,
        structure.mu0,
        structure.e_coul_tmp,
        structure.f_coul,
        structure.dq_p1,
        structure.stress_coulomb,
        structure.scf_iter_count,  # <-- NEW: iteration count or -1
    ) = SCFx(...)
```

**Guard pattern – named exception** (lines 68-85):
```python
def _require_closed_shell_f_system(
    structure, const, dftorch_params, context: str
) -> None:
    """Reject spin-polarized calculations for systems containing f orbitals.
    
    Phase 4 supports closed-shell occupation only for f systems (decision D-12)...
    """
    if SPIN_POLARIZATION_AVAILABLE:
        return
    type_ids = getattr(structure, "TYPE", None)
    if type_ids is None:
        return
    # ... check for f shell ...
    if bool((valid & (counts == 16)).any()):
        raise FSpinPolarizationUnsupportedError(...)
```

---

### `src/dftorch/_coulomb_matrix.py` (utility/service, matrix construction)

**Analog:** Self; existing `ewald_real_space_vectorized_sr` function at lines 723+ with nine implemented blocks (lines 836-1076)

**Guard pattern – named exception** (lines 690-719):
```python
def _require_no_f_shell_resolved_coulomb(structure, TYPE, context: str) -> None:
    """Reject shell-resolved Coulomb assembly for systems containing f orbitals.
    
    :func:`ewald_real_space_vectorized_sr` covers ``max_ang`` 1, 2 and 3 only...
    """
    const = getattr(structure, "const", None)
    if const is None:
        return
    n_orb = getattr(const, "n_orb", None)
    if n_orb is None or TYPE is None:
        return
    valid = TYPE >= 0
    if not bool(valid.any()):
        return
    counts = n_orb[TYPE.clamp(min=0)]
    if bool((valid & (counts == 16)).any()):
        raise FShellResolvedCoulombUnsupportedError(
            f"{context}: this system contains at least one atom with "
            f"n_orb == 16 (an f-shell element).\n"
            f"{F_SHELL_RESOLVED_COULOMB_UNSUPPORTED_MESSAGE}"
        )
```

**Existing s-s block pattern** (lines 836-858):
```python
### s-s ###
Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I]]
Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J]]
mask_same_elem = structure.TYPE[neighbor_I] == structure.TYPE[neighbor_J]
if mask_same_elem.any():
    dR_mskd_same = dR_mskd[mask_same_elem]
    Ti_same_el = Ti[mask_same_elem]
    t1, dt1 = coul_same_elem_and_ang(Ti_same_el, dR_mskd_same)
    tmp1[mask_same_elem] -= t1
    dtmp1[mask_same_elem] -= dt1
if (~mask_same_elem).any():
    dR_mskd_diff = dR_mskd[~mask_same_elem]
    Ti_diff_el = Ti[~mask_same_elem]
    Tj_diff_el = Tj[~mask_same_elem]
    t1, dt1 = coul_diff_elem_and_ang(Ti_diff_el, Tj_diff_el, dR_mskd_diff)
    tmp1[~mask_same_elem] -= t1
    dtmp1[~mask_same_elem] -= dt1
tmp1 *= KECONST
dtmp1 *= KECONST
idx_row = structure.H_INDEX_START_U[neighbor_I] * CDIM
idx_col = structure.H_INDEX_START_U[neighbor_J]
CC_real.index_add_(0, (idx_row + idx_col), tmp1)
dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd)
```

**Block pattern for f (e.g., s-f):** Repeat above with `structure.const.Uf` instead of `U` for j-element:
```python
### s-f (example) ###
# Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I]]   # s-shell for I
# Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J]]  # f-shell for J
# ... same mask logic, same damping functions, same index_add_ ...
```

**Existing blocks implemented:**
- s-s (line 836)
- s-p (line 860) 
- p-s (line 890)
- p-p (line 920)
- s-d (line 953)
- d-s (line 976)
- p-d (line 999)
- d-p (line 1022)
- d-d (line 1045)

**Missing blocks for f (D-6.06 conditional):**
- s-f (H-Z): `pair_mask_HZ`, U[I] * Uf[J]
- f-s (Z-H): `pair_mask_ZH`, Uf[I] * U[J]
- p-f (X-Z): `pair_mask_XZ`, Up[I] * Uf[J]
- f-p (Z-X): `pair_mask_ZX`, Uf[I] * Up[J]
- d-f (Y-Z): `pair_mask_YZ`, Ud[I] * Uf[J]
- f-d (Z-Y): `pair_mask_ZY`, Uf[I] * Ud[J]
- f-f (Z-Z): `pair_mask_ZZ`, Uf[I] * Uf[J] (with same-element special case)

---

### `src/dftorch/Structure.py` (model, data structure)

**Analog:** Self; existing attributes like `H`, `Hcoul`, etc. set by ESDriver.forward

**D-6.04 modification required:** Add `scf_iter_count` attribute:
```python
# In Structure.__init__ or as a property:
self.scf_iter_count = None  # Will be set by forward(do_scf=True)
```

**Location reference:** `Structure.py:435` and `:651` show existing `Hubbard_U_sr` construction. Add `scf_iter_count` near other SCF bookkeeping attributes set at the unpack site.

---

### `tests/test_scf_convergence_f.py` (test, integration/convergence)

**Analog:** `tests/test_single_shot_energy.py` (harness) + `tests/test_eu_n_scan.py` (structure)

**Import and environment setup** (from test_single_shot_energy.py lines 15-21):
```python
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import sys
from pathlib import Path
import torch
```

**Module-reset fixture `run_with_float64`** (from test_single_shot_energy.py lines 29-52):
```python
def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.
    
    Copied from ``tests/test_f_orbital_skf.py`` — the established harness for
    every f-orbital test module in this project.
    """
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        for name in [
            name
            for name in sys.modules
            if name == "dftorch" or name.startswith("dftorch.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
```

**Geometry writer and parameters** (from test_single_shot_energy.py lines 59-87):
```python
EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}

def _write_eu_n_xyz(path: Path, separation: float = EU_N_SEPARATION) -> None:
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 4 single-shot reference case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )

def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"
```

**Convergence test assertion pattern** (D-6.01 requirement):
```python
# In test_eu_n_scf_converges_at_target_separation:
assert structure.scf_iter_count > 0, "converged (not -1)"
assert structure.scf_iter_count < 100, "did not exhaust MaxIt"
assert structure.scf_iter_count <= 20, "converged reasonably fast"
```

**Non-convergence assertion pattern** (D-6.04 requirement):
```python
# In test_non_converged_separation_returns_minus_one:
# At a separation that the blocking diagnosis shows will not converge:
assert structure.scf_iter_count == -1, "returned -1 for non-convergence"
assert structure.e_tot is not None, "last iterate returned despite failure"
```

---

### `tests/test_shell_resolved_coulomb_f.py` (test, unit/validation) — CONDITIONAL on D-6.06

**Analog:** `tests/test_shell_resolved_u.py` (existing shell-resolved tests)

**Test structure pattern** (conditional on D-6.06):
```python
# From test_shell_resolved_u.py pattern, adapted for f blocks
def test_f_coulomb_blocks_exist_and_are_finite():
    """Seven f angular blocks are populated in the shell-resolved Coulomb matrix."""
    # Build Eu-N system with MAGNETIC_HUBBARD_LDEP = True (triggers shell-resolved path)
    # Call ewald_real_space_vectorized_sr directly
    # Assert that C_sr shape is (n_shells, n_shells)
    # Assert that C_sr is finite everywhere (no NaN, no inf, no zero in f-f block)

def test_shell_resolved_f_blocks_enable_convergence():
    """With f blocks, SCF loop converges; without, it does not."""
    # Run ESDriver.forward(do_scf=True) with MAGNETIC_HUBBARD_LDEP=True
    # Assert structure.scf_iter_count > 0 (loop converged)
```

**Guard removal (if D-6.06 fires):** Remove or narrow the check in `_require_no_f_shell_resolved_coulomb` at `_coulomb_matrix.py:690-719`.

---

### Plot/script deliverable (utility, batch/analysis)

**Analog:** `experiments/diatomic_scans/plot_energies.py` (lines 1-59)

**Plotting infrastructure** (from plot_energies.py):
```python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style
style.apply()
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = Path(...); FIGS.mkdir(exist_ok=True)
NAVY, SLATE, CRIMSON, GREEN = style.NAVY, style.SLATE, style.CRIMSON, style.GREEN

fig, ax = plt.subplots(...)
fig.suptitle("...", fontsize=18, fontweight="bold")
# ... plot setup ...
ax.plot(separations, energies_singleshot, 'b-', label='single-shot (pinned)')
ax.plot(separations, energies_scf, 'r-', label='self-consistent')
# Mark non-converged points with scatter or different marker
ax.scatter(failed_seps, failed_energies, marker='x', color='red', s=100, label='did not converge')
ax.legend()
ax.set_xlabel('Separation (Angstrom)')
ax.set_ylabel('Energy (eV)')
plt.savefig(out, dpi=200); print("wrote", out)
```

**Key requirement (D-6.09):** Both curves overlaid, non-converged points visibly marked (e.g., red X, dashed line, legend note).

**Key requirement (D-6.02):** Graph regenerated from live code at phase completion, for human visual review — never pre-computed or recorded.

---

## Shared Patterns

### Authentication / Access Guards
**Source:** `src/dftorch/ESDriver.py` (lines 40-85) and `src/dftorch/_coulomb_matrix.py` (lines 690-719)  
**Apply to:** All modified source files that add new capabilities  

**Pattern:**
```python
def _require_CAPABILITY(structure, const, context: str) -> None:
    """Reject unsupported paths for systems containing f orbitals.
    
    Modelled on existing guards; tolerates a structure with no ``const``
    so it can be called unconditionally as the guarded function's first statement.
    """
    if CAPABILITY_AVAILABLE:
        return
    const = getattr(structure, "const", None)
    if const is None:
        return
    n_orb = getattr(const, "n_orb", None)
    if n_orb is None or TYPE is None:
        return
    valid = TYPE >= 0
    if not bool(valid.any()):
        return
    counts = n_orb[TYPE.clamp(min=0)]
    if bool((valid & (counts == 16)).any()):
        raise FUnsupportedError(
            f"{context}: this system contains at least one atom with "
            f"n_orb == 16 (an f-shell element).\n"
            f"{UNSUPPORTED_MESSAGE}"
        )
```

### Convergence Status Reporting
**Source:** `src/dftorch/_scf.py` (lines 520-554)  
**Apply to:** SCFx and any other loop returning convergence  

**Pattern — compute flag at exit:**
```python
# At loop exit:
converged_iter = it if (
    ResNorm <= dftorch_params.get("SCF_TOL", 1e-6)
    and dEc <= dftorch_params.get("SCF_TOL", 1e-6) * 100
) else -1
return ..., converged_iter
```

**Pattern — unpack and store on structure:**
```python
# In ESDriver.forward:
(
    structure.H,
    # ... 13 more elements ...
    structure.scf_iter_count,  # -1 if not converged, else iteration count
) = SCFx(...)
```

### Test Harness for f Orbital Systems
**Source:** `tests/test_single_shot_energy.py` (lines 15-100)  
**Apply to:** All new f-orbital test modules  

**Pattern — imports and environment:**
```python
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import sys
from pathlib import Path

def run_with_float64(fn):
    """Run fn under float64 defaults, restoring dftorch module state after."""
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        for name in list(sys.modules.keys()):
            if name == "dftorch" or name.startswith("dftorch."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
```

**Pattern — scan with shared Constants instance:**
```python
# From test_eu_n_scan.py lines 227-249:
def eu_n_scan(tmp_path_factory):
    """Compute whole scan once, reusing one Constants instance."""
    def compute():
        from dftorch.Constants import Constants
        from dftorch.ESDriver import ESDriver
        
        tmp_dir = tmp_path_factory.mktemp("eu_n_scan")
        separations = scan_separations()
        
        # Built once and reused; verified bit-identical to per-point construction
        params = _eu_n_params(separations[0], tmp_dir)
        const = Constants(params).to("cpu")
        driver = ESDriver(params, device="cpu")
        
        energies = []
        for separation in separations:
            params = _eu_n_params(separation, tmp_dir)
            structure = Structure(params, const, device="cpu")
            driver(structure, const, do_scf=False)
            energies.append(float(structure.e_tot.item()))
        
        return {"separations": separations, "energies": energies}
    
    return run_with_float64(compute)
```

### Three-Branch Diagnostic Failure Message
**Source:** `tests/test_eu_n_scan.py` (lines 360-419)  
**Apply to:** Test modules reporting convergence or band-check failures  

**Pattern:**
```python
context = (
    f"\n  located minimum: {separation:.2f} A at grid index {index} of {last}"
    f"\n  tolerance band:  [{BAND_MIN:.3f}, {BAND_MAX:.3f}] A"
    f"  (target {TARGET} A, +/-{FRACTION * 100:.0f}%)"
    f"\n  scanned range:   [{SCAN_MIN:.2f}, {SCAN_MAX:.2f}] A"
    f"\n  full curve:\n{_format_curve(separations, energies)}"
)

if index == 0 or index == last:
    no_interior = (
        "GENUINE RED FLAG: no interior minimum...\n"
        "[diagnosis and remediation advice]\n"
        + context
    )
    raise AssertionError(no_interior)

if separation < BAND_MIN:
    short_of_band = (
        "MINIMUM SHORT OF THE BAND - ESCALATE TO HUMAN JUDGEMENT.\n"
        "[plausible physics argument]\n"
        + context
    )
    raise AssertionError(short_of_band)

if separation > BAND_MAX:
    long_of_band = (
        "GENUINE RED FLAG: minimum long of the band.\n"
        "[defect diagnosis]\n"
        + context
    )
    raise AssertionError(long_of_band)
```

---

## No Analog Found

No files in this phase have zero analog coverage. All files either reuse existing patterns from the same module or follow established conventions from related test modules.

---

## Integration Points & Cautions

### Critical: Tuple Unpacking Changes
- **File:** `ESDriver.py:663-678`  
- **Scope:** Only place SCFx output is unpacked in production code  
- **Change:** 14 → 15 elements  
- **Verification:** grep for other unpack sites; if any exist outside ESDriver, they are in blast radius

### Conditional on D-6.06 Diagnosis
If the Krylov mixer fix alone does not stabilize the loop AND coarse-grain charges are identified as the root cause, then shell-resolved Coulomb blocks become in-scope:
- Remove/narrow guard at `_coulomb_matrix.py:690`  
- Implement seven f blocks following existing pattern  
- Update `docs/F-SUPPORT-STATUS.md` to remove `FShellResolvedCoulombUnsupportedError` entry  
- Update `tests/test_support_documentation.py` exception list

### Non-Convergence is Honest, Not an Error
- D-6.04 requires returning -1, not raising  
- Failing separations are reported via graph marking (D-6.09)  
- Test suite does not fail on non-converged separations away from 2.655 Å (D-6.05)  
- Only 2.655 Å neighbourhood must converge (D-6.01)

### Reference Numbers Are Not Frozen
- D-6.08 forbids pinning converged energies or charges  
- Single-shot reference `-17.510444238744924` eV at 2.655 Å **stays pinned** (Phase 4 decision)  
- Self-consistent curve is validated by human visual inspection, not numeric gate

---

## Metadata

**Analog search scope:** 
- `tests/test_*.py` (all test modules, emphasis on f-orbital harnesses)
- `src/dftorch/_scf.py`, `ESDriver.py`, `_coulomb_matrix.py`, `Structure.py` (source modules)
- `experiments/diatomic_scans/*.py` (plotting scripts)

**Files scanned:** ~15 core source/test files  
**Pattern extraction date:** 2026-08-04  
**Confidence level:** HIGH (all files self-analogs or direct copies of established patterns)

---

*Phase: 6 - Self-Consistent SCF for f Systems*  
*Pattern mapping complete — ready for planner*
