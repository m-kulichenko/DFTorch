# Phase 4: SCF and Reference Simulation Validation - Pattern Map

**Mapped:** 2026-07-29  
**Files analyzed:** 8 (4 modifications + 4 new tests)  
**Analogs found:** 8 / 8 with concrete matches

---

## File Classification

| File | Role | Data Flow | Closest Analog | Match Quality |
|------|------|-----------|----------------|---------------|
| `src/dftorch/ESDriver.py` | service | request-response | `src/dftorch/ESDriver.py:615-620 (calc_forces)` | exact |
| `src/dftorch/ewald_pme/PME_torch.py` | utility | dtype transform | `src/dftorch/ewald_pme/PME_torch.py:261` | exact (bug location) |
| `src/dftorch/Constants.py` | config | configuration gate | `src/dftorch/Constants.py:137-140` | exact |
| `src/dftorch/Structure.py` | model | data construction | `src/dftorch/Structure.py:413-441` | exact |
| `tests/test_single_shot_energy.py` | test | integration | `tests/test_scf.py:13-69` | role-match |
| `tests/test_spin_guard.py` | test | unit error-handling | `tests/test_f_orbital_skf.py:9-23` | role-match |
| `tests/test_shell_resolved_u.py` | test | unit data validation | `tests/test_f_orbital_skf.py:9-23` | role-match |
| `tests/test_eu_n_scan.py` | test | integration | `tests/test_f_orbital_skf.py:116-176` | role-match |

---

## Pattern Assignments

### `src/dftorch/ESDriver.py` — Add do_scf=False Energy Path and Spin Guard

**Primary Analogs:**
- `src/dftorch/ESDriver.py:32-57` — Guard pattern for f derivatives
- `src/dftorch/ESDriver.py:615-620` — Guard call at method start
- `src/dftorch/ESDriver.py:544-571` — Energy computation pattern
- `src/dftorch/_slater_koster_pair.py:234-251` — Exception class definition

**Guard Pattern — Precedent from calc_forces** (lines 615-620):

```python
def calc_forces(self, structure, const):
    # Phase 3 supplies source-locked f angular *values* only; dH0/dS are
    # exactly zero inside every f block. Forces consume those derivatives,
    # so an f-containing system would silently integrate a wrong (zero)
    # gradient. Refuse before any force term is assembled.
    _require_f_derivatives(structure, const, "ESDriver.calc_forces")
```

**Guard Function Pattern** (lines 32-57):

```python
def _require_f_derivatives(structure, const, context: str) -> None:
    """Reject derivative-consuming paths for systems containing f orbitals.

    Phase 3 implements source-locked f angular *values* only, so ``dH0``/``dS``
    are exactly zero throughout every f block. A force or stress assembled from
    those zeros looks perfectly well formed, which is precisely why it must not
    be produced. See :class:`FDerivativeUnsupportedError`.
    """
    if F_ANGULAR_DERIVATIVES_AVAILABLE:
        return
    type_ids = getattr(structure, "TYPE", None)
    if type_ids is None:
        return
    n_orb = getattr(const, "n_orb", None)
    if n_orb is None:
        return
    valid = type_ids >= 0
    if not bool(valid.any()):
        return
    counts = n_orb[type_ids.clamp(min=0)]
    if bool((valid & (counts == 16)).any()):
        raise FDerivativeUnsupportedError(
            f"{context}: this system contains at least one atom with "
            f"n_orb == 16 (an f-shell element).\n"
            f"{F_DERIVATIVE_UNSUPPORTED_MESSAGE}"
        )
```

**Exception Class Pattern** (from `_slater_koster_pair.py:234-251`):

```python
class FDerivativeUnsupportedError(NotImplementedError):
    """Raised when a path would consume f-orbital H0/S *derivatives*.

    Phase 3 implements the f angular values (Takegahara Table 2) but not their
    Cartesian derivatives.  ``dH0``/``dS`` therefore carry exact zeros in every
    f-containing block.  Zero is a legal-looking derivative, so any consumer
    (forces, stress, MD, geometry optimisation) must fail loudly rather than
    integrate a silently wrong gradient.
    """


F_DERIVATIVE_UNSUPPORTED_MESSAGE: Final[str] = (
    "f-orbital Slater-Koster angular derivatives are not implemented.\n"
    "Phase 3 provides source-locked f angular *values* only, so dH0/dS are "
    "exactly zero inside every f block. That is indistinguishable from a real "
    "vanishing gradient, which is why this path refuses to run instead of "
    "returning an f-incomplete result.\n"
    "Use energies/H0/S for f-containing systems; forces, stress and MD for "
    "those systems require the deferred f derivative work."
)
```

**Spin-Polarization Guard — To be Created** (copy structure from _require_f_derivatives but adapted for spin check):

New exception class (add to `_slater_koster_pair.py` after FDerivativeUnsupportedError):

```python
class FSpinPolarizationUnsupportedError(NotImplementedError):
    """Raised when spin-polarized f-orbital calculations are requested.

    Phase 4 supports closed-shell f-orbital systems only. Open-shell treatment
    requires separate α/β density matrices and spin-dependent Hubbard coupling.
    """
```

New guard function (add to `ESDriver.py` after _require_f_derivatives):

```python
def _require_closed_shell_f_systems(structure, const, context: str) -> None:
    """Reject spin-polarized calculations for f-orbital systems.
    
    Phase 4 supports closed-shell only. Open-shell treatment requires
    spin-dependent Hubbard coupling and separate α/β charge grids.
    """
    type_ids = getattr(structure, "TYPE", None)
    if type_ids is None:
        return
    n_orb = getattr(const, "n_orb", None)
    if n_orb is None:
        return
    valid = type_ids >= 0
    if not bool(valid.any()):
        return
    counts = n_orb[type_ids.clamp(min=0)]
    has_f_system = bool((valid & (counts == 16)).any())
    if not has_f_system:
        return  # Not an f system; no constraint
    
    is_unrestricted = self.dftorch_params.get("UNRESTRICTED", False)
    if is_unrestricted:
        raise FSpinPolarizationUnsupportedError(
            f"{context}: system contains f-orbital atoms but UNRESTRICTED=True.\n"
            f"Spin-polarized f-orbital support deferred to Phase 5+."
        )
```

**do_scf=False Energy Path — Insert after line 326, before `if do_scf:` block**:

Pattern based on lines 544-571 (energy computation in SCF path):

```python
        # ─────────────────────────────────────────────────────────────────
        # Single-shot (non-SCF) energy path: use D0 directly, no charge loop
        # ─────────────────────────────────────────────────────────────────
        if not do_scf:
            # Use reference density D0 (set in Structure.__init__)
            # No SCF charge update; evaluate energy at fixed charge state
            structure.e_field = 0.0  # No external field for now
            structure.D = structure.D0.clone()
            structure.q = torch.zeros(structure.Nats, device=self.device, dtype=torch.get_default_dtype())
            structure.dq_p1 = torch.zeros_like(structure.q)
            structure.f = torch.zeros_like(structure.diagonal)
            structure.Te = self.dftorch_params.get("T_ELECTRONIC", 0.0)
            
            # For single-shot, set f (occupation) from reference diagonal
            # This is a simplification; full SCF would converge charges
            structure.f = torch.zeros(structure.HDIM, device=self.device)
            
            # Compute energy using reference density
            (
                structure.e_elec_tot,
                structure.e_band0,
                structure.e_coul,
                structure.e_dipole,
                structure.e_entropy,
                structure.s_ent,
            ) = energy(
                structure.H0,
                structure.Hubbard_U,
                structure.e_field,
                structure.D0,
                structure.C,
                structure.dq_p1,
                structure.D0,  # Use D0 instead of SCF-updated D
                structure.q,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.f,
                structure.Te,
                structure.dU_dq,
                thirdorder=structure.thirdorder,
            )
            
            structure.e_tot = (
                structure.e_elec_tot + structure.e_repulsion
            )
            
            # No solvation or DFTB3 for single-shot path (can be added later)
            structure.e_gb = 0.0
            structure.e_sasa = 0.0
            structure.e_solv = 0.0
            structure.e_d3 = 0.0
            structure.e_spin = 0.0
            return
```

**Location Guide:**
- Exception definitions: `src/dftorch/_slater_koster_pair.py` (import at top of ESDriver.py)
- Guard function: `src/dftorch/ESDriver.py` (add after `_require_f_derivatives`, before class ESDriver)
- Guard call: `src/dftorch/ESDriver.py:forward()` line ~210 (after repulsion, before Coulomb block)
- do_scf=False branch: `src/dftorch/ESDriver.py:forward()` line 328 (insert complete if-not-do_scf block)

---

### `src/dftorch/ewald_pme/PME_torch.py` — Fix dtype Bug

**Location:** Line 261

**Current Code (lines 260-264):**

```python
    # Zero out the G=0 contribution (m_2 == 0)
    g_mask = (m_2 > 0).float()  # (K1,K2,K3)

    # σ_{αβ} = (1/V) Σ_G  E_G · T_{αβ}(G)
    sigma = torch.einsum("ijk,ijk,ijkab->ab", E_G, g_mask, metric) / V
```

**Fix:** Replace line 261 with dtype-aware conversion:

```python
    # Zero out the G=0 contribution (m_2 == 0)
    g_mask = (m_2 > 0).to(dtype=E_G.dtype)  # Match einsum operand dtype

    # σ_{αβ} = (1/V) Σ_G  E_G · T_{αβ}(G)
    sigma = torch.einsum("ijk,ijk,ijkab->ab", E_G, g_mask, metric) / V
```

**Rationale:** PyTorch's `.float()` defaults to float32 regardless of project default dtype (float64). The einsum operation at line 264 requires all operands to have the same dtype. `E_G` and `metric` are float64 under the project default; `g_mask` must match.

---

### `src/dftorch/Constants.py` — Extend MAGNETIC_HUBBARD_LDEP Flag

**Existing Gating Pattern** (lines 137-140):

```python
            if self.magnetic_hubbard_ldep:
                self.w = torch.nn.Parameter(w_shell.clone(), requires_grad=False)
            else:
                self.w = torch.nn.Parameter(w_atom.clone(), requires_grad=False)
```

**Integration Point:** Phase 4 does NOT modify Constants.py directly; the flag already exists (line 57-58):

```python
        self.magnetic_hubbard_ldep = dftorch_params.get("MAGNETIC_HUBBARD_LDEP", False)
```

**Note:** The flag's docstring (line 38-39) already documents shell-dependent Hubbard U:

```python
        ``MAGNETIC_HUBBARD_LDEP`` : bool, default False
            Use shell-dependent (l-dependent) Hubbard U parameters.
```

**No changes needed in Constants.py** — the infrastructure is complete. Phase 4 will thread the `const.magnetic_hubbard_ldep` flag through ESDriver and coulomb_matrix_vectorized to gate shell-resolved Hubbard U usage.

---

### `src/dftorch/Structure.py` — Shell-Resolved Hubbard_U_sr (Already Exists)

**Pattern** (lines 413-441):

```python
        # Shell on-site Hubbard U per atom.
        UsA = const.U[self.TYPE]       # s shell U
        UpA = const.Up[self.TYPE]      # p shell U
        UdA = const.Ud[self.TYPE]      # d shell U
        UfA = const.Uf[self.TYPE]      # f shell U
        ns = const.n_s[self.TYPE]      # electron count per s shell
        np = const.n_p[self.TYPE]      # electron count per p shell
        nd = const.n_d[self.TYPE]      # electron count per d shell
        nf = const.n_f[self.TYPE]      # electron count per f shell

        # Stack into template (per-atom, per-shell)
        template_U = torch.stack((UsA, UpA, UdA, UfA), dim=1)
        template_ang = torch.stack(
            (
                torch.ones_like(UsA, dtype=torch.int64),
                torch.ones_like(UsA, dtype=torch.int64) + 1,
                torch.ones_like(UsA, dtype=torch.int64) + 2,
                torch.ones_like(UsA, dtype=torch.int64) + 3,
            ),
            dim=1,
        )
        template_el_per_shell = torch.stack((ns, np, nd, nf), dim=1)

        # Apply shell_present mask to get active shells only
        shell_mask = self.shell_present
        self.Hubbard_U_sr = template_U[shell_mask]       # Shape: (sum of active shells,)
        self.shell_types = template_ang[shell_mask]      # Angular momentum labels
        self.el_per_shell = template_el_per_shell[shell_mask]
        self.n_shells_per_atom = self.shell_present.sum(dim=1).to(torch.int64)
        # Cumulative indices for shell-resolved Hubbard U access
        self.H_INDEX_START_U = torch.zeros(self.Nats, dtype=torch.int64, device=device)
        self.H_INDEX_START_U[1:] = torch.cumsum(self.n_shells_per_atom, dim=0)[:-1]
        self.H_INDEX_END_U = self.H_INDEX_START_U + self.n_shells_per_atom - 1
```

**No changes needed in Structure.py** — shell-resolved construction already includes f shells (line 416: `UfA = const.Uf[self.TYPE]`). Phase 4 tests (test_shell_resolved_u.py) will verify this works end-to-end.

---

## Test Files — Pattern Template

All new tests follow the harness pattern from `tests/test_f_orbital_skf.py` (lines 9-23):

**Common Test Harness Pattern** (from test_f_orbital_skf.py):

```python
import importlib.util
import sys
from pathlib import Path

import torch


def run_with_float64(fn):
    """Decorator: run test function with torch.float64 default dtype and clean module reload."""
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
        for name in [name for name in sys.modules if name == "dftorch" or name.startswith("dftorch.")]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def load_validation_script():
    """Load the dftorch validation/test helper module (script.py)."""
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "dftorch" / "script.py"
    spec = importlib.util.spec_from_file_location("dftorch_phase1_validation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation script from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

**Usage in Test Functions:**

```python
def test_example():
    """Test that runs with float64 and module reset."""
    def check():
        # Test code here
        validation = load_validation_script()
        # ... rest of test
        return result
    
    assert run_with_float64(check) == expected
```

---

### `tests/test_single_shot_energy.py` — Single-Shot Energy Calculation

**Analog:** `tests/test_scf.py:13-69` (energy smoke test)

**Pattern** (from test_scf.py):

```python
@pytest.mark.parametrize("device", ["cpu"])
def test_energy_smoke_import_and_call(device):
    """
    CI smoke test:
    - Imports core modules.
    - Runs a *small* _scf + forces calculation on CPU.
    - Skips automatically if required SKF test data is not present.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = root / "tests" / "ch4.xyz"
    skf_dir = root / "tests" / "data_skf_mio-1-1"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

    import torch

    torch.set_default_dtype(torch.float64)

    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

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

    const = Constants(dftorch_params).to(device)
    structure1 = Structure(dftorch_params, const, device=device)

    es_driver = ESDriver(dftorch_params, device=device)
    es_driver(structure1, const, do_scf=True)  # or do_scf=False for single-shot

    assert hasattr(structure1, "e_tot")
    assert torch.isfinite(structure1.e_tot).all()
```

**Adaptation for test_single_shot_energy.py:**
- Use f_orbital_data SKF files (Eu, Ga, N)
- Call `do_scf=False` instead of `do_scf=True`
- Assert that e_tot is populated and finite

---

### `tests/test_spin_guard.py` — Spin-Polarization Guard

**Analog:** Test pattern from test_f_orbital_skf.py with exception assertion

**Pattern (pytest exception check):**

```python
import pytest

def test_spin_polarization_error_on_f_systems():
    """Verify that UNRESTRICTED=True raises on f-containing systems."""
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        skf_dir = project_root / "tests" / "f_orbital_data"
        xyz_path = project_root / "tests" / "eu_single.xyz"  # Single Eu atom
        
        from dftorch.Constants import Constants
        from dftorch.ESDriver import ESDriver
        from dftorch.Structure import Structure
        from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError
        
        dftorch_params = {
            "FILENAME": str(xyz_path),
            "SKFPATH": str(skf_dir),
            "T_ELECTRONIC": 300.0,
            "RCUT_ELECTRONIC": 10.0,
            "RCUT_REPULSIVE": 6.0,
            "COUL_METHOD": "FULL",
            "UNRESTRICTED": True,  # Trigger the guard
        }
        
        const = Constants(dftorch_params).to("cpu")
        struct = Structure(dftorch_params, const, device="cpu")
        
        es_driver = ESDriver(dftorch_params, device="cpu")
        
        # Should raise FSpinPolarizationUnsupportedError
        with pytest.raises(FSpinPolarizationUnsupportedError):
            es_driver(struct, const, do_scf=False)
        
        return True
    
    assert run_with_float64(check)
```

---

### `tests/test_shell_resolved_u.py` — Shell-Resolved Hubbard U Data Validation

**Analog:** test_f_orbital_skf.py shape/metadata assertions

**Pattern (from test_f_orbital_skf.py metadata checks):**

```python
def test_shell_resolved_hubbard_u_shapes():
    """Verify Hubbard_U_sr has correct shape for systems with s/p/d/f shells."""
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        skf_dir = project_root / "tests" / "f_orbital_data"
        xyz_path = project_root / "tests" / "eu_n.xyz"  # Eu-N pair
        
        from dftorch.Constants import Constants
        from dftorch.Structure import Structure
        
        dftorch_params = {
            "FILENAME": str(xyz_path),
            "SKFPATH": str(skf_dir),
            "T_ELECTRONIC": 300.0,
            "RCUT_ELECTRONIC": 10.0,
            "RCUT_REPULSIVE": 6.0,
            "MAGNETIC_HUBBARD_LDEP": True,  # Enable shell-resolved
        }
        
        const = Constants(dftorch_params).to("cpu")
        struct = Structure(dftorch_params, const, device="cpu")
        
        # Assertions on shell-resolved U structure
        assert hasattr(struct, "Hubbard_U_sr"), "Hubbard_U_sr must be set"
        assert hasattr(struct, "n_shells_per_atom"), "n_shells_per_atom must be set"
        assert struct.Hubbard_U_sr.shape[0] == struct.n_shells_per_atom.sum(), \
            "Hubbard_U_sr length must match total active shells"
        
        # Verify f shell is included for Eu (has 4 shells: s/p/d/f)
        eu_idx = 0  # Eu is first atom
        eu_shells = struct.n_shells_per_atom[eu_idx].item()
        assert eu_shells == 4, f"Eu should have 4 shells (s/p/d/f), got {eu_shells}"
        
        # Verify N has 3 shells (s/p/d)
        n_idx = 1  # N is second atom
        n_shells = struct.n_shells_per_atom[n_idx].item()
        assert n_shells == 3, f"N should have 3 shells (s/p/d), got {n_shells}"
        
        return True
    
    assert run_with_float64(check)
```

---

### `tests/test_eu_n_scan.py` — Eu-N Energy Scan Validation

**Analog:** test_f_orbital_skf.py build_h0_and_s pattern (lines 116-176)

**Pattern (adapted for energy scan):**

```python
def write_xyz_euN(path: Path, separation: float) -> None:
    """Write Eu-N diatom with specified separation in Angstroms."""
    lines = ["2", "Eu-N diatom"]
    lines.append("Eu 0.0 0.0 0.0")
    lines.append(f"N {separation:.8f} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")


def test_f_orbital_eu_n_energy_scan():
    """Validate f-orbital energy prediction against Eu-N bond-length minimum.
    
    Performs a uniform energy scan over Eu-N separation and asserts that
    the energy minimum falls within the loose tolerance band (~10-20% of
    literature value ~2.4 Å).
    """
    def check():
        validation = load_validation_script()
        project_root = validation.find_project_root()
        skf_dir = project_root / "tests" / "f_orbital_data"
        
        from dftorch.Constants import Constants
        from dftorch.ESDriver import ESDriver
        from dftorch.Structure import Structure
        
        # Scan parameters (D-20 discretion: uniform grid, modest resolution)
        separations = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]  # 0.2 Å steps
        energies = []
        
        for sep in separations:
            xyz_path = Path("/tmp") / f"eu_n_{sep:.1f}.xyz"
            write_xyz_euN(xyz_path, sep)
            
            dftorch_params = {
                "FILENAME": str(xyz_path),
                "SKFPATH": str(skf_dir),
                "T_ELECTRONIC": 300.0,
                "RCUT_ELECTRONIC": 10.0,
                "RCUT_REPULSIVE": 6.0,
                "COUL_METHOD": "FULL",
            }
            
            const = Constants(dftorch_params).to("cpu")
            struct = Structure(dftorch_params, const, device="cpu")
            
            es_driver = ESDriver(dftorch_params, device="cpu")
            es_driver(struct, const, do_scf=False)  # Single-shot energy
            
            energies.append(struct.e_tot.item())
            print(f"Separation {sep:.2f} Å: E = {struct.e_tot.item():.6f} eV")
        
        # Find minimum (D-20/D-22: loose ~10-20% band on 2.4 Å)
        min_idx = energies.index(min(energies))
        min_separation = separations[min_idx]
        
        TOLERANCE_MIN = 2.4 * 0.80  # –20% = 1.92 Å
        TOLERANCE_MAX = 2.4 * 1.20  # +20% = 2.88 Å
        
        print(f"\nEu-N energy minimum at: {min_separation:.2f} Å")
        print(f"Tolerance band: [{TOLERANCE_MIN:.2f}, {TOLERANCE_MAX:.2f}] Å")
        
        assert (TOLERANCE_MIN <= min_separation <= TOLERANCE_MAX), \
            f"Minimum {min_separation:.2f} Å outside tolerance band"
        
        return True
    
    assert run_with_float64(check)
```

---

## Shared Patterns

### Authentication / Guards (applies to ESDriver)

**Source:** `src/dftorch/_slater_koster_pair.py:32-57` and `234-251`

**Apply to:** ESDriver.forward() — add spin-polarization guard before SCF block

**Pattern:**
1. Define exception class inheriting from NotImplementedError
2. Define guard function that checks feature flag + raises named exception
3. Call guard as FIRST statement in method that would silently produce wrong results

---

### Error Handling — Named Exceptions

**Source:** `src/dftorch/_slater_koster_pair.py:173-195`, `234-251`

**Apply to:** All new guards in Phase 4 (spin polarization)

**Pattern:**
1. Exception class with docstring explaining the constraint
2. Final message constant explaining workaround
3. Guard function with context parameter for informative error

---

### Configuration Gating

**Source:** `src/dftorch/Constants.py:57-58, 137-140`

**Apply to:** ESDriver.forward() energy path selection

**Pattern:**
```python
if const.magnetic_hubbard_ldep:
    # Use shell-resolved variant
else:
    # Use per-atom variant
```

---

### Data Structure Construction with f Shells

**Source:** `src/dftorch/Structure.py:413-441`

**Apply to:** All tests that construct structures with Eu atoms

**Pattern:**
1. Extract per-shell parameter (U, on-site energy, electron count) from Constants
2. Stack into template tensor (4 shells: s/p/d/f)
3. Apply shell_present mask to get active shells only
4. Compute cumulative indices for access

---

## No Analog Found

**None.** All Phase 4 files have direct analogs in existing codebase or clear precedent patterns.

---

## Metadata

**Analog search scope:** `src/dftorch/`, `tests/`  
**Files scanned:** ~120  
**Pattern extraction date:** 2026-07-29  

**Confidence summary:**
- **ESDriver modifications:** HIGH — Direct guard precedent from Phase 3; energy pattern from existing SCF code
- **PME dtype fix:** HIGH — Bug location confirmed; fix pattern standard in codebase
- **Constants/Structure:** HIGH — Existing infrastructure already complete; no new patterns needed
- **Test patterns:** HIGH — Direct adaptation from test_f_orbital_skf.py and test_scf.py

---

## Key Integration Points

### 1. Exception Imports in ESDriver.py (Top of File)

Current (line 22-26):
```python
from ._slater_koster_pair import (
    F_ANGULAR_DERIVATIVES_AVAILABLE,
    F_DERIVATIVE_UNSUPPORTED_MESSAGE,
    FDerivativeUnsupportedError,
)
```

Must add (after line 26):
```python
    FSpinPolarizationUnsupportedError,
)
```

### 2. Guard Function Definitions in ESDriver.py (Before Class ESDriver)

Pattern from _require_f_derivatives (lines 32-57) — add new spin guard function

### 3. Guard Call Location in ESDriver.py (In forward() method)

Call spin guard after repulsion calculation, before Coulomb block (~line 210)

### 4. do_scf=False Branch in ESDriver.py (In forward() method)

Insert complete branch at line 328 (before existing `if do_scf:` block)

---

## Ready for Planning

Pattern mapping complete. All concrete analog excerpts with file paths and line numbers are now available for planner to reference in PLAN.md action sections.
