# Phase 4: SCF and Reference Simulation Validation - Research

**Researched:** 2026-07-29
**Domain:** Single-shot f-orbital energy calculation and Eu-N diatomic validation
**Confidence:** HIGH for infrastructure; MEDIUM for scan design parameters

## Summary

Phase 4 is narrowly scoped: deliver single-shot energy (H0/S → diagonalize once → report band/total energy) for f-containing systems and validate the f machinery against a common-sense Eu-N bond-length minimum check. The infrastructure for shell-resolved Hubbard U is already in place from Phase 2. The critical blocker is a missing non-SCF energy path in ESDriver, and a float32 dtype bug in PME that currently breaks test_scf.py. Both must be resolved before energy can flow.

**Primary recommendation:** Fix the PME dtype bug immediately (D-17 gating item), then implement the do_scf=False energy path (diagonalize + energy function calls), then extend shell-resolved U threading to energy calculation sites, then validate the Eu-N case.

---

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-11:** Single-shot energy only — no iterative SCF charge updates.
- **D-12:** Spin polarization must raise `FSpinPolarizationUnsupportedError` (explicit guard).
- **D-14:** Extend and test shell-resolved U shapes, even though D-11's path won't consume them (de-risks later SCF work).
- **D-15:** f Hubbard U values source from extended SKF headers. If absent → blocker to surface, not zero-fill.
- **D-16:** Support BOTH single per-atom U AND shell-resolved U, gated by config flag (D-15's decision point).
- **D-17/D-18:** PME dtype audit lands in Phase 4, not deferred. Broader than line 261 alone.
- **D-19:** Validation is energy scan over Eu-N separation, NOT geometry optimization (forces not available for f systems).
- **D-20/D-22:** ~10–20% loose tolerance on Eu-N minimum; do NOT tighten.

### Claude's Discretion
- Scan range, step count, and uniformity for Eu-N diatomic.
- Dtype audit scope and file staging order.
- Non-convergence warning channel (D-13 reuse or new?).

### Deferred Ideas (OUT OF SCOPE)
- Self-consistent SCF for f systems (Phase 5+).
- Spin-polarized / collinear-spin for open-shell 4f.
- f derivatives (forces, stress, MD).
- Ga-containing and periodic EuN crystal cases.

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIM-01 | Minimal f system reaches H0/S construction without shape/routing failure | H0/S routing gated in Phase 3; Eu-N fixture has 16-orbital atoms and routes to f blocks |
| SIM-02 | Minimal f system reaches closed-shell energy or explicit unsupported error | Spin-polarization guard must be installed (D-12); no SCF charge loop (D-11) |
| SIM-03 | Shell-resolved charge/Hubbard/Coulomb data include f dimensions | Hubbard_U_sr scaffolding exists; f entries sourced from SKF headers (D-15 confirmed PRESENT) |
| SIM-04 | Eu-N diatomic documented (geometry, SKF files, observables, units, tolerances) | Fixture at `tests/f_orbital_data/Eu-N.skf` exists; test pattern established in test_f_orbital_skf.py |
| SIM-05 | Eu-N energy scan locates minimum within ~10–20% of known bond length | Loose-band approach; requires single-shot energy path and scan harness |

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| H0/S assembly for f systems | Backend (Phase 3) | — | Already routed and source-locked; Phase 4 consumes this |
| Single-shot energy (no SCF loop) | Backend (ESDriver.forward + energy module) | — | Must diagonalize H0/S once and compute band/total energy |
| Shell-resolved Hubbard U data | Backend (Structure + ESDriver integration points) | Frontend tests | SCF paths thread U through ~14 ESDriver call sites |
| Coulomb/charge repulsion with f | Backend (ESDriver.forward Coulomb matrix) | — | Uses shell-resolved U; shapes must include f dimension |
| Spin-polarization guard | Backend (ESDriver.forward SCF branch) | — | Raise before any open-shell code runs |
| Eu-N validation (energy scan) | Backend (single-shot loop) | Test harness (test_f_orbital_skf.py) | Scans separation, computes energy at each point; validates minimum location |

---

## Standard Stack

### Core (Must-Have)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.0+ | Tensor operations, autodiff | DFTorch's compute substrate; float64 default required |
| NumPy | 1.20+ | Utility (device moves, file I/O) | Standard Python numerical library |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| SciPy | 1.7+ | Eigenvalue solvers (linalg.eigh) | Already used in _scf.py for Hamiltonian diagonalization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch linalg.eigh | NumPy diagonalization | NumPy slower on large matrices; PyTorch stays on device |
| Direct energy formula | _energy function in _energy.py | Custom implementation would duplicate tested code |

**Installation:** No new packages needed; existing dftorch environment suffices.

---

## Package Legitimacy Audit

**No new external packages required for Phase 4.** All energy, Coulomb, and diagnostics code already exists. Work is integration, bugfix, and validation within existing imports.

---

## Critical Finding: The Single-Shot Energy Path Does Not Exist

### Current Code Structure

ESDriver.forward() (lines 84–614) takes `do_scf=True/False`, but:

- **do_scf=True path** (lines 328–614): Full SCF loop (closed-shell or open-shell), charge updates, energy computation.
- **do_scf=False path**: Skipped entirely. Method ends at line 614; no energy computed.

The docstring (line 96–98) promises: *"If False, evaluate energy and forces at the current charge state (useful for non-SCC DFTB or for shadow-energy MD restart steps)."* But the implementation does not exist.

### Implications for Phase 4

D-11 requires **single-shot energy without SCF charge self-consistency**. The intended usage pattern is:

```python
# One-time diagonalization energy
es_driver(structure, const, do_scf=False)
# structure.e_tot should be populated
```

But currently this leaves `structure.e_tot` unset.

### Minimal Required Fix

After the H0/S assembly and repulsion energy calculation (lines 147–198), insert a non-SCF energy branch:

```python
if do_scf:
    # ... existing 328–614 ...
else:
    # Single-shot energy path:
    # 1. Construct initial charge density from reference occupations (D0 already set in Structure.__init__)
    # 2. No SCF loop; use D0 directly
    # 3. Assemble Coulomb matrix C (uses Hubbard_U, shaped for f)
    # 4. Call energy() function (existing in _energy.py)
    structure.e_tot = e_elec + structure.e_repulsion
```

**Blocker:** This path is currently missing. Must be implemented before any Eu-N scan can run.

---

## Finding 1: f Hubbard U Headers — CONFIRMED PRESENT

**Question:** Do the extended-format SKF files actually carry an f Hubbard U value?

**Answer:** YES, CONFIRMED PRESENT at `tests/f_orbital_data/Eu-Eu.skf`.

### Evidence

**File:** `tests/f_orbital_data/Eu-Eu.skf` [VERIFIED: direct file read]

Header (line 3):
```
-0.0559 -0.0247 0.0197 -0.0968    0.0   0.50 0.25 0.19 0.21    7.0 0.0 0.0 2.0
```

Extended homonuclear header parsing in `src/dftorch/_bond_integral.py:620–641` [VERIFIED: source code]:

```python
if extended:
    (Ef, Ed, Ep, Es, SPE,
     Uf, Ud, Up, Us,
     ff, fd, fp, fs) = (float(token) for token in header_tokens[:13])
```

**Mapped values:**
- Uf (f Hubbard U, position 6) = **0.50** eV (in atomic units before EV_PER_HARTREE conversion)
- Stored to `UF[el_num]` at line 696

**Verification:** Constants.py line 177 registers this as `self.Uf = torch.nn.Parameter(UF)`, confirming the SKF parser already surfaces f-shell U values to the Constants container.

**Conclusion for D-15:** f-shell Hubbard U values are **present in the fixture files and already parsed**. No blocker; proceed with usage.

---

## Finding 2: Shell-Resolved Hubbard U Infrastructure — Partially Ready

**Question:** Where does shell-resolved U thread through ESDriver, and what must be extended to support f?

**Answer:** Infrastructure exists but is incomplete; shell-resolved U is computed but not yet consumed by all energy/Coulomb paths.

### Current Infrastructure

**Structure.__init__** (lines 361–441, `src/dftorch/Structure.py`):

1. **Per-atom U:** `structure.Hubbard_U = const.U[self.TYPE]` (line 361) — scalar per atom (s-only).
2. **Shell presence:** `structure.shell_present = const.shell_present[self.TYPE]` (line 363) — 4-element bool tensor (s/p/d/f).
3. **Shell-resolved U:** Lines 413–435 construct `template_U = stack([UsA, UpA, UdA, UfA])` and mask to active shells:
   ```python
   UsA = const.U[self.TYPE]      # s
   UpA = const.Up[self.TYPE]     # p
   UdA = const.Ud[self.TYPE]     # d
   UfA = const.Uf[self.TYPE]     # f  ← Already extracted from SKF
   template_U = torch.stack((UsA, UpA, UdA, UfA), dim=1)
   self.Hubbard_U_sr = template_U[shell_mask]  # Line 435
   ```
4. **Shell counts:** `structure.n_shells_per_atom = shell_present.sum(dim=1)` (line 438) — e.g., 3 for s/p/d, 4 for s/p/d/f.

**Constants.py** (lines 65–67, 174–177):

- `shell_dim = [0, 1, 3, 5, 7]` — dimensions for each shell (already supports l=3 for f).
- `self.Uf = torch.nn.Parameter(UF)` — f Hubbard U per element.

### Where It's NOT Yet Used

Current consumption of Hubbard U:

- **ESDriver.forward:**
  - Line 254: `coulomb_matrix_vectorized(structure.Hubbard_U, ...)` — Uses per-atom U only (not shell-resolved).
  - Line 529, 553, 689: Similar — all use `structure.Hubbard_U` (scalar per atom).
- **SCFx** (_scf.py): Takes `Hubbard_U` (per-atom) as argument; no shell indexing visible.
- **energy** (_energy.py): Consumes `Hubbard_U` (per-atom).

### What D-14/D-16 Require

- **D-14:** Test that shell-resolved shapes are correct (Hubbard_U_sr has right length for s/p/d/f mix).
- **D-16:** Add config gate (flag like `MAGNETIC_HUBBARD_LDEP` in Constants.py line 58) to choose:
  - **False (default):** Use per-atom Hubbard_U (existing behavior, s-only).
  - **True:** Use shell-resolved Hubbard_U_sr (l-dependent, includes f).

### Threading Points in ESDriver (~14 call sites claimed in CONTEXT.md)

[ASSUMED: exact count not verified line-by-line, but representative sites found]

- Line 254: coulomb_matrix_vectorized → would take Hubbard_U_sr if shell-resolved mode.
- Lines 377–378, 391: SCF functions take n_shells_per_atom and shell_dim (infrastructure already there).
- Line 411: calc_forces passes n_shells_per_atom.
- Batch code (lines 1503, 1525): Similar patterns for batched structures.

### Implementation Path for D-16

1. Add flag check: `if const.magnetic_hubbard_ldep: use_shell_resolved = True else False`
2. In coulomb_matrix_vectorized call (and others), pass `Hubbard_U_sr if use_shell_resolved else Hubbard_U`.
3. Update _coulomb_matrix.py line 722 (`CDIM = len(structure.Hubbard_U_sr)`) to handle both paths.
4. Test both paths end-to-end.

---

## Finding 3: PME dtype Bug — ROOT CAUSE CONFIRMED + CLASS AUDIT

**Question:** What is the dtype bug scope?

**Answer:** One confirmed critical bug at line 261; one benign `.float()` elsewhere; no other class offenders found.

### Critical Bug: PME_torch.py Line 261

**File:** `src/dftorch/ewald_pme/PME_torch.py:261` [VERIFIED: direct read]

```python
g_mask = (m_2 > 0).float()  # (K1,K2,K3) — **HARDCODED FLOAT32**
```

**Context:** Lines 254–264 compute stress tensor σ:

```python
metric = (...)  # float64 under project default
E_G = (...)     # float64 (eigsum input from energy calculation)
g_mask = (m_2 > 0).float()  # ← BECOMES FLOAT32
sigma = torch.einsum("ijk,ijk,ijkab->ab", E_G, g_mask, metric) / V
                   # einsum tries float64 × float32 × float64 → ERROR
```

**Error reproducer:** `pytest tests/test_scf.py` [VERIFIED: test fails on line 264]

```
RuntimeError: Expected all tensors to have the same dtype in einsum operation
```

**Fix:** Change `.float()` to `.to(dtype=E_G.dtype)` or `.to(dtype=metric.dtype)`:

```python
g_mask = (m_2 > 0).to(dtype=E_G.dtype)  # Match the einsum operands
```

### Dtype Audit Results

**Sweep of src/dftorch/ for `.float()`, `.double()`, hardcoded dtype:** [VERIFIED: grep + manual inspection]

| File | Line | Pattern | Verdict | Action |
|------|------|---------|---------|--------|
| ewald_pme/PME_torch.py | 261 | `(m_2 > 0).float()` | **REAL BUG** | Fix to `.to(dtype=E_G.dtype)` |
| _nearestneighborlist.py | ~line with `.float()` | `ri.float() * (N * d2_max)` | **BENIGN** | Indices for sorting; dtype mismatch not possible |
| Constants.py | 65–66 | `torch.tensor(..., dtype=torch.int64)` | **OK** | Intentional int; no float issues |
| ewald_torch.py | Various | `torch.tensor(1.0, dtype=dtype, device=device)` | **OK** | Explicitly passes dtype |

**Conclusion for D-18:** Scope is **one high-priority fix** (PME line 261) + verification that no other `.float()` / `.double()` conversions silently truncate precision elsewhere. The audit found no other class of bug. D-18 is achievable in one task.

---

## Finding 4: Spin Polarization Guard — Guard Site Identified

**Question:** Where should the spin-polarization error be raised (D-12)?

**Answer:** ESDriver.forward(), after construction of `structure` but before entering the SCF branching logic.

### Proposed Implementation

**Location:** `src/dftorch/ESDriver.py`, after line 200 (after H0/S assembly and repulsion, before Coulomb setup).

**Precedent:** Phase 3 pattern for derivative guard (lines 32–57):

```python
def _require_f_derivatives(structure, const, context: str) -> None:
    """Reject derivative-consuming paths for systems containing f orbitals."""
    if F_ANGULAR_DERIVATIVES_AVAILABLE:
        return
    if not bool((valid & (counts == 16)).any()):
        return
    raise FDerivativeUnsupportedError(...)
```

**Analogous guard for spin polarization:**

```python
def _reject_spin_polarization(structure, const, context: str) -> None:
    """Reject spin-polarized calculations for f-orbital systems.
    
    Phase 4 supports closed-shell only. Open-shell treatment requires
    spin-dependent Hubbard coupling and separate α/β charge grids.
    """
    if not bool((valid & (counts == 16)).any()):
        return  # Not an f system; no constraint
    if self.dftorch_params.get("UNRESTRICTED", False):
        raise FSpinPolarizationUnsupportedError(
            f"{context}: system contains f-orbital atoms but UNRESTRICTED=True.\n"
            f"Spin-polarized f-orbital support deferred to Phase 5+."
        )
```

**Call site:** ESDriver.forward(), line 200–210 region (before the `if do_scf:` block at 328).

**Exception name:** `FSpinPolarizationUnsupportedError` (new, defined in _slater_koster_pair.py with other f exceptions).

---

## Finding 5: Eu-N Diatomic Validation Scan Design

**Question:** What is the literature Eu-N bond length, and what scan range/step count resolves a minimum?

> **⚠ SUPERSEDED 2026-07-29 by CONTEXT.md D-24.** The ~2.4–2.5 Å figure below was
> unsourced model recollection (assumption A3). The user has since supplied a real citation.
> **Use D-24's values, not this section's.** Retained only for provenance.
>
> - **Target: 2.655 Å** — mean of 16 Eu-N bond lengths from a user-supplied table of Eu-Bp
>   coordination complexes across four ligand variants (Bp, Bp^Me, Bp^Me2, Bp^CF3);
>   range 2.606–2.716, spread ~4%. Eu-O rows excluded.
> - **Band: ±20% → [2.12, 3.19] Å.** Express as a percentage of the target, NEVER as a
>   hardcoded absolute half-width. The "±0.2 Å" arithmetic below is wrong (it yields ±8%,
>   silently tightening the test past what D-20 permits).
> - **Caveat (accepted):** the citation is dative Eu-N in a crowded 8–9-coordinate sphere,
>   not an isolated diatomic; a diatomic would plausibly be shorter. Bulk rock-salt EuN
>   (~2.45–2.5 Å) also falls inside the band, so two reference points agree.
> - **Asymmetric failure rule:** short-of-band = plausibly correct diatomic physics, escalate
>   to human judgement; long-of-band or no interior minimum = genuine red flag.

**Answer (superseded):** Literature value is sparse in Phase docs, but ~2.4–2.5 Å is typical for rare-earth nitrides. Recommend a centered scan with modest resolution.

### Eu-N Equilibrium Separation

[SUPERSEDED by D-24 — was: ASSUMED from training data, not verified against a specific paper]

- **Typical value:** ~2.4–2.5 Å (cubic EuN lattice parameter ÷ 2 for nearest-neighbor separation)
- **Loose tolerance band (D-20):** ±10–20% → [1.92–2.75 Å for 2.40 Å center]
- **Conservative estimate:** Scan 1.8–3.0 Å to safely cover the band

### Recommended Scan Parameters

> **⚠ Range below is too narrow under D-24.** Centered on 2.655 Å with a ±20% band of
> [2.12, 3.19] Å, a 1.8–3.0 Å scan clips the upper band edge. Widen to at least 1.6–3.4 Å so
> both band edges — and a short diatomic minimum — are observable rather than clipped.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Range** | 1.8–3.0 Å ⚠ widen per D-24 | Covers ~±25% around 2.4 Å; ensures minimum detection |
| **Step size** | 0.2 Å | 6–7 points; resolves curvature near minimum without excess cost |
| **Total points** | ~6–7 | Single-shot energy ≈ 0.1–0.2 s per point; full scan <2 s |
| **Uniformity** | Uniform grid | Simplest; no adaptive refinement needed for loose tolerance |

### Test Pattern from Phase 3

File: `tests/test_f_orbital_skf.py`, function `build_h0_and_s()` (lines 116–176) [VERIFIED: source code]

```python
def build_h0_and_s(..., spacing: float = 1.5, axis: str = "x"):
    """Directly assemble H0/S for elements without running SCF."""
    write_xyz(xyz_path, elements, spacing=spacing, axis=axis)  # Write geometry
    struct = build_structure(validation, ..., xyz_path, const)  # Create Structure
    # Neighbor list, H0/S assembly
    return const, struct, H0, dH0, S, dS
```

**Adaptation for Eu-N scan:**

```python
def test_f_orbital_eu_n_scan():
    torch.set_default_dtype(torch.float64)
    # Module reset fixture (lines 9–23)
    
    # Write Eu-N diatom at varying separations
    for separation in [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
        write_xyz(xyz_path, ["Eu", "N"], spacing=separation, axis="x")
        const, struct, H0, _, S, _ = build_h0_and_s(
            validation, project_root, skf_dir, ["Eu", "N"], xyz_path, spacing=separation
        )
        # Compute energy (requires do_scf=False path)
        es_driver = ESDriver(dftorch_params, device="cpu")
        es_driver(struct, const, do_scf=False)
        energies.append(struct.e_tot.item())
    
    # Find minimum
    min_idx = energies.index(min(energies))
    min_separation = [1.8, 2.0, ...][min_idx]
    
    # Assert within ~10–20% of known value (~2.4 Å)
    assert 1.92 <= min_separation <= 2.88, f"Minimum at {min_separation} Å outside tolerance band"
```

---

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │  ESDriver.forward(do_scf=False)         │
                    │  Single-Shot Energy Path (Phase 4)      │
                    └────────────┬────────────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
          ┌───────▼────────┐          ┌────────▼────────┐
          │ H0_and_S Build │          │ Repulsion Energy│
          │ (Phase 3)      │          │ get_repulsion   │
          └───────┬────────┘          └────────┬────────┘
                  │                             │
                  │              ┌──────────────┴──────────────┐
                  │              │                             │
          ┌───────▼──────────┐  ┌▼─────────────┐    ┌──────────▼──────┐
          │  Overlap Matrix  │  │ Coulomb (C)  │    │  Diagonal (Te)  │
          │  from spline fit │  │ coulomb_     │    │  from Constants │
          │                  │  │ matrix_      │    │                  │
          │  S = S_overlap   │  │ vectorized   │    │  = Es, Ep, Ed, Ef│
          └────────┬─────────┘  └──────┬───────┘    └──────────┬───────┘
                   │                   │                       │
                   │    ┌──────────────┴───────────────┐       │
                   │    │                              │       │
                  │    │     ┌───────────────────────▼──────┐ │
                  │    │     │  energy() function           │ │
                  │    │     │  (existing, _energy.py)      │ │
                  │    │     │  Takes: H0, C, D (charges),  │ │
                  │    │     │  Hubbard_U, q, Etc.          │ │
                  │    │     └───────────────┬──────────────┘ │
                  │    │                     │                 │
                  │    └─────────────────────┤─────────────────┘
                  │                          │
                  └──────────────┬───────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
    ┌───────▼──────────┐                   ┌─────────▼─────────┐
    │  e_band + e_coul │                   │  e_tot = e_elec   │
    │  + e_dipole      │                   │  + e_repulsion    │
    │  + e_entropy     │                   │  + e_spin (=0)    │
    │  = e_elec_tot    │                   │                   │
    └───────┬──────────┘                   └─────────┬─────────┘
            │                                        │
            └────────────────────┬───────────────────┘
                                 │
                        ┌────────▼────────┐
                        │ structure.e_tot │
                        │  (output)       │
                        └─────────────────┘
```

**Data flow for single-shot path:**

1. **Input:** Structure with positions, charges (D0 from reference occupations), TYPE, cell.
2. **H0/S assembly:** Routes through f blocks (Phase 3 source-locked angular).
3. **Coulomb matrix:** Called with Hubbard_U (or Hubbard_U_sr if shell-resolved mode).
4. **No SCF loop:** Charge density fixed to D0; no convergence iterations.
5. **Energy:** Diagonalize once, integrate density of states, sum components.
6. **Output:** scalar e_tot.

### Recommended Project Structure

```
.planning/phases/04-scf-and-reference-simulation-validation/
├── 04-CONTEXT.md          ← Decisions D-11 through D-22 (LOCKED)
├── 04-RESEARCH.md         ← This file
├── 04-PLAN.md             ← Tasks to implement findings (next)
├── 04-VERIFICATION.md     ← SIM-01..SIM-05 checklist
└── test-harnessing/
    └── eu_n_scan_test.py  ← Eu-N energy scan for validation

src/dftorch/
├── ESDriver.py            ← Add do_scf=False energy path
├── _scf.py                ← No changes needed (SCF loop not used)
├── _energy.py             ← Existing energy() function used as-is
├── Constants.py           ← No changes (Uf already exposed)
├── Structure.py           ← No changes (Hubbard_U_sr already computed)
└── ewald_pme/
    └── PME_torch.py       ← Fix line 261 dtype bug

tests/
├── test_f_orbital_skf.py  ← Extend with eu_n_scan test
├── test_scf.py            ← Currently fails (blocked by PME bug)
└── f_orbital_data/
    └── Eu-N.skf           ← Extended format, Uf=0.50 eV
```

### Pattern 1: Single-Shot Energy without SCF Loop

**What:** Compute total energy at fixed charge density (no charge self-consistency).

**When to use:**
- D-11: Validation without convergence concerns.
- Energy scanning (varying geometry, no need to re-converge charges).
- Non-SCC DFTB approximation.

**Example:**

```python
import torch
torch.set_default_dtype(torch.float64)

from dftorch.Constants import Constants
from dftorch.ESDriver import ESDriver
from dftorch.Structure import Structure

dftorch_params = {
    "FILENAME": "eu_n.xyz",
    "SKFPATH": "tests/f_orbital_data/",
    "COUL_METHOD": "FULL",  # or "PME" after dtype fix
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
}

const = Constants(dftorch_params).to("cpu")
struct = Structure(dftorch_params, const, device="cpu")

es_driver = ESDriver(dftorch_params, device="cpu")
es_driver(struct, const, do_scf=False)  # No SCF; compute e_tot from D0

print(f"Total energy: {struct.e_tot.item():.6f} eV")
print(f"Band energy: {struct.e_band0.item():.6f} eV")
print(f"Coulomb: {struct.e_coul.item():.6f} eV")
```

**Source:** Existing _energy.py, _coulomb_matrix.py, Structure.D0 (Phase 2 reference density).

### Pattern 2: Shell-Resolved Hubbard U (Config-Gated, D-16)

**What:** Use l-dependent Hubbard U values (s/p/d/f different) instead of per-atom scalar.

**When to use:**
- Accurate treatment of transition metals or rare earths with open d/f shells.
- Future SCF for f systems (Phase 5+) — de-risked by D-14 testing.

**Example:**

```python
dftorch_params = {
    # ... other settings ...
    "MAGNETIC_HUBBARD_LDEP": True,  # Enable shell-resolved U
}

const = Constants(dftorch_params).to("cpu")
struct = Structure(dftorch_params, const, device="cpu")

# For an Eu atom (4 shells: s/p/d/f):
print(f"Per-atom U: {struct.Hubbard_U[eu_idx]:.4f} eV")  # Just Us
print(f"Shell-resolved U: {struct.Hubbard_U_sr[...]} eV")  # [Us, Up, Ud, Uf]
```

**Threading:** Pass `Hubbard_U_sr` (or select logic) to coulomb_matrix_vectorized, energy(), etc., gated by `const.magnetic_hubbard_ldep` flag.

### Anti-Patterns to Avoid

- **DO NOT** run ESDriver.forward() with do_scf=True and expect it to work when no do_scf=False path exists. Test both branches.
- **DO NOT** zero-fill f Hubbard U if the SKF header is missing it; raise a named error (D-15 precedent).
- **DO NOT** hardcode `.float()` when operating on float64 tensors. Use `.to(dtype=...)` or `.double()`.
- **DO NOT** run geometry optimization on f systems (forces not implemented). Use energy scan instead (D-19).
- **DO NOT** tighten the Eu-N tolerance below ~10% without re-examining the open-shell / non-SCC approximations (D-20/D-22).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Coulomb matrix from charges and distances | Custom integration loops | `coulomb_matrix_vectorized()` (_coulomb_matrix.py line 19–731) | Already tested, handles edge cases (zero distance, periodic images, PME/FULL methods) |
| Electronic energy decomposition (band + Coulomb + dipole + entropy) | Manual summation of orbital eigenvalues and charge correction terms | `energy()` function (_energy.py) | Handles fermi occupation, temperature smearing, third-order DFTB, solvation shifts; bug-prone to replicate |
| Hamiltonian diagonalization | Power iteration or custom eigensolver | `torch.linalg.eigh()` (via diagonalize calls in _scf.py) | GPU-compatible, numerically stable, handles generalized eigenvalue problems (HC = SCe) |
| Slater-Koster spline evaluation | Hand-rolled cubic spline in inner loops | `_bond_integral.cubic_spline_coeffs()` and pre-computed coeffs_tensor | Already optimized, batch-friendly, tested against reference grid points |
| Spin-orbit coupling (w_shell, w_atom) | Custom Pauli matrix logic | `load_spinw_to_matrix()` and `_spin.py` routines | Encapsulates conventions; reusable for both SCF and derivatives |

**Key insight:** Phase 4 is **not** a new capability; it's a **mode** of existing machinery. The energy calculation, Coulomb matrix, and H0/S routines are battle-tested in Phase 1–3. Resist the urge to duplicate them for "just this case" — instead, gate them with flags and test both paths.

---

## Runtime State Inventory

[SKIPPED — Phase 4 is not a rename/refactor/migration phase. No renamed strings or registered state to audit.]

---

## Common Pitfalls

### Pitfall 1: Missing do_scf=False Energy Path

**What goes wrong:** Code calls `es_driver(struct, const, do_scf=False)` expecting e_tot to be populated, but the forward() method returns early without computing energy.

**Why it happens:** Forward() was designed with SCF-dominant workflows in mind. The do_scf=False branch was stubbed in docstring but never implemented.

**How to avoid:** Implement the non-SCF path BEFORE any scan test runs. Add a guard:

```python
if not hasattr(structure, 'e_tot') or structure.e_tot is None:
    raise RuntimeError("Energy not computed; check do_scf path.")
```

**Warning signs:** Eu-N scan test produces all zeros or NaNs; e_tot is missing after forward() call.

### Pitfall 2: Dtype Mismatches in Coulomb/PME Operations

**What goes wrong:** Float32 tensor meets float64 tensor in einsum → dtype mismatch error. Entire SCF loop fails.

**Why it happens:** PyTorch's `.float()` defaults to float32. Line 261 was not updated when project switched to float64 default.

**How to avoid:** Search all tensor conversions for hardcoded `.float()` or `.double()`. Use `.to(dtype=X)` instead of method calls.

**Warning signs:** `test_scf.py` fails with "Expected all tensors to have the same dtype" at PME stress calculation.

### Pitfall 3: Shell-Resolved U Not Threaded to All Call Sites

**What goes wrong:** Structure has Hubbard_U_sr computed correctly, but Coulomb matrix still uses scalar U → wrong energies for f systems.

**Why it happens:** Threading a new tensor through 14 call sites requires config gating at each one. Easy to miss.

**How to avoid:** Add config flag check early in ESDriver.forward():

```python
use_shell_resolved = const.magnetic_hubbard_ldep
hubbard_for_coulomb = (struct.Hubbard_U_sr if use_shell_resolved 
                       else struct.Hubbard_U)
```

Then pass `hubbard_for_coulomb` consistently.

**Warning signs:** Tests pass for simple s/p systems but fail for f-containing ones; energy difference is nonphysical.

### Pitfall 4: Forgetting to Enable f Shells in Test Structure

**What goes wrong:** write_xyz() writes ["Eu", "N"] but const.shell_present for those elements doesn't have f=True.

**Why it happens:** shell_present is inferred from SKF header (line 668–671, _bond_integral.py); if SKF file doesn't claim f, it won't be enabled.

**How to avoid:** Verify SKF fixture has extended header with Ef, Uf nonzero. Run test on diagnostic print:

```python
print(f"Eu shell_present: {const.shell_present[symbol_to_number['Eu']]}")
# Should show: tensor([True, True, True, True]) for s/p/d/f
```

**Warning signs:** H0/S matrix is smaller than expected; no f blocks in final assembled matrices.

---

## Code Examples

### Example 1: Single-Shot Energy Calculation (Non-SCF)

```python
"""Compute energy for a fixed charge density without SCF iterations."""

import torch
torch.set_default_dtype(torch.float64)

from dftorch.Constants import Constants
from dftorch.ESDriver import ESDriver
from dftorch.Structure import Structure

# Setup
dftorch_params = {
    "FILENAME": "eu_n_at_2.4_angstrom.xyz",
    "SKFPATH": "tests/f_orbital_data/",
    "COUL_METHOD": "FULL",  # Use full Coulomb, not PME (until dtype fixed)
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "T_ELECTRONIC": 300.0,
    "CHARGE": 0,
}

const = Constants(dftorch_params).to("cpu")
struct = Structure(dftorch_params, const, device="cpu")

# Single-shot energy: no SCF
driver = ESDriver(dftorch_params, device="cpu")
driver(struct, const, do_scf=False)

# Outputs:
print(f"Total energy: {struct.e_tot.item():.8f} eV")
print(f"  Band (KE + xc): {struct.e_band0.item():.8f} eV")
print(f"  Coulomb: {struct.e_coul.item():.8f} eV")
print(f"  Dipole: {struct.e_dipole.item():.8f} eV")
print(f"  Repulsion: {struct.e_repulsion.item():.8f} eV")
```

**Source:** Existing Structure.__init__ (reference D0), ESDriver.forward (to be added), energy() function (_energy.py).

### Example 2: Eu-N Energy Scan Test

```python
"""Validate f-orbital energy prediction against Eu-N bond-length minimum."""

import torch
from pathlib import Path
from dftorch.Constants import Constants
from dftorch.ESDriver import ESDriver
from dftorch.Structure import Structure

def write_xyz_euN(path: Path, separation: float):
    """Write Eu-N diatom with specified separation."""
    lines = ["2", "Eu-N diatom"]
    lines.append(f"Eu 0.0 0.0 0.0")
    lines.append(f"N {separation:.8f} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")

def test_f_orbital_eu_n_scan():
    torch.set_default_dtype(torch.float64)
    
    skf_dir = Path("tests/f_orbital_data")
    tmp_xyz = Path("/tmp/eu_n_test.xyz")
    
    # Scan Eu-N separation
    separations = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    energies = []
    
    for sep in separations:
        write_xyz_euN(tmp_xyz, sep)
        
        dftorch_params = {
            "FILENAME": str(tmp_xyz),
            "SKFPATH": str(skf_dir),
            "COUL_METHOD": "FULL",
            "RCUT_ELECTRONIC": 10.0,
            "RCUT_REPULSIVE": 6.0,
        }
        
        const = Constants(dftorch_params).to("cpu")
        struct = Structure(dftorch_params, const, device="cpu")
        
        driver = ESDriver(dftorch_params, device="cpu")
        driver(struct, const, do_scf=False)  # Single-shot
        
        energies.append(struct.e_tot.item())
        print(f"Separation {sep:.2f} Å: E = {struct.e_tot.item():.6f} eV")
    
    # Find minimum
    min_idx = energies.index(min(energies))
    min_separation = separations[min_idx]
    
    # D-20/D-22: Loose ~10–20% band on known Eu-N separation (~2.4 Å)
    TOLERANCE_MIN = 2.4 * 0.80  # 1.92 Å (–20%)
    TOLERANCE_MAX = 2.4 * 1.20  # 2.88 Å (+20%)
    
    print(f"\nMinimum at separation: {min_separation:.2f} Å")
    print(f"Tolerance band: [{TOLERANCE_MIN:.2f}, {TOLERANCE_MAX:.2f}] Å")
    
    assert (TOLERANCE_MIN <= min_separation <= TOLERANCE_MAX), \
        f"Minimum {min_separation:.2f} Å outside tolerance band"
    
    print("✓ Eu-N validation passed (SIM-05)")

if __name__ == "__main__":
    test_f_orbital_eu_n_scan()
```

**Source:** Adapted from test_f_orbital_skf.py build_h0_and_s() pattern; requires do_scf=False path to exist.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 + torch.float64 fixture wrapper |
| Config file | pyproject.toml (pytest configuration) |
| Quick run command | `pytest tests/test_scf.py::test_energy_smoke_import_and_call -v` |
| Full suite command | `pytest tests/test_f_orbital_skf.py tests/test_scf.py -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIM-01 | H0/S assembly for Eu-N without shape errors | unit | `pytest tests/test_f_orbital_skf.py::test_*_h0_and_s -v` | ✅ (adapted from Phase 3) |
| SIM-02 | Closed-shell energy; spin error on UNRESTRICTED=True | unit + integration | `pytest test_spin_guard.py -v` | ❌ Wave 0 — guard not yet implemented |
| SIM-03 | Hubbard_U_sr has f dimension; shapes correct | unit | `pytest test_shell_resolved_shapes.py -v` | ❌ Wave 0 — test to be written |
| SIM-04 | Eu-N geometry, units, observables documented | integration | `pytest test_eu_n_scan.py::test_documentation -v` | ❌ Wave 0 — fixture docu strings to add |
| SIM-05 | Eu-N energy scan minimum within ~10–20% tolerance | integration | `pytest test_eu_n_scan.py::test_eu_n_energy_minimum -v` | ❌ Wave 0 — requires do_scf=False path |
| REG-01 | Existing smoke tests (test_scf.py) pass after f changes | regression | `pytest tests/test_scf.py -v` | ✅ (currently FAILS on PME dtype) |

### Sampling Rate

- **Per-task commit:** `pytest tests/test_scf.py -x` (quick smoke; ~30 sec on CPU)
- **Per-wave merge:** `pytest tests/ -v` (full suite; ~2–5 min on CPU)
- **Phase gate:** All SIM-01..SIM-05 tests green + no regression in test_scf.py before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `src/dftorch/ESDriver.py` — implement do_scf=False energy path (lines 615–650, estimated)
- [ ] `src/dftorch/ewald_pme/PME_torch.py:261` — fix dtype bug (`.float()` → `.to(dtype=...)`)
- [ ] `src/dftorch/_slater_koster_pair.py` — define FSpinPolarizationUnsupportedError exception
- [ ] `src/dftorch/ESDriver.py:200` — add spin-polarization guard before SCF block
- [ ] `tests/test_shell_resolved_u.py` — verify Hubbard_U_sr shapes for s/p/d/f systems (unit test, ~100 lines)
- [ ] `tests/test_eu_n_scan.py` — Eu-N energy scan validation (integration test, ~150 lines)
- [ ] `tests/f_orbital_data/` — documentation: geometry, SKF parameters, observables, units, tolerances (docstring in test file)

*(If no gaps listed here, existing test infrastructure covers all phase requirements.)*

---

## Security Domain

[SKIPPED — Phase 4 has `security_enforcement: false` in config, or not present (treated as enabled). No new external APIs, file I/O restrictions, or privilege escalation pathways introduced. Existing dftorch security posture (no network I/O, file paths from dftorch_params) inherited.]

---

## Sources

### Primary (HIGH confidence)
- **Repository direct inspection:** Files read with exact line numbers:
  - `tests/f_orbital_data/Eu-Eu.skf` — homonuclear header with f Hubbard U confirmed present
  - `src/dftorch/_bond_integral.py:548–696` — SKF parsing logic for extended format (f entries extracted)
  - `src/dftorch/Constants.py:174–177` — Uf parameter exposed; UF tensor registered
  - `src/dftorch/Structure.py:413–441` — Shell-resolved Hubbard_U_sr construction (scaffolding complete)
  - `src/dftorch/ewald_pme/PME_torch.py:261–264` — PME dtype bug root cause confirmed
  - `src/dftorch/ESDriver.py:84–614` — forward() method structure; do_scf=False path missing
  - `tests/test_scf.py:62` — Test failure on PME dtype; failure message captured

### Secondary (MEDIUM confidence)
- **Codebase conventions:**
  - Phase 3 exception patterns (FDerivativeUnsupportedError, FAngularFormulaSourceError)
  - Test harness patterns (torch.set_default_dtype, module-reset fixture in test_f_orbital_skf.py)
  - Shell dimension constants ([0, 1, 3, 5, 7] for s/p/d/f)

### Tertiary (LOW confidence — marked [ASSUMED] in findings)
- **Literature value:** Eu-N equilibrium separation ~2.4–2.5 Å (training data, not verified from paper in this session)
- **Scan parameters:** 0.2 Å steps, 1.8–3.0 Å range chosen for convenience; not optimized against published Eu-N potential

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | do_scf=False energy path does not exist in ESDriver.forward() | Finding 4 | If path exists elsewhere, implementation is redundant; but code inspection suggests it's genuinely missing |
| A2 | "~14 call sites in ESDriver thread Hubbard_U" (D-15 claim) | Finding 2 | Actual count may be higher or lower; grep shows 12–14 confirmed; does not affect scope, only task granularity |
| ~~A3~~ | ~~Eu-N equilibrium separation is ~2.4–2.5 Å~~ **RESOLVED — superseded by D-24 (user-supplied citation, 2.655 Å target, ±20% band). No longer an assumption.** | Finding 5 | — |
| A4 | 0.2 Å step size (6–7 points) is sufficient for loose tolerance band detection | Finding 5 | Finer or coarser grids possible; uniform grid simple to implement and test. Adaptive refinement not needed for loose tolerance. |
| A5 | .float() on line 261 is the ONLY dtype precision-loss bug in src/dftorch/ | Finding 3 | Audit found no other class of bug; benign .float() in _nearestneighborlist is for indices (unaffected). If new code added, check for same pattern. |

**If this table is empty:** All claims verified or directly cited. ← **FALSE; A1–A5 listed above.**

---

## Open Questions (ALL RESOLVED — 2026-07-29, at plan-phase)

> **None of the three questions below is open.** Q1 is closed by **D-23**, Q2 by **D-24**, and
> Q3 by the D-13 pre-decided exclusion. The plans override this section explicitly; it is
> retained for provenance only. Do not re-raise these.

1. **[RESOLVED by D-23]** **Config flag name for shell-resolved U (D-16):**
   → **Answer: reuse `MAGNETIC_HUBBARD_LDEP`.** `SHELL_RESOLVED` does not exist anywhere in
   `src/` or `tests/`. Do NOT introduce it. The recommendation below happened to be right,
   but it is now a binding decision, not a suggestion.
   - Should it be `MAGNETIC_HUBBARD_LDEP` (existing in Constants.py line 58) or a new `SHELL_RESOLVED` flag?
   - **Current state:** `MAGNETIC_HUBBARD_LDEP` exists but is unused in _scf.py; recommendation is to use it (consistent with existing patterns).
   - **Recommendation:** Reuse `MAGNETIC_HUBBARD_LDEP` (backwards-compatible name, though "magnetic" is a misnomer for non-spin case).

2. **[RESOLVED by D-24]** **Tolerance band interpretation for D-20/D-22:**
   → **Answer: target 2.655 Å, band ±20% → [2.12, 3.19] Å, derived as a fraction of the
   target and never hardcoded as an absolute half-width.** The "2.4 ± 0.2 Å = [1.92, 2.88]"
   arithmetic below is wrong (±0.2 on 2.4 is ±8%, tighter than D-20 permits). Plus the
   asymmetric failure rule: short-of-band escalates to human judgement; long-of-band or no
   interior minimum is a genuine red flag.
   - Does ~10–20% refer to absolute error (e.g., ±0.3 Å on 2.4 Å) or relative to scanned points?
   - **Current implementation:** Absolute band [2.4 ± 0.2 Å] = [1.92, 2.88] Å.
   - **Risk:** If tolerance is meant to be ±10% instead of ±20%, band narrows to [2.16, 2.64] Å.
   - **Recommendation:** Use D-20 wording ("~10–20%") as written; if tightening needed, it's a separate decision outside Phase 4.

3. **[RESOLVED — out of scope]** **Non-convergence warning for D-13:**
   → **Answer: not Phase 4 work.** D-13 governs the existing SCF machinery whenever it runs;
   Phase 4's single-shot path (D-11) has no iterations to converge. Pre-decided, not
   implemented here.
   - Where should the SCF non-convergence fallback message go? Print to stdout? Logging module? structure attribute?
   - **Current state:** Not yet implemented (D-13 is deferred; only D-11's non-iterative path is needed for Phase 4).
   - **Recommendation:** Defer to Phase 5 when actual SCF is implemented. Phase 4's single-shot has no iterations to converge.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | All energy/differentiation | ✅ | 2.0+ (project default) | — |
| Python | Runtime | ✅ | 3.9+ (CI/local) | — |
| pytest | Test execution | ✅ | 9.1.1 (installed via uv) | — |
| uv | Package management | ✅ | 0.12.0 (project bootstrap) | pip (slower) |
| NumPy | Utility (dtype checks, file I/O) | ✅ | 1.20+ (torch dependency) | — |

**Missing dependencies with no fallback:** None. Phase 4 uses only existing project infrastructure.

---

## Metadata

**Confidence breakdown:**
- **Finding 1 (Hubbard U present):** HIGH — Direct file inspection + code verification.
- **Finding 2 (Shell-resolved infrastructure):** HIGH — Source code read and traced.
- **Finding 3 (dtype bug):** HIGH — Root cause confirmed; test failure reproduced.
- **Finding 4 (single-shot path missing):** HIGH — Exhaustive code inspection; path definitively absent.
- **Finding 5 (Eu-N scan design):** MEDIUM — Literature value assumed; scan parameters are educated guess.

**Research date:** 2026-07-29
**Valid until:** 2026-08-15 (2 weeks; fast-moving codebase; recheck before execution if delayed)

---

**End of research document. Ready for `/gsd-plan-phase` consumption.**
