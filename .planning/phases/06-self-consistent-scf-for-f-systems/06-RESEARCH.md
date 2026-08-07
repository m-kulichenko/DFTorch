# Phase 6: Self-Consistent SCF for f Systems - Research

**Researched:** 2026-08-04  
**Domain:** Self-consistent charge loop debugging and shell-resolved Coulomb implementation  
**Confidence:** HIGH (code-based findings) / MEDIUM (diagnosis assumptions)

> ## ⚠ SUPERSEDING ADDENDUM — 2026-08-04, post-research measurement
>
> **The SCF divergence has been root-caused and it is not what the body of this document
> assumes.** Sections below that describe the divergence as an open diagnosis task, or that
> treat the shell-resolved f Coulomb blocks as a candidate cause, are superseded by
> [§ Blocking Diagnosis](#blocking-diagnosis-2026-08-04) at the end of this file. Read that
> section first. In particular:
>
> * The divergence is the **low-rank Krylov charge mixer**, not the f orbitals, not the
>   charge description, and not the Coulomb assembly.
> * The "10.6 eV gap" this document flags for investigation is **explained** and is not a defect.
> * Neither the single-shot nor the self-consistent path dissociates to neutral atoms, for two
>   different and now-understood reasons.
> * Mulliken charge **is** conserved (worst case 1.3e-09); an earlier concern that it was not
>   is not borne out.

## Summary

Phase 6 must stabilize the self-consistent charge loop for f-orbital systems, which currently runs away at 9 of 21 separation points in the Eu-N benchmark, returning unphysical energies as high as +198 eV. The loop **already runs** for f systems at 2.655 Å but diverges elsewhere. Primary work is diagnosis and bug repair. Secondary work — implementing the seven missing f angular blocks for shell-resolved Coulomb interactions — is **conditional** on whether the coarse per-atom charge description is the root cause. If it is not, shell-resolved work becomes a separate phase. The loop must return a convergence flag (iteration count on success, -1 on failure) to make non-convergence machine-readable rather than just printing. All changes must preserve the Phase 4 single-shot reference energy (-17.510444238744924 eV) and the Eu-N binding curve shape.

**Primary recommendation:** Diagnose the SCF instability first (mixture of charge loop, mixing strategy, and Fermi occupation handling); implement the convergence flag return value; compute and visually inspect the self-consistent Eu-N binding curve; then decide whether shell-resolved charges are needed based on the diagnosis, not the roadmap.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-6.01:** Loop must converge in the **neighbourhood of 2.655 Å** — not necessarily all 21 points. Other separations may remain unstable; that is accepted if honestly reported.
- **D-6.02:** Final gate is **human visual inspection of the recomputed graph**, not a numeric threshold. Human reads the plot and declares whether it looks correct.
- **D-6.03:** The 10.6 eV single-shot / self-consistent energy difference **gets investigated and a written verdict is produced**: either "this is real physics, here is the mechanism" or "this is a defect, here is where it lives."
- **D-6.04:** SCF loop **returns** convergence result as iteration count (on success) or -1 (on failure), modelled on scipy's iterative solvers.
- **D-6.05:** A separation that fails to converge away from 2.655 Å does not fail the test suite — reported via D-6.04's -1, marked on the graph.
- **D-6.06:** Seven f Coulomb blocks and shell-resolved plumbing are **conditional on diagnosis**. If coarse per-atom description is the cause, they belong in Phase 6. Otherwise, Phase 6 does not touch them.
- **D-6.07:** Scan stays **1.60 to 3.60 Å, same 21 points** as Phase 4, for point-for-point comparability.
- **D-6.08:** **No reference numbers frozen** this phase. Neither -6.920 eV nor the converged charges — we do not yet know if there is a bug.
- **D-6.09:** Graph shows **both curves overlaid** (single-shot and self-consistent) with non-converged separations **visibly marked**.
- **D-6.10:** `Constants.py:232` takes per-atom Hubbard U from s shell for every element; for Eu this is 5.714 eV but 7 of 9 valence electrons are in f shell (13.606 eV). This is a **genuine separate defect, NOT the instability cause** (tested and disproven). Record it; do not present it as the fix.

### Claude's Discretion

- Whether all four SCF loops (`SCFx`, `scf_x_os`, `SCFx_batch`, `delta_scf_x_os`) get the returned convergence flag, or only the closed-shell single-system one. Consistency argues for all four; the positional tuple churn at every call site argues for restraint.
- Concrete shape of returned convergence result (tuple element, dataclass, mirrored attribute on structure) — subject to D-6.04 semantics.
- Whether diagnosis is one task or multiple, and staging relative to the fix.
- How the graph is produced and where written (docs/assets/ already exists).
- Whether 10.6 eV verdict lives in its own document or in the phase summary.
- Whether `Constants.py:232` is fixed, guarded, or left as documentation.

### Deferred Ideas (OUT OF SCOPE)

- Extending scan to ~6 Å (flagged in Phase 4 as worth doing once self-consistency lands).
- Requiring self-consistent minimum inside [2.124, 3.186] Å band.
- Fixing `Constants.py:232` Hubbard U selection (not a blocker for Phase 6).
- Spin-polarized / open-shell treatment of Eu 4f7 (Phase 4 D-12, still deferred).

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCC-01 | f systems run real self-consistent loop to convergence; non-convergence warns/returns iterate with flag, never raises | Convergence criterion already defined (`ResNorm <= SCF_TOL` AND `dEc <= SCF_TOL * 100`); flag return plumbing identified |
| SCC-02 | Seven f angular blocks of shell-resolved Coulomb (s-f, f-s, p-f, f-p, d-f, f-d, f-f) implemented; `FShellResolvedCoulombUnsupportedError` no longer fires | Conditional on D-6.06 diagnosis; blocks identified; existing 9-block pattern documented |
| SCC-03 | Shell-resolved f plumbing from Phase 4 D-14 actually consumed by SCF | Conditional on D-6.06 diagnosis; consumer machinery located in open-shell routine (not yet in closed-shell) |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SCF convergence detection | Backend (charge loop in _scf.py) | Caller API (ESDriver unpacks flag) | Loop produces convergence criterion; caller must make it visible outside print statements |
| Per-atom charge calculation | Backend (SCFx loop body) | — | Charge assembled from density matrix; already in place |
| Shell-resolved charge calculation (if D-6.06 fires) | Backend (new path inside SCFx) | — | Detailed charge needed only if coarse description is the instability root cause |
| Shell-resolved Coulomb matrix assembly (if D-6.06 fires) | Backend (_coulomb_matrix.ewald_real_space_vectorized_sr) | Structure metadata (shell ranges, indices) | Coulomb computed pair-wise; needs shell-wise indexing already present |
| SCF loop diagnostics / energy investigation | Developer task (analysis) | — | Diagnosis requires running the loop and comparing band energy, Coulomb energy, and charge direction |

## Standard Stack

### Core Libraries (Already in Use)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | (current, float64-capable) | Dense linear algebra, automatic differentiation | Project standard; SCF loop relies on tensor operations |
| (no new external deps) | — | — | Phase 6 modifies existing loops, does not add new ones |

### Internal Modules (Phase 6 Must Modify or Extend)

| Module | Location | Purpose | Modification Required |
|--------|----------|---------|----------------------|
| `_scf.py:SCFx` | src/dftorch/_scf.py:158 | Closed-shell single-system SCF loop | (1) Return convergence flag; (2) optionally add shell-resolved charge consumer if D-6.06 fires |
| `_scf.py:scf_x_os` | src/dftorch/_scf.py:557 | Open-shell SCF (refused for f, but may get flag for consistency) | Return convergence flag (if decision made to update all loops) |
| `ESDriver.forward` | src/dftorch/ESDriver.py:489 | Main driver; unpacks SCFx return tuple | Unpack new 15th tuple element (convergence flag) |
| `_coulomb_matrix.ewald_real_space_vectorized_sr` | src/dftorch/_coulomb_matrix.py:723 | Shell-resolved Coulomb assembly | Add seven f angular blocks (s-f, f-s, p-f, f-p, d-f, f-d, f-f) **only if D-6.06 fires** |
| `_require_no_f_shell_resolved_coulomb` | src/dftorch/_coulomb_matrix.py:690 | Guard that refuses f systems on shell-resolved path | Remove or narrow guard **only if D-6.06 fires** |
| `Structure` class | src/dftorch/Structure.py | Holds convergence flag alongside energy / charge results | Add `scf_converged` attribute (optional, depends on shape choice) |

### Supporting

| Module | Location | Purpose | When Used |
|--------|----------|---------|-----------|
| `_AndersonMixer` | src/dftorch/_scf.py:36 | Pre-Krylov charge mixer | Parameterised by `ANDERSON_DEPTH` and `ANDERSON_ALPHA` — tuning these may help diagnosis |
| `dm_fermi_x` | src/dftorch/_scf.py | Fermi occupation filling | Unchanged unless diagnosis points to Fermi level handling |
| Test harness `run_with_float64` | tests/test_single_shot_energy.py:29 | Ensures deterministic float64 default for f-orbital tests | Reuse for new SCF convergence tests |
| Test harness `_write_eu_n_xyz`, `_run_single_shot` | tests/test_single_shot_energy.py | Geometry setup and parameter standardisation for Eu-N | Reuse for SCF convergence test |

### Alternative Approaches Considered (Not Chosen)

| Instead of | Could Use | Tradeoff | Decision |
|-----------|-----------|----------|----------|
| Returning convergence flag in tuple | Mirroring flag on `structure` as separate attribute | Tuple approach matches scipy precedent; attribute approach avoids re-unpacking at every call site | D-6.04 locks tuple return; shape is discretionary |
| Conditional shell-resolved work | Building it regardless / keeping it out of Phase 6 | User chose conditional: diagnosis first, then decide. Avoids solving wrong problem and prevents changing machinery underneath bug. | D-6.06 locks this choice |

## The Seven Missing f Angular Blocks

### Existing Blocks (Implemented in _coulomb_matrix.py:723 onwards)

The shell-resolved Coulomb matrix construction currently builds the following (l_a, l_b) shell-pair blocks using a vectorized neighbor-list approach. Code uses 'H' = s only (max_ang 1), 'X' = s+p (max_ang 2), 'Y' = s+p+d (max_ang 3):

**Constructed blocks:**
- s-s (line 836-858): `pair_mask_HH`, U[I] * U[J] damping
- s-p (line 860-888): `pair_mask_HX`, U[I] * Up[J] damping
- p-s (line 890-918): `pair_mask_XH`, Up[I] * U[J] damping
- p-p (line 920-951): `pair_mask_XX`, Up[I] * Up[J] damping with same-element special case
- s-d (line 953-974): `pair_mask_HY`, U[I] * Ud[J] damping
- d-s (line 976-997): `pair_mask_YH`, Ud[I] * U[J] damping
- p-d (line 999-1020): `pair_mask_XY`, Up[I] * Ud[J] damping
- d-p (line 1022-1043): `pair_mask_YX`, Ud[I] * Up[J] damping
- d-d (line 1045-1076): `pair_mask_YY`, Ud[I] * Ud[J] damping with same-element special case

Each block follows the pattern:
```
1. Create pair mask for the (max_ang_I, max_ang_J) combination
2. Clone the bare Coulomb CA = erfc(alpha*r)/r
3. Apply short-range damping correction (coul_diff_elem_and_ang or coul_same_elem_and_ang)
4. Multiply by Hubbard U values for each shell
5. Accumulate into output matrix at shell-row/shell-column indices
```

**Missing blocks (for f, max_ang 4 = 'Z'):**
- s-f (H-Z): max_ang_I == 1, max_ang_J == 4 → pair_mask_HZ, U[I] * Uf[J] damping
- f-s (Z-H): max_ang_I == 4, max_ang_J == 1 → pair_mask_ZH, Uf[I] * U[J] damping
- p-f (X-Z): max_ang_I == 2, max_ang_J == 4 → pair_mask_XZ, Up[I] * Uf[J] damping
- f-p (Z-X): max_ang_I == 4, max_ang_J == 2 → pair_mask_ZX, Uf[I] * Up[J] damping
- d-f (Y-Z): max_ang_I == 3, max_ang_J == 4 → pair_mask_YZ, Ud[I] * Uf[J] damping
- f-d (Z-Y): max_ang_I == 4, max_ang_J == 3 → pair_mask_ZY, Uf[I] * Ud[J] damping
- f-f (Z-Z): max_ang_I == 4, max_ang_J == 4 → pair_mask_ZZ, Uf[I] * Uf[J] damping with same-element special case

Each f block must follow the existing block's exact structure: pair mask, CA clone, damping call, Hubbard U product, index scatter-add. No reference implementation exists for f damping — they must be **derived from first principles** (Ewald screening plus short-range tail) or **no test can validate them**. This is NOT a Slater-Koster angular-block question (Phase 3 source-locked those); this is a Coulomb-damping question specific to DFTB that has no published f formula in the literature. Phase 3's `f_orbital_SlaterKosterAngularTransformations.pdf` (Takegahara 1980) does not cover Coulomb damping.

### How f Blocks Differ from Existing Ones

The seven blocks cannot be generated by pattern-extension alone — they require new short-range damping formula. The existing `coul_diff_elem_and_ang` (line 1082) and `coul_same_elem_and_ang` (line 1109) parameterize damping by a single Hubbard U value per element and use empirical exponential screening. **The same formula applies to all blocks** (it is not l-dependent). So the f blocks mechanically reuse the damping calculation with Uf instead of U/Up/Ud:

```python
# s-f example (pseudo-code)
pair_mask_HZ = (max_ang_I == 1) * (max_ang_J == 4)
tmp1 = CA[pair_mask_HZ].clone()
dtmp1 = -(CA[pair_mask_HZ] + ...) / dR_mskd[pair_mask_HZ]
Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[pair_mask_HZ]]]
Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J[pair_mask_HZ]]]  # <-- Uf, not U/Up/Ud
dR_mskd_diff = dR_mskd[pair_mask_HZ]
t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)  # <-- same function
# ... scale and accumulate ...
```

**The validation challenge:** Without a published f damping reference, validation must be **charge self-consistency** (does adding f blocks make the loop converge?) and **physical reasonableness** (are converged charges in the ballpark?). There is no "golden numbers" test like Phase 3's Slater-Koster formula verification.

## Convergence Flag Plumbing

### Current Return Signature

`SCFx` at `_scf.py:158` returns 14-element tuple (line 554):

```python
return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul
```

Named as:
1. H — Hamiltonian including Coulomb/dipole
2. Hcoul — Coulomb contribution to H
3. Hdipole — External field dipole correction
4. KK — Mixing/precondition kernel
5. D — Density matrix
6. Q — ??? (Q-matrix, appears to be intermediate Dorth before transformation back to AO basis)
7. e — Eigenvalues (Fermi occupations)
8. q — Atomic charges (Nats,)
9. f — Eigenvalues of orthogonal density
10. mu0 — Fermi level / chemical potential
11. Ecoul — Coulomb energy (PME only, else None)
12. forces1 — Electrostatic forces (PME only, else None)
13. dq_p1 — Charge-response PME output (optional, else None)
14. stress_coul — Coulomb stress (optional, else None)

### Where the Tuple is Unpacked

**Single-system path (only place f systems use):**
- `ESDriver.py:663-678` — unpacks into `structure.H`, `structure.Hcoul`, etc. This is the **primary unpack site**. Adding element 15 here requires 14 new attribute assignments and changes the unpack order.

**Batch paths (refusal for f, but may need consistency update):**
- `_scf.py:977` `SCFx_batch` — has its own loop, also returns same 14-tuple
- `_scf.py:557` `scf_x_os` — open-shell path, also 14 elements
- `_scf.py:1233` `delta_scf_x_os` — delta SCF path, also 14 elements

**Print statements (non-convergence signals):**
- `_scf.py:521` — `print("Did not converge")` after SCFx loop exits via MaxIt
- `_scf.py:911-912`, `_scf.py:1216-1217`, `_scf.py:1551-1552` — same for other three loops

**Build artifact copies:**
- `build/lib/dftorch/ESDriver.py:663-678` — same unpack, must be updated (generated from src/, so rebuilding is automatic if src/ is edited)

### D-6.04 Semantics: Return Value Shape

**Requirement:** Iteration count when converged, -1 when not.

**Options for shape:**

| Option | Signature | Blast Radius | Precedent |
|--------|-----------|--------------|-----------|
| 15th tuple element | `return ..., scf_iter_count` | Moderate: one new unpack site per loop, four loops total; matches scipy style | `scipy.sparse.linalg.gmres` |
| New return type: `SCFResult` dataclass | `class SCFResult(NamedTuple): H; Hcoul; ...; scf_iter_count` | High: destructures all call sites; cleaner long-term | Breaks backward compat slightly, but improves clarity |
| Mirrored attribute on structure | `return same_14, structure.scf_iter_count = it` | Low: no new return element, but requires structure passed through; inconsistent with non-PME return | — |

**User preference:** D-6.04 explicitly names scipy's approach (iteration count / -1). Tuple element is the most conservative choice.

**Specific unpack site to edit:**

```python
# ESDriver.py:663-678 (BEFORE)
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
) = SCFx(...)

# ESDriver.py:663-679 (AFTER - add one element)
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

The converged check at `_scf.py:520-521` already has `it`, the iteration counter. At non-convergence (MaxIt exit), `it == SCF_MAX_ITER` but we want to return -1 to signal failure, so:

```python
# _scf.py:554 (AFTER)
converged_iter = it if (ResNorm <= dftorch_params.get("SCF_TOL", 1e-6) 
                        and dEc <= dftorch_params.get("SCF_TOL", 1e-6) * 100) else -1
return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul, converged_iter
```

## Shell-Resolved Charge Plumbing (Conditional on D-6.06)

### What Exists (Phase 4 D-14: Validated but Unconsumed)

**Structures with shell-resolved (per-shell) charge/Hubbard data:**
- `structure.Hubbard_U_sr` — shape (n_shells, ) from Phase 4 D-15, extracted from extended SKF headers
- `structure.q_spin_sr` — exists **ONLY in open-shell routine** `scf_x_os` (line 715), shape (2, n_shells)
- `structure.dCC_sr` — shell-resolved Coulomb derivative (set to None at ESDriver.py:323 for non-shell-resolved path)
- `structure.C_sr` — shell-resolved Coulomb matrix (set to None at ESDriver.py:322 for non-shell-resolved path)

**Phase 4 D-14 says these were validated against f dimensions but never fed into the loop.**

### What Does NOT Exist (What D-6.06 Would Require)

If the coarse per-atom charge description is the root cause of the instability, Phase 6 must build a **shell-resolved charge path inside the closed-shell loop**, because:

**Measurement 7 (from CONTEXT.md):** Shell-resolved charge calculation (`q_spin_sr` and `net_spin_sr`) lives **ONLY** in `scf_x_os` (open-shell), at lines 715-739. It computes:

```python
q_spin_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
q_spin_sr.scatter_add_(1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS)
```

where `DS` is the diagonal of the shell-resolved density matrix. **Phase 4 D-12 refuses any open-shell f system**, so this charge path is unreachable by f systems today.

**Building it inside `SCFx` (closed-shell) would require:**
1. Computing shell-resolved density matrix from `Dorth` (already exists per AO, must aggregate to per-shell)
2. Extracting diagonal per shell
3. Accumulating scattered charges per shell-to-atom mapping
4. Using per-shell charges to build shell-resolved Coulomb matrix (ewald_real_space_vectorized_sr)
5. Folding per-shell Coulomb back into per-atom Hamiltonian (or using it directly if Coulomb is also computed per-shell)

**This is not "add seven missing blocks." It is "build a new charge path,"** which is the cost estimate behind D-6.06.

### How to Determine if D-6.06's Condition Fires

1. Run diagnostic on charge loop: inspect `q` evolution per iteration, compare against known-good cases
2. Study whether charge oscillates / diverges in a way that fine-graining should fix
3. Test: artificially clamp per-shell charges to see if loop stabilizes (experiment, not committed)
4. Compare coarse and fine-grained damping: does s-s damping alone cause the problem?

If diagnosis points to "coarse Hubbard U missing the f shell's large penalty," then D-6.06's condition fires.

## The 10.6 eV Single-Shot / Self-Consistent Energy Gap

### What Must Be Investigated

Measured at Eu-N 2.655 Å (CONTEXT.md, measurement 2):

| Quantity | Single-shot | Self-consistent | Difference |
|----------|-------------|-----------------|-----------|
| E_tot | -17.510444 eV | -6.920251 eV | **+10.590 eV** |
| E_band0 | -16.948082 eV | -8.030036 eV | -8.918 eV |
| E_coul | 0 (by D-11) | +1.698983 eV | +1.699 eV |
| q_Eu, q_N | ~-2.7 (diag only) | -0.600, +0.600 | — |

A chemical bond is 2-5 eV. This gap is **two to five times larger** than a bond energy, not a self-consistency correction.

### Why It Matters

Phase 4 D-11 says single-shot energy uses only band energy (no Coulomb), so the two calculations start from different Hamiltonians. But the band energy difference alone (-16.95 → -8.03, about -8.9 eV) does not account for the full +10.6 eV gap. The additional +1.7 eV from the new Coulomb term should not flip the total by 10.6 eV.

**This points to either:**
1. A real physics difference: self-consistent charge relaxation at 2.655 Å genuinely flips the bonding picture
2. A defect in how Coulomb energy is computed
3. A defect in how band energy is computed with the self-consistent charges
4. A defect in how Fermi level is determined (shifts all orbital occupations)

### What Must Be Delivered

**Not:** "This is real" or "This is a bug" as a guess.

**Delivered:** Traced investigation showing:
- Which energy term changed most (band vs Coulomb vs Fermi shift)
- Whether the charge direction is correct (Eu is cation in both — ✓ from measurement 3)
- Whether the number makes sense (comparison against known Eu-N reference, if any exists)
- Explicit verdict: "This is real physics because X, or this is a defect at Y in the code"

**Severity for Phase 6:** If it is a defect and fixing it lands inside D-6.01's "converge at 2.655 Å" bar, it gets fixed. If fixing it moves the minimum away from 2.655 Å or unlocks work beyond that bar, it gets documented as a known issue.

## Regression Surface

### Tests That Currently Pin Unsupported Behavior (Will Flip if D-6.06 Fires)

These tests **expect** `FShellResolvedCoulombUnsupportedError` to be raised. If shell-resolved f blocks are implemented, these **must flip to expecting green**, not raising:

**test_shell_resolved_u.py:**
- `test_coulomb_matrix_sr_raises_for_f_systems_when_flag_unset` (line ~502) — expects error when MAGNETIC_HUBBARD_LDEP is False
- `test_coulomb_matrix_sr_raises_clear_message_for_f_systems` (line ~519) — expects error, checks message clarity
- `test_driver_refuses_f_system_when_flag_set` (line 613) — expects error when driver calls shell-resolved Coulomb with MAGNETIC_HUBBARD_LDEP = True

**test_orbital_count_guards.py:**
- `_case_shell_resolved_coulomb` (line 315) — registers FShellResolvedCoulombUnsupportedError as one of the four expected exceptions; will fail if exception class is removed without audit sync

### Tests That Must Stay Unbroken (No Matter What Phase 6 Does)

**Single-shot reference (Phase 4 D-11 / SIM-05):**
- `tests/test_single_shot_energy.py::test_eu_n_single_shot_has_no_coulomb_term` — reference energy at Eu-N 2.655 Å **must stay** -17.510444238744924 eV
- `tests/test_eu_n_scan.py::test_eu_n_binding_curve_minimum_is_in_band` — single-shot 21-point scan minimum must stay inside [2.124, 3.186] Å band (D-24)

**f-free regression (Phase 5 REG-01, REG-02):**
- `tests/test_scf.py::test_scf_ch4_convergence` — CH4 on mio-1-1 must converge in **6 iterations** (pinned in Phase 5, currently green)
- `tests/test_single_shot_energy.py::_run_single_shot` harness is reused by new SCF tests — must not modify existing test code except to add new tests

**Exception taxonomy (Phase 5 D-04 / CLN-03):**
- `tests/test_orbital_count_guards.py::test_defined_exceptions_match_inventory` — exactly four f exception classes, named [FAngularFormulaSourceError, FDerivativeUnsupportedError, FShellResolvedCoulombUnsupportedError, FSpinPolarizationUnsupportedError]. Removing one fails the test; if Phase 6 removes FShellResolvedCoulombUnsupportedError (D-6.06 fires), must update `docs/ORBITAL-COUNT-INVENTORY.md` **and** `tests/test_support_documentation.py` simultaneously.

**Spin guard (Phase 4 D-12):**
- `tests/test_spin_guard.py` — spin-polarized f systems must still refuse with FSpinPolarizationUnsupportedError. Unchanged.

## Validation Architecture

> **Trigger:** Workflow.nyquist_validation is enabled (default). This section gates VALIDATION.md creation.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 (suite baseline: 216 passed / 0 failed, 18 files) |
| Config file | none (no pytest.ini or setup.cfg in tests/; defaults apply) |
| Quick run command | `uv run pytest tests/test_scf.py -x` (CH4 f-free convergence, ~10s) |
| Full suite command | `uv run pytest` (all 18 test files, ~45s including 14.5s Eu-N 21-point scan) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCC-01 | f-containing Eu-N converges real loop at 2.655 Å; non-convergence returns -1 flag and last iterate | smoke/integration | `uv run pytest tests/test_scf_convergence_f.py::test_eu_n_scf_converges_at_target_separation -x` | ❌ Wave 0 |
| SCC-01 | Convergence flag -1 is returned at separations that do not converge | smoke/integration | `uv run pytest tests/test_scf_convergence_f.py::test_non_converged_separation_returns_minus_one -x` | ❌ Wave 0 |
| SCC-02 | Seven f Coulomb blocks are populated (s-f, f-s, p-f, f-p, d-f, f-d, f-f) — **only if D-6.06 fires** | unit | `uv run pytest tests/test_shell_resolved_u.py::test_coulomb_matrix_sr_f_blocks_populated -x` | ❌ Wave 0 (conditional) |
| SCC-02 | FShellResolvedCoulombUnsupportedError no longer fires for supported f systems — **only if D-6.06 fires** | integration | `uv run pytest tests/test_shell_resolved_u.py::test_driver_accepts_f_system_when_sr_implemented -x` | ❌ Wave 0 (conditional) |
| SCC-03 | Shell-resolved charges flow into SCF Hamiltonian — **only if D-6.06 fires** | integration | `uv run pytest tests/test_scf_convergence_f.py::test_shell_resolved_charge_consumed -x` | ❌ Wave 0 (conditional) |
| SCC-01 (10.6 eV) | Written verdict on 10.6 eV gap delivered (separate document, not test) | manual / doc | (human review of PHASE-VERDICT.md or summary section) | ❌ Wave 0 |
| SCC-01 (graph) | Two-curve overlay graph (single-shot + self-consistent) with non-converged points marked | manual / visual | (human inspection of plots/eu_n_self_consistent_binding_curve.png) | ❌ Wave 0 |

### Sampling Rate

- **Per-task commit:** `uv run pytest tests/test_scf_convergence_f.py -x` (focused f convergence, ~5s)
- **Per-wave merge:** `uv run pytest` (full suite including 21-point scan and all regression tests)
- **Phase gate:** Full suite green before `/gsd-verify-work`; human visual inspection of Eu-N graph before `/gsd-plan-phase 07`

### Wave 0 Gaps

- [ ] `tests/test_scf_convergence_f.py` — new test module covering SCC-01 (convergence flag, target separation convergence, non-convergence honesty)
- [ ] `tests/test_shell_resolved_coulomb_f.py` — new test module covering SCC-02/SCC-03 (seven f blocks, grid structure, Coulomb energy contribution) — **only if D-6.06 fires**
- [ ] `src/dftorch/Structure.py` — add `scf_iter_count` attribute or equivalent (depends on D-6.04 shape choice)
- [ ] `docs/ORBITAL-COUNT-INVENTORY.md` — update if FShellResolvedCoulombUnsupportedError is removed (D-6.06 fires)
- [ ] `tests/test_support_documentation.py` — sync exception list if taxonomy changes
- [ ] Integration with matplotlib or plotting tool for graph generation (not a test, but delivery requirement per D-6.02/D-6.09)

*Existing test infrastructure covers f-free CH4 and all Phase 4 single-shot gates; Phase 6 adds only f-specific SCF convergence and conditional shell-resolved coverage.*

## Validation: What Passing Tests Prove and Don't Prove

### What a Green SCC-01 Test Proves

- At 2.655 Å, the closed-shell SCF loop on Eu-N reaches `ResNorm <= 1e-6` and `dEc <= 1e-4` (default convergence criteria)
- The loop converges in a finite number of iterations, not by exhausting MaxIt
- The convergence flag is machine-readable (-1 vs iteration count)
- The loop does not raise an exception at this separation
- Per-atom charges settle to a stable value (Eu→-0.6, N→+0.6, matching measurement 3)

### What It Does NOT Prove

- The loop converges at any other separation (only 2.655 Å is required by D-6.01)
- The converged energy is physically correct (we do not freeze reference numbers per D-6.08)
- Shell-resolved charges are correct (no reference implementation exists)
- The 10.6 eV gap is understood or fixed
- The self-consistent minimum lands inside the band (explicitly out of scope per D-6.01)
- Eu 4f7 is being treated correctly (spin is deferred per D-12)

### How to Assert Genuine Convergence (Not MaxIt Exhaustion)

```python
# In test_scf_convergence_f.py
def test_eu_n_scf_converges_at_target_separation():
    """Verify loop reaches tolerance before MaxIt."""
    params = EU_N_PARAMS.copy()
    params["SCF_MAX_ITER"] = 100  # standard
    driver = ESDriver(...)
    structure = driver.forward(do_scf=True)
    
    # MUST pass, not just "no exception raised":
    assert structure.scf_iter_count > 0, "Converged: has iteration count"
    assert structure.scf_iter_count < 100, "Did not exhaust MaxIt"
    assert structure.scf_iter_count <= 20, "Converged reasonably fast (sanity for 2.655 A)"
    # The last assertion is optional but helps catch a loop that converges only at MaxIt-1.
```

### How to Validate Seven f Blocks (If D-6.06 Fires)

Without a published f damping reference, validation is **constructive:** run the SCF loop and check charge stability.

```python
# In test_shell_resolved_coulomb_f.py (if implemented)
def test_f_coulomb_blocks_enable_convergence():
    """Proof: with blocks, loop converges; without, it does not (or slower)."""
    params = EU_N_PARAMS.copy()
    params["MAGNETIC_HUBBARD_LDEP"] = True
    driver = ESDriver(params, ...)
    
    # Must NOT raise FShellResolvedCoulombUnsupportedError:
    structure = driver.forward(do_scf=True)
    assert structure.scf_iter_count > 0
    
def test_f_coulomb_blocks_have_correct_shape():
    """Proof: blocks are correctly indexed and scaled."""
    # Build shell-resolved Coulomb matrix directly
    C_sr = ewald_real_space_vectorized_sr(
        structure, dR, dR_dxyz, TYPE, nnType, neighbor_I, neighbor_J, CALPHA
    )
    n_shells = len(structure.H_INDEX_START_U)
    assert C_sr.shape == (n_shells, n_shells), "Matrix is (n_shells, n_shells)"
    assert torch.isfinite(C_sr).all(), "All entries finite"
    # Check Cauchy-Schwarz: C_{ij}^2 <= C_{ii} * C_{jj}
    for i in range(n_shells):
        for j in range(n_shells):
            assert C_sr[i,j]**2 <= C_sr[i,i] * C_sr[j,j] + 1e-10, f"Block ({i},{j}) violates C-S"
```

## Common Pitfalls

### Pitfall 1: Adding Shell-Resolved Blocks While Root Cause Unknown

**What goes wrong:** Building seven new blocks under debugging pressure makes the loop's behavior more complex, layering new code paths on top of an unsolved bug. A loop that crashed before now hangs or oscillates differently, but the diagnosis is now clouded by "did the blocks help, or did I inadvertently fix something else?"

**Why it happens:** The roadmap promises SCC-02 and SCC-03 in Phase 6, and the conditional is buried in CONTEXT.md, so the temptation is to "just build them" and see if it fixes things.

**How to avoid:** Follow D-6.06 literally. Diagnosis **precedes** implementation. If the diagnosis says "coarse charges are the root cause," build blocks. If it says "mixing strategy instability" or "Fermi handling bug," fix that instead and do not touch blocks.

**Warning signs:** A commit message that says "add f Coulomb blocks" without a preceding commit that says "diagnosis: charge description is the root cause." If the first commit appears before the second, the phase is off-track.

### Pitfall 2: Freezing the -6.92 eV Number or 0.6 e Transfer

**What goes wrong:** A test asserts `E_scf == -6.920251 eV` at 2.655 Å, then a later fix to the instability changes it to -6.821 eV. The test fails, and now the phase is blocked on "why did the number drift?" instead of "does the loop converge?"

**Why it happens:** D-6.08 explicitly forbids freezing numbers, but it is easy to miss and add a quick sanity check that becomes a regression gate.

**How to avoid:** In test code, **assert convergence**, not energy. Use `assert structure.scf_iter_count > 0, "converged"` not `assert abs(E_tot - (-6.920251)) < 1e-4, "energy matches"`. The graph is the energy gate (D-6.02).

**Warning signs:** A test file with a `EU_N_SCF_REFERENCE_E_TOT =` constant. Do not write this. Tests for SCC-01 assert convergence; tests for SCC-02/SCC-03 assert the new blocks are populated; only the graph asserts the energy makes sense.

### Pitfall 3: Misinterpreting Measurement 5 (Hubbard U from s Shell)

**What goes wrong:** The phase opens with a fix to `Constants.py:232`, replacing the s-shell U with Uf for Eu. Then the loop still runs away at 9 separations. The phase stalls because "the obvious fix didn't work."

**Why it happens:** D-6.10 records that swapping U did change the failure pattern but did not restore convergence. This is easy to forget, and measurement 5 points a clear arrow at a bug.

**How to avoid:** Read measurement 5 in full. It explicitly says the substitution was tested and did not fix the instability (though it revealed a separate defect). Do not re-test it.

**Warning signs:** A commit that says "fix Hubbard U for f elements" as the opening move. Check whether diagnosis says this is the root cause. If not, note it as D-6.10 and move on.

### Pitfall 4: Not Marking Non-Converged Separations on the Graph

**What goes wrong:** The graph shows a beautiful two-curve overlay with no indication which points are garbage. A human looks at it and says "why does the self-consistent curve flatten after 3.2 Å?" not realizing the loop gave up there.

**Why it happens:** D-6.09 asks for visible marking, but "visible" is ambiguous (color? markers? vertical lines?). If the implementation skips this, the graph is misleading.

**How to avoid:** Before writing graph code, define exactly how to mark failure: e.g., red X on failed points, dashed line for non-converged region, or legend entry "converged: 12/21 separations." Get user feedback on the mockup before committing to the plot.

**Warning signs:** A graph generated without a legend or note explaining which separations failed.

## Code Examples

### Setting Up the Eu-N Convergence Test

```python
# tests/test_scf_convergence_f.py (skeleton)
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from pathlib import Path
import pytest

def run_with_float64(fn):
    """Reused from test_single_shot_energy.py; do not re-invent."""
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        # Reload dftorch modules...

def _write_eu_n_xyz(path: Path, separation: float = 2.655) -> None:
    """Reused from test_single_shot_energy.py."""
    lines = ["2", "Eu-N diatomic", f"Eu 0.0 0.0 0.0", f"N {separation} 0.0 0.0"]
    path.write_text("\n".join(lines) + "\n")

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}

def test_eu_n_scf_converges_at_target_separation(tmp_path):
    """Eu-N at 2.655 A reaches convergence with do_scf=True (D-6.01, SCC-01)."""
    def check():
        geo_file = tmp_path / "eu_n.xyz"
        _write_eu_n_xyz(geo_file, 2.655)
        
        from dftorch import Constants, Structure, ESDriver
        skf_dir = Path(__file__).parent / "f_orbital_data"
        
        const = Constants(str(skf_dir), ["Eu", "N"])
        structure = Structure(str(geo_file), const)
        driver = ESDriver(EU_N_PARAMS)
        
        # do_scf=True activates closed-shell SCF loop
        result = driver.forward(structure, do_scf=True)
        
        # ASSERTIONS (per D-6.01 and D-6.04)
        assert hasattr(result, 'scf_iter_count'), "convergence flag returned"
        assert result.scf_iter_count > 0, "converged (not -1)"
        assert result.scf_iter_count < 100, "did not exhaust MaxIt"
        assert result.scf_iter_count <= 20, "converged reasonably fast"
        
        # Charge sanity (measurement 3: correct direction)
        q_Eu = result.q[0].item()
        q_N = result.q[1].item()
        assert -1.0 < q_Eu < 0.0, f"Eu is cation (q={q_Eu})"
        assert 0.0 < q_N < 1.0, f"N is anion (q={q_N})"
    
    run_with_float64(check)
```

### Convergence Flag Return Pattern

```python
# _scf.py:540-555 (pseudocode for return modification)

# Loop exit criterion already in place (line 392-395)
while (
    (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
    or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
) and it < dftorch_params.get("SCF_MAX_ITER", 100):
    # ... loop body ...
    it += 1

# At exit, it is either:
# - the iteration count where convergence was reached (ResNorm and dEc both satisfied), or
# - SCF_MAX_ITER (loop exhausted, convergence failed)

# Compute convergence flag
converged_iter = it if (
    ResNorm <= dftorch_params.get("SCF_TOL", 1e-6)
    and dEc <= dftorch_params.get("SCF_TOL", 1e-6) * 100
) else -1

# Return tuple with new 15th element
return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul, converged_iter
```

### Detecting Which Separations Fail (For Graph Marking)

```python
# Pseudo-code for full 21-point scan with convergence tracking
separations = torch.linspace(1.60, 3.60, 21)
energies_singleshot = []
energies_scf = []
convergence_flags = []

for sep in separations:
    # Single-shot (always converges, no loop)
    E_ss = driver.forward(structure_at(sep), do_scf=False).e_tot
    energies_singleshot.append(E_ss)
    
    # Self-consistent (may fail)
    result = driver.forward(structure_at(sep), do_scf=True)
    if result.scf_iter_count > 0:
        energies_scf.append(result.e_tot)
        convergence_flags.append(True)  # converged
    else:
        energies_scf.append(result.e_tot)  # last iterate, unphysical
        convergence_flags.append(False)  # did not converge

# Graph: plot both curves, mark False points with X or different color
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(separations, energies_singleshot, 'b-', label='single-shot (pinned)')
ax.plot(separations, energies_scf, 'r-', label='self-consistent')
# Mark non-converged points
failed_seps = separations[~torch.tensor(convergence_flags)]
failed_energies = [E for E, flag in zip(energies_scf, convergence_flags) if not flag]
ax.scatter(failed_seps, failed_energies, marker='x', color='red', s=100, label='did not converge')
ax.legend()
ax.set_xlabel('Separation (Angstrom)')
ax.set_ylabel('Energy (eV)')
plt.savefig('docs/assets/eu_n_scf_binding_curve.png')
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SCF convergence reported via print() | Return iteration count (-1 on failure) | Phase 6 (D-6.04) | Non-convergence now machine-readable; callers can make decisions without parsing stdout |
| Single-shot reference energy approved by design | Single-shot pinned; self-consistent approved by human visual inspection | Phase 4 → Phase 6 | Self-consistent curve is validated by eye, not by frozen numbers; improves resilience to fix-induced energy shifts |
| Shell-resolved Coulomb refused for f; guard raised FShellResolvedCoulombUnsupportedError | Conditional: build blocks only if coarse charges are root cause | Phase 6 (D-6.06) | Avoids solving wrong problem; prevents machinery rewrites during debugging |
| Eu's per-atom Hubbard U taken from s-shell regardless of electron distribution | Documented as defect (D-6.10); fix is conditional | Phase 6 (deferred) | Bug is visible; fix blocked only on cost/impact, not on forgetting it exists |

### Deprecated/Outdated

- Phase 4 roadmap framing: "Phase 6 must still switch on self-consistency" — False; it already works at one separation (measurement 1). Phase 6's actual job is to fix the instability across the range.
- Roadmap SCC-02 and SCC-03 as unconditional Phase 6 deliverables — Superseded by D-6.06's conditional. Roadmap is accurate but two of three requirements are now conditional. Resolved when diagnosis lands.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CH4 on mio-1-1 converges in 6 iterations post-Phase-5 (commit 4dbffaa) | Summary; Pitfalls | If CH4 no longer converges, regression in base f-free case; blocks whole phase |
| A2 | ESDriver.py:663-678 is the only unpack site for SCFx in production code | Convergence Flag Plumbing | If other caller sites exist, tuple change breaks them silently without test coverage |
| A3 | build/lib/dftorch/ESDriver.py auto-regenerates from src/dftorch/ESDriver.py | Convergence Flag Plumbing | If build is stale, deployment carries two different unpack signatures; confusing failure modes |
| A4 | Constants.Uf exists and is populated from extended SKF headers | Shell-Resolved Charge Plumbing | If Uf is not available on the Eu element, f damping formula cannot be applied; phase stalls on f block implementation |
| A5 | The seven f Coulomb blocks can be validated by closed-loop SCF convergence without published reference | Validation Architecture | If true validation requires a reference implementation (none exists), the blocks cannot be verified before Phase 6 closes; gates on Phase 7 or later |

**If any assumption A1-A5 is overturned during execution, research assumptions must be updated before continuing.**

## Open Questions

1. **Root cause of SCF instability at large separations (> 2.80 Å)**
   - What we know: at 3.10+ Å, Eu absorbs all five of N's valence electrons (q_Eu → +5.0). Loop does not refuse; it converges to this unphysical state.
   - What's unclear: Is this a charge-damping issue (coarse U too small for f), a mixing strategy (Anderson mixer depth/alpha), a Fermi-level/filling algorithm, or something else?
   - Recommendation: Trace the per-iteration charge evolution at a failing separation (e.g., 3.10 Å). If q_Eu smoothly climbs to +5 over iterations, it is convergence (loop thinks it is correct). Inspect why. If q_Eu oscillates wildly, loop is unstable (different diagnosis).

2. **Whether the 10.6 eV gap is real physics or a bug**
   - What we know: Single-shot uses only band energy (no Coulomb); self-consistent includes Coulomb. Band energy differs by -8.9 eV; Coulomb adds +1.7 eV. Total swing is +10.6 eV.
   - What's unclear: Is the self-consistent charge (-0.6, +0.6) correct for the self-consistent Hamiltonian? (Compare against another source if one exists.) Does the Fermi level swing by the full energy difference, suggesting an artifact?
   - Recommendation: At 2.655 Å, compare (1) band energy at self-consistent charge with (2) band energy at reference charge. If (1) is -8.0 eV and (2) is -16.9 eV, that is the physics. If not, the difference is an artifact.

3. **Whether all four SCF loops should return convergence flags for consistency, or only SCFx**
   - What we know: SCFx is used by f systems (only closed-shell path). Others are used by open-shell (f-refused) and batch (f-refused). User declined this decision as too small.
   - What's unclear: Will a later phase break because the batch or open-shell loops did not get the flag?
   - Recommendation: Plan for "all four" and implement that way unless a later phase explicitly says "batch convergence is always successful" or similar. One extra tuple element across four loops is cheap; undoing it is not.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | Core SCF loop, tensor ops | ✓ | (float64-capable) | — |
| uv | Test runner | ✓ | 0.12.0 | `python -m pytest` (slower, less deterministic) |
| pytest | Test execution | ✓ | 9.1.1 | — |
| matplotlib | Graph generation (D-6.09) | ✓ (or install `pip install matplotlib`) | — | Fallback: save data as CSV, plot manually or with inline tool |
| CUDA (optional) | GPU acceleration | ✗ | — | CPU is fine; Warp prints warning, harmless. Spinw.txt warning expected. |

**Missing dependencies with no fallback:** None identified; Phase 6 is code-only except graph rendering.

**Missing dependencies with fallback:** matplotlib may need explicit install; graph can be plotted external to the phase if needed.

## Sources

### Primary (HIGH Confidence)

- **Code audit, _scf.py:** Convergence criterion at lines 393-395 is hardwired `(ResNorm > SCF_TOL) or (dEc > SCF_TOL * 100)` with defaults SCF_TOL=1e-6, SCF_MAX_ITER=100. Verified live code 2026-08-04.
- **Code audit, _coulomb_matrix.py:** Nine existing shell-pair blocks (lines 836-1076) documented; seven missing blocks for f identified by enumeration of (max_ang_I, max_ang_J) pairs.
- **Code audit, _scf.py:715-739:** Shell-resolved charge calculation lives **only** in open-shell routine `scf_x_os`, not in closed-shell `SCFx`. Verified live code.
- **Code audit, ESDriver.py:663-678:** Single unpack site for SCFx; 14-element tuple to 14 attributes. Verified live code.
- **Phase 4 decision D-11 (stored at `.planning/phases/04-scf-and-reference-simulation-validation/04-CONTEXT.md`):** Single-shot energy uses only band + repulsion; e_coul exactly 0. Verified in source.
- **Phase 5 commit 4dbffaa:** CH4 divergence root cause (phantom p shell on H) fixed; CH4 now converges in 6 iterations. Verified in git history.
- **Measurements 1-7 (CONTEXT.md `<measurements>`):** Eu-N scan results, charge direction, instability pattern, Hubbard U substitution test. All reproducible with code + provided geometry/parameters.

### Secondary (MEDIUM Confidence)

- **CONTEXT.md D-6.03:** Investigation of 10.6 eV gap is a locked decision. Cross-referenced with DISCUSSION-LOG.md where user asked for it explicitly.
- **CONTEXT.md D-6.06:** Shell-resolved work conditional on diagnosis. Cross-referenced with user's rejection of unconditional build.
- **ROADMAP.md Phase 6:** SCC-02, SCC-03 listed as requirements. Tension with D-6.06 recorded in CONTEXT.md `<open_tensions>` 1. Not contradicted; tension is noted for planner.

### Tertiary (LOW Confidence)

- **Takegahara et al. 1980 (Phase 3 SOURCE-LOCK.md):** f angular transforms source. Does not cover Coulomb damping; f damping formula must be derived separately.
- **Training knowledge on Ewald summation:** Assumed that short-range damping follows the pattern `exp(-l*r)` with l depending on Hubbard U. Not verified for f in this codebase; treated as `[ASSUMED]` for f blocks.

## Metadata

**Confidence breakdown:**

| Area | Level | Reason |
|------|-------|--------|
| Convergence flag plumbing | HIGH | Code trace confirms 14-element tuple, one unpack site, return mechanics |
| Seven f Coulomb blocks (existence, indexing) | HIGH | Enumeration of 3×3 max_ang grid; current 9 blocks plus missing 7 = 16 total (confirmed) |
| Shell-resolved charge location | HIGH | Code search confirms `q_spin_sr` only in open-shell; closed-shell has none |
| Root cause of SCF instability | MEDIUM | Diagnosis not yet performed; only symptoms measured (charge divergence, energy runaway) |
| 10.6 eV gap explanation | MEDIUM | Multiple causes possible; investigation deferred to Phase 6 |
| f damping formula correctness | LOW | No published formula for f in DFTB; formula must be derived or validated empirically |

**Research date:** 2026-08-04  
**Context gathered:** 2026-08-04  
**Valid until:** 2026-09-03 (Phase 6 is slow-moving diagnosis work; 30-day validity)

---

<a id="blocking-diagnosis-2026-08-04"></a>

# Blocking Diagnosis (2026-08-04)

Four objections were raised against the first self-consistent Eu–N curve. Each was tested
directly. This section records what was measured, with the commands and numbers, and
supersedes any conflicting statement earlier in this document.

**Everything below was produced with the same driver parameters as
`tests/test_eu_n_scan.py`** — closed-shell, `COUL_METHOD="FULL"`, `T_ELECTRONIC=1000.0`,
`RCUT_ELECTRONIC=10.0`, `RCUT_REPULSIVE=6.0`, `CHARGE=0`, float64, CPU, a fresh `Structure`
per point — changing only `do_scf` and the mixer keys where stated.

## Objection 1 — "Mulliken charges should conserve charge regardless of convergence"

**Not violated.** Measured `q_Eu + q_N` across all 21 separations of the original diverging
scan:

| worst \|q_Eu + q_N\| | where | converged points | diverged points |
|---|---|---|---|
| 1.302e-09 | r = 2.10 Å | ~1e-10 to 1e-09 | ~1e-15 to 1e-14 |

Charge is conserved on **both** converged and diverged points. Counter-intuitively the
*converged* points carry the larger residual. That is traceable, not mysterious: the electron
count is set by the Fermi-level search, and every call site passes `eps=1e-9` —
`dm_fermi_x(..., eps=1e-9, MaxIt=50)` at `_xl_tools.py:104` and `_scf.py:352`. The 1e-9 floor
is the Fermi bisection tolerance, not a charge leak. The diverged points sit at saturated
occupations where the bisection terminates more exactly.

**Consequence for planning:** no charge-conservation work is needed. If a tighter number is
ever wanted, `eps` is the knob, and it is a one-line change in two places.

## Objection 2 — "A two-atom system should not have convergence problems"

**Correct, and it does not.** The divergence is the **low-rank Krylov charge mixer**, which
engages once `it > KRYLOV_START` (default `10`) at `_scf.py:452`:

```python
use_krylov = it > dftorch_params.get("KRYLOV_START", 10)
if use_krylov:
    K0Res = KK @ Res
    K0Res = kernel_update_lr(...)      # <-- this is what diverges
```

Every point that converges does so in 12–22 iterations under the default, i.e. only a few
iterations past the Krylov switch. Every point that fails runs to `SCF_MAX_ITER = 100`.

Setting `KRYLOV_START` above `SCF_MAX_ITER` (Anderson/DIIS mixing retained, nothing else
changed) gives:

| | default | Krylov disabled |
|---|---|---|
| converged | 12 / 21 | **21 / 21** |
| iterations | 12–22, or 100 | 13–29 |
| worst energy | +197.98 eV | −0.41 eV (the 1.60 Å endpoint) |

**The decisive check — it is the same fixed point, not a different answer.** At the 12
separations where the default *did* converge, the Krylov-disabled run reproduces the total
energy to `|ΔE| < 1e-6 eV` at every one. Krylov acceleration is not finding a different
solution; it is failing to find the solution the plain mixer finds reliably.

Corrected curve, all 21 points converged:

| r (Å) | E_tot (eV) | it | q_Eu (e) | | r (Å) | E_tot (eV) | it | q_Eu (e) |
|---|---|---|---|---|---|---|---|---|
| 1.60 | −0.410807 | 13 | −0.5420 | | 2.70 | −6.636423 | 27 | −0.5905 |
| 1.70 | −6.409461 | 14 | −0.5690 | | 2.80 | −6.059714 | 28 | −0.5695 |
| 1.80 | −9.439071 | 14 | −0.5902 | | 2.90 | −5.557098 | 29 | −0.5496 |
| 1.90 | −10.719673 | 17 | −0.6058 | | 3.00 | −5.090483 | 27 | −0.5320 |
| **2.00** | **−10.997697** | 19 | −0.6172 | | 3.10 | −4.659414 | 29 | −0.5169 |
| 2.10 | −10.717228 | 20 | −0.6259 | | 3.20 | −4.275004 | 29 | −0.5039 |
| 2.20 | −10.129834 | 27 | −0.6324 | | 3.30 | −3.931544 | 28 | −0.4928 |
| 2.30 | −9.424637 | 28 | −0.6351 | | 3.40 | −3.625795 | 28 | −0.4831 |
| 2.40 | −8.696156 | 29 | −0.6323 | | 3.50 | −3.356250 | 28 | −0.4745 |
| 2.50 | −7.977710 | 29 | −0.6236 | | 3.60 | −3.122492 | 28 | −0.4667 |
| 2.60 | −7.283848 | 29 | −0.6096 | | | | | |

This is a **single clean well**: strictly decreasing to 2.00 Å, strictly increasing after.
Both the shape checks in `test_eu_n_curve_is_a_single_well` would pass on it.

**The minimum sits at 2.00 Å, SHORT of the D-24 band [2.124, 3.186] Å.** Per the asymmetric
failure rule in `tests/f_orbital_data/README-EU-N-CASE.md`, a short minimum is escalated to
human judgement and is explicitly *not* evidence that the f implementation is broken.

**Consequence for planning:** the SCF stabilisation task is not open-ended diagnosis. It is a
bounded decision about the Krylov mixer — fix `kernel_update_lr`, or change its default
engagement, or gate it. The seven f angular blocks are **not implicated**: with
`COUL_METHOD="FULL"` and `MAGNETIC_HUBBARD_LDEP` unset, `_select_coulomb_hubbard`
(`ESDriver.py:129-131`) returns the atom-resolved branch, so
`ewald_real_space_vectorized_sr` is never called and
`FShellResolvedCoulombUnsupportedError` never fires on this path.

## Objection 3 — "Huge discrepancy between the SCF energy and the single-shot energy"

**Explained; it is a definitional difference, not a defect.** The two paths are computing
different quantities:

| | single-shot (`do_scf=False`) | self-consistent (`do_scf=True`) |
|---|---|---|
| minimum | 2.40 Å, −17.6414 eV | 2.00 Å, −10.9977 eV |
| q_Eu there | −2.5506 e | −0.6172 e |
| Coulomb energy | **0 by construction** | included |

`ESDriver.py:783-792` passes `C=None` and `dq_p1=None` into `energy()`, selecting its
`Ecoul = 0` arm, and the comment there states the reason outright: the available charges are
first-iterate Mulliken charges with no self-consistency behind them, and feeding them into the
electrostatics "would not produce a single-shot energy, it would produce one broken SCF step,
and it destroys the binding curve."

So the single-shot number banks a large band-structure gain from moving 2.55 electrons onto N
while paying **no** electrostatic penalty for the resulting ±2.55 e dipole. Turning the penalty
on collapses the transfer to 0.62 e, which gives back most of that band gain. A 6.6 eV
difference between the two is the expected size of that effect, not an anomaly.

**Consequence for planning:** the two numbers must never be compared as if they were the same
observable, and no plan task should try to make them agree. The Phase 4 pinned reference
(−17.510444238744924 eV at 2.655 Å, single-shot) remains valid as a regression pin **for the
single-shot path only**.

## Objection 4 — "At 10 Å I should get basically neutral Mulliken charges from both paths"

**Neither path delivers this. The two failures have different causes and different standing.**

| r (Å) | 6.0 | 8.0 | 10.0 | 15.0 | 20.0 | 30.0 | 40.0 |
|---|---|---|---|---|---|---|---|
| single-shot q_Eu | −3.0000 | −3.0000 | −3.0000 | −3.0000 | −3.0000 | −3.0000 | −3.0000 |
| self-consistent q_Eu | −0.3704 | −0.3422 | −0.3270 | −0.3088 | −0.3004 | −0.2925 | −0.2887 |

### 4a. Single-shot pins at exactly −3.0000 e — expected for non-SCC, not a bug

At 10 Å the two atoms are uncoupled, so the eigenvalues are the bare on-site energies. Measured
spectrum and occupations at 10 Å (µ = −1.6001 eV, `Nocc` = 7 pairs = 14 electrons):

| rank | eigenvalue (eV) | occupation f | orbital |
|---|---|---|---|
| 0 | −21.5623 | 1.0000 | N 2s |
| 1–3 | −6.8355 | 1.0000 | N 2p ×3 |
| 4 | −2.6341 | 1.0000 | Eu 6s |
| 5–11 | −1.5211 | 0.2857 | Eu 4f ×7 |

N's four orbitals hold 2 + 6 = **8 electrons** (a filled octet, N³⁻); Eu holds 2 in 6s plus
2 spread over seven 4f orbitals (7 × 2 × 0.2857 = 4), total 6, so q_Eu = 6 − 9 = **−3 exactly**.

The cause is structural: **every** N level lies below **every** Eu level. Deepest Eu level is
4f at −1.5211 eV; N 2p is at −6.8355 eV — a **5.3144 eV** mismatch. With one common Fermi level
and no charge penalty, N fills completely at any separation. This is the textbook non-SCC
charge-transfer catastrophe and is exactly why SCC-DFTB exists. It is **expected DFTB1
behaviour, not a parser bug** — parsing was verified correct, see § Parameter audit below.

### 4b. Self-consistent converges to −0.2887 e — DFTB2's known dissociation error

Quantitatively predicted. At large r the intersite Coulomb term vanishes, so the charge-transfer
energy is `E(Δ) = −gap·Δ + ½(U_Eu + U_N)·Δ²`, minimised at `Δ = gap / (U_Eu + U_N)`:

| Eu Hubbard U used | source | predicted q_Eu | **measured at 40 Å** | error |
|---|---|---|---|---|
| 5.714 eV (U_s = 0.21 Ha) | what the code uses | −0.2790 | **−0.2887** | 3.5 % |
| 13.606 eV (U_f = 0.50 Ha) | Eu's f shell | −0.1973 | **−0.2021** | 2.4 % |

(gap = 5.3144 eV, U_N = 13.3336 eV. The residual few-percent error is Fermi smearing at
`T_ELECTRONIC = 1000 K`.)

The model reproduces both measurements, so the mechanism is identified: this is DFTB2's known
fractional-charge-at-dissociation behaviour, driven by the 5.31 eV level mismatch against a
finite Hubbard penalty. **It does not go to zero for any choice of U** — a better U shrinks it
from 0.29 e to 0.20 e, no more.

**This is the one finding that should change the phase's scope conversation.** Exact
neutrality at dissociation is not reachable by tuning inside the current model; it needs the
Eu on-site energies to align with N's, or a fragment-based / DFTB+U treatment. That is a
parameter-set and method question, not a Phase 6 implementation task.

## Parameter audit — parsing verified correct

Raw SKF header lines and what dftorch loaded from them:

```
Eu-Eu.skf line 3:  -0.0559 -0.0247 0.0197 -0.0968   0.0   0.50 0.25 0.19 0.21   7.0 0.0 0.0 2.0
                   \___ Ef  Ed     Ep     Es ___/  SPE  \_ Uf  Ud   Up   Us _/  \_ ff fd fp fs _/
N-N.skf   line 2:  0.0 -0.2512 -0.7924   0.00   0.490 0.490 0.490   0.0 3.0 2.0
                   \_ Ed Ep    Es ____/  SPE   \_ Ud   Up    Us __/ \_ fd fp fs _/
```

| quantity | SKF (Ha) | loaded (eV) | check |
|---|---|---|---|
| Eu E_f | −0.0559 | −1.5211 | ✓ |
| Eu E_d | −0.0247 | −0.6721 | ✓ |
| Eu E_p | +0.0197 | +0.5361 | ✓ (positive in the source file) |
| Eu E_s | −0.0968 | −2.6341 | ✓ |
| N E_p | −0.2512 | −6.8355 | ✓ (standard mio value) |
| N E_s | −0.7924 | −21.5623 | ✓ |
| Eu Hubbard U | U_s = 0.21 | 5.7144 | ✓ **but see below** |
| N Hubbard U | 0.490 | 13.3336 | ✓ |

Occupations `ff fd fp fs = 7,0,0,2` sum to 9 = `Znuc[Eu]`; `fd fp fs = 0,3,2` sum to 5 =
`Znuc[N]`; `el_per_shell = [2,0,0,7 | 2,3]` matches. Descending-l ordering confirmed by the
occupations (Eu = [Xe]4f⁷6s²). **No parsing defect found — this is not a repeat of the Phase 5
phantom-p-shell bug.**

### The one parameter finding that matters

**The atom-resolved Hubbard U is always the s-shell value**, unconditionally:

* `Constants.py:232` — `self.U = torch.nn.Parameter(US, ...)`, where `US` is the s-shell column.
  `Up`, `Ud`, `Uf` are loaded into separate tensors at lines 233–235.
* `Structure.py:361` and `Structure.py:549` — `self.Hubbard_U = const.U[self.TYPE]`.

For N this is harmless: the SKF gives `U_d = U_p = U_s = 0.490`, all equal. For Eu they are
**not** equal — `U_f = 0.50`, `U_d = 0.25`, `U_p = 0.19`, `U_s = 0.21` Ha. The code therefore
charges Eu at **5.71 eV** per unit charge while **7 of Eu's 9 valence electrons live in the f
shell, whose U is 13.61 eV — 2.4× larger**.

This is the concrete, measurable case for the shell-resolved Coulomb work: substituting U_f
changes the dissociation charge from −0.2887 e to −0.2021 e (a 30 % reduction) and the 40 Å
energy from −1.96171 eV to −1.73157 eV. It is a real effect on a real observable, and it is
independent of the Krylov bug.

## What this changes for Phase 6 planning

1. **The SCF stabilisation task is bounded, not exploratory.** Root cause is `kernel_update_lr`
   / `KRYLOV_START`. A plan task can name it. The open decision is the fix shape (repair the
   kernel, change the default, or gate it), not the diagnosis.
2. **Success criterion 1 is already reachable today** — 21/21 converge with one config key
   changed. The f SCF convergence test can be written now.
3. **The seven f angular blocks are decoupled from the convergence bug.** They were never on
   the code path. Their justification is the U_s/U_f discrepancy above, which is a genuine
   30 % effect — but it is a *quality* argument, not a *convergence* argument.
4. **Success criterion 5 is safe.** Nothing here touches the single-shot path; its pinned
   reference energy is unaffected.
5. **A new, un-roadmapped question exists:** neither path dissociates to neutral, and the
   self-consistent residual is a method limitation rather than a bug. Whether Phase 6 accepts
   ≈0.2–0.3 e at dissociation, or opens a parameter/method investigation, is a scope decision
   that needs a human ruling before planning.

## Cross-check against the mio-1-1 set — the Eu–N findings generalise

Sixteen diatomics were scanned over the mio-1-1 parameter set at 61 separations each, on both
the `H0` and self-consistent paths, with the Krylov accelerator disabled
(`experiments/diatomic_scans/mio_suite.py`). Only even-valence-electron systems appear —
`Structure.py:355` raises "Closed shell systems require even number of electrons", so the
common radicals OH, CH, CN, NO and SH cannot run closed-shell at all; the anions stand in.

**Every system converged at every point: 61/61 × 16.** No mixer failures anywhere once Krylov
is off, which is independent corroboration that the divergence was the accelerator and not
anything specific to f orbitals.

Equilibrium separations against experimental `r_e` (reference markers only, not tolerances):

| system | exp. r_e | H0 min | dev | SCC min | dev | | system | exp. r_e | H0 min | dev | SCC min | dev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H₂ | 0.7414 | 0.7300 | −1.54 % | 0.7300 | −1.54 % | | PN | 1.4909 | 1.5167 | +1.73 % | 1.4750 | −1.07 % |
| C₂ | 1.2425 | 1.2450 | +0.20 % | 1.2450 | +0.20 % | | P₂ | 1.8934 | 1.9200 | +1.40 % | 1.9200 | +1.40 % |
| N₂ | 1.0977 | 1.0933 | −0.40 % | 1.0933 | −0.40 % | | S₂ | 1.8892 | 1.8767 | −0.66 % | 1.8767 | −0.66 % |
| O₂ | 1.2075 | 1.2083 | +0.07 % | 1.2083 | +0.07 % | | SO | 1.4811 | 1.5000 | +1.28 % | 1.5000 | +1.28 % |
| CO | 1.1283 | 1.0933 | −3.10 % | 1.0933 | −3.10 % | | CS | 1.5349 | 1.5167 | −1.19 % | 1.5167 | −1.19 % |
| CN⁻ | 1.1772 | 1.1367 | −3.44 % | 1.1725 | −0.40 % | | PH | 1.4223 | 1.4233 | +0.07 % | 1.4233 | +0.07 % |
| NH | 1.0362 | 1.0667 | +2.94 % | 1.0667 | +2.94 % | | SH⁻ | 1.3400 | 1.3500 | +0.75 % | 1.3500 | +0.75 % |
| CH⁻ | 1.1190 | 1.1200 | +0.09 % | 1.1550 | +3.22 % | | OH⁻ | 0.9640 | 1.0000 | +3.73 % | 1.0000 | +3.73 % |

Every deviation lands inside ±3.8 %. Note the grid resolution is ~0.03–0.04 Å, so several
heteronuclear rows show an identical H0 and SCC minimum only because the shift is smaller than
one grid step — do not read that as "SCC did nothing" without refining `N_POINTS`.

Bond lengths were verified against the **NIST CCCBDB** list of experimental diatomic bond
lengths (`https://cccbdb.nist.gov/diatomicexpbondx.asp`), not quoted from memory: all twelve
neutrals match to the digits shown, CN⁻ (1.177 ± 0.004 Å) and OH⁻ (0.964 Å) were confirmed
separately. **CH⁻ and SH⁻ remain unverified** — no experimental r_e was located for either, so
their reference markers (1.119 and 1.340 Å) are unsourced and must not be quoted.

### Atomization energies — and the reference trap that makes them look wrong

`E(6 Å)` is **not** a valid dissociation reference. At 6 Å the restricted closed-shell
calculation still shares one Fermi level across both fragments and smears occupations at
`T_ELECTRONIC = 1000 K`, so it is not two ground-state atoms. Measured `E(6 Å)` runs −0.24 to
−1.01 eV rather than 0.

The correct reference is already built into the code: `_energy.py` computes
`Eband0 = 2·Tr[H0 (D − D0)]` with `D0` the free-atom reference density, so **`structure.e_tot`
is already an atomization-style energy** and `D_e = −E_tot(min)` directly. Anyone reading
`e_tot` as an absolute total energy will draw wrong conclusions.

That reference is the *spin-unpolarised* atom. Real atoms are spin-polarised and lie lower, by
the free-atom spin-polarisation energy carried in SKF field 4 (`SPE`). Correcting for it:

| system | D_e, unpolarised ref | D_e, +SPE | exp. D_e | err (+SPE) | | system | D_e, unpol. | D_e, +SPE | exp. D_e | err (+SPE) |
|---|---|---|---|---|---|---|---|---|---|---|
| H₂ | 5.378 | 3.582 | 4.75 | −1.17 | | PN | 11.340 | 6.713 | 6.36 | +0.35 |
| C₂ | 9.997 | 7.608 | 6.32 | +1.29 | | P₂ | 8.030 | 4.828 | 5.08 | −0.25 |
| N₂ | 17.400 | 11.348 | 9.90 | +1.44 | | S₂ | 6.532 | 6.532 | 4.37 | +2.16 |
| O₂ | 8.522 | 5.575 | 5.21 | +0.36 | | SO | 8.761 | 7.288 | 5.43 | +1.86 |
| CO | 15.375 | 12.707 | 11.23 | +1.48 | | CS | 10.416 | 9.222 | 7.36 | +1.86 |
| NH | 6.373 | 2.449 | 3.47 | −1.02 | | PH | 4.749 | 2.249 | 3.10 | −0.85 |

| atomic reference | mean error | MAE |
|---|---|---|
| spin-unpolarised (`D0` as-is) | **+3.36 eV** | 3.36 eV |
| spin-polarised (`D0` + SKF `SPE`) | **+0.63 eV** | **1.17 eV** |

**Read the +3.36 eV row as the real result, not the +0.63 eV row.** The published benchmark for
this parameter set gives an atomization-energy MAD of **68.7 kcal/mol = 2.98 eV for
DFTB3/MIO**, against 4.7 kcal/mol for DFTB3/3OB (Gaus, Cui & Elstner, *Parametrization and
Benchmark of DFTB3 for Organic Molecules*, JCTC 2013, and the 3OB S/P parametrization paper,
JCTC 2014). The uncorrected measurement here — **+3.36 eV = 77 kcal/mol** — reproduces that
published figure. The overbinding is therefore mio's documented, expected failure mode, not a
defect in the SCC path and not an artefact to be corrected away. The same literature explains
why mio remains useful: its per-bond overbindings are balanced so that they **cancel in
reaction energies**, which is what mio was parameterised for. It was never intended to give
good atomization energies.

**The `+SPE` column is UNVERIFIED and should not be quoted.** The spin-polarisation convention
applied to produce it was reconstructed from the SKF field layout, not checked against DFTB+
documentation. That it lands nearer experiment is not evidence it is right — the literature
comparison above suggests the uncorrected column is the one that corresponds to how mio is
actually characterised. Settle this by checking the DFTB+ treatment of the SKF `SPE` field
before either column is used for anything.

Two contributions were isolated:

* **Electronic temperature is not a factor.** All twelve systems were rerun at 1000 K, 300 K
  and 10 K: MAE moves 1.18 → 1.16 → 1.16 eV. Only the five systems with degenerate frontier
  orbitals shift at all (a −0.2389 eV entropy term at 1000 K — O₂, NH, S₂, SO, PH, exactly the
  triplet ground states), and lowering the temperature makes agreement marginally *worse*.
* **Sulfur is a distinct sub-cluster.** Excluding S₂/SO/CS drops MAE from 1.18 to 0.92 eV and
  mean error from +0.63 to +0.19 eV. Sulfur is also the only element whose `SPE` field is
  `0.0` in `S-S.skf`, so it receives no correction under any convention.

**Caveat on the experimental D_e column:** these are quoted Huber–Herzberg values and, unlike
the bond lengths, were **not** hard-verified against a primary source. Spot-check before
quoting.

**None of this bears on Phase 6 scope.** It characterises the mio parameter set, not the SCC
implementation. The evidence that the code is correct is the geometry and the invariants —
bond lengths within ±3.7 %, homonuclear SCC ≡ H0 to 1e-13 eV, charge conserved to 1e-9, and
976/976 SCF points converged. Defensible atomization energies would require the **3OB**
parameter set, which is not present in this repository (only mio-1-1 plus Zn, and the
f-orbital Eu/N/Ga set).

### Two results that are worth turning into tests

1. **Homonuclear diatomics give SCC ≡ H0 identically.** For all six (H₂, C₂, N₂, O₂, P₂, S₂),
   `max|E_SCC − E_H0| ≈ 1e-13 eV` and `max|q| ≈ 1e-9 e` across the whole scan. Symmetry forbids
   charge transfer, so the SCC correction must vanish exactly — and it does. This is a sharp,
   free regression test on the entire self-consistent path, and it costs nothing to run.

2. **The non-SCC charge-transfer catastrophe is universal, not an Eu–N artefact.** Every
   heteronuclear system shows the same runaway on the `H0` path — at 3 Å, `q` reaches −2.0 e
   (CO), −3.0 e (PN), −2.0 e (SO, CS), −2.0 e (CN⁻). The self-consistent path holds all of them
   in the −0.3 to +0.05 e band for the neutral species. So the Eu–N behaviour documented above
   is the method behaving as designed, in a well-established parameter set, and is not evidence
   of a defect in the f implementation.

## Scope rulings (2026-08-04, human decision)

Three questions above were left open for a human. All three are now settled:

1. **Krylov accelerator — deferred, disabled in the interim.** Repairing `kernel_update_lr` is
   moved out of Phase 6 to a later phase. Phase 6 sets `KRYLOV_START` above `SCF_MAX_ITER` and
   proceeds. The `UNRESOLVED` entry on fix shape is withdrawn: the decision is "defer", not
   "choose a repair".
2. **Dissociation charge — accepted.** ≈0.2–0.3 e residual transfer at dissociation is accepted
   as within the method's limits. No parameter or method investigation is opened. The
   `UNRESOLVED` entry is closed.
3. **Hubbard-U shell selection — in scope for Phase 6.** The s-shell/f-shell discrepancy
   (`U_s = 0.21` vs `U_f = 0.50` Ha for Eu) is to be addressed by the shell-resolved Coulomb
   work in this phase, which is therefore confirmed as Phase 6 scope rather than conditional.

## Reproduction

Scripts and data: `experiments/diatomic_scans/` (see its README). Figures: `figures/`.

| figure | content |
|---|---|
| `figures/eu_n_binding_curve.png` | Eu–N total energy and charge transfer, H0 vs SCC |
| `figures/eu_n_dissociation_limit.png` | Eu–N charge out to 40 Å, both paths |
| `figures/mio_diatomic_binding_curves.png` | 16 mio-1-1 diatomics, H0 vs SCC |
| `figures/mio_equilibrium_bond_lengths.png` | computed vs experimental r_e, absolute and % |
| `figures/mio_charge_transfer.png` | Mulliken charge, 10 heteronuclear systems |

Superseded and safe to delete: `eu_n_binding_curve_scf.png` and
`eu_n_binding_curve_scf_fixed.png` at the repository root.

To reproduce the fix in one line, add to the driver params dict:

```python
"KRYLOV_START": 10**6,   # keeps Anderson/DIIS, never engages kernel_update_lr
```

## Confidence

| Finding | Level | Basis |
|---|---|---|
| Krylov mixer is the divergence cause | **HIGH** | 21/21 vs 12/21; identical energies at all 12 overlap points |
| Charge conservation is not violated | **HIGH** | Direct measurement, worst 1.3e-09, traced to `eps=1e-9` |
| Single-shot/SCF energy gap is definitional | **HIGH** | `C=None` at `ESDriver.py:783-792` plus measured charges |
| Single-shot −3.0000 e is non-SCC filling | **HIGH** | Eigenvalue spectrum and occupations measured at 10 Å |
| SCC dissociation residual is DFTB2 error | **HIGH** | Analytic model matches measurement to 2.4–3.5 % across two U values |
| SKF parsing is correct | **HIGH** | Every header field cross-checked against loaded values |
| Krylov behaviour generalises beyond f systems | **HIGH** | 16 mio-1-1 diatomics × 61 points, 61/61 converged in every system with the accelerator off |
| Homonuclear SCC ≡ H0 identity | **HIGH** | Measured across all six homonuclear systems: ΔE ~ 1e-13 eV, q ~ 1e-9 e |
| Correct fix shape for the Krylov mixer | **CLOSED — deferred** | Human ruling 2026-08-04: repair moved to a later phase; Phase 6 disables the accelerator. |
| Whether ≈0.29 e at dissociation is acceptable | **CLOSED — accepted** | Human ruling 2026-08-04: accepted as within method limits; no investigation opened. |
| Shell-resolved Hubbard U in Phase 6 scope | **CLOSED — in scope** | Human ruling 2026-08-04: the U_s/U_f discrepancy is to be fixed by this phase's shell-resolved work. |

**Diagnosis date:** 2026-08-04

---

*Phase: 6-Self-Consistent SCF for f Systems*  
*Research complete; blocking diagnosis complete; scope ruling needed before planning.*
