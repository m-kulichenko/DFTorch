from __future__ import annotations
import math
import torch

from ._tools import _maybe_compile

from ._nearestneighborlist import (
    vectorized_nearestneighborlist,
)
from ._slater_koster_pair import Slater_Koster_Pair_SKF_vectorized


def _cell_volume(cell: torch.Tensor) -> torch.Tensor:
    if cell is None:
        raise ValueError("Stress tensor requires a periodic cell.")
    if cell.dim() != 2 or cell.shape != (3, 3):
        raise ValueError(f"Unsupported cell shape for stress: {tuple(cell.shape)}")
    return torch.abs(torch.det(cell))


def _symmetrize_stress(stress: torch.Tensor) -> torch.Tensor:
    return 0.5 * (stress + stress.transpose(-1, -2))


def _positions_from_components(
    RX: torch.Tensor, RY: torch.Tensor, RZ: torch.Tensor
) -> torch.Tensor:
    return torch.stack((RX, RY, RZ), dim=-1)


def _orbital_atom_ids(TYPE: torch.Tensor, const) -> torch.Tensor:
    n_orbitals_per_atom = const.n_orb[TYPE]
    return torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=TYPE.device), n_orbitals_per_atom
    )


def _stress_from_matrix_cell_derivative(
    dA_dxyz: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    atom_ids: torch.Tensor,
    cell: torch.Tensor,
    *,
    prefactor: float = 1.0,
) -> torch.Tensor:
    """Contract a matrix derivative wrt atomic coordinates into cell stress.

    If fractional coordinates are held fixed under a cell deformation, one has
    ``dr_i / dh_{alpha,beta} = delta_{alpha,gamma} r_{i,beta}``, which turns a
    Cartesian coordinate derivative into the cell derivative used for the
    physical stress tensor.
    """
    volume = _cell_volume(cell)
    sigma = torch.zeros((3, 3), dtype=weight.dtype, device=weight.device)

    for beta in range(3):
        weighted = dA_dxyz * positions[atom_ids, beta].view(1, -1, 1)
        sigma[:, beta] = (
            prefactor * torch.sum(weight.unsqueeze(0) * weighted, dim=(1, 2)) / volume
        )

    return _symmetrize_stress(sigma)


def _stress_from_force_component(
    forces_xyz: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    volume = _cell_volume(cell)
    sigma = positions.transpose(-1, -2) @ forces_xyz.transpose(-1, -2) / volume
    return _symmetrize_stress(sigma)


def _overlap_stress_from_force_logic(
    dS: torch.Tensor,
    H: torch.Tensor,
    Z: torch.Tensor,
    D: torch.Tensor,
    q: torch.Tensor,
    C: torch.Tensor,
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    const,
    n: torch.Tensor | None = None,
) -> torch.Tensor:
    positions = _positions_from_components(RX, RY, RZ)
    atom_ids = _orbital_atom_ids(TYPE, const)

    if D.dim() == 3:
        D_tot = D.sum(0) / 2
    else:
        D_tot = D

    n_eff = n if n is not None else q
    CoulPot = C @ n_eff if C is not None else torch.zeros_like(q)
    factor = 2.0 * (Hubbard_U * n_eff + CoulPot)

    dS_times_D = D_tot.unsqueeze(0) * dS * factor[atom_ids].unsqueeze(-1)
    FScoul = torch.zeros((3, len(TYPE)), dtype=dS.dtype, device=dS.device)
    dDS_xyz_row_sum = torch.sum(dS_times_D, dim=2)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_xyz_row_sum)
    dDS_xyz_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_xyz_col_sum)

    SIHD = 4.0 * Z @ Z.transpose(-1, -2) @ H @ D
    FPulay = torch.zeros((3, len(TYPE)), dtype=dS.dtype, device=dS.device)
    if SIHD.dim() == 2:
        tmp = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    else:
        tmp = -0.5 * torch.matmul(dS.unsqueeze(0), SIHD.unsqueeze(1)).sum(0).diagonal(
            offset=0, dim1=1, dim2=2
        )
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), tmp)

    return -(
        _stress_from_force_component(FPulay, positions, cell)
        + _stress_from_force_component(FScoul, positions, cell)
    )


def _pair_grad_from_sk(
    metadata: dict[str, torch.Tensor],
    const,
    stress_weight: torch.Tensor,
    cell: torch.Tensor,
    SH_shift: int = 0,
    extra_scale: float = 1.0,
) -> torch.Tensor:
    """Compute pairwise stress by piggybacking on the Slater-Koster builder.

    Instead of re-evaluating splines and angular derivatives from scratch,
    this reconstructs the SK inputs from the stored metadata and calls
    ``Slater_Koster_Pair_SKF_vectorized`` with ``stress_weight``.  The SK
    function accumulates the weighted pair gradients on-the-fly — no
    separate Python loops over orbital blocks are needed.

    Returns the symmetrised 3×3 stress tensor (Ha/Bohr³).
    """
    Rab = metadata["Rab_mskd"]  # (P, 3)
    dev = Rab.device

    Rab_X = Rab[:, 0]
    Rab_Y = Rab[:, 1]
    Rab_Z = Rab[:, 2]
    dR = torch.norm(Rab, dim=-1)

    L = Rab_X / dR
    M = Rab_Y / dR
    N = Rab_Z / dR

    dR3 = dR**3
    L_dx = (Rab_Y**2 + Rab_Z**2) / dR3
    L_dy = -Rab_X * Rab_Y / dR3
    L_dz = -Rab_X * Rab_Z / dR3
    M_dx = -Rab_Y * Rab_X / dR3
    M_dy = (Rab_X**2 + Rab_Z**2) / dR3
    M_dz = -Rab_Y * Rab_Z / dR3
    N_dx = -Rab_Z * Rab_X / dR3
    N_dy = -Rab_Z * Rab_Y / dR3
    N_dz = (Rab_X**2 + Rab_Y**2) / dR3

    dR_dxyz = (Rab / dR.unsqueeze(-1)).T  # (3, P)
    L_dxyz = torch.stack((L_dx, L_dy, L_dz))  # (3, P)
    M_dxyz = torch.stack((M_dx, M_dy, M_dz))
    N_dxyz = torch.stack((N_dx, N_dy, N_dz))

    idx = metadata["idx"]
    dx = dR - const.R_orb[idx]
    IJ_pair_type = metadata["IJ_pair_type"]
    JI_pair_type = metadata["JI_pair_type"]
    i0 = metadata["i0"]
    j0 = metadata["j0"]
    nI = metadata["n_orb_I"]  # uint8
    nJ = metadata["n_orb_J"]  # uint8

    # Reconstruct pair masks
    pair_mask_HH = (nI == 1) & (nJ == 1)
    pair_mask_HX = (nI == 1) & (nJ == 4)
    pair_mask_XH = (nI == 4) & (nJ == 1)
    pair_mask_XX = (nI == 4) & (nJ == 4)
    pair_mask_HY = (nI == 1) & (nJ == 9)
    pair_mask_XY = (nI == 4) & (nJ == 9)
    pair_mask_YH = (nI == 9) & (nJ == 1)
    pair_mask_YX = (nI == 9) & (nJ == 4)
    pair_mask_YY = (nI == 9) & (nJ == 9)

    # The SK function needs neighbor_I / neighbor_J only through
    # H_INDEX_START[neighbor_I/J].  We create identity arrays so that
    # H_INDEX_START_fake[p] == i0[p] and j0[p] respectively.
    HDIM = stress_weight.shape[0]

    # Instead, we pass i0/j0 directly as neighbor arrays and use
    # an identity H_INDEX_START over the AO dimension so H_INDEX_START_id[k] == k.
    H_INDEX_START_id = torch.arange(HDIM, device=dev, dtype=i0.dtype)

    _, _, pair_grad = Slater_Koster_Pair_SKF_vectorized(
        HDIM,
        dR_dxyz,
        L,
        M,
        N,
        L_dxyz,
        M_dxyz,
        N_dxyz,
        pair_mask_HH,
        pair_mask_HX,
        pair_mask_XH,
        pair_mask_XX,
        pair_mask_HY,
        pair_mask_XY,
        pair_mask_YH,
        pair_mask_YX,
        pair_mask_YY,
        dx,
        idx,
        IJ_pair_type,
        JI_pair_type,
        const.coeffs_tensor,
        i0,
        j0,  # "neighbor_I", "neighbor_J"
        H_INDEX_START_id,  # identity mapping
        SH_shift,
        stress_weight=stress_weight,
        i0_stress=i0,
        j0_stress=j0,
    )

    sigma = extra_scale * torch.einsum("pi,pj->ij", pair_grad, Rab) / _cell_volume(cell)
    return _symmetrize_stress(sigma)


def get_electronic_stress_analytical(
    H0: torch.Tensor,
    H: torch.Tensor,
    Z: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    D0: torch.Tensor,
    dH0: torch.Tensor,
    dS: torch.Tensor,
    dCC: torch.Tensor,
    Hubbard_U: torch.Tensor,
    q: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    Efield: torch.Tensor,
    TYPE: torch.Tensor,
    const,
    dU_dq: torch.Tensor | None = None,
    n: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a nonorthogonal analytical electronic stress estimate.

    This computes the electronic stress using the pairwise Slater-Koster
    derivative framework, matching the force decomposition used in
    ``_forces.py``:

    1. **Band stress**: ``Tr[2(D-D0) * dH0/deps]`` using H0-channel SK
       derivatives with density weight ``2*(D_eff - D0_eff)``.

    2. **Pulay stress**: ``-0.5 * Tr[SIHD * dS_raw/deps]`` using S-channel
       SK derivatives.  The factor of 0.5 comes from the overlap-matrix
       symmetrisation ``S = (S_raw + S_raw^T)/2 / 27.21 + I``, and the
       minus sign from the standard Pulay/EWDM contribution to the stress.

    3. **SCC-overlap stress**: ``Tr[D*factor * dS_raw/deps]`` using the
       same S-channel framework, where ``factor`` is the per-orbital SCC
       shift ``2*(U*q + V_Coul)``.

    Notes
    -----
    - The neighbor list uses ``upper_tri_only=False``, so both I->J and
      J->I pairs appear.  For a symmetric density weight the double sum
      exactly reproduces the correct stress; for the antisymmetric Pulay
      weight the ``-0.5`` factor handles the doubling and sign.
    - D0 is purely diagonal (on-site) so ``(D-D0)`` and ``D`` produce the
      same result in the pairwise formula (only off-diagonal pairs contribute).
    """
    positions = _positions_from_components(RX, RY, RZ)
    atom_ids = _orbital_atom_ids(TYPE, const)

    if D.dim() == 3:
        D_eff = D.sum(0) / 2
    else:
        D_eff = D

    if D0.dim() == 1:
        D0_eff = torch.diag(D0)
    else:
        D0_eff = D0

    band_weight = 2.0 * (D_eff - D0_eff)
    metadata = getattr(const, "_stress_metadata", None)
    if metadata is not None:
        # Band (H0) pairwise stress via the SK builder with stress_weight
        band_stress = _pair_grad_from_sk(
            metadata,
            const,
            band_weight,
            cell,
        )

        # Overlap contributions (Pulay + SCC-overlap) computed in the same
        # pairwise SK framework by building appropriate density/weight
        # matrices that multiply the S-channel derivatives. The SK builder
        # scales raw S-values by 1/27.21138625, so pass that as extra_scale.
        hartree_to_eV = 27.21138625

        # Effective density matrix used in SCC overlap (D summed if batched)
        if D.dim() == 3:
            D_tot = D.sum(0) / 2
        else:
            D_tot = D

        # Pulay weight: SIHD = 4 Z Z^T H D from the force logic.
        # The overlap matrix is symmetrized: S = (S_raw + S_raw^T)/2 / 27.21 + I.
        # The Pulay stress involves Tr[SIHD * dS/deps] which, after accounting
        # for the S symmetrization, gives a factor of 1/2. Together with the
        # overall minus sign of the Pulay term: W_pulay = -0.5 * SIHD.
        # (SIHD is essentially symmetric since S^{-1}HD is symmetric at
        # self-consistency, so SIHD.T ≈ SIHD.)
        SIHD = 4.0 * Z @ Z.transpose(-1, -2) @ H @ D
        # Collapse spin channels for open-shell, mirroring _forces.py:
        # closed-shell forces use  -Tr[SIHD @ dS],
        # open-shell  forces use  -0.5 * Tr[SIHD.sum(0) @ dS].
        # Stress adds an extra 0.5 from the overlap symmetrisation,
        # giving -0.5 (cs) and -0.25 (os).
        if SIHD.dim() == 3:  # open-shell
            W_pulay = -0.25 * SIHD.sum(0)
        else:
            W_pulay = -0.5 * SIHD

        # SCC-overlap weight: row-weighted D by the per-atom factor.
        # The force code applies factor[atom_of_row_i] to each row of D*dS,
        # so W_scc[i,j] = D_tot[i,j] * factor[atom_of_i].
        #
        # In XL-BOMD the shadow energy uses extrapolated charges n in place
        # of SCF charges q for the Coulomb potential and on-site Hubbard
        # contributions, matching the shadow force expression in _forces.py:
        #   factor = 2*(U*n + C@n)  (+ DFTB3 correction with 2q-n and n)
        # When n is None (standard SCF), q is used everywhere.
        n_eff = n if n is not None else q
        CoulPot = C @ n_eff if C is not None else torch.zeros_like(q)
        if dU_dq is not None:
            factor = 2.0 * (
                Hubbard_U * n_eff + CoulPot + 0.5 * dU_dq * (2.0 * q - n_eff) * n_eff
            )
        else:
            factor = 2.0 * (Hubbard_U * n_eff + CoulPot)
        factor_orb = factor[atom_ids]
        W_scc = D_tot * factor_orb.unsqueeze(1)

        # Merge Pulay + SCC into a single pass — the function is linear in
        # density_weight, so stress(W_pulay) + stress(W_scc) = stress(W_pulay + W_scc).
        # This halves the number of spline evaluations for the overlap channel.
        overlap_stress = _pair_grad_from_sk(
            metadata,
            const,
            W_pulay + W_scc,
            cell,
            SH_shift=1,
            extra_scale=1.0 / hartree_to_eV,
        )
    else:
        # Fallback: use the existing force-based overlap stress helper
        band_stress = _stress_from_matrix_cell_derivative(
            dH0,
            band_weight,
            positions,
            atom_ids,
            cell,
            prefactor=-1.0,
        )
        overlap_stress = _overlap_stress_from_force_logic(
            dS,
            H,
            Z,
            D,
            q,
            C,
            Hubbard_U,
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            const,
            n=n,
        )

    out = {
        "band_analytical": band_stress,
        "overlap_analytical": overlap_stress,
        "electronic_total": _symmetrize_stress(band_stress + overlap_stress),
    }

    if Efield is not None and torch.any(Efield != 0):
        out["dipole_field_note"] = torch.zeros_like(out["electronic_total"])

    return out


# ---------------------------------------------------------------------------
# Coulomb (Ewald) stress — real-space virial + reciprocal-space cell derivative
# ---------------------------------------------------------------------------


def get_coulomb_stress_real(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    q: torch.Tensor,
    Nats: int,
    const,
    Coulomb_acc: float,
    CALPHA: float,
    COULCUT: float,
    n: torch.Tensor | None = None,
) -> torch.Tensor:
    """Real-space Ewald Coulomb stress from the pairwise virial.

    Uses the same neighbor list and short-range damping as the Coulomb matrix
    builder, but contracts with charges and forms the outer-product virial
    instead of assembling a force tensor.

    When *n* is provided (XL-BOMD shadow stress), the pair weight uses the
    shadow energy expression ``0.5 * (2q_i - n_i) * n_j`` instead of the
    standard ``0.5 * q_i * q_j``.

    Returns stress in eV/Å³.
    """

    _, _, nnRx, nnRy, nnRz, nnType, _, _, neighbor_I, neighbor_J, _, _ = (
        vectorized_nearestneighborlist(
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            COULCUT,
            Nats,
            const,
            upper_tri_only=False,
        )
    )

    Ra = torch.stack((RX.unsqueeze(-1), RY.unsqueeze(-1), RZ.unsqueeze(-1)), dim=-1)
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)

    nn_mask = nnType != -1
    dR_mskd = dR[nn_mask]
    Rab_mskd = Rab[nn_mask]  # (Npairs, 3)

    # Recompute the real-space Coulomb derivative dC/dR for each pair
    CALPHA2 = CALPHA**2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)

    Ti = TFACT * Hubbard_U[neighbor_I]
    Tj = TFACT * Hubbard_U[neighbor_J]

    # Ewald screened Coulomb and derivative
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
    dCA = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI) / dR_mskd
    dtmp1 = dCA.clone()

    # Short-range damping corrections
    mask_same = TYPE[neighbor_I] == TYPE[neighbor_J]
    if mask_same.any():
        dR_same = dR_mskd[mask_same]
        Ti_same = Ti[mask_same]
        TI2 = Ti_same**2
        TI3 = TI2 * Ti_same
        SSB = TI3 / 48.0
        SSC = 3 * TI2 / 16.0
        SSD = 11 * Ti_same / 16.0
        EXPTI = torch.exp(-Ti_same * dR_same)
        tmp = SSB * dR_same**2 + SSC * dR_same + SSD + 1.0 / dR_same
        dtmp1[mask_same] -= EXPTI * (
            (-Ti_same) * tmp + (2 * SSB * dR_same + SSC - 1.0 / dR_same**2)
        )
    if (~mask_same).any():
        dR_diff = dR_mskd[~mask_same]
        Ti_d = Ti[~mask_same]
        Tj_d = Tj[~mask_same]
        TI2 = Ti_d**2
        TI4 = TI2**2
        TI6 = TI4 * TI2
        TJ2 = Tj_d**2
        TJ4 = TJ2**2
        TJ6 = TJ4 * TJ2
        EXPTI = torch.exp(-Ti_d * dR_diff)
        EXPTJ = torch.exp(-Tj_d * dR_diff)
        TI2MTJ2 = TI2 - TJ2
        TJ2MTI2 = -TI2MTJ2
        SB = TJ4 * Ti_d / (2 * TI2MTJ2**2)
        SC = (TJ6 - 3 * TJ4 * TI2) / (TI2MTJ2**3)
        SE = TI4 * Tj_d / (2 * TJ2MTI2**2)
        SF = (TI6 - 3 * TI4 * TJ2) / (TJ2MTI2**3)
        COULOMBV_tmp1 = SB - SC / dR_diff
        COULOMBV_tmp2 = SE - SF / dR_diff
        dtmp1[~mask_same] -= EXPTI * (
            (-Ti_d) * COULOMBV_tmp1 + SC / dR_diff**2
        ) + EXPTJ * ((-Tj_d) * COULOMBV_tmp2 + SF / dR_diff**2)

    dtmp1 *= KECONST  # dC_ij / dR_ij  (scalar per pair)

    # Virial: sigma_ab = (1/V) sum_{pairs} w_ij * (dC/dR) * Rab_a * Rab_b / R
    # Standard SCC:  w_ij = 0.5 * q_i * q_j   (from E = 0.5 q^T C q)
    # Shadow XL-BOMD: w_ij = 0.5 * (2q_i - n_i) * n_j  (from E = 0.5 (2q-n)^T C n)
    qi = q[neighbor_I]
    qj = q[neighbor_J]
    if n is not None:
        ni = n[neighbor_I]
        nj = n[neighbor_J]
        weight = 0.5 * (2 * qi - ni) * nj * dtmp1 / dR_mskd
    else:
        weight = 0.5 * qi * qj * dtmp1 / dR_mskd

    Vcell = torch.abs(torch.det(cell))
    sigma = torch.einsum("p,pa,pb->ab", weight, Rab_mskd, Rab_mskd) / Vcell
    return _symmetrize_stress(sigma)


def get_coulomb_stress_kspace(
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    q: torch.Tensor,
    Coulomb_acc: float,
    CALPHA: float,
    n: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reciprocal-space Ewald **metric** stress correction.

    This computes ONLY the stress from the explicit cell dependence of the
    k-space Ewald sum (volume prefactor and reciprocal-vector changes).
    The position-dependent part (structure-factor variation) is already
    captured by the force virial of the k-space Coulomb forces.

    For each reciprocal vector G in the half-sphere sum, the energy is:

        E_G = (K_e / 2) · (8π / (V G²)) · exp(-G²/4α²) · |S(G)|²

    Under cell strain ε_{αβ} with fixed fractional coords, the metric
    (non-position) derivative is:

        dE_G/dε_{αβ}|_metric = E_G × (-δ_{αβ} + 2 G_α G_β / G² + G_α G_β / (2α²))

    When *n* is provided (XL-BOMD shadow stress), the structure-factor
    product ``|S_q|²`` is replaced by ``Re[S_{2q-n}(G) · S_n*(G)]``.

    Returns stress in eV/Å³.
    """
    KECONST = 14.3996437701414
    CALPHA2 = CALPHA**2

    COULVOL = torch.abs(torch.det(cell))
    SQRTX = math.sqrt(-math.log(Coulomb_acc))
    KCUTOFF = 2 * CALPHA * SQRTX
    KCUTOFF2 = KCUTOFF**2

    cell_inv = torch.linalg.inv(cell)
    RECIPVECS = 2.0 * math.pi * cell_inv.T

    g1_norm = torch.norm(RECIPVECS[:, 0])
    g2_norm = torch.norm(RECIPVECS[:, 1])
    g3_norm = torch.norm(RECIPVECS[:, 2])
    LMAX = int(torch.ceil(KCUTOFF / g1_norm).item())
    MMAX = int(torch.ceil(KCUTOFF / g2_norm).item())
    NMAX = int(torch.ceil(KCUTOFF / g3_norm).item())

    sigma = torch.zeros((3, 3), dtype=RX.dtype, device=RX.device)
    eye = torch.eye(3, dtype=RX.dtype, device=RX.device)

    # Shadow charges: when n is provided, the k-space energy uses
    # Re[S_{2q-n}(G) S_n*(G)] instead of |S_q(G)|²
    q_eff = 2 * q - n if n is not None else q
    n_eff = n if n is not None else q

    for L in range(0, LMAX + 1):
        MMIN = 0 if L == 0 else -MMAX
        for M in range(MMIN, MMAX + 1):
            NMIN = 1 if (L == 0 and M == 0) else -NMAX
            for N in range(NMIN, NMAX + 1):
                kvec = (
                    torch.tensor([L, M, N], dtype=RX.dtype, device=RX.device)
                    @ RECIPVECS
                )
                K2 = torch.dot(kvec, kvec)
                if K2 > KCUTOFF2:
                    continue

                exp_factor = torch.exp(-K2 / (4 * CALPHA2))
                # Half-sphere prefactor (8π already includes factor of 2)
                prefactor = 8.0 * math.pi * exp_factor / (COULVOL * K2)

                # Structure factors
                dot = kvec[0] * RX + kvec[1] * RY + kvec[2] * RZ
                cos_dot = torch.cos(dot)
                sin_dot = torch.sin(dot)

                # S_product = Re[S_{q_eff}(G) · S_{n_eff}*(G)]
                #           = S_{q_eff}^cos · S_{n_eff}^cos + S_{q_eff}^sin · S_{n_eff}^sin
                S_q_cos = torch.dot(q_eff, cos_dot)
                S_q_sin = torch.dot(q_eff, sin_dot)
                S_n_cos = torch.dot(n_eff, cos_dot)
                S_n_sin = torch.dot(n_eff, sin_dot)
                S_product = S_q_cos * S_n_cos + S_q_sin * S_n_sin

                # Energy from this G: E_G = 0.5 * KECONST * prefactor * S_product
                E_G = 0.5 * KECONST * prefactor * S_product

                # Metric correction: dE_G/dε_ab = E_G * (-δ_ab + 2 k_a k_b/K² + k_a k_b/(2α²))
                kk = kvec.unsqueeze(-1) * kvec.unsqueeze(0)
                sigma += E_G * (-eye + 2.0 * kk / K2 + kk / (2.0 * CALPHA2))

    return _symmetrize_stress(sigma / COULVOL)


def get_coulomb_stress(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    q: torch.Tensor,
    Nats: int,
    const,
    Coulomb_acc: float,
    cutoff: float | None = None,
    n: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full Ewald Coulomb stress (real + k-space + on-site).

    The on-site Hubbard term E_onsite = 0.5 Σ q²U has no cell dependence
    and contributes zero stress.

    When *n* is provided, shadow XL-BOMD charge expressions are used.

    Returns the total Coulomb stress in eV/Å³.
    """
    SQRTX = math.sqrt(-math.log(Coulomb_acc))
    COULCUT_MAX = 50.0
    if cutoff is not None:
        COULCUT = min(cutoff, COULCUT_MAX)
    else:
        COULCUT = min(
            const._coulcut if hasattr(const, "_coulcut") else 10.0, COULCUT_MAX
        )
    CALPHA = SQRTX / COULCUT

    sigma_real = get_coulomb_stress_real(
        Hubbard_U,
        TYPE,
        RX,
        RY,
        RZ,
        cell,
        q,
        Nats,
        const,
        Coulomb_acc,
        CALPHA,
        COULCUT,
        n=n,
    )
    sigma_kspace = get_coulomb_stress_kspace(
        RX,
        RY,
        RZ,
        cell,
        q,
        Coulomb_acc,
        CALPHA,
        n=n,
    )

    return sigma_real + sigma_kspace


# ---------------------------------------------------------------------------
# Complete analytical stress tensor
# ---------------------------------------------------------------------------


def get_total_stress_analytical(
    structure,
    const,
    repulsive_rcut: float,
    Coulomb_acc: float | None = None,
    dftorch_params: dict | None = None,
    verbose: bool = False,
    n: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the full analytical SCC-DFTB stress tensor.

    The stress is the derivative of the total energy with respect to a
    homogeneous cell deformation at fixed fractional coordinates:

        σ_{αβ} = (1/V) ∂E_tot / ∂ε_{αβ}

    Implementation — fully explicit analytical formulas (no autograd)
    ------------------------------------------------------------------
    1. **Repulsive stress** — pair virial from short-range repulsive potential.
    2. **Band (H0) stress** — pairwise SK derivative with density weight
       ``2*(D - D0)`` (D0 is on-site so equivalent to using the full D).
    3. **Pulay stress** — pairwise S-channel derivative with weight
       ``-0.5 * SIHD`` (SIHD = 4 Z Z^T H D, the energy-weighted density
       matrix factor from the force code).
    4. **SCC-overlap stress** — pairwise S-channel derivative with weight
       ``D * factor_orb`` (SCC shift per orbital row).
    5. **Coulomb stress** — real-space pair virial + k-space metric.

    Parameters
    ----------
    n : torch.Tensor, optional
        Extrapolated (shadow) charges for XL-BOMD.  When provided, the
        SCC-overlap and Coulomb stress components use the shadow energy
        expression ``E = 0.5 (2q-n)^T C n + …`` instead of ``E = 0.5 q^T C q``,
        consistent with the forces computed by ``forces_shadow``.
        When None (default), standard SCF stress with ``q`` is computed.

    Validated against DFTB+ reference to < 2×10⁻⁷ Ha/Bohr³.

    Returns a dict with per-component and total stress tensors in eV/Å³.
    """
    cell = structure.cell
    if cell is None:
        raise ValueError("Stress tensor requires a periodic cell.")

    RX, RY, RZ = structure.RX, structure.RY, structure.RZ
    TYPE = structure.TYPE
    Nats = structure.Nats
    out: dict[str, torch.Tensor] = {}

    # --- 1. Repulsive stress (analytical pair virial, validated) ---
    if structure.stress_repulsion is None:
        from ._repulsive_spline import get_repulsion_energy

        _, _, out["repulsion"] = get_repulsion_energy(
            const.R_rep_tensor,
            const.rep_splines_tensor,
            const.close_exp_tensor,
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            repulsive_rcut,
            Nats,
            const,
            verbose=verbose,
            compute_stress=True,
        )
    else:
        out["repulsion"] = structure.stress_repulsion

    # --- 2-4. Electronic stress (band + Pulay + SCC-overlap) ---
    electronic_dict = get_electronic_stress_analytical(
        structure.H0,
        structure.H,
        structure.Z,
        structure.C,
        structure.D,
        structure.D0,
        structure.dH0,
        structure.dS,
        None,  # dCC not needed for pairwise
        structure.Hubbard_U,
        structure.q,
        RX,
        RY,
        RZ,
        cell,
        structure.e_field,
        TYPE,
        const,
        dU_dq=getattr(structure, "dU_dq", None),
        n=n,
    )
    out["band"] = electronic_dict["band_analytical"]
    out["overlap"] = electronic_dict["overlap_analytical"]

    # --- 5. Coulomb stress (real-space pair virial + k-space metric) ---
    if Coulomb_acc is None:
        Coulomb_acc = (
            dftorch_params.get("COULOMB_ACC", 1e-5) if dftorch_params else 1e-5
        )
    coulomb_cutoff = (
        dftorch_params.get("COULOMB_CUTOFF", None) if dftorch_params else None
    )

    if structure.stress_coulomb is None:
        out["coulomb"] = get_coulomb_stress(
            structure.Hubbard_U,
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            structure.q,
            Nats,
            const,
            Coulomb_acc,
            cutoff=coulomb_cutoff,
            n=n,
        )
    else:
        out["coulomb"] = structure.stress_coulomb

    # --- Total ---
    out["total"] = out["repulsion"] + out["band"] + out["overlap"] + out["coulomb"]

    return out


get_coulomb_stress_real_eager = get_coulomb_stress_real

get_coulomb_stress_real = _maybe_compile(
    get_coulomb_stress_real,
    fullgraph=False,
    dynamic=False,
)
