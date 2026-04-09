# ruff: noqa
import math
import torch
import numpy as np
import time
from typing import Optional, Tuple, Dict
from .util import CONV_FACTOR


# ── Van der Waals radii & H5 constants (mirrored from _coulomb_matrix.py) ─
_VDW_RADII_PM: Dict[int, int] = {
    1: 120,
    2: 140,
    3: 182,
    4: 153,
    5: 192,
    6: 170,
    7: 155,
    8: 152,
    9: 147,
    10: 154,
    11: 227,
    12: 173,
    13: 184,
    14: 210,
    15: 180,
    16: 180,
    17: 175,
    18: 188,
    19: 275,
    20: 231,
    28: 163,
    29: 140,
    30: 139,
    31: 187,
    32: 211,
    33: 185,
    34: 190,
    35: 185,
    36: 202,
    46: 163,
    47: 172,
    48: 158,
    49: 193,
    50: 217,
    51: 206,
    52: 206,
    53: 198,
    54: 216,
    79: 166,
    80: 155,
    92: 186,
}
_H5_DEFAULT_SCALING: Dict[int, float] = {7: 0.18, 8: 0.06, 16: 0.21}
_GAUSS_WIDTH_FACTOR = 0.5 / math.sqrt(2.0 * math.log(2.0))


@torch.compile
def ewald_real(
    nbr_inds: torch.Tensor,
    nbr_diff_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int,
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the real-space contribution to the Ewald summation.

    This function calculates the electrostatic interaction energy in the real-space
    portion of the Ewald summation. It also optionally computes forces and derivatives
    with respect to charge if `calculate_forces` or `calculate_dq` are set.

    Args:
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `N` is the number of local atoms.
            - `K` is the maximum number of neighbors per atom.
        nbr_diff_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)`, where:
            - `3` represents the x, y, and z components of the displacement.
            - `N` is the number of local atoms.
            - `K` is the number of neighbors per atom.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Real-space energy contribution (scalar).
            - **(torch.Tensor, shape `(3, N)`)** Forces on atoms if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(N,)`)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """
    # TODO: finalize DUMMY_ATOM_IND
    DUMMY_NBR_IND = -1
    N = len(charges)
    q_sq = charges[nbr_inds] * charges[:, None]
    qq_over_dist = q_sq / nbr_dists
    qq_over_dist = qq_over_dist * ((nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff))
    erfc = torch.erfc(alpha * nbr_dists)
    res = erfc * qq_over_dist
    # de_dq = erfc * charges[nbr_inds] * (nbr_inds != DUMMY_NBR_IND) / nbr_dists
    # de_dq = torch.sum(de_dq, dim=1)
    if calculate_forces:
        nbr_dists_sq = nbr_dists**2
        f = qq_over_dist * (
            erfc / nbr_dists_sq
            + (2.0 * alpha / math.sqrt(torch.pi))
            * torch.exp(-alpha * alpha * nbr_dists_sq)
            / nbr_dists
        )
        f = -1.0 * torch.sum(f[None, ...] * nbr_diff_vecs, dim=2)
    else:
        f = None

    if calculate_dq:
        de_dq = erfc * charges[nbr_inds] * (nbr_inds != DUMMY_NBR_IND) / nbr_dists
        de_dq = torch.sum(de_dq, dim=1)
    else:
        de_dq = None

    return torch.sum(res) / 2.0, f, de_dq


@torch.compile
def ewald_real_screening(
    nbr_inds,
    nbr_diff_vecs,
    nbr_dists,
    charges,
    hubbard_u,
    atomtypes,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int,
):
    """
    Computes the real-space contribution with the Hubbard-U screening correction to the Ewald summation.

    This function calculates the electrostatic interaction energy in the real-space
    portion of the Ewald summation. It also optionally computes forces and derivatives
    with respect to charge if `calculate_forces` or `calculate_dq` are set.

    Args:
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `N` is the number of local atoms.
            - `K` is the maximum number of neighbors per atom.
        nbr_diff_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)`, where:
            - `3` represents the x, y, and z components of the displacement.
            - `N` is the number of local atoms.
            - `K` is the number of neighbors per atom.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        hubbard_u (torch.Tensor): Hubbard U values for each atom. Shape: `(N,)`.
        atomtypes (torch.Tensor): Atomic types for each atom. Shape: `(N,)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Real-space energy contribution (scalar).
            - **(torch.Tensor, shape `(3, N)`)** Forces on atoms if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(N,)`)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """
    # TODO: finalize DUMMY_ATOM_IND
    KECONST = 14.3996437701414
    device = nbr_dists.device
    dtype = nbr_dists.dtype

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    DUMMY_NBR_IND = -1
    # symbols = torch.Tensor(sy.symbols)[atomtypes]
    mask = (nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff)
    same_element_mask = mask & (
        atomtypes.unsqueeze(1) == atomtypes[nbr_inds]
    )  # (Nr_atoms, Max_Nr_Neigh)
    different_element_mask = mask & ~same_element_mask

    TFACT = 16.0 / (5.0 * KECONST)
    # TI = TFACT * U.unsqueeze(1) * mask # (Nr_atoms, Max_Nr_Neigh)
    TI = torch.where(mask, TFACT * hubbard_u.unsqueeze(1) * mask, one)
    TI2 = TI * TI
    TI3 = TI2 * TI
    TI4 = TI2 * TI2
    TI6 = TI4 * TI2

    SSA = TI
    SSB = TI3 / 48.0
    SSC = 3.0 * TI2 / 16.0
    SSD = 11.0 * TI / 16.0
    SSE = 1.0

    MAGR = torch.where(mask, nbr_dists, one)
    MAGR2 = MAGR * MAGR
    Z = abs(alpha * MAGR)
    NUMREP_ERFC = torch.special.erfc(Z)

    J0 = torch.where(mask, NUMREP_ERFC / MAGR, zero)

    EXPTI = torch.exp(-TI * MAGR)

    J0[same_element_mask] = (
        J0[same_element_mask]
        - (EXPTI * (SSB * MAGR2 + SSC * MAGR + SSD + SSE / MAGR))[same_element_mask]
    )

    # TJ = TFACT * U[nbr_inds] * different_element_mask     # (Nr_atoms, Max_Nr_Neigh)
    TJ = torch.where(
        different_element_mask,
        TFACT * hubbard_u[nbr_inds] * different_element_mask,
        one,
    )
    TJ2 = TJ * TJ
    TJ4 = TJ2 * TJ2
    TJ6 = TJ4 * TJ2
    EXPTJ = torch.exp(-TJ * MAGR)
    TI2MTJ2 = TI2 - TJ2
    TI2MTJ2 = torch.where(different_element_mask, TI2MTJ2, one)
    SA = TI
    SB = EXPTI * TJ4 * TI / 2.0 / TI2MTJ2 / TI2MTJ2
    SC = EXPTI * (TJ6 - 3.0 * TJ4 * TI2) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    SD = TJ
    SE = EXPTJ * TI4 * TJ / 2.0 / TI2MTJ2 / TI2MTJ2
    SF = EXPTJ * (-(TI6 - 3.0 * TI4 * TJ2)) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    J0[different_element_mask] = (
        J0[different_element_mask]
        - (1.0 * (SB - SC / MAGR) + 1.0 * (SE - SF / MAGR))[different_element_mask]
    )

    energy = charges[:, None] * J0 * charges[nbr_inds]

    if calculate_forces:
        nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 2).contiguous()
        nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 1).contiguous()
        alpha2 = alpha * alpha
        DC = torch.where(
            mask.unsqueeze(2), nbr_diff_vecs / nbr_dists.unsqueeze(2), zero
        )
        CA = torch.where(mask, NUMREP_ERFC / MAGR, zero)
        CA = CA + 2.0 * alpha * torch.exp(-alpha2 * MAGR2) / math.sqrt(math.pi)
        FORCE = -torch.sum(
            (
                charges[:, None]
                * charges[nbr_inds]
                * torch.where(mask, CA / MAGR, zero)
            ).unsqueeze(2)
            * DC
            * mask.unsqueeze(2),
            dim=1,
        )

        FORCE = FORCE + torch.sum(
            (
                (charges[:, None] * charges[nbr_inds] * EXPTI)
                * (
                    (
                        torch.where(same_element_mask, SSE / MAGR2, zero)
                        - 2.0 * SSB * MAGR
                        - SSC
                    )
                    + SSA
                    * (
                        SSB * MAGR2
                        + SSC * MAGR
                        + SSD
                        + torch.where(same_element_mask, SSE / MAGR, zero)
                    )
                )
            ).unsqueeze(2)
            * DC
            * same_element_mask.unsqueeze(2),
            dim=1,
        )
        FORCE = FORCE + torch.sum(
            (
                charges[:, None]
                * charges[nbr_inds]
                * (
                    (
                        1.0
                        * (
                            SA
                            * (
                                SB
                                - torch.where(different_element_mask, SC / MAGR, zero)
                            )
                            - torch.where(different_element_mask, SC / MAGR2, zero)
                        )
                    )
                    + (
                        1.0
                        * (
                            SD
                            * (
                                SE
                                - torch.where(different_element_mask, SF / MAGR, zero)
                            )
                            - torch.where(different_element_mask, SF / MAGR2, zero)
                        )
                    )
                )
            ).unsqueeze(2)
            * DC
            * different_element_mask.unsqueeze(2),
            dim=1,
        )

    else:
        FORCE = None

    if calculate_dq:
        COULOMBV = torch.sum(J0 * charges[nbr_inds], dim=1)
    else:
        COULOMBV = None

    return torch.sum(energy) / 2.0, FORCE, COULOMBV


def ewald_real_screening_stress(
    nbr_inds: torch.Tensor,
    nbr_diff_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    hubbard_u: torch.Tensor,
    atomtypes: torch.Tensor,
    alpha: float,
    cutoff: float,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Real-space Ewald Coulomb stress with Hubbard-U screening.

    Computes the virial contribution to the stress tensor from the real-space
    Ewald sum, including the short-range damping corrections for same-element
    and different-element pairs (matching ``ewald_real_screening``).

    Parameters
    ----------
    nbr_inds : (N, K) neighbor indices (-1 = padding)
    nbr_diff_vecs : (3, N, K) displacement vectors R_j - R_i
    nbr_dists : (N, K) pair distances
    charges : (N,) atomic charges
    hubbard_u : (N,) Hubbard U per atom
    atomtypes : (N,) element types
    alpha : Ewald splitting parameter
    cutoff : real-space cutoff
    cell : (3, 3) lattice vectors

    Returns
    -------
    sigma : (3, 3) real-space Coulomb stress in the *same internal units*
            as the real-space energy (i.e. before CONV_FACTOR multiplication).
            Caller must multiply by CONV_FACTOR to get eV/Å³.
    """
    KECONST = 14.3996437701414
    device = nbr_dists.device
    dtype = nbr_dists.dtype

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    DUMMY_NBR_IND = -1
    mask = (nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff)
    same_element_mask = mask & (atomtypes.unsqueeze(1) == atomtypes[nbr_inds])
    different_element_mask = mask & ~same_element_mask

    TFACT = 16.0 / (5.0 * KECONST)
    TI = torch.where(mask, TFACT * hubbard_u.unsqueeze(1) * mask, one)
    TI2 = TI * TI
    TI3 = TI2 * TI
    TI4 = TI2 * TI2
    TI6 = TI4 * TI2

    SSA = TI
    SSB = TI3 / 48.0
    SSC = 3.0 * TI2 / 16.0
    SSD = 11.0 * TI / 16.0
    SSE = 1.0

    MAGR = torch.where(mask, nbr_dists, one)
    MAGR2 = MAGR * MAGR
    NUMREP_ERFC = torch.special.erfc(abs(alpha * MAGR))
    EXPTI = torch.exp(-TI * MAGR)

    # --- Compute dJ0/dR (radial derivative of the screened potential) ---
    alpha2 = alpha * alpha
    # Ewald part: d(erfc(aR)/R)/dR = -(erfc(aR)/R + 2a*exp(-a²R²)/√π) / R
    CA = NUMREP_ERFC / MAGR + 2.0 * alpha * torch.exp(-alpha2 * MAGR2) / math.sqrt(
        math.pi
    )
    dJ0_dR = torch.where(mask, -CA / MAGR, zero)

    # Same-element damping derivative
    dJ0_dR = dJ0_dR + torch.where(
        same_element_mask,
        EXPTI
        * (
            (torch.where(same_element_mask, SSE / MAGR2, zero) - 2.0 * SSB * MAGR - SSC)
            + SSA
            * (
                SSB * MAGR2
                + SSC * MAGR
                + SSD
                + torch.where(same_element_mask, SSE / MAGR, zero)
            )
        ),
        zero,
    )

    # Different-element damping derivative
    TJ = torch.where(
        different_element_mask,
        TFACT * hubbard_u[nbr_inds] * different_element_mask,
        one,
    )
    TJ2 = TJ * TJ
    TJ4 = TJ2 * TJ2
    TJ6 = TJ4 * TJ2
    EXPTJ = torch.exp(-TJ * MAGR)
    TI2MTJ2 = torch.where(different_element_mask, TI2 - TJ2, one)
    SA = TI
    SB = EXPTI * TJ4 * TI / 2.0 / TI2MTJ2 / TI2MTJ2
    SC = EXPTI * (TJ6 - 3.0 * TJ4 * TI2) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    SD = TJ
    SE = EXPTJ * TI4 * TJ / 2.0 / TI2MTJ2 / TI2MTJ2
    SF = EXPTJ * (-(TI6 - 3.0 * TI4 * TJ2)) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2

    dJ0_dR = dJ0_dR + torch.where(
        different_element_mask,
        (
            SA * (SB - torch.where(different_element_mask, SC / MAGR, zero))
            - torch.where(different_element_mask, SC / MAGR2, zero)
        )
        + (
            SD * (SE - torch.where(different_element_mask, SF / MAGR, zero))
            - torch.where(different_element_mask, SF / MAGR2, zero)
        ),
        zero,
    )

    # --- Virial stress: σ_αβ = (1/2V) Σ_{ij} q_i q_j (dJ0/dR) R_α R_β / R ---
    # Ensure nbr_diff_vecs is (N, K, 3) for the outer product
    if nbr_diff_vecs.shape[0] == 3:
        Rab = nbr_diff_vecs.permute(1, 2, 0)  # (N, K, 3)
    else:
        Rab = nbr_diff_vecs  # already (N, K, 3)

    qq = charges.unsqueeze(1) * charges[nbr_inds]  # (N, K)
    weight = qq * dJ0_dR / MAGR * mask  # (N, K)

    # Outer product sum: σ_αβ = Σ weight * R_α * R_β
    # Rab shape (N, K, 3), weight shape (N, K)
    Vcell = torch.abs(torch.det(cell))
    sigma = torch.einsum("ij,ija,ijb->ab", weight, Rab, Rab) / (2.0 * Vcell)
    sigma = 0.5 * (sigma + sigma.T)
    return sigma

    sigma = torch.einsum("ij,ija,ijb->ab", weight, Rab, Rab) / (2.0 * Vcell)
    sigma = 0.5 * (sigma + sigma.T)
    return sigma


# ═══════════════════════════════════════════════════════════════════════════
# H-damping / H5 correction to Ewald real-space screening
# ═══════════════════════════════════════════════════════════════════════════


def h_damp_h5_correction(
    nbr_inds: torch.Tensor,
    nbr_diff_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    hubbard_u: torch.Tensor,
    atomtypes: torch.Tensor,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int,
    h_damp_exp: Optional[float] = None,
    h5_params: Optional[Dict] = None,
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute the energy / force / CoulPot correction from H-damping or H5.

    This function mirrors ``ewald_real_screening`` exactly, computing J0
    twice — once without damping (undamped, matching what ``ewald_real_screening``
    already computed) and once with damping — and returns the *difference*
    (damped − undamped) as an additive correction.

    Returns (energy_corr, force_corr, dq_corr) already multiplied by
    ``CONV_FACTOR``.
    """
    KECONST = 14.3996437701414
    EV_TO_HA = 1.0 / 27.211386245988
    ANG_TO_BOHR = 1.0 / 0.52917721067
    device = nbr_dists.device
    dtype = nbr_dists.dtype

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    DUMMY_NBR_IND = -1
    mask = (nbr_inds != DUMMY_NBR_IND) & (nbr_dists <= cutoff)
    MAGR = torch.where(mask, nbr_dists, one)
    MAGR2 = MAGR * MAGR

    TFACT = 16.0 / (5.0 * KECONST)

    # ── Replicate the full ewald_real_screening computation ──────────────
    # We rebuild J0 (undamped) and J0_damped, then return the difference.
    # This guarantees bit-for-bit consistency with ewald_real_screening.
    same_element_mask = mask & (atomtypes.unsqueeze(1) == atomtypes[nbr_inds])
    different_element_mask = mask & ~same_element_mask

    TI = torch.where(mask, TFACT * hubbard_u.unsqueeze(1) * mask, one)
    TI2 = TI * TI
    TI3 = TI2 * TI
    TI4 = TI2 * TI2
    TI6 = TI4 * TI2

    SSA = TI
    SSB = TI3 / 48.0
    SSC = 3.0 * TI2 / 16.0
    SSD = 11.0 * TI / 16.0
    SSE = 1.0

    EXPTI = torch.exp(-TI * MAGR)

    # ── J0_undamped (same as ewald_real_screening) ───────────────────────
    # We only need the screening part (γ subtracted from erfc/r).
    # Since we're computing the *difference*, the erfc/r terms cancel.
    # So we only need the γ values and their force contributions.

    # Same-element γ
    gamma_same = EXPTI * (SSB * MAGR2 + SSC * MAGR + SSD + SSE / MAGR)

    # Different-element γ
    TJ = torch.where(
        different_element_mask,
        TFACT * hubbard_u[nbr_inds] * different_element_mask,
        one,
    )
    TJ2 = TJ * TJ
    TJ4 = TJ2 * TJ2
    TJ6 = TJ4 * TJ2
    EXPTJ = torch.exp(-TJ * MAGR)
    TI2MTJ2 = TI2 - TJ2
    TI2MTJ2 = torch.where(different_element_mask, TI2MTJ2, one)
    SA = TI
    SB = EXPTI * TJ4 * TI / 2.0 / TI2MTJ2 / TI2MTJ2
    SC = EXPTI * (TJ6 - 3.0 * TJ4 * TI2) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    SD = TJ
    SE = EXPTJ * TI4 * TJ / 2.0 / TI2MTJ2 / TI2MTJ2
    SF = EXPTJ * (-(TI6 - 3.0 * TI4 * TJ2)) / TI2MTJ2 / TI2MTJ2 / TI2MTJ2
    gamma_diff = (SB - SC / MAGR) + (SE - SF / MAGR)

    # ── H-damping: identify affected pairs ───────────────────────────────
    is_H_I = atomtypes == 1
    is_H_J_lookup = is_H_I[nbr_inds.clamp(min=0)]
    is_H_I_exp = is_H_I.unsqueeze(1).expand_as(nbr_inds)

    # ── Build J0_undamped and J0_damped (screening part only) ────────────
    # J0 = erfc/r - γ   ⟹  correction = J0_damped - J0_undamped = γ_undamped - γ_damped
    # (erfc/r cancels in the difference)
    J0_undamped = torch.zeros_like(MAGR)
    J0_undamped[same_element_mask] = gamma_same[same_element_mask]
    J0_undamped[different_element_mask] = gamma_diff[different_element_mask]

    J0_damped = J0_undamped.clone()

    _apply_h5 = False

    if h_damp_exp is not None:
        h_mask = mask & (is_H_I_exp | is_H_J_lookup)
        if h_mask.any():
            Ui_au = hubbard_u.unsqueeze(1).expand_as(nbr_inds)[h_mask] * EV_TO_HA
            Uj_au = hubbard_u[nbr_inds.clamp(min=0)][h_mask] * EV_TO_HA
            r_au = MAGR[h_mask] * ANG_TO_BOHR
            rTmp = -((0.5 * (Ui_au + Uj_au)) ** h_damp_exp)
            D = torch.exp(rTmp * r_au**2)
            # γ_damped = γ * D
            J0_damped[h_mask] = J0_undamped[h_mask] * D

    elif h5_params is not None:
        _h5_r_scaling = h5_params.get("r_scaling", 0.714)
        _h5_w_scaling = h5_params.get("w_scaling", 0.25)
        _h5_scaling = h5_params.get("h5_scaling", _H5_DEFAULT_SCALING)
        _vdw_H_ang = _VDW_RADII_PM.get(1, 120) * 0.01

        h5_mask_all = mask & (is_H_I_exp ^ is_H_J_lookup)
        if h5_mask_all.any():
            heavy_Z = torch.where(
                is_H_I_exp[h5_mask_all],
                atomtypes[nbr_inds.clamp(min=0)][h5_mask_all],
                atomtypes.unsqueeze(1).expand_as(nbr_inds)[h5_mask_all],
            )
            k_XH_all = torch.zeros(int(h5_mask_all.sum()), dtype=dtype, device=device)
            sumVdW_all = torch.zeros_like(k_XH_all)
            for z_heavy, k_val in _h5_scaling.items():
                z_mask = heavy_Z == z_heavy
                if z_mask.any():
                    vdw_heavy_ang = _VDW_RADII_PM.get(z_heavy, -1) * 0.01
                    if vdw_heavy_ang > 0:
                        k_XH_all[z_mask] = k_val
                        sumVdW_all[z_mask] = _vdw_H_ang + vdw_heavy_ang
            active = k_XH_all > 0.0
            if active.any():
                _apply_h5 = True
                r_h5 = MAGR[h5_mask_all][active]
                k_h5 = k_XH_all[active]
                svdw = sumVdW_all[active]
                r0 = _h5_r_scaling * svdw
                cc = _h5_w_scaling * svdw * _GAUSS_WIDTH_FACTOR
                gauss = k_h5 * torch.exp(-0.5 * (r_h5 - r0) ** 2 / cc**2)

                g = J0_undamped[h5_mask_all][active]
                # γ_H5 = γ*(1+G) - G/r
                g_damped = g * (1.0 + gauss) - gauss / r_h5

                idx_full = torch.where(h5_mask_all.flatten())[0]
                idx_active = idx_full[active]
                J0_damped.view(-1)[idx_active] = g_damped

    # ── DeltaJ0 = γ_undamped - γ_damped ─────────────────────────────────
    # ewald_real_screening computed: J0 = erfc/r - γ_undamped
    # correct J0 should be:          J0 = erfc/r - γ_damped
    # correction = (erfc/r - γ_damped) - (erfc/r - γ_undamped) = γ_undamped - γ_damped
    DeltaJ0 = J0_undamped - J0_damped

    # ── Energy correction: (1/2) Σ q_i ΔJ0 q_j ─────────────────────────
    energy_corr = (
        torch.sum(charges[:, None] * DeltaJ0 * charges[nbr_inds.clamp(min=0)] * mask)
        / 2.0
    )

    # ── CoulPot (dq) correction ──────────────────────────────────────────
    dq_corr = None
    if calculate_dq:
        dq_corr = torch.sum(DeltaJ0 * charges[nbr_inds.clamp(min=0)] * mask, dim=1)

    # ── Force correction ─────────────────────────────────────────────────
    # Mirror ewald_real_screening force assembly exactly.
    # ewald_real_screening computes three FORCE contributions:
    #   1. erfc/r part:      FORCE = -sum(qq * CA/MAGR * DC)
    #   2. same-elem γ:      FORCE += +sum(qq * [-dγ/dr] * DC)
    #   3. diff-elem γ:      FORCE += +sum(qq * [-dγ/dr] * DC)
    # The erfc/r part cancels in the difference.  So the correction force is:
    #   F_corr = [γ_undamped force terms] - [γ_damped force terms]
    # which equals sum(qq * (-dγ_undamped/dr + dγ_damped/dr) * DC)
    force_corr = None
    if calculate_forces:
        # Ensure nbr_diff_vecs is (N, K, 3)
        if nbr_diff_vecs.shape[0] == 3:
            nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 2).contiguous()
            nbr_diff_vecs = torch.transpose(nbr_diff_vecs, 0, 1).contiguous()
        DC = torch.where(mask.unsqueeze(2), nbr_diff_vecs / MAGR.unsqueeze(2), zero)

        # --- Undamped force (same as ewald_real_screening terms 2+3) ---
        # Same-element screening force contribution (from ewald_real_screening):
        #   +sum(qq * EXPTI * [(1/r² - 2*SSB*r - SSC) + SSA*(SSB*r²+SSC*r+SSD+1/r)] * DC)
        FORCE_undamped = torch.sum(
            (
                (charges[:, None] * charges[nbr_inds.clamp(min=0)] * EXPTI)
                * (
                    (
                        torch.where(same_element_mask, SSE / MAGR2, zero)
                        - 2.0 * SSB * MAGR
                        - SSC
                    )
                    + SSA
                    * (
                        SSB * MAGR2
                        + SSC * MAGR
                        + SSD
                        + torch.where(same_element_mask, SSE / MAGR, zero)
                    )
                )
            ).unsqueeze(2)
            * DC
            * same_element_mask.unsqueeze(2),
            dim=1,
        )
        # Different-element screening force contribution:
        FORCE_undamped = FORCE_undamped + torch.sum(
            (
                charges[:, None]
                * charges[nbr_inds.clamp(min=0)]
                * (
                    (
                        SA * (SB - torch.where(different_element_mask, SC / MAGR, zero))
                        - torch.where(different_element_mask, SC / MAGR2, zero)
                    )
                    + (
                        SD * (SE - torch.where(different_element_mask, SF / MAGR, zero))
                        - torch.where(different_element_mask, SF / MAGR2, zero)
                    )
                )
            ).unsqueeze(2)
            * DC
            * different_element_mask.unsqueeze(2),
            dim=1,
        )

        # --- Damped force: recompute screening with damping applied ---
        if h_damp_exp is not None:
            h_mask = mask & (is_H_I_exp | is_H_J_lookup)
            Ui_au_full = hubbard_u.unsqueeze(1).expand_as(nbr_inds) * EV_TO_HA
            Uj_au_full = hubbard_u[nbr_inds.clamp(min=0)] * EV_TO_HA
            r_au_full = MAGR * ANG_TO_BOHR
            rTmp_full = -((0.5 * (Ui_au_full + Uj_au_full)) ** h_damp_exp)
            D_full = torch.where(h_mask, torch.exp(rTmp_full * r_au_full**2), one)
            Dprime_ang_full = torch.where(
                h_mask,
                2.0 * rTmp_full * r_au_full * D_full * ANG_TO_BOHR,
                zero,
            )  # dD/dr in 1/Å

            # Damped same-element: γ_d = γ*D, so the force term becomes:
            #   qq * [γ'*D + γ*D'] * DC   (product rule on γ*D)
            # But ewald_real_screening computes it using the expanded form.
            # We mimic: for damped, the EXPTI*tmp becomes EXPTI*tmp*D,
            # and its derivative via product rule.
            # Simplest: F_damped_same = F_undamped_same * D + qq * γ * D' * DC
            #
            # Actually, to be perfectly safe, we compute:
            #   FORCE_damped_same = sum(qq * (-dγ_d/dr) * DC)
            # where γ_d = γ*D, so -dγ_d/dr = -(γ'*D + γ*D') = -γ'*D - γ*D'
            # The undamped term was: -dγ/dr = -γ'
            # Factor: -dγ_d/dr = (-dγ/dr)*D + (-γ)*D' = (-dγ/dr)*D - γ*D'

            # For same-element screening force, the "(-dγ/dr)" part is what
            # ewald_real_screening computes.  We can factor:
            #   F_damped = F_undamped * D  +  sum(qq * (-γ) * D' * DC)

            FORCE_damped_same = torch.sum(
                (
                    (charges[:, None] * charges[nbr_inds.clamp(min=0)] * EXPTI)
                    * (
                        (
                            torch.where(same_element_mask, SSE / MAGR2, zero)
                            - 2.0 * SSB * MAGR
                            - SSC
                        )
                        + SSA
                        * (
                            SSB * MAGR2
                            + SSC * MAGR
                            + SSD
                            + torch.where(same_element_mask, SSE / MAGR, zero)
                        )
                    )
                    * torch.where(h_mask & same_element_mask, D_full, one)
                ).unsqueeze(2)
                * DC
                * same_element_mask.unsqueeze(2),
                dim=1,
            )
            # Add the γ * D' contribution:
            FORCE_damped_same = FORCE_damped_same + torch.sum(
                (
                    charges[:, None]
                    * charges[nbr_inds.clamp(min=0)]
                    * (-gamma_same)
                    * torch.where(h_mask, Dprime_ang_full, zero)
                ).unsqueeze(2)
                * DC
                * same_element_mask.unsqueeze(2),
                dim=1,
            )

            # For different-element: same approach
            FORCE_damped_diff = torch.sum(
                (
                    charges[:, None]
                    * charges[nbr_inds.clamp(min=0)]
                    * (
                        (
                            SA
                            * (
                                SB
                                - torch.where(different_element_mask, SC / MAGR, zero)
                            )
                            - torch.where(different_element_mask, SC / MAGR2, zero)
                        )
                        + (
                            SD
                            * (
                                SE
                                - torch.where(different_element_mask, SF / MAGR, zero)
                            )
                            - torch.where(different_element_mask, SF / MAGR2, zero)
                        )
                    )
                    * torch.where(h_mask & different_element_mask, D_full, one)
                ).unsqueeze(2)
                * DC
                * different_element_mask.unsqueeze(2),
                dim=1,
            )
            FORCE_damped_diff = FORCE_damped_diff + torch.sum(
                (
                    charges[:, None]
                    * charges[nbr_inds.clamp(min=0)]
                    * (-gamma_diff)
                    * torch.where(h_mask, Dprime_ang_full, zero)
                ).unsqueeze(2)
                * DC
                * different_element_mask.unsqueeze(2),
                dim=1,
            )

            FORCE_damped = FORCE_damped_same + FORCE_damped_diff

        elif _apply_h5:
            # H5: γ_H5 = γ*(1+G) - G/r
            # dγ_H5/dr = γ'*(1+G) + γ*G' - (G'/r - G/r²)
            # -dγ_H5/dr = -γ'*(1+G) - γ*G' + G'/r - G/r²
            # The difference vs undamped (-dγ/dr = -γ'):
            #   Δ(-dγ/dr) = [-γ'*(1+G) - γ*G' + G'/r - G/r²] - [-γ']
            #             = -γ'*G - γ*G' + G'/r - G/r²
            # This is complex; for safety we compute FORCE_damped from scratch

            # Build G and G' as full (N,K) arrays, zero where not active
            G_full = torch.zeros_like(MAGR)
            Gp_full = torch.zeros_like(MAGR)
            if _apply_h5:
                # Reconstruct from saved h5 data
                _h5_r_scaling2 = h5_params.get("r_scaling", 0.714)
                _h5_w_scaling2 = h5_params.get("w_scaling", 0.25)
                _h5_scaling2 = h5_params.get("h5_scaling", _H5_DEFAULT_SCALING)
                _vdw_H_ang2 = _VDW_RADII_PM.get(1, 120) * 0.01

                h5_mask_all2 = mask & (is_H_I_exp ^ is_H_J_lookup)
                heavy_Z2 = torch.where(
                    is_H_I_exp[h5_mask_all2],
                    atomtypes[nbr_inds.clamp(min=0)][h5_mask_all2],
                    atomtypes.unsqueeze(1).expand_as(nbr_inds)[h5_mask_all2],
                )
                k_XH2 = torch.zeros(int(h5_mask_all2.sum()), dtype=dtype, device=device)
                svdw2 = torch.zeros_like(k_XH2)
                for z_h, k_v in _h5_scaling2.items():
                    zm = heavy_Z2 == z_h
                    if zm.any():
                        vdw_h = _VDW_RADII_PM.get(z_h, -1) * 0.01
                        if vdw_h > 0:
                            k_XH2[zm] = k_v
                            svdw2[zm] = _vdw_H_ang2 + vdw_h
                active2 = k_XH2 > 0.0
                if active2.any():
                    r_h5_2 = MAGR[h5_mask_all2][active2]
                    r0_2 = _h5_r_scaling2 * svdw2[active2]
                    cc_2 = _h5_w_scaling2 * svdw2[active2] * _GAUSS_WIDTH_FACTOR
                    gauss2 = k_XH2[active2] * torch.exp(
                        -0.5 * (r_h5_2 - r0_2) ** 2 / cc_2**2
                    )
                    dgauss2 = -gauss2 * (r_h5_2 - r0_2) / cc_2**2

                    idx_f = torch.where(h5_mask_all2.flatten())[0]
                    idx_a = idx_f[active2]
                    G_full.view(-1)[idx_a] = gauss2
                    Gp_full.view(-1)[idx_a] = dgauss2

            # h5_active_mask marks pairs where G_full != 0
            h5_active_mask = G_full.abs() > 0.0

            # For H5 pairs: force term is -dγ_H5/dr * DC
            #   = [-γ'*(1+G) - γ*G' + G'/r - G/r²] * DC
            # For non-H5 pairs: force term is -dγ/dr * DC (same as undamped)
            # So the damped force = undamped force (for non-H5) + H5 force (for H5)

            # Same-element: no H5 (H5 is exactly-one-H → always diff element)
            FORCE_damped = FORCE_undamped.clone()
            # Note: same-element is not affected by H5.

            # Different-element H5 correction:
            # extra_force = sum(qq * [-γ'*G - γ*G' + G'/r - G/r²] * DC) for H5 pairs
            h5_diff_mask = h5_active_mask & different_element_mask
            if h5_diff_mask.any():
                # Undamped -dγ/dr for diff element is what's already in FORCE_undamped
                # We need to ADD the extra H5 terms:
                #   undamped: -γ'  →  damped: -γ'*(1+G) - γ*G' + G'/r - G/r²
                #   extra = -γ'*G - γ*G' + G'/r - G/r²
                FORCE_h5_extra = torch.sum(
                    (
                        charges[:, None]
                        * charges[nbr_inds.clamp(min=0)]
                        * (
                            # The "undamped" diff-elem force kernel = (-dγ/dr)
                            # We need extra = (-dγ/dr)*G + (-γ)*G' + G'/r - G/r²
                            (
                                (
                                    SA
                                    * (
                                        SB
                                        - torch.where(
                                            different_element_mask, SC / MAGR, zero
                                        )
                                    )
                                    - torch.where(
                                        different_element_mask, SC / MAGR2, zero
                                    )
                                )
                                + (
                                    SD
                                    * (
                                        SE
                                        - torch.where(
                                            different_element_mask, SF / MAGR, zero
                                        )
                                    )
                                    - torch.where(
                                        different_element_mask, SF / MAGR2, zero
                                    )
                                )
                            )
                            * G_full
                            + (-gamma_diff) * Gp_full
                            + torch.where(
                                h5_diff_mask, Gp_full / MAGR - G_full / MAGR2, zero
                            )
                        )
                    ).unsqueeze(2)
                    * DC
                    * h5_diff_mask.unsqueeze(2),
                    dim=1,
                )
                FORCE_damped = FORCE_damped + FORCE_h5_extra
        else:
            FORCE_damped = FORCE_undamped

        # force_corr = FORCE_undamped - FORCE_damped
        # In ewald_real_screening: FORCE += screening terms (positive addition)
        # The screening terms are (-dγ/dr) summed.  The correction should
        # add the undamped screening and subtract the damped screening.
        # undamped screening was already applied → we correct by (damped - undamped)
        # Wait — ewald_real_screening already added +(-dγ_undamped/dr).
        # We want it to have been +(-dγ_damped/dr).
        # So correction = [+(-dγ_damped/dr)] - [+(-dγ_undamped/dr)] = FORCE_damped - FORCE_undamped
        force_corr = (FORCE_damped - FORCE_undamped).T  # (N,3) → (3,N)

    # Apply CONV_FACTOR (same as calculate_PME_ewald does for the base terms)
    energy_corr = energy_corr * CONV_FACTOR
    if dq_corr is not None:
        dq_corr = dq_corr * CONV_FACTOR
    if force_corr is not None:
        force_corr = force_corr * CONV_FACTOR

    return energy_corr, force_corr, dq_corr


def ewald_real_matrix(
    my_inds, nbr_inds, nbr_diff_vecs, nbr_dists, charges, alpha: float
):
    # TODO: finalize DUMMY_ATOM_IND
    DUMMY_NBR_IND = -1
    N = len(charges)
    q_sq = charges[nbr_inds] * charges[my_inds, None]
    qq_over_dist = q_sq / nbr_dists
    qq_over_dist = qq_over_dist * (nbr_inds != DUMMY_NBR_IND)
    erfc = torch.erfc(alpha * nbr_dists)
    res = erfc * qq_over_dist
    A = torch.vmap(
        lambda j: torch.vmap(
            lambda sub_res, sub_inds: torch.sum(sub_res * (sub_inds == j))
        )(res, nbr_inds)
    )(torch.arange(N))
    # de/dq = erfc * charges[nbr_inds]/dist (double check, this is needed for the solver)
    return A


def ewald_kspace_matrix():
    pass


def ewald_self_energy(
    charges: torch.Tensor, alpha: float, calculate_dq: int = 0
) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Computes the self-energy contribution in the Ewald summation.

    The self-energy term accounts for the interaction of each charge with its own
    periodic images in an Ewald summation.

    Args:
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`, where `N` is the number of atoms.
        alpha (float): Ewald screening parameter (scalar).
        calculate_dq (int, optional): Flag to compute charge derivatives.
            - `1`: Compute derivatives (`dq`).
            - `0`: Do not compute (`dq` is `None`).
            Defaults to `0`.

    Returns:
        Tuple[float, Optional[torch.Tensor]]:
            - Self-energy contribution (scalar, `float`).
            - Charge derivatives (`torch.Tensor` of shape `(N,)`) if `calculate_dq` is enabled, otherwise `None`.
    """
    en = -1.0 * alpha * torch.sum(charges**2) / math.sqrt(torch.pi)
    dq = None
    if calculate_dq == 1:
        dq = -2.0 * alpha * charges / math.sqrt(torch.pi)
    return en, dq


@torch.compile
def ewald_kspace_part1(positions, charges, kvecs):
    """
    Part 1 of the ewald sum. Calculate intermediate values to share with other processes
    for reduction.
    """
    # mmul is M x N, M: # kvectors and N: number of (local) atoms
    mmul = kvecs @ positions
    r_vals = torch.cos(mmul) * charges
    i_vals = torch.sin(mmul) * charges
    return r_vals, i_vals


@torch.compile
def ewald_kspace_part2(
    sum_r: torch.Tensor,
    sum_i: torch.Tensor,
    r_vals: torch.Tensor,
    i_vals: torch.Tensor,
    vol: float,
    kvecs: torch.Tensor,
    I: torch.Tensor,
    charges: torch.Tensor,
    positions: torch.Tensor,
    calculate_forces: int,
    calculate_dq: int,
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the reciprocal-space contribution to the Ewald summation.

    This function calculates the electrostatic interaction energy in reciprocal space,
    as well as optional force and charge derivative calculations.

    Args:
        sum_r (torch.Tensor): Real part of the structure factor sum. Shape: `(K,)`, where:
            - `K` is the number of k-vectors.
        sum_i (torch.Tensor): Imaginary part of the structure factor sum. Shape: `(K,)`.
        r_vals (torch.Tensor): Real part of exponential terms per atom. Shape: `(K, n)`, where:
            - `n` is the number of local atoms.
        i_vals (torch.Tensor): Imaginary part of exponential terms per atom. Shape: `(K, n)`.
        vol (float): Volume of the simulation box.
        kvecs (torch.Tensor): Reciprocal space vectors. Shape: `(K, 3)`, where:
            - `3` represents the x, y, and z components of each k-vector.
        I (torch.Tensor): Fourier-space prefactors. Shape: `(K,)`.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`, where:
            - `N` is the total number of atoms.
        positions (torch.Tensor): Atomic positions. Shape: `(3, N)`.
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int): Flag to compute charge derivatives (`1` for True, `0` for False`).

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Total k-space energy contribution.
            - **(torch.Tensor, shape `(3, n)`, optional)** Computed forces if `calculate_forces` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(n,)`, optional)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
    """

    # sum_r, sum_i: [M,]
    abs_fac_sq = sum_r**2 + sum_i**2
    """
    dE/dq = (2 * 2 * pi * I * sum_r).reshape(-1, M) @ torch.cos(mmul)/vol
          + (2 * 2 * pi * I * sum_i).reshape(-1, M) @ torch.sin(mmul)/vol 
    We either need to return cos(mmul) and sin(mmul), or do a division to restore the values
    One potential issue: this can cause numerical issues 
    # if torch.cos(mmul) * charges / charges != torch.cos(mmul) (in terms of num. precision)
    """
    # de_dq = (4.0 * torch.pi * I * sum_r).reshape(-1, M) @ (r_vals / charges)/vol
    # de_dq += (4.0 * torch.pi * I * sum_i).reshape(-1, M) @ (i_vals / charges)/vol

    if calculate_forces:
        # local N
        N = r_vals.shape[1]
        # cos_sin_ln is M x N
        cos_sin_ln = r_vals * sum_i.reshape(-1, 1)
        sin_cos_ln = i_vals * sum_r.reshape(-1, 1)

        prefac_ln = I.reshape(-1, 1) * (cos_sin_ln - sin_cos_ln)
        # convert from Nx3 to 3xN
        f_nc = torch.sum(kvecs.reshape(-1, 1, 3) * prefac_ln.reshape(-1, N, 1), dim=0).T
        forces = -4 * torch.pi * f_nc / vol
    else:
        forces = None

    if calculate_dq:
        M = len(abs_fac_sq)
        charges = charges.reshape(1, -1)
        # TODO: find better solution to this,
        # fix zero charge issue
        charges = torch.where(charges != 0, charges, 1.0)
        de_dq = (4.0 * torch.pi * I * sum_r).reshape(-1, M) @ (r_vals / charges) / vol
        de_dq += (4.0 * torch.pi * I * sum_i).reshape(-1, M) @ (i_vals / charges) / vol
        de_dq = de_dq.flatten()
    else:
        de_dq = (None,)
    return 2 * torch.pi * torch.sum(I * abs_fac_sq) / vol, forces, de_dq


def construct_kspace(cell, kcounts, cutoff, alpha, transpose_kvec=False):
    """
    k-vectors: Mx3
    """
    nx = torch.arange(-kcounts[0], kcounts[0] + 1)
    ny = torch.arange(-kcounts[1], kcounts[1] + 1)
    nz = torch.arange(-kcounts[2], kcounts[2] + 1)
    n_lc = torch.stack(torch.meshgrid(nx, ny, nz, indexing="xy"))
    n_lc = n_lc.permute(*torch.arange(n_lc.ndim - 1, -1, -1))
    n_lc = n_lc.reshape(-1, 3).to(cell.device)
    k_lc = (
        2 * torch.pi * torch.matmul(torch.linalg.inv(cell), n_lc.T.type(cell.dtype)).T
    )
    k = torch.linalg.norm(k_lc, dim=1)
    mask = torch.logical_and(k <= cutoff, k != 0)

    kvecs = k_lc[mask]
    if transpose_kvec:
        kvecs = kvecs.T.contiguous()

    return torch.exp(-((k[mask] / (2 * alpha)) ** 2)) / k[mask] ** 2, kvecs


@torch.compile
def ewald_kspace(positions, charges, vol, kvecs, I, calculate_forces=0, calculate_dq=0):
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, charges, kvecs)
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)
    en, out_f, out_dq = ewald_kspace_part2(
        r_sum,
        i_sum,
        my_r_vals,
        my_i_vals,
        vol,
        kvecs,
        I,
        charges,
        positions,
        calculate_forces,
        calculate_dq,
    )
    return en, out_f, out_dq


def ewald_benchmark(
    positions,
    charges,
    nbr_inds,
    nbr_disp_vecs,
    nbr_dists,
    alpha,
    cutoff,
    vol,
    kvecs,
    I,
    calculate_forces=1,
    calculate_dq=0,
):

    my_e_real, my_f_real, my_dq_real = ewald_real(
        0,
        len(charges),
        nbr_inds,
        nbr_disp_vecs,
        nbr_dists,
        charges,
        alpha,
        cutoff,
        calculate_forces,
        calculate_dq,
    )
    my_charges = charges[0 : 0 + len(charges)]
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, my_charges, kvecs)
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)
    en, out_f, out_dq = ewald_kspace_part2(
        r_sum,
        i_sum,
        my_r_vals,
        my_i_vals,
        vol,
        kvecs,
        I,
        my_charges,
        positions,
        calculate_forces,
        calculate_dq,
    )

    if calculate_forces:
        out_f = my_f_real + out_f
    if calculate_dq:
        out_dq = my_dq_real + out_dq
    return en + my_e_real, out_f, out_dq


def ewald_energy(
    positions: torch.Tensor,
    cell: torch.Tensor,
    nbr_inds: torch.Tensor,
    nbr_disp_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    charges: torch.Tensor,
    kvecs: torch.Tensor,
    I: torch.Tensor,
    alpha: float,
    cutoff: float,
    calculate_forces: int,
    calculate_dq: int = 0,
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the Ewald sum energy and forces in a distributed way.

    This function calculates the real-space and reciprocal-space contributions to the
    Ewald summation, including optional force and charge derivative calculations.

    The computed forces will have the same shape as the positions

    Args:
        positions (torch.Tensor): Atomic positions. Shape: `(3, N)` or `(3, N)`, where:
            - `N` is the total number of atoms.
            - `3` represents x, y, z coordinates.
        cell (torch.Tensor): Simulation cell matrix. Shape: `(3, 3)`.
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `K` is the max number of neighbors per atom.
        nbr_disp_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)` or `(N, K, 3)`.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        charges (torch.Tensor): Charge per atom. Shape: `(N,)`.
        kvecs (torch.Tensor): Reciprocal space vectors. Shape: `(M, 3)`, where:
            - `M` is the number of k-space vectors.
        I (torch.Tensor): Fourier-space prefactors. Shape: `(M,)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for real-space interactions (scalar).
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int, optional): Flag to compute charge derivatives (`1` for True, `0` for False`). Defaults to `0`.
    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **total_ewald_e (float)**: Total ewald energy.
            - **forces (torch.Tensor, shape `(3, N)`, optional)**: Computed forces if `calculate_forces` is enabled, otherwise `None`.
                If the positions are provided as `(N, 3)`, the forces will be also  `(N, 3)`.
            - **dq (torch.Tensor, shape `(N,)`, optional)**: Charge derivatives if `calculate_dq` is enabled, otherwise `None`.

    """
    # As the internal functions expects (3, N), transpose the position tensor as needed
    transpose = False
    if positions.shape[1] == 3:
        transpose = True
        positions = positions.T.contiguous()

    # transpose the disp. vectors as needed
    if nbr_disp_vecs.shape[2] == 3:
        nbr_disp_vecs = nbr_disp_vecs.permute(2, 0, 1).contiguous()

    device = positions.device
    N = positions.shape[1]
    my_e_real, my_f_real, my_dq_real = ewald_real(
        nbr_inds,
        nbr_disp_vecs,
        nbr_dists,
        charges,
        alpha,
        cutoff,
        calculate_forces,
        calculate_dq,
    )
    vol = torch.det(cell)
    alpha = torch.tensor(alpha)
    my_r_vals, my_i_vals = ewald_kspace_part1(positions, charges, kvecs)
    # size K vectors
    r_sum = torch.sum(my_r_vals, axis=1)
    i_sum = torch.sum(my_i_vals, axis=1)

    total_e_kspace, my_f_kspace, my_dq_kspace = ewald_kspace_part2(
        r_sum,
        i_sum,
        my_r_vals,
        my_i_vals,
        vol,
        kvecs,
        I,
        charges,
        positions,
        calculate_forces,
        calculate_dq,
    )

    self_e, self_dq = ewald_self_energy(charges, alpha, calculate_dq)

    if calculate_forces:
        forces = (my_f_real + my_f_kspace) * CONV_FACTOR
    else:
        forces = None
    if calculate_dq:
        dq = (my_dq_kspace + my_dq_real + self_dq) * CONV_FACTOR
    else:
        dq = None

    total_ewald_e = (my_e_real + total_e_kspace + self_e) * CONV_FACTOR

    # if user provided [N,3] positions, tranpose the forces to [N, 3]
    if transpose and calculate_forces:
        forces = forces.T.contiguous()

    return total_ewald_e, forces, dq
