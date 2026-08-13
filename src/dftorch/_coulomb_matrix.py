from __future__ import annotations

import math
import time
from typing import Dict, Optional  # <-- add Optional, Dict

import torch

from ._tools import _maybe_compile

# ── Van der Waals radii (Bondi / Mantina) in Angstrom ───────────────────
# Source: Bondi, J. Phys. Chem. 68:441 (1964);
#         Mantina et al., J. Phys. Chem. A 113:5806 (2009).
# -1 means unavailable.  Only elements needed for DFTB are listed;
# the full table mirrors DFTB+ `vdwdata.F90`.
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
    21: 211,
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
    55: 343,
    56: 268,
    78: 175,
    79: 166,
    80: 155,
    81: 196,
    82: 202,
    83: 207,
    84: 197,
    85: 202,
    86: 220,
    92: 186,
}

# Default H5 element-specific scaling factors k_XH (Řezáč, JCTC 13, 4804, 2017)
_H5_DEFAULT_SCALING: Dict[int, float] = {
    7: 0.18,  # N
    8: 0.06,  # O
    16: 0.21,  # S
}

# gaussianWidthFactor = 0.5 / sqrt(2 * ln(2))  ≈ 0.42466
_GAUSS_WIDTH_FACTOR = 0.5 / math.sqrt(2.0 * math.log(2.0))

_H_DAMP_BASE_EPS = 1.0e-20


def _h_damp_base_power(base: torch.Tensor, exponent) -> torch.Tensor:
    """Stable positive-base exponentiation for trainable H-damping exponents."""
    base_safe = torch.clamp(base, min=_H_DAMP_BASE_EPS)
    return torch.exp(exponent * torch.log(base_safe))


def coulomb_matrix_vectorized(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    Nr_atoms: int,
    Coulomb_acc: float,
    nnRx: torch.Tensor,
    nnRy: torch.Tensor,
    nnRz: torch.Tensor,
    nnType: torch.Tensor,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    CALPHA: float,
    verbose: bool = False,
    h_damp_exp: Optional[float] = None,
    h5_params: Optional[Dict] = None,
    library_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Ewald-summed Coulomb matrix and its Cartesian derivatives.

    This routine computes the Coulomb interaction matrix `CC` using an Ewald split:

    - Real-space contribution computed using the provided neighbor list.
    - Reciprocal-space (k-space) contribution computed for periodic systems.

    The returned derivative tensor follows the existing convention used in this
    codebase: the function returns `-dCC_dxyz`.

    Parameters
    ----------
    Hubbard_U:
        Float tensor of shape `(Nr_atoms,)` with per-atom Hubbard U values used in
        short-range corrections.
    TYPE:
        Long tensor of shape `(Nr_atoms,)` with per-atom element/type ids.
    RX, RY, RZ:
        Float tensors of shape `(Nr_atoms,)` giving Cartesian coordinates in Å.
    cell:
        Float tensor of shape `(3, 3)` with lattice vectors; used for cell volume.
    Nr_atoms:
        Number of atoms in the system.
    Coulomb_acc:
        Desired accuracy for the k-space sum (used to determine reciprocal cutoff).
    nnRx, nnRy, nnRz:
        Neighbor coordinates (shape matches `nnType`; commonly `(Nr_atoms, MAXNN)`).
    nnType:
        Neighbor type/index tensor. `-1` indicates padded/invalid neighbor entries.
    neighbor_I, neighbor_J:
        1D long tensors containing the flattened neighbor pair indices into `[0, Nr_atoms)`.
        The ordering must correspond to the masked neighbor entries.
    CALPHA:
        Ewald splitting parameter α.
    verbose:
        If True, prints progress from the k-space summation.
    library_output : bool, default True
        Whether to emit this function's status output.  Threaded from the
        ``VERBOSE_LIBRARY_OUTPUT`` key via
        :func:`dftorch._tools.library_output_enabled` (decision D-02).
        **The default is True on purpose.**  This function receives no
        parameter dictionary, so a call site that forgets to thread the
        flag keeps printing -- which is today's behaviour.  A default of
        ``False`` would silently go quiet at every missed call site, the
        exact regression D-02's noisy default exists to prevent.

    Returns
    -------
    CC:
        Coulomb matrix of shape `(Nr_atoms, Nr_atoms)`.
    neg_dCC_dxyz:
        Tensor of shape `(3, Nr_atoms, Nr_atoms)` equal to `-dCC_dxyz`.

    Notes
    -----
    - This function intentionally keeps existing behavior (including print timing output),
      but removes the unconditional banner print.
    """

    if library_output:
        print("coulomb_matrix_vectorized")
    if verbose:
        print("  Do Coulomb Real")
    start_time1 = time.perf_counter()

    Ra = torch.stack((RX.unsqueeze(-1), RY.unsqueeze(-1), RZ.unsqueeze(-1)), dim=-1)
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)
    dR_dxyz = Rab / dR.unsqueeze(-1)
    # dist_mask = (dR <= COULCUT)*(dR > 1e-12)

    use_ewald = cell is not None

    ##################
    CC_real, dCC_dxyz_real = ewald_real_space_vectorized(
        Hubbard_U,
        TYPE,
        dR,
        dR_dxyz,
        nnType,
        neighbor_I,
        neighbor_J,
        CALPHA,
        use_ewald,
        h_damp_exp=h_damp_exp,
        h5_params=h5_params,
    )
    ##################

    dq_J = torch.zeros(Nr_atoms, dtype=dR.dtype, device=dR.device)
    if library_output:
        print("  Coulomb_Real t {:.1f} s".format(time.perf_counter() - start_time1))

    ## Second, k-space
    start_time1 = time.perf_counter()
    if cell is None:
        CC_k, dCC_dR_k = 0.0, 0.0
    else:
        if verbose:
            print("  Doing Coulomb k")
        CC_k, dCC_dR_k = ewald_k_space_vectorized(
            RX,
            RY,
            RZ,
            cell,
            dq_J,
            Nr_atoms,
            Coulomb_acc,
            CALPHA,
            verbose,
            library_output=library_output,
        )
        if library_output:
            print("  Coulomb_k t {:.1f} s\n".format(time.perf_counter() - start_time1))

    CC = CC_real + CC_k
    dCC_dxyz = dCC_dxyz_real + dCC_dR_k

    return CC, -dCC_dxyz  # , CC_sr, -dCC_dxyz_sr


def ewald_real_space_vectorized(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    dR: torch.Tensor,
    dR_dxyz: torch.Tensor,
    nnType: torch.Tensor,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    CALPHA: float,
    use_ewald: bool,
    h_damp_exp: Optional[float] = None,
    h5_params: Optional[Dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-space Ewald contribution using a neighbor list (vectorized).

    Parameters
    ----------
    Hubbard_U:
        `(Nr_atoms,)` float tensor (eV).
    TYPE:
        `(Nr_atoms,)` long tensor (atomic numbers, 1-based).
    dR:
        Distance tensor with same leading shape as `nnType` (typically `(Nr_atoms, MAXNN)`).
    dR_dxyz:
        Unit-vector tensor with shape `dR.shape + (3,)`.
    nnType:
        Neighbor type/index tensor used to mask padded neighbors (`-1` padding).
    neighbor_I, neighbor_J:
        1D long tensors defining the flattened pair list (aligned with `nnType != -1`).
    CALPHA:
        Ewald splitting parameter α.
    h_damp_exp:
        Hydrogen damping exponent ζ for DFTB3.  When not None the short-range
        gamma for every pair that involves at least one hydrogen atom is
        multiplied by ``exp(−((U_A + U_B) / 2)^ζ · r²)`` (Hubbard U in Hartree,
        r in Bohr).  Typical values: 4.0 (mio-1-1) or 4.05 (3ob-3-1).
        See J. Phys. Chem. A 111, 10865 (2007).
    h5_params:
        H5 hydrogen bond correction parameters (Řezáč, JCTC 13, 4804, 2017).
        Dict with optional keys ``r_scaling`` (default 0.714), ``w_scaling``
        (default 0.25), and ``h5_scaling`` (dict mapping atomic number → k_XH,
        defaults: {7: 0.18, 8: 0.06, 16: 0.21}).  Mutually exclusive with
        *h_damp_exp* (DFTB+ convention).

    Returns
    -------
    CC_real:
        `(Nr_atoms, Nr_atoms)` tensor.
    dCC_dxyz_real:
        `(3, Nr_atoms, Nr_atoms)` tensor.
    """

    Nats = TYPE.shape[-1]

    # Constants
    CALPHA2 = CALPHA**2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)

    # Pair indices
    nn_mask = (
        nnType != -1
    )  # & dist_mask # mask to exclude zero padding from the neigh list

    dR_mskd = dR[nn_mask]
    Ti = TFACT * Hubbard_U[neighbor_I]
    Tj = TFACT * Hubbard_U[neighbor_J]
    CC_real = torch.zeros((Nats * Nats), device=dR.device, dtype=dR.dtype)

    if use_ewald:
        CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
        dCA = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI) / dR_mskd
        tmp1 = CA.clone()
        dtmp1 = dCA.clone()

    else:
        CA = 1.0 / dR_mskd
        dCA = -1.0 / (dR_mskd**2)
        tmp1 = CA.clone()
        dtmp1 = dCA.clone()

    # ── Hydrogen gamma damping (DFTB3) ──────────────────────────────────
    # For pairs involving at least one H atom, the short-range gamma is
    # multiplied by exp(rTmp · r²) where rTmp = −((U_A+U_B)/2)^ζ  (a.u.).
    # The product rule gives:
    #   γ_h  = γ · D         and     γ_h' = γ'·D + γ · D'
    # with D = exp(rTmp·r²), D' = 2·rTmp·r·D.
    if h_damp_exp is not None:
        EV_TO_HA = 1.0 / 27.211386245988
        ANG_TO_BOHR = 1.0 / 0.52917721067
        is_H_I = TYPE[neighbor_I] == 1  # H has atomic number 1
        is_H_J = TYPE[neighbor_J] == 1
        h_mask = is_H_I | is_H_J  # pairs with at least one hydrogen
    # ────────────────────────────────────────────────────────────────────

    # ── H5 correction (Řezáč, JCTC 13, 4804, 2017) ─────────────────────
    # For pairs where *exactly one* atom is H, the short-range gamma is
    # scaled: γ_H5 = γ·(1+G) − G/r  and  γ'_H5 = γ·G' + γ'·(1+G) − (G'/r − G/r²)
    # with G = k_XH · exp(−0.5·(r−r₀)²/c²), G' = −G·(r−r₀)/c²
    # where r₀ = s_r·(rVdW_X + rVdW_H), c = s_w·(rVdW_X + rVdW_H)·gaussWidthFactor
    # Mutually exclusive with h_damp_exp (DFTB+ convention).
    _apply_h5 = False
    if h5_params is not None and h_damp_exp is not None:
        import warnings

        warnings.warn(
            "Both h_damp_exp and h5_params are set. These are mutually exclusive "
            "(DFTB+ convention). h5_params will be ignored; only gamma damping is applied.",
            stacklevel=2,
        )
    if h5_params is not None and h_damp_exp is None:
        _h5_r_scaling = h5_params.get("r_scaling", 0.714)
        _h5_w_scaling = h5_params.get("w_scaling", 0.25)
        _h5_scaling = h5_params.get("h5_scaling", _H5_DEFAULT_SCALING)  # {Z: k}
        # vdW radius of H in Angstrom
        _vdw_H_ang = _VDW_RADII_PM.get(1, 120) * 0.01  # pm → Å = 1.20 Å
        is_H_I_h5 = TYPE[neighbor_I] == 1
        is_H_J_h5 = TYPE[neighbor_J] == 1
        # exactly one H
        h5_mask_all = is_H_I_h5 ^ is_H_J_h5  # XOR
        if h5_mask_all.any():
            _apply_h5 = True
            # For each H-X pair, identify heavy-atom Z and look up k_XH
            heavy_Z = torch.where(is_H_I_h5, TYPE[neighbor_J], TYPE[neighbor_I])
            # Build per-pair k_XH and sumVdW (Å) tensors
            k_XH_all = torch.zeros_like(dR_mskd)
            sumVdW_all = torch.zeros_like(dR_mskd)
            for z_heavy, k_val in _h5_scaling.items():
                z_mask = h5_mask_all & (heavy_Z == z_heavy)
                if z_mask.any():
                    vdw_heavy_ang = _VDW_RADII_PM.get(z_heavy, -1) * 0.01
                    if vdw_heavy_ang > 0:
                        k_XH_all[z_mask] = k_val
                        sumVdW_all[z_mask] = _vdw_H_ang + vdw_heavy_ang
            # Redefine h5_mask to only include pairs with positive k and vdW
            h5_mask_active = h5_mask_all & (k_XH_all > 0.0)
            if not h5_mask_active.any():
                _apply_h5 = False
    # ────────────────────────────────────────────────────────────────────

    mask_same_elem = TYPE[neighbor_I] == TYPE[neighbor_J]
    mask_diff_elem = ~mask_same_elem
    zero_pair = torch.zeros_like(tmp1)

    TI2_same = Ti**2
    TI3_same = TI2_same * Ti
    SSB_same = TI3_same / 48.0
    SSC_same = 3 * TI2_same / 16.0
    SSD_same = 11 * Ti / 16.0
    EXPTI_same = torch.exp(-Ti * dR_mskd)
    same_tmp = SSB_same * dR_mskd**2 + SSC_same * dR_mskd + SSD_same + 1.0 / dR_mskd
    same_sr_val = EXPTI_same * same_tmp
    same_sr_deriv = EXPTI_same * (
        (-Ti) * same_tmp + (2 * SSB_same * dR_mskd + SSC_same - 1.0 / dR_mskd**2)
    )

    TI2_diff = Ti**2
    TI4_diff = TI2_diff**2
    TI6_diff = TI4_diff * TI2_diff
    TJ2_diff = Tj**2
    TJ4_diff = TJ2_diff**2
    TJ6_diff = TJ4_diff * TJ2_diff
    EXPTI_diff = torch.exp(-Ti * dR_mskd)
    EXPTJ_diff = torch.exp(-Tj * dR_mskd)
    TI2MTJ2 = torch.where(mask_diff_elem, TI2_diff - TJ2_diff, torch.ones_like(tmp1))
    TJ2MTI2 = -TI2MTJ2
    SB = TJ4_diff * Ti / (2 * TI2MTJ2**2)
    SC = (TJ6_diff - 3 * TJ4_diff * TI2_diff) / (TI2MTJ2**3)
    SE = TI4_diff * Tj / (2 * TJ2MTI2**2)
    SF = (TI6_diff - 3 * TI4_diff * TJ2_diff) / (TJ2MTI2**3)
    COULOMBV_tmp1 = SB - SC / dR_mskd
    COULOMBV_tmp2 = SE - SF / dR_mskd
    diff_sr_val = EXPTI_diff * COULOMBV_tmp1 + EXPTJ_diff * COULOMBV_tmp2
    diff_sr_deriv = EXPTI_diff * (
        (-Ti) * COULOMBV_tmp1 + SC / dR_mskd**2
    ) + EXPTJ_diff * ((-Tj) * COULOMBV_tmp2 + SF / dR_mskd**2)

    if h_damp_exp is not None:
        Ui_au = Hubbard_U[neighbor_I] * EV_TO_HA
        Uj_au = Hubbard_U[neighbor_J] * EV_TO_HA
        r_au = dR_mskd * ANG_TO_BOHR
        rTmp = -_h_damp_base_power(0.5 * (Ui_au + Uj_au), h_damp_exp)
        D = torch.exp(rTmp * r_au**2)
        Dprime = 2.0 * rTmp * r_au * D * ANG_TO_BOHR  # d(D)/d(r_Ang)

        same_h_mask = mask_same_elem & h_mask
        diff_h_mask = mask_diff_elem & h_mask
        same_sr_deriv = torch.where(
            same_h_mask, same_sr_deriv * D + same_sr_val * Dprime, same_sr_deriv
        )
        same_sr_val = torch.where(same_h_mask, same_sr_val * D, same_sr_val)
        diff_sr_deriv = torch.where(
            diff_h_mask, diff_sr_deriv * D + diff_sr_val * Dprime, diff_sr_deriv
        )
        diff_sr_val = torch.where(diff_h_mask, diff_sr_val * D, diff_sr_val)

    elif _apply_h5:
        h5_mask_diff = mask_diff_elem & h5_mask_active
        safe_sum_vdw = torch.where(
            h5_mask_diff, sumVdW_all, torch.ones_like(sumVdW_all)
        )
        r0 = _h5_r_scaling * safe_sum_vdw
        cc = _h5_w_scaling * safe_sum_vdw * _GAUSS_WIDTH_FACTOR
        gauss = torch.where(
            h5_mask_diff,
            k_XH_all * torch.exp(-0.5 * (dR_mskd - r0) ** 2 / cc**2),
            zero_pair,
        )
        dgauss = torch.where(
            h5_mask_diff,
            -gauss * (dR_mskd - r0) / cc**2,
            zero_pair,
        )

        # scaled value:  γ·(1+G) − G/r
        diff_sr_deriv = torch.where(
            h5_mask_diff,
            diff_sr_val * dgauss
            + diff_sr_deriv * (1.0 + gauss)
            - (dgauss / dR_mskd - gauss / dR_mskd**2),
            diff_sr_deriv,
        )
        diff_sr_val = torch.where(
            h5_mask_diff,
            diff_sr_val * (1.0 + gauss) - gauss / dR_mskd,
            diff_sr_val,
        )

    tmp1 = tmp1 - torch.where(mask_same_elem, same_sr_val, zero_pair)
    dtmp1 = dtmp1 - torch.where(mask_same_elem, same_sr_deriv, zero_pair)
    tmp1 = tmp1 - torch.where(mask_diff_elem, diff_sr_val, zero_pair)
    dtmp1 = dtmp1 - torch.where(mask_diff_elem, diff_sr_deriv, zero_pair)

    tmp1 *= KECONST
    dtmp1 *= KECONST
    pair_index = neighbor_I * Nats + neighbor_J
    dtmp1_xyz = dtmp1.unsqueeze(-1) * dR_dxyz[nn_mask]

    # bincount is a good no-grad reduction here, but it does not implement
    # backward for weighted inputs. Keep the differentiable accumulation path
    # when autograd is tracking these weights.
    if tmp1.requires_grad or dtmp1_xyz.requires_grad:
        CC_real = torch.zeros(
            (Nats * Nats), device=dR.device, dtype=dR.dtype
        ).scatter_add(0, pair_index, tmp1)

        pair_index_xyz = pair_index.unsqueeze(0).expand(3, -1)
        dCC_dxyz_real = torch.zeros(
            (3, Nats * Nats), device=dR.device, dtype=dR.dtype
        ).scatter_add(1, pair_index_xyz, dtmp1_xyz.T)
    else:
        CC_real = torch.bincount(pair_index, weights=tmp1, minlength=Nats * Nats)
        dCC_dxyz_real = torch.stack(
            [
                torch.bincount(
                    pair_index, weights=dtmp1_xyz[:, axis], minlength=Nats * Nats
                )
                for axis in range(3)
            ],
            dim=0,
        )

    CC_real = CC_real.reshape(Nats, Nats)
    dCC_dxyz_real = dCC_dxyz_real.reshape(3, Nats, Nats)
    return CC_real, dCC_dxyz_real


def ewald_k_space_vectorized(
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    DELTAQ: torch.Tensor,
    Nr_atoms: int,
    COULACC: float,
    CALPHA: float,
    verbose: bool,
    do_vec: bool = False,
    library_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reciprocal-space (k-space) Ewald contribution and derivatives.

    Parameters
    ----------
    RX, RY, RZ:
        `(Nr_atoms,)` coordinate tensors in Å.
    cell:
        `(3, 3)` lattice vectors (used for volume determinant).
    DELTAQ:
        Charge-difference tensor (not used; kept for API compatibility).
    Nr_atoms:
        Number of atoms.
    COULACC:
        Accuracy target controlling reciprocal-space cutoff.
    CALPHA:
        Ewald α.
    verbose:
        Print loop counters if True.
    do_vec:
        If True, run a fully-vectorized k-space path (memory heavy).
    library_output : bool, default True
        Whether to emit this function's status output.  Threaded from the
        ``VERBOSE_LIBRARY_OUTPUT`` key via
        :func:`dftorch._tools.library_output_enabled` (decision D-02).
        **The default is True on purpose.**  This function receives no
        parameter dictionary, so a call site that forgets to thread the
        flag keeps printing -- which is today's behaviour.  A default of
        ``False`` would silently go quiet at every missed call site, the
        exact regression D-02's noisy default exists to prevent.

    Returns
    -------
    COULOMBV:
        `(Nr_atoms, Nr_atoms)` k-space Coulomb matrix contribution.
    dC_dR:
        `(3, Nr_atoms, Nr_atoms)` derivative tensor.
    """

    device = RX.device

    COULVOL = torch.abs(torch.det(cell))
    SQRTX = math.sqrt(-math.log(COULACC))

    CALPHA2 = CALPHA * CALPHA
    KCUTOFF = 2 * CALPHA * SQRTX
    KCUTOFF2 = KCUTOFF * KCUTOFF

    cell_inv = torch.linalg.inv(cell)
    RECIPVECS = 2.0 * math.pi * cell_inv.T

    # LMAX = int(KCUTOFF / RECIPVECS[0, 0])
    # MMAX = int(KCUTOFF / RECIPVECS[1, 1])
    # NMAX = int(KCUTOFF / RECIPVECS[2, 2])

    g1_norm = torch.norm(RECIPVECS[:, 0])
    g2_norm = torch.norm(RECIPVECS[:, 1])
    g3_norm = torch.norm(RECIPVECS[:, 2])

    LMAX = int(torch.ceil(KCUTOFF / g1_norm).item())
    MMAX = int(torch.ceil(KCUTOFF / g2_norm).item())
    NMAX = int(torch.ceil(KCUTOFF / g3_norm).item())

    KECONST = 14.3996437701414  # in eV·Å/e²
    SQRTPI = math.sqrt(math.pi)

    COULOMBV = torch.zeros((Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

    dC_dR = torch.zeros((3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

    if do_vec:
        # Create meshgrid of all combinations
        if library_output:
            print("  init L,M,N,K")
        L_vals = torch.arange(0, LMAX + 1)
        M_vals = torch.arange(-MMAX, MMAX + 1)
        N_vals = torch.arange(-NMAX, NMAX + 1)
        L_vec, M_vec, N_vec = torch.meshgrid(L_vals, M_vals, N_vals, indexing="ij")

        L_vec = L_vec.flatten()
        M_vec = M_vec.flatten()
        N_vec = N_vec.flatten()

        mask = ~((L_vec == 0) * (M_vec < 0))  # exclude L==0 and M<0
        mask &= ~((L_vec == 0) * (M_vec == 0) * (N_vec < 1))  # exclude L==0, M==0, N<1
        L_vec = L_vec[mask]
        M_vec = M_vec[mask]
        N_vec = N_vec[mask]

        # Step 3: Stack into a (N, 3) tensor of LMN vectors
        LMN = torch.stack([L_vec, M_vec, N_vec], dim=1).to(
            dtype=RX.dtype, device=device
        )  # shape: (num_valid, 3)
        K_vectors = LMN @ RECIPVECS
        K2 = torch.sum(K_vectors**2, dim=1)

        exp_factor = torch.exp(-K2 / (4 * CALPHA2))
        prefactor = 8 * torch.pi * exp_factor / (COULVOL * K2)
        KEPREF = KECONST * prefactor

        dot = (
            K_vectors[:, 0] * RX.unsqueeze(-1)
            + K_vectors[:, 1] * RY.unsqueeze(-1)
            + K_vectors[:, 2] * RZ.unsqueeze(-1)
        )
        sin_list = torch.sin(dot)
        cos_list = torch.cos(dot)

        COULOMBV += (
            KEPREF
            * (
                cos_list.unsqueeze(1) * cos_list.unsqueeze(0)
                + sin_list.unsqueeze(1) * sin_list.unsqueeze(0)
            )
        ).sum(-1)
        force_tmp = KEPREF * (
            -cos_list.unsqueeze(1) * sin_list.unsqueeze(0)
            + sin_list.unsqueeze(1) * cos_list.unsqueeze(0)
        )
        dC_dR[0] += (force_tmp * K_vectors[:, 0]).sum(-1)
        dC_dR[1] += (force_tmp * K_vectors[:, 1]).sum(-1)
        dC_dR[2] += (force_tmp * K_vectors[:, 2]).sum(-1)
    else:
        # if verbose: print('   LMAX:', LMAX)
        if library_output:
            print("   LMAX:", LMAX)
        for L in range(0, LMAX + 1):
            if verbose:
                print("  ", L)
            MMIN = 0 if L == 0 else -MMAX
            for M in range(MMIN, MMAX + 1):
                NMIN = 1 if (L == 0 and M == 0) else -NMAX
                for N in range(NMIN, NMAX + 1):
                    kvec = (
                        torch.tensor([L, M, N], dtype=RX.dtype, device=device)
                        @ RECIPVECS
                    )
                    K2 = torch.dot(kvec, kvec)
                    if K2 > KCUTOFF2:
                        continue
                    exp_factor = torch.exp(-K2 / (4 * CALPHA2))
                    prefactor = 8 * math.pi * exp_factor / (COULVOL * K2)
                    KEPREF = 14.3996437701414 * prefactor  # KECONST in eV·Å/e²

                    dot = torch.matmul(
                        kvec.view(1, 3), torch.stack((RX, RY, RZ), dim=0)
                    ).squeeze(0)  # shape (N,)
                    sin_list = torch.sin(dot)
                    cos_list = torch.cos(dot)

                    # Use broadcasting for outer products
                    sin_i = sin_list.view(-1, 1)
                    sin_j = sin_list.view(1, -1)
                    cos_i = cos_list.view(-1, 1)
                    cos_j = cos_list.view(1, -1)

                    COULOMBV += KEPREF * (cos_i * cos_j + sin_i * sin_j)
                    force_term = KEPREF * (-cos_i * sin_j + sin_i * cos_j)
                    dC_dR += force_term * kvec.view(3, 1, 1)

    # Self-interaction correction
    DELTAQ_vec = torch.eye(Nr_atoms, device=device)
    CORRFACT = 2 * KECONST * CALPHA / SQRTPI
    COULOMBV -= CORRFACT * DELTAQ_vec
    return COULOMBV, dC_dR


ewald_real_space_vectorized_eager = ewald_real_space_vectorized
ewald_k_space_vectorized_eager = ewald_k_space_vectorized
ewald_real_space_vectorized = _maybe_compile(ewald_real_space_vectorized)
ewald_k_space_vectorized = _maybe_compile(ewald_k_space_vectorized)


def _shell_pair_mask(max_ang_I, max_ang_J, shell_i: int, shell_j: int):
    """Neighbour pairs where atom I carries shell ``shell_i`` and J carries ``shell_j``.

    One uniform rule in place of the sixteen hand-enumerated named masks the
    sixteen blocks would otherwise need.  In this codebase's encoding an atom's
    ``max_ang`` is 1 for s-only, 2 for s+p, 3 for s+p+d and 4 for s+p+d+f, while
    a shell is numbered 0 for s, 1 for p, 2 for d and 3 for f.  So "atom I
    actually carries shell ``shell_i``" is exactly ``max_ang_I > shell_i``.

    This reproduces every one of the nine pre-existing blocks exactly, including
    the unmasked s-s case (``max_ang > 0`` is true for every atom).  Worked
    through, with H, X and Y the old names for ``max_ang`` 1, 2 and 3:

    ==========  ==============================  ==========================
    block       old mask                        this rule
    ==========  ==============================  ==========================
    s-s         no mask (all pairs)             always true - all pairs
    s-p         HX + XX + HY + YY + XY + YX     J in {X, Y}   - same six
    p-s         XH + XX + YH + YY + XY + YX     I in {X, Y}   - same six
    p-p         XX + YY + XY + YX               both in {X, Y} - same four
    s-d         HY + XY + YY                    J = Y         - same three
    d-s         YH + YX + YY                    I = Y         - same three
    p-d         XY + YY                         I in {X, Y}, J = Y - same two
    d-p         YX + YY                         I = Y, J in {X, Y} - same two
    d-d         YY                              I = J = Y     - same one
    ==========  ==============================  ==========================

    The old masks tested ``max_ang`` against 1, 2 and 3 only, so an f atom
    (``max_ang == 4``) matched none of the six off-diagonal ones and fell
    straight through - which is why an f system used to come back with every row
    zero except the two s rows.  Requirement SCC-02 in Phase 6.
    """
    return (max_ang_I > shell_i) & (max_ang_J > shell_j)


def _coul_shell_pair_term(Ti, Tj, dR_pair):
    """Short-range damping term for one shell pair, and its distance derivative.

    Selects between the two closed forms by whether the two damping exponents
    are **equal**, not by whether the two atoms are the same element.

    Those two conditions coincide for the s-s, p-p, d-d and f-f blocks, where
    both sides read the same strength table, so this reproduces the pre-existing
    same-element branches exactly.  They do **not** coincide for a block that
    reads two different tables: an atom paired with another atom of its own
    element still contributes to the s-p block, and if that element's s and p
    strengths happen to be equal - which is the common case, since a
    non-extended SKF file writes one Hubbard U per shell and they are usually
    identical - then ``coul_diff_elem_and_ang`` divides by ``Ti**2 - Tj**2``,
    i.e. by zero.  ``tests/f_orbital_data/N-N.skf`` carries Us = Up = Ud = 0.490
    Ha, so an N2 molecule was one such case.

    ``coul_same_elem_and_ang`` is the limit of ``coul_diff_elem_and_ang`` as the
    two exponents approach each other, so this only ever replaces a non-finite
    value with the value it was the limit of.  No pair that produced a finite
    number before produces a different one now.
    """
    t1 = torch.zeros_like(dR_pair)
    dt1 = torch.zeros_like(dR_pair)

    equal_strength = Ti == Tj
    if bool(equal_strength.any()):
        same_t1, same_dt1 = coul_same_elem_and_ang(
            Ti[equal_strength], dR_pair[equal_strength]
        )
        t1[equal_strength] = same_t1
        dt1[equal_strength] = same_dt1

    different_strength = ~equal_strength
    if bool(different_strength.any()):
        diff_t1, diff_dt1 = coul_diff_elem_and_ang(
            Ti[different_strength],
            Tj[different_strength],
            dR_pair[different_strength],
        )
        t1[different_strength] = diff_t1
        dt1[different_strength] = diff_dt1

    return t1, dt1


def onsite_shell_pair_coulomb(Ui: torch.Tensor, Uj: torch.Tensor) -> torch.Tensor:
    """Electron repulsion between two orbital groups sitting on the *same* atom.

    What this is, in plain words
    ---------------------------
    Every block built by :func:`ewald_real_space_vectorized_sr` describes two
    orbital groups on two *different* atoms, because the neighbour list it is
    driven with never pairs an atom with itself.  The interaction between, say,
    europium's s group and europium's own f group is therefore missing from that
    matrix entirely - and it is not small: it is the dominant repulsion either
    group feels.

    The per-atom path has the same hole and fills it the same way, one line
    later: ``Ecoul = 0.5 * q @ (C @ q) + 0.5 * sum(q**2 * Hubbard_U)``.  That
    second term *is* the same-atom interaction, and ``Hubbard_U`` is the R -> 0
    limit of the very function ``C``'s off-site entries are built from.  With
    one charge per atom there is only one such number per atom, so a single
    ``U`` suffices.  With one charge per orbital group there are as many as
    sixteen per atom, and only the four diagonal ones are ``U_l``.

    What this returns
    -----------------
    The same limit, for two groups whose strengths differ: the R -> 0 value of
    ``1/R - S(R)`` where ``S`` is exactly the short-range form
    :func:`_coul_shell_pair_term` uses off-site.  Nothing new is introduced -
    this is the existing formula read at zero separation:

    * with ``Ui == Uj`` it returns ``Ui`` (so a molecule whose groups all share
      one strength reproduces the per-atom expression entry for entry);
    * otherwise it returns a value strictly between the two, e.g. 7.78 eV for
      europium's s group (5.71 eV) against its f group (13.61 eV).

    Deriving it
    -----------
    With ``a`` and ``b`` the two damping exponents, ``S(R) = e^-aR (SB - SC/R)
    + e^-bR (SE - SF/R)`` as written in :func:`coul_diff_elem_and_ang`.  The
    two ``1/R`` pieces cancel against the ``1/R`` in ``gamma`` because
    ``SC + SF == -1`` identically, leaving

        gamma(0) = -(SB + SE) - (a * SC + b * SF).

    For ``a == b`` that expression is 0/0; the limit is ``5a/16``, which is the
    same limit :func:`coul_same_elem_and_ang` has, and which equals ``U`` after
    the ``TFACT`` scaling.  Both branches were checked against the numerical
    value of the codebase's own functions at R = 1e-5 A.

    Parameters
    ----------
    Ui, Uj : torch.Tensor
        Hubbard U of each group, in eV.  Broadcast against each other, so
        passing a column and a row vector produces the full pair matrix.

    Returns
    -------
    torch.Tensor
        The same-atom repulsion for each pair, in eV, shaped by broadcasting.
    """
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)

    a, b = torch.broadcast_tensors(TFACT * Ui, TFACT * Uj)
    # Dispatch on strength equality, exactly as _coul_shell_pair_term does: the
    # different-strength form divides by (a**2 - b**2), which is zero whenever
    # two of an element's groups happen to carry the same U -- the common case
    # for a non-extended SKF file.
    equal_strength = a == b
    # Substitute a value that cannot collide with `a` so the arithmetic below
    # stays finite everywhere; those entries are discarded by the where().
    b_safe = torch.where(equal_strength, b + 1.0, b)

    a2 = a * a
    b2 = b_safe * b_safe
    d = a2 - b2
    SB = b2 * b2 * a / (2 * d * d)
    SC = (b2**3 - 3 * b2 * b2 * a2) / (d**3)
    SE = a2 * a2 * b_safe / (2 * d * d)
    SF = (a2**3 - 3 * a2 * a2 * b2) / ((-d) ** 3)
    different = -(SB + SE) - (a * SC + b_safe * SF)

    same = 5.0 * a / 16.0
    return KECONST * torch.where(equal_strength, same, different)


def onsite_shell_coulomb_matrix(
    Hubbard_U_sr: torch.Tensor, shell_to_atom: torch.Tensor
) -> torch.Tensor:
    """The same-atom, different-group block of the shell-resolved Coulomb matrix.

    Zero everywhere except between two *different* orbital groups of one atom,
    which is exactly the set of entries
    :func:`ewald_real_space_vectorized_sr` cannot produce (its neighbour list
    never pairs an atom with itself) and that the ``0.5 * sum(q**2 * U)`` term
    does not cover either (that term is each group against *itself*).

    Adding this to the off-site matrix gives a shell-resolved Coulomb operator
    with the same convention as the per-atom ``C``: every interaction except
    each charge against itself, which stays in the ``U`` term.

    Leaving it out is not a simplification, it is a hole.  Measured on the Eu-N
    diatomic, without it the charge loop never settles: europium's f level
    responds to its own group's charge at 13.6 eV per electron with nothing on
    the same atom to screen it, and the charges oscillate between integer
    fillings.  With it the loop settles.
    """
    same_atom = shell_to_atom.unsqueeze(1) == shell_to_atom.unsqueeze(0)
    different_group = ~torch.eye(
        len(Hubbard_U_sr), dtype=torch.bool, device=Hubbard_U_sr.device
    )
    pair = onsite_shell_pair_coulomb(
        Hubbard_U_sr.unsqueeze(1), Hubbard_U_sr.unsqueeze(0)
    )
    return pair * (same_atom & different_group)


def ewald_real_space_vectorized_sr(
    structure, dR, dR_dxyz, TYPE, nnType, neighbor_I, neighbor_J, CALPHA
):
    """
    Shell-resolved Coulomb matrix. This one is vectorized in a fashion of _slater_koster_pair.py.
    Computes the real-space component of the Ewald-summed Coulomb interaction matrix and its
    derivatives using a fully vectorized implementation with neighbor lists.

    This function evaluates pairwise interactions between atoms and their neighbors within a
    specified real-space cutoff. It includes analytical short-range damping corrections for
    same-element and different-element pairs as required in DFTB-like models.

    Parameters
    ----------
    RX, RY, RZ : torch.Tensor of shape (Nr_atoms,)
        Cartesian coordinates of atoms along x, y, and z directions.
    dR : torch.Tensor of shape (Nr_atoms, MAXNN)
        Scalar distances between atoms and their neighbors.
    dR_dxyz : torch.Tensor of shape (Nr_atoms, MAXNN, 3)
        Normalized displacement vectors (dR_x, dR_y, dR_z) between atoms and their neighbors (d_dR/dxyz).
    dist_mask : torch.BoolTensor of shape (Nr_atoms, MAXNN)
        Boolean mask indicating which neighbor distances fall within the real-space Ewald cutoff.
    Hubbard_U : torch.Tensor of shape (Nr_atoms,)
        Hubbard U parameters for the atoms, used in short-range corrections.
    TYPE : torch.Tensor of shape (Nr_atoms,)
        Integer element type identifiers for atoms.
    Nr_atoms : int
        Total number of atoms in the system.
    HDIM : int
        Hamiltonian matrix size (used for context, but not used directly in this function).
    Coulomb_acc : float
        Desired accuracy threshold for the Ewald summation.
    TIMERATIO : float
        Empirical scaling constant used to determine the Ewald damping parameter.
    nnRx, nnRy, nnRz : torch.Tensor
        Neighbor coordinates (not used directly here but passed for API consistency).
    nrnnlist : torch.Tensor
        Number of neighbors per atom (not used directly).
    nnType : torch.Tensor of shape (Nr_atoms, MAXNN)
        Indices of neighbor atoms for each atom.
    H_INDEX_START, H_INDEX_END : torch.Tensor
        Index mappings for block matrix ranges (not used directly).
    CALPHA : float
        Ewald real-space damping parameter, typically precomputed externally.

    Returns
    -------
    CC_real : torch.Tensor of shape (n_shells, n_shells)
        Real-space contribution to the shell-resolved Coulomb interaction matrix.
    dCC_dxyz_real : torch.Tensor of shape (3, n_shells, n_shells)
        Derivatives of the real-space Coulomb interaction with respect to x, y, and z.

    Notes
    -----
    - All **sixteen** ordered shell-pair blocks are built: s-s, s-p, s-d, s-f,
      p-s, ... , f-f.  Requirement SCC-02 in Phase 6 added the seven involving
      f; before it there were nine, and an f atom matched none of the six
      off-diagonal masks and fell through them to zero.  The refusal that used
      to guard this function is retired and its class has been deleted.
    - Each block selects its pairs with :func:`_shell_pair_mask`, which encodes
      "atom I carries shell l_i and atom J carries shell l_j" as
      ``(max_ang_I > l_i) & (max_ang_J > l_j)``.  A block writes into row
      ``H_INDEX_START_U[atom] + l_i`` and column ``H_INDEX_START_U[atom] + l_j``;
      that fixed offset is only a valid address because every shipped element's
      shells are a contiguous run starting at s, which
      ``tests/test_shell_resolved_coulomb_f.py::test_orbital_groups_are_a_contiguous_run_from_s``
      states out loud.
    - This function computes the pairwise Coulomb interactions between atoms and their neighbors
      within a real-space cutoff derived from the Ewald alpha parameter.
    - It includes analytical short-range corrections for both same-strength and
      different-strength shell pairs using shell-dependent Hubbard U parameters.
    - Derivatives (dCC/dR) are calculated analytically using the chain rule applied to screened
      Coulomb functions and short-range exponential terms.
    - Output matrices are assembled via scatter operations with accumulation.
    """
    CALPHA2 = CALPHA**2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)

    CDIM = len(structure.Hubbard_U_sr)
    max_ang_I = structure.const.max_ang[TYPE[neighbor_I]]
    max_ang_J = structure.const.max_ang[TYPE[neighbor_J]]

    CC_real = torch.zeros((CDIM**2), device=dR.device, dtype=dR.dtype)
    dCC_dxyz_real = torch.zeros((3, CDIM**2), device=dR.device, dtype=dR.dtype)

    # Pair indices
    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    dR_dxyz_mskd = dR_dxyz[nn_mask].T
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd

    # The sixteen ordered shell-pair blocks, in the order s, p, d, f for the row
    # shell and the same order for the column shell.  Every block has the same
    # shape on purpose: a block that differed structurally from its neighbours
    # is how the seven f blocks went missing in the first place.

    ### s-s ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 0, 0)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 0) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 0
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### s-p ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 0, 1)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 0) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### s-d ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 0, 2)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 0) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### s-f ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 0, 3)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 0) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 3
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-s ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 1, 0)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 0
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-p ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 1, 1)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-d ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 1, 2)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-f ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 1, 3)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 3
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-s ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 2, 0)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 0
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-p ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 2, 1)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-d ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 2, 2)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-f ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 2, 3)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 3
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### f-s ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 3, 0)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Uf[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 3) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 0
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### f-p ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 3, 1)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Uf[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 3) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### f-d ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 3, 2)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Uf[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 3) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### f-f ###
    tmp_mask = _shell_pair_mask(max_ang_I, max_ang_J, 3, 3)
    dR_pair = dR_mskd[tmp_mask]
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(CA[tmp_mask] + 2 * CALPHA * torch.exp(-CALPHA2 * dR_pair**2) / SQRTPI)
        / dR_pair
    )
    Ti = TFACT * structure.const.Uf[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Uf[structure.TYPE[neighbor_J[tmp_mask]]]
    t1, dt1 = _coul_shell_pair_term(Ti, Tj, dR_pair)
    tmp1 -= t1
    dtmp1 -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 3) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 3
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    CC_real = CC_real.reshape(CDIM, CDIM)
    dCC_dxyz_real = dCC_dxyz_real.reshape(3, CDIM, CDIM)
    return CC_real, dCC_dxyz_real


def coul_diff_elem_and_ang(
    Ti_diff_el: torch.Tensor, Tj_diff_el: torch.Tensor, dR_mskd_diff: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short-range damping term for different-element pairs and its derivative."""
    TI2 = Ti_diff_el**2
    TI4 = TI2**2
    TI6 = TI4 * TI2
    TJ2 = Tj_diff_el**2
    TJ4 = TJ2**2
    TJ6 = TJ4 * TJ2
    EXPTI = torch.exp(-Ti_diff_el * dR_mskd_diff)
    EXPTJ = torch.exp(-Tj_diff_el * dR_mskd_diff)
    TI2MTJ2 = TI2 - TJ2
    TJ2MTI2 = -TI2MTJ2
    SB = TJ4 * Ti_diff_el / (2 * TI2MTJ2**2)
    SC = (TJ6 - 3 * TJ4 * TI2) / (TI2MTJ2**3)
    SE = TI4 * Tj_diff_el / (2 * TJ2MTI2**2)
    SF = (TI6 - 3 * TI4 * TJ2) / (TJ2MTI2**3)
    COULOMBV_tmp1 = SB - SC / dR_mskd_diff
    COULOMBV_tmp2 = SE - SF / dR_mskd_diff
    t1 = EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2
    dt1 = EXPTI * ((-Ti_diff_el) * COULOMBV_tmp1 + SC / dR_mskd_diff**2) + EXPTJ * (
        (-Tj_diff_el) * COULOMBV_tmp2 + SF / dR_mskd_diff**2
    )
    return t1, dt1


def coul_same_elem_and_ang(
    Ti_same_el: torch.Tensor, dR_mskd_same: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short-range damping term for same-element pairs and its derivative."""
    TI2 = Ti_same_el**2
    TI3 = TI2 * Ti_same_el
    SSB = TI3 / 48.0
    SSC = 3 * TI2 / 16.0
    SSD = 11 * Ti_same_el / 16.0
    EXPTI = torch.exp(-Ti_same_el * dR_mskd_same)
    tmp = SSB * dR_mskd_same**2 + SSC * dR_mskd_same + SSD + 1.0 / dR_mskd_same
    t1 = EXPTI * tmp
    dt1 = EXPTI * (
        (-Ti_same_el) * tmp + (2 * SSB * dR_mskd_same + SSC - 1.0 / dR_mskd_same**2)
    )
    return t1, dt1
