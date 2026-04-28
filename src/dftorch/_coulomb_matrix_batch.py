from __future__ import annotations

import math
from typing import Optional, Dict  # <-- add Optional, Dict


import torch

from ._coulomb_matrix import (
    _VDW_RADII_PM,
    _H5_DEFAULT_SCALING,
    _GAUSS_WIDTH_FACTOR,
)
from ._tools import _maybe_compile


def coulomb_matrix_vectorized_batch(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    lattice_vecs: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batched Coulomb matrix and its Cartesian derivatives.

    This function builds the Coulomb interaction matrix `CC` for each batch item
    using an Ewald-like split into:

    - Real-space sum over a neighbor list (always applied)
    - Reciprocal-space (k-space) contribution (only if `lattice_vecs` is provided)

    Parameters
    ----------
    Hubbard_U:
        Tensor of shape `(B, N)` with per-atom Hubbard U parameters.
    TYPE:
        Long tensor of shape `(B, N)` with per-atom type indices.
    RX, RY, RZ:
        Coordinate tensors of shape `(B, N)` for x/y/z in Angstrom.
    lattice_vecs:
        Tensor of shape `(B, 3, 3)` describing lattice vectors. Used for volume and
        reciprocal vectors in k-space.
    Nr_atoms:
        Number of atoms `N` (used for shaping/allocations).
    Coulomb_acc:
        Ewald accuracy parameter (called `COULACC` elsewhere).
    nnRx, nnRy, nnRz:
        Neighbor coordinates for each atom. Shape must broadcast with `RX`.
        Typical shape: `(B, N, N)` or `(B, N, Nn)` depending on neighbor list layout.
    nnType:
        Neighbor type tensor; used to mask padded neighbors (`-1` indicates padding).
    neighbor_I, neighbor_J:
        Index tensors defining neighbor pairs in the flattened neighbor list.
        Negative values are treated as padding and excluded.
    CALPHA:
        Ewald splitting parameter alpha.
    verbose:
        If True, prints additional info in k-space routine.

    Returns
    -------
    CC:
        Coulomb matrices of shape `(B, N, N)`.
    dCC_dxyz:
        Negative Cartesian derivatives (forces convention) of shape `(B, 3, N, N)`.

    Notes
    -----
    - The returned derivative matches existing behavior: `return CC, -dCC_dxyz`.
    - Division by distance uses `dR_dxyz = Rab / dR`; if any `dR` is zero, this can
      produce NaNs. Existing logic assumes those pairs are masked out by `nnType`
      and neighbor indices.
    """
    Ra = torch.stack((RX.unsqueeze(-1), RY.unsqueeze(-1), RZ.unsqueeze(-1)), dim=-1)
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)
    dR_dxyz = Rab / dR.unsqueeze(-1)

    use_ewald = lattice_vecs is not None

    CC_real, dCC_dxyz_real = ewald_real_space_vectorized_batch(
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

    if lattice_vecs is None:
        CC_k, dCC_dR_k = 0.0, 0.0
    else:
        dq_J = torch.zeros(Nr_atoms, dtype=dR.dtype, device=dR.device)
        CC_k, dCC_dR_k = ewald_k_space_vectorized(
            RX, RY, RZ, lattice_vecs, dq_J, Nr_atoms, Coulomb_acc, CALPHA, verbose
        )

    CC = CC_real + CC_k
    dCC_dxyz = dCC_dxyz_real + dCC_dR_k

    return CC, -dCC_dxyz


def ewald_real_space_vectorized_batch(
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
    """Compute the real-space Ewald contribution for a neighbor list (batched).

    Parameters
    ----------
    Hubbard_U:
        `(B, N)` float tensor (eV).
    TYPE:
        `(B, N)` long tensor (atomic numbers, 1-based).
    dR:
        Distances in neighbor-grid layout, typically `(B, N, max_neighbors)`.
    dR_dxyz:
        Unit vectors for each neighbor-grid entry; shape `dR.shape + (3,)`.
    nnType:
        Neighbor-grid padding mask with the same layout as `dR`; `-1` marks padding.
    neighbor_I, neighbor_J:
        Flattened pair lists of shape `(B, max_pairs)` with `-1` padding.
        These are aligned with `dR[nnType != -1]` within each batch item.
    CALPHA:
        Real-space Ewald splitting parameter alpha.
    h_damp_exp:
        Hydrogen damping exponent ζ for DFTB3.  See :func:`ewald_real_space_vectorized`.
    h5_params:
        H5 hydrogen-bond correction parameters (Řezáč, JCTC 13, 4804, 2017).
        Mutually exclusive with *h_damp_exp*.

    Returns
    -------
    CC_real:
        `(B, N, N)` real-space Coulomb matrices.
    dCC_dxyz_real:
        `(B, 3, N, N)` derivatives with respect to coordinates.
    """

    batch_size = Hubbard_U.shape[0]
    Nats = TYPE.shape[-1]
    nn_mask = nnType != -1
    valid_pairs = (neighbor_I >= 0) & (neighbor_J >= 0)
    safe_I = neighbor_I.clamp(min=0)
    safe_J = neighbor_J.clamp(min=0)

    # Constants
    CALPHA2 = CALPHA**2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)

    # Pair indices
    dR_mskd = dR[nn_mask]
    dR_dxyz_mskd = dR_dxyz[nn_mask]
    TYPE_I = TYPE.gather(1, safe_I)[valid_pairs]
    TYPE_J = TYPE.gather(1, safe_J)[valid_pairs]
    Ti = TFACT * Hubbard_U.gather(1, safe_I)[valid_pairs]
    Tj = TFACT * Hubbard_U.gather(1, safe_J)[valid_pairs]
    CC_real = torch.zeros((batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype)

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
    if h_damp_exp is not None:
        EV_TO_HA = 1.0 / 27.211386245988
        ANG_TO_BOHR = 1.0 / 0.52917721067
        is_H_I = TYPE_I == 1
        is_H_J = TYPE_J == 1
        h_mask = is_H_I | is_H_J
    # ────────────────────────────────────────────────────────────────────

    # ── H5 correction (Řezáč, JCTC 13, 4804, 2017) ─────────────────────
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
        _vdw_H_ang = _VDW_RADII_PM.get(1, 120) * 0.01  # pm → Å
        TYPE_I_h5 = TYPE_I
        TYPE_J_h5 = TYPE_J
        is_H_I_h5 = TYPE_I_h5 == 1
        is_H_J_h5 = TYPE_J_h5 == 1
        h5_mask_all = is_H_I_h5 ^ is_H_J_h5  # exactly one H
        if h5_mask_all.any():
            _apply_h5 = True
            heavy_Z = torch.where(is_H_I_h5, TYPE_J_h5, TYPE_I_h5)
            k_XH_all = torch.zeros_like(dR_mskd)
            sumVdW_all = torch.zeros_like(dR_mskd)
            for z_heavy, k_val in _h5_scaling.items():
                z_mask = h5_mask_all & (heavy_Z == z_heavy)
                if z_mask.any():
                    vdw_heavy_ang = _VDW_RADII_PM.get(z_heavy, -1) * 0.01
                    if vdw_heavy_ang > 0:
                        k_XH_all[z_mask] = k_val
                        sumVdW_all[z_mask] = _vdw_H_ang + vdw_heavy_ang
            h5_mask_active = h5_mask_all & (k_XH_all > 0.0)
            if not h5_mask_active.any():
                _apply_h5 = False
    # ────────────────────────────────────────────────────────────────────

    mask_same_elem = TYPE_I == TYPE_J
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
        Ui_au = Hubbard_U.gather(1, safe_I)[valid_pairs] * EV_TO_HA
        Uj_au = Hubbard_U.gather(1, safe_J)[valid_pairs] * EV_TO_HA
        r_au = dR_mskd * ANG_TO_BOHR
        rTmp = -((0.5 * (Ui_au + Uj_au)) ** h_damp_exp)
        D = torch.exp(rTmp * r_au**2)
        Dprime = 2.0 * rTmp * r_au * D * ANG_TO_BOHR

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
    batch_ids = (
        torch.arange(batch_size, device=dR.device).unsqueeze(1).expand_as(safe_I)
    )
    pair_index = (
        safe_I[valid_pairs] * Nats
        + safe_J[valid_pairs]
        + batch_ids[valid_pairs] * (Nats * Nats)
    )
    dtmp1_xyz = dtmp1.unsqueeze(-1) * dR_dxyz_mskd

    if tmp1.requires_grad or dtmp1_xyz.requires_grad:
        CC_real = torch.zeros(
            (batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype
        ).scatter_add(0, pair_index, tmp1)
        pair_index_xyz = pair_index.unsqueeze(0).expand(3, -1)
        dCC_dxyz_real = torch.zeros(
            (3, batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype
        ).scatter_add(1, pair_index_xyz, dtmp1_xyz.T)
    else:
        CC_real = torch.bincount(
            pair_index, weights=tmp1, minlength=batch_size * Nats * Nats
        )
        dCC_dxyz_real = torch.stack(
            [
                torch.bincount(
                    pair_index,
                    weights=dtmp1_xyz[:, axis],
                    minlength=batch_size * Nats * Nats,
                )
                for axis in range(3)
            ],
            dim=0,
        )

    CC_real = CC_real.reshape(batch_size, Nats, Nats)
    dCC_dxyz_real = dCC_dxyz_real.view(3, batch_size, Nats, Nats)
    dCC_dxyz_real = dCC_dxyz_real.permute(1, 0, 2, 3).contiguous()
    return CC_real, dCC_dxyz_real


def ewald_k_space_vectorized(
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    lattice_vecs: torch.Tensor,
    DELTAQ: torch.Tensor,
    Nr_atoms: int,
    COULACC: float,
    CALPHA: float,
    verbose: bool,
    do_vec: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute k-space Ewald contribution (batched).

    Parameters
    ----------
    RX, RY, RZ:
        `(B, N)` coordinates.
    lattice_vecs:
        `(B, 3, 3)` lattice vectors. Used for volume.
    DELTAQ:
        Charge tensor (currently only used for shape; kept for API compatibility).
    Nr_atoms:
        Number of atoms `N`.
    COULACC:
        Accuracy parameter.
    CALPHA:
        Ewald alpha.
    verbose:
        If True, prints progress.
    do_vec:
        Placeholder flag. Vectorized k-space path is not implemented for batch.

    Returns
    -------
    COULOMBV:
        `(B, N, N)` k-space Coulomb matrix contribution.
    dC_dR:
        `(B, 3, N, N)` Cartesian derivatives.
    """

    batch_size = RX.shape[0]
    device = RX.device

    COULVOL = torch.abs(torch.det(lattice_vecs))
    SQRTX = math.sqrt(-math.log(COULACC))

    CALPHA2 = CALPHA * CALPHA
    KCUTOFF = 2 * CALPHA * SQRTX
    KCUTOFF2 = KCUTOFF * KCUTOFF

    cell_inv = torch.linalg.inv(lattice_vecs)  # (B,3,3)
    RECIPVECS = 2.0 * math.pi * cell_inv.transpose(1, 2)  # (B,3,3)

    g1_norm = torch.linalg.norm(RECIPVECS[:, :, 0], dim=1)
    g2_norm = torch.linalg.norm(RECIPVECS[:, :, 1], dim=1)
    g3_norm = torch.linalg.norm(RECIPVECS[:, :, 2], dim=1)

    LMAX = torch.ceil(KCUTOFF / g1_norm).to(torch.int64)
    MMAX = torch.ceil(KCUTOFF / g2_norm).to(torch.int64)
    NMAX = torch.ceil(KCUTOFF / g3_norm).to(torch.int64)

    KECONST = 14.3996437701414  # in eV·Å/e²
    SQRTPI = math.sqrt(math.pi)

    COULOMBV = torch.zeros(
        (batch_size, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device
    )

    dC_dR = torch.zeros(
        (batch_size, 3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device
    )

    if do_vec:
        # Create meshgrid of all combinations
        print("vectorized k-space is not implemented for batched data")
        return
    else:
        # if verbose: print('   LMAX:', LMAX)
        print("   LMAX:", LMAX)
        for L in range(0, torch.max(LMAX) + 1):
            if verbose:
                print("  ", L)
            MMIN = 0 if L == 0 else -torch.max(MMAX)
            for M in range(MMIN, torch.max(MMAX) + 1):
                NMIN = 1 if (L == 0 and M == 0) else -torch.max(NMAX)
                for N in range(NMIN, torch.max(NMAX) + 1):
                    LMN = torch.tensor([L, M, N], dtype=RX.dtype, device=device)
                    kvec = torch.einsum("k,bkj->bj", LMN, RECIPVECS)

                    K2 = (kvec * kvec).sum(
                        dim=1
                    )  # similar to K2 = torch.dot(kvec, kvec) for non-batched

                    cutoff_mask = K2 > KCUTOFF2
                    if cutoff_mask.all():
                        continue

                    # print(K2, KCUTOFF2)
                    exp_factor = torch.exp(-K2 / (4 * CALPHA2))
                    prefactor = 8 * math.pi * exp_factor / (COULVOL * K2)
                    KEPREF = 14.3996437701414 * prefactor  # KECONST in eV·Å/e²

                    dot = torch.matmul(
                        kvec.view(batch_size, 1, 3), torch.stack((RX, RY, RZ), dim=1)
                    ).squeeze(0)

                    # coords = torch.stack((RX, RY, RZ), dim=1)              # (B,3,N)
                    # dot = torch.matmul(kvec.view(batch_size, 1, 3), coords).squeeze(1)   # (B,N)

                    sin_list = torch.sin(dot)
                    cos_list = torch.cos(dot)

                    # Use broadcasting for outer products
                    sin_i = sin_list.view(batch_size, -1, 1)
                    sin_j = sin_list.view(batch_size, 1, -1)
                    cos_i = cos_list.view(batch_size, -1, 1)
                    cos_j = cos_list.view(batch_size, 1, -1)

                    COULOMBV[~cutoff_mask] += (
                        KEPREF.unsqueeze(-1).unsqueeze(-1)
                        * (cos_i * cos_j + sin_i * sin_j)
                    )[~cutoff_mask]
                    force_term = KEPREF.unsqueeze(-1).unsqueeze(-1) * (
                        -cos_i * sin_j + sin_i * cos_j
                    )

                    dC_dR[~cutoff_mask] += (
                        force_term.unsqueeze(1) * kvec.view(batch_size, 3, 1, 1)
                    )[~cutoff_mask]

    # Self-interaction correction
    DELTAQ_vec = torch.eye(Nr_atoms, device=device)
    CORRFACT = 2 * KECONST * CALPHA / SQRTPI
    COULOMBV -= CORRFACT * DELTAQ_vec
    return COULOMBV, dC_dR


ewald_real_space_vectorized_batch_eager = ewald_real_space_vectorized_batch
ewald_k_space_vectorized_eager = ewald_k_space_vectorized
ewald_real_space_vectorized_batch = _maybe_compile(ewald_real_space_vectorized_batch)
ewald_k_space_vectorized = _maybe_compile(ewald_k_space_vectorized)
