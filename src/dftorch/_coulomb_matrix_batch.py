from __future__ import annotations

import math
from typing import Optional, Dict  # <-- add Optional, Dict


import torch

from ._coulomb_matrix import (
    _VDW_RADII_PM,
    _H5_DEFAULT_SCALING,
    _GAUSS_WIDTH_FACTOR,
)


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

    CC_real, dCC_dxyz_real = ewald_real_space_vectorized_batch(
        Hubbard_U,
        TYPE,
        dR,
        dR_dxyz,
        nnType,
        neighbor_I,
        neighbor_J,
        CALPHA,
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
        Distances for neighbor list entries (shape depends on neighbor list layout).
    dR_dxyz:
        Unit vectors for each neighbor entry; must have shape `dR.shape + (3,)`.
    nnType:
        Tensor used as padding mask (`-1` indicates invalid/unfilled neighbor entries).
    neighbor_I, neighbor_J:
        Flattened neighbor index lists (same shape), containing `-1` as padding.
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
    nn_mask = (
        nnType != -1
    )  # & dist_mask # mask to exclude zero padding from the neigh list

    dR_mskd = dR[nn_mask]
    Ti = TFACT * Hubbard_U.gather(1, safe_I)[valid_pairs]
    Tj = TFACT * Hubbard_U.gather(1, safe_J)[valid_pairs]
    CC_real = torch.zeros((batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype)
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
    tmp1 = CA.clone()
    dtmp1 = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI) / dR_mskd

    # ── Hydrogen gamma damping (DFTB3) ──────────────────────────────────
    if h_damp_exp is not None:
        EV_TO_HA = 1.0 / 27.211386245988
        ANG_TO_BOHR = 1.0 / 0.52917721067
        TYPE_I = TYPE.gather(1, safe_I)[valid_pairs]
        TYPE_J = TYPE.gather(1, safe_J)[valid_pairs]
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
        TYPE_I_h5 = TYPE.gather(1, safe_I)[valid_pairs]
        TYPE_J_h5 = TYPE.gather(1, safe_J)[valid_pairs]
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

    mask_same_elem = (
        TYPE.gather(1, safe_I)[valid_pairs] == TYPE.gather(1, safe_J)[valid_pairs]
    )
    if mask_same_elem.any():
        dR_mskd_same = dR_mskd[mask_same_elem]
        Ti_same_el = Ti[mask_same_elem]
        TI2 = Ti_same_el**2
        TI3 = TI2 * Ti_same_el
        SSB = TI3 / 48.0
        SSC = 3 * TI2 / 16.0
        SSD = 11 * Ti_same_el / 16.0
        EXPTI = torch.exp(-Ti_same_el * dR_mskd_same)
        tmp = SSB * dR_mskd_same**2 + SSC * dR_mskd_same + SSD + 1.0 / dR_mskd_same
        sr_val = EXPTI * tmp
        sr_deriv = EXPTI * (
            (-Ti_same_el) * tmp + (2 * SSB * dR_mskd_same + SSC - 1.0 / dR_mskd_same**2)
        )

        # Apply H-damping to same-element pairs (H-H only)
        if h_damp_exp is not None:
            h_mask_same = h_mask[mask_same_elem]
            if h_mask_same.any():
                Ui_au = (
                    Hubbard_U.gather(1, safe_I)[valid_pairs][mask_same_elem][
                        h_mask_same
                    ]
                    * EV_TO_HA
                )
                Uj_au = (
                    Hubbard_U.gather(1, safe_J)[valid_pairs][mask_same_elem][
                        h_mask_same
                    ]
                    * EV_TO_HA
                )
                r_au = dR_mskd_same[h_mask_same] * ANG_TO_BOHR
                rTmp = -((0.5 * (Ui_au + Uj_au)) ** h_damp_exp)
                D = torch.exp(rTmp * r_au**2)
                Dprime = 2.0 * rTmp * r_au * D * ANG_TO_BOHR
                sr_val_h = sr_val[h_mask_same]
                sr_deriv_h = sr_deriv[h_mask_same]
                sr_val[h_mask_same] = sr_val_h * D
                sr_deriv[h_mask_same] = sr_deriv_h * D + sr_val_h * Dprime

        tmp1[mask_same_elem] -= sr_val
        dtmp1[mask_same_elem] -= sr_deriv

    if (~mask_same_elem).any():
        dR_mskd_diff = dR_mskd[~mask_same_elem]
        Ti_diff_el = Ti[~mask_same_elem]
        Tj_diff_el = Tj[~mask_same_elem]
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
        sr_val = EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2
        sr_deriv = EXPTI * (
            (-Ti_diff_el) * COULOMBV_tmp1 + SC / dR_mskd_diff**2
        ) + EXPTJ * ((-Tj_diff_el) * COULOMBV_tmp2 + SF / dR_mskd_diff**2)

        # Apply H-damping to different-element pairs (H-X or X-H)
        if h_damp_exp is not None:
            h_mask_diff = h_mask[~mask_same_elem]
            if h_mask_diff.any():
                Ui_au = (
                    Hubbard_U.gather(1, safe_I)[valid_pairs][~mask_same_elem][
                        h_mask_diff
                    ]
                    * EV_TO_HA
                )
                Uj_au = (
                    Hubbard_U.gather(1, safe_J)[valid_pairs][~mask_same_elem][
                        h_mask_diff
                    ]
                    * EV_TO_HA
                )
                r_au = dR_mskd_diff[h_mask_diff] * ANG_TO_BOHR
                rTmp = -((0.5 * (Ui_au + Uj_au)) ** h_damp_exp)
                D = torch.exp(rTmp * r_au**2)
                Dprime = 2.0 * rTmp * r_au * D * ANG_TO_BOHR
                sr_val_h = sr_val[h_mask_diff]
                sr_deriv_h = sr_deriv[h_mask_diff]
                sr_val[h_mask_diff] = sr_val_h * D
                sr_deriv[h_mask_diff] = sr_deriv_h * D + sr_val_h * Dprime

        # Apply H5 correction to different-element pairs (exactly one H)
        # γ_H5 = γ·(1+G) − G/r ;  γ'_H5 = γ·G' + γ'·(1+G) − (G'/r − G/r²)
        elif _apply_h5:
            h5_mask_diff = h5_mask_active[~mask_same_elem]
            if h5_mask_diff.any():
                r_h5 = dR_mskd_diff[h5_mask_diff]  # in Å
                k_h5 = k_XH_all[~mask_same_elem][h5_mask_diff]
                svdw = sumVdW_all[~mask_same_elem][h5_mask_diff]  # in Å
                r0 = _h5_r_scaling * svdw
                cc = _h5_w_scaling * svdw * _GAUSS_WIDTH_FACTOR
                gauss = k_h5 * torch.exp(-0.5 * (r_h5 - r0) ** 2 / cc**2)
                dgauss = -gauss * (r_h5 - r0) / cc**2

                sr_val_h5 = sr_val[h5_mask_diff]
                sr_deriv_h5 = sr_deriv[h5_mask_diff]
                sr_val[h5_mask_diff] = sr_val_h5 * (1.0 + gauss) - gauss / r_h5
                deriv1 = sr_val_h5 * dgauss + sr_deriv_h5 * (1.0 + gauss)
                deriv2 = dgauss / r_h5 - gauss / r_h5**2
                sr_deriv[h5_mask_diff] = deriv1 - deriv2

        tmp1[~mask_same_elem] -= sr_val
        dtmp1[~mask_same_elem] -= sr_deriv

    tmp1 *= KECONST
    dtmp1 *= KECONST
    batch_ids = (
        torch.arange(batch_size, device=dR.device).unsqueeze(1).expand_as(Hubbard_U)
    )
    batch_ids = batch_ids.gather(1, safe_J)[valid_pairs]
    batch_block_offset = batch_ids * (Nats * Nats)
    CC_real.index_add_(
        0, safe_I[valid_pairs] * (Nats) + safe_J[valid_pairs] + batch_block_offset, tmp1
    )
    CC_real = CC_real.reshape(batch_size, Nats, Nats)

    dCC_dxyz_real = torch.zeros(
        (3, batch_size * Nats * Nats), device=dR.device, dtype=dR.dtype
    )
    dCC_dxyz_real.index_add_(
        1,
        safe_I[valid_pairs] * (Nats) + safe_J[valid_pairs] + batch_block_offset,
        dtmp1 * dR_dxyz[nn_mask].T,
    )
    dCC_dxyz_real = dCC_dxyz_real.view(3, batch_size, Nats, Nats)
    dCC_dxyz_real = dCC_dxyz_real.permute(1, 0, 2, 3).contiguous()
    return CC_real, dCC_dxyz_real


@torch.compile(dynamic=False)
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
