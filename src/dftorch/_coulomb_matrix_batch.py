from __future__ import annotations

import math
from typing import Optional  # <-- add Optional


import torch


def coulomb_matrix_vectorized_batch(
    Hubbard_U: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    LBox: Optional[torch.Tensor],  # <-- was: torch.Tensor | None
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batched Coulomb matrix and its Cartesian derivatives.

    This function builds the Coulomb interaction matrix `CC` for each batch item
    using an Ewald-like split into:

    - Real-space sum over a neighbor list (always applied)
    - Reciprocal-space (k-space) contribution (only if `LBox` is provided)

    Parameters
    ----------
    Hubbard_U:
        Tensor of shape `(B, N)` with per-atom Hubbard U parameters.
    TYPE:
        Long tensor of shape `(B, N)` with per-atom type indices.
    RX, RY, RZ:
        Coordinate tensors of shape `(B, N)` for x/y/z in Angstrom.
    LBox:
        Tensor of shape `(B, 3)` with box lengths. If `None`, k-space is skipped.
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
        Hubbard_U, TYPE, dR, dR_dxyz, nnType, neighbor_I, neighbor_J, CALPHA
    )

    if LBox is None:
        CC_k, dCC_dR_k = 0.0, 0.0
    else:
        dq_J = torch.zeros(Nr_atoms, dtype=dR.dtype, device=dR.device)
        CC_k, dCC_dR_k = ewald_k_space_vectorized(
            RX, RY, RZ, LBox, lattice_vecs, dq_J, Nr_atoms, Coulomb_acc, CALPHA, verbose
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the real-space Ewald contribution for a neighbor list (batched).

    Parameters
    ----------
    Hubbard_U:
        `(B, N)` float tensor
    TYPE:
        `(B, N)` long tensor
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
        tmp1[mask_same_elem] -= EXPTI * tmp
        dtmp1[mask_same_elem] -= EXPTI * (
            (-Ti_same_el) * tmp + (2 * SSB * dR_mskd_same + SSC - 1.0 / dR_mskd_same**2)
        )
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
        tmp1[~mask_same_elem] -= EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2
        dtmp1[~mask_same_elem] -= EXPTI * (
            (-Ti_diff_el) * COULOMBV_tmp1 + SC / dR_mskd_diff**2
        ) + EXPTJ * ((-Tj_diff_el) * COULOMBV_tmp2 + SF / dR_mskd_diff**2)

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
    LBox: torch.Tensor,
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
    LBox:
        `(B, 3)` box lengths (orthorhombic).
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

    RECIPVECS = torch.zeros((batch_size, 3, 3), dtype=RX.dtype, device=device)
    RECIPVECS[:, 0, 0] = 2 * math.pi / LBox[:, 0]
    RECIPVECS[:, 1, 1] = 2 * math.pi / LBox[:, 1]
    RECIPVECS[:, 2, 2] = 2 * math.pi / LBox[:, 2]

    LMAX = (KCUTOFF / RECIPVECS[:, 0, 0]).int()
    MMAX = (KCUTOFF / RECIPVECS[:, 1, 1]).int()
    NMAX = (KCUTOFF / RECIPVECS[:, 2, 2]).int()

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
                    kvec = (
                        L * RECIPVECS[:, :, 0]
                        + M * RECIPVECS[:, :, 1]
                        + N * RECIPVECS[:, :, 2]
                    )
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
