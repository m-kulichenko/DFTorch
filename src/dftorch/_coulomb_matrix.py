from __future__ import annotations

import math
import time
from typing import Optional  # <-- add Optional

import torch


def coulomb_matrix_vectorized(
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
    LBox:
        Float tensor of shape `(3,)` (orthorhombic box lengths in Å). If `None`,
        the k-space term is skipped.
    lattice_vecs:
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

    ##################
    CC_real, dCC_dxyz_real = ewald_real_space_vectorized(
        Hubbard_U, TYPE, dR, dR_dxyz, nnType, neighbor_I, neighbor_J, CALPHA
    )
    ##################

    dq_J = torch.zeros(Nr_atoms, dtype=dR.dtype, device=dR.device)
    print("  Coulomb_Real t {:.1f} s".format(time.perf_counter() - start_time1))

    ## Second, k-space
    start_time1 = time.perf_counter()
    if LBox is None:
        CC_k, dCC_dR_k = 0.0, 0.0
    else:
        if verbose:
            print("  Doing Coulomb k")
        CC_k, dCC_dR_k = ewald_k_space_vectorized(
            RX, RY, RZ, LBox, lattice_vecs, dq_J, Nr_atoms, Coulomb_acc, CALPHA, verbose
        )
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-space Ewald contribution using a neighbor list (vectorized).

    Parameters
    ----------
    Hubbard_U:
        `(Nr_atoms,)` float tensor.
    TYPE:
        `(Nr_atoms,)` long tensor.
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
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
    tmp1 = CA.clone()
    dtmp1 = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI) / dR_mskd

    mask_same_elem = TYPE[neighbor_I] == TYPE[neighbor_J]
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
    CC_real.index_add_(0, neighbor_I * (Nats) + neighbor_J, tmp1)
    CC_real = CC_real.reshape(Nats, Nats)

    dCC_dxyz_real = torch.zeros((3, Nats * Nats), device=dR.device, dtype=dR.dtype)
    dCC_dxyz_real.index_add_(
        1, neighbor_I * (Nats) + neighbor_J, dtmp1 * dR_dxyz[nn_mask].T
    )
    dCC_dxyz_real = dCC_dxyz_real.reshape(3, Nats, Nats)
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
    """Reciprocal-space (k-space) Ewald contribution and derivatives.

    Parameters
    ----------
    RX, RY, RZ:
        `(Nr_atoms,)` coordinate tensors in Å.
    LBox:
        `(3,)` box lengths in Å (orthorhombic).
    lattice_vecs:
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

    Returns
    -------
    COULOMBV:
        `(Nr_atoms, Nr_atoms)` k-space Coulomb matrix contribution.
    dC_dR:
        `(3, Nr_atoms, Nr_atoms)` derivative tensor.
    """

    device = RX.device

    COULVOL = torch.abs(torch.det(lattice_vecs))
    SQRTX = math.sqrt(-math.log(COULACC))

    CALPHA2 = CALPHA * CALPHA
    KCUTOFF = 2 * CALPHA * SQRTX
    KCUTOFF2 = KCUTOFF * KCUTOFF

    RECIPVECS = torch.zeros((3, 3), dtype=RX.dtype, device=device)
    RECIPVECS[0, 0] = 2 * math.pi / LBox[0]
    RECIPVECS[1, 1] = 2 * math.pi / LBox[1]
    RECIPVECS[2, 2] = 2 * math.pi / LBox[2]

    LMAX = int(KCUTOFF / RECIPVECS[0, 0])
    MMAX = int(KCUTOFF / RECIPVECS[1, 1])
    NMAX = int(KCUTOFF / RECIPVECS[2, 2])

    KECONST = 14.3996437701414  # in eV·Å/e²
    SQRTPI = math.sqrt(math.pi)

    COULOMBV = torch.zeros((Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

    dC_dR = torch.zeros((3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

    if do_vec:
        # Create meshgrid of all combinations
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
        print("   LMAX:", LMAX)
        for L in range(0, LMAX + 1):
            if verbose:
                print("  ", L)
            MMIN = 0 if L == 0 else -MMAX
            for M in range(MMIN, MMAX + 1):
                NMIN = 1 if (L == 0 and M == 0) else -NMAX
                for N in range(NMIN, NMAX + 1):
                    kvec = (
                        L * RECIPVECS[:, 0] + M * RECIPVECS[:, 1] + N * RECIPVECS[:, 2]
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


### not working shell-resolved ###
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
    LBox : tuple of floats
        Simulation box lengths (Lx, Ly, Lz) used to define periodic boundary conditions.
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
        Ewald real-space damping parameter (α), typically precomputed externally.

    Returns
    -------
    CC_real : torch.Tensor of shape (Nr_atoms, Nr_atoms)
        Real-space contribution to the Coulomb interaction matrix.
    dCC_dxyz_real : torch.Tensor of shape (3, Nr_atoms, Nr_atoms)
        Derivatives of the real-space Coulomb interaction with respect to x, y, and z.

    Notes
    -----
    - This function computes the pairwise Coulomb interactions between atoms and their neighbors
      within a real-space cutoff derived from the Ewald α parameter.
    - It includes analytical short-range corrections for both same-element and different-element
      atomic pairs using atom-dependent Hubbard U parameters.
    - Derivatives (dCC/dR) are calculated analytically using the chain rule applied to screened
      Coulomb functions and short-range exponential terms.
    - Output matrices are assembled via scatter operations using index_put_ with accumulation.
    - Only the upper triangle of the interaction matrix is filled; symmetry must be enforced externally if needed.
    """
    CALPHA2 = CALPHA**2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)

    CDIM = len(structure.Hubbard_U_sr)
    max_ang_I = structure.const.max_ang[TYPE[neighbor_I]]
    max_ang_J = structure.const.max_ang[TYPE[neighbor_J]]

    # pair_mask_HH = (max_ang_I == 1) * (max_ang_J == 1)
    pair_mask_HX = (max_ang_I == 1) * (max_ang_J == 2)
    pair_mask_XH = (max_ang_I == 2) * (max_ang_J == 1)
    pair_mask_XX = (max_ang_I == 2) * (max_ang_J == 2)

    pair_mask_HY = (max_ang_I == 1) * (max_ang_J == 3)
    pair_mask_XY = (max_ang_I == 2) * (max_ang_J == 3)
    pair_mask_YH = (max_ang_I == 3) * (max_ang_J == 1)
    pair_mask_YX = (max_ang_I == 3) * (max_ang_J == 2)
    pair_mask_YY = (max_ang_I == 3) * (max_ang_J == 3)
    CC_real = torch.zeros((CDIM**2), device=dR.device, dtype=dR.dtype)
    dCC_dxyz_real = torch.zeros((3, CDIM**2), device=dR.device, dtype=dR.dtype)

    # Pair indices
    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    dR_dxyz_mskd = dR_dxyz[nn_mask].T
    CA = torch.erfc(CALPHA * dR_mskd) / dR_mskd
    tmp1 = CA.clone()
    dtmp1 = -(CA + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd**2) / SQRTPI) / dR_mskd

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

    ### s-p ###
    tmp_mask = (
        pair_mask_HX
        + pair_mask_XX
        + pair_mask_HY
        + pair_mask_YY
        + pair_mask_XY
        + pair_mask_YX
    )
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = structure.H_INDEX_START_U[neighbor_I[tmp_mask]] * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-s ###
    tmp_mask = (
        pair_mask_XH
        + pair_mask_XX
        + pair_mask_YH
        + pair_mask_YY
        + pair_mask_XY
        + pair_mask_YX
    )
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]]
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-p ###
    tmp_mask = pair_mask_XX + pair_mask_YY + pair_mask_XY + pair_mask_YX
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    # mask_same_elem = (structure.TYPE[neighbor_I] == structure.TYPE[neighbor_J])
    if mask_same_elem.any():
        dR_mskd_same = dR_mskd[mask_same_elem & tmp_mask]
        Ti_same_el = Ti[mask_same_elem[tmp_mask]]
        t1, dt1 = coul_same_elem_and_ang(Ti_same_el, dR_mskd_same)
        tmp1[mask_same_elem[tmp_mask]] -= t1
        dtmp1[mask_same_elem[tmp_mask]] -= dt1
    if (~mask_same_elem).any():
        dR_mskd_diff = dR_mskd[(~mask_same_elem) & tmp_mask]
        Ti_diff_el = Ti[~mask_same_elem[tmp_mask]]
        Tj_diff_el = Tj[~mask_same_elem[tmp_mask]]
        t1, dt1 = coul_diff_elem_and_ang(Ti_diff_el, Tj_diff_el, dR_mskd_diff)
        tmp1[~mask_same_elem[tmp_mask]] -= t1
        dtmp1[~mask_same_elem[tmp_mask]] -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### s-d ###
    tmp_mask = pair_mask_HY + pair_mask_XY + pair_mask_YY
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.U[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = structure.H_INDEX_START_U[neighbor_I[tmp_mask]] * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-s ###
    tmp_mask = pair_mask_YH + pair_mask_YX + pair_mask_YY
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.U[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]]
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### p-d ###
    tmp_mask = pair_mask_XY + pair_mask_YY
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Up[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 1) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-p ###
    tmp_mask = pair_mask_YX + pair_mask_YY
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Up[structure.TYPE[neighbor_J[tmp_mask]]]
    dR_mskd_diff = dR_mskd[tmp_mask]
    t1, dt1 = coul_diff_elem_and_ang(Ti, Tj, dR_mskd_diff)
    tmp1 -= t1
    tmp1 *= KECONST
    dtmp1 -= dt1
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 1
    CC_real.index_add_(0, (idx_row + idx_col), tmp1)
    dCC_dxyz_real.index_add_(1, (idx_row + idx_col), dtmp1 * dR_dxyz_mskd[:, tmp_mask])

    ### d-d ###
    tmp_mask = pair_mask_YY
    tmp1 = CA[tmp_mask].clone()
    dtmp1 = (
        -(
            CA[tmp_mask]
            + 2 * CALPHA * torch.exp(-CALPHA2 * dR_mskd[tmp_mask] ** 2) / SQRTPI
        )
        / dR_mskd[tmp_mask]
    )
    Ti = TFACT * structure.const.Ud[structure.TYPE[neighbor_I[tmp_mask]]]
    Tj = TFACT * structure.const.Ud[structure.TYPE[neighbor_J[tmp_mask]]]
    # mask_same_elem = (structure.TYPE[neighbor_I] == structure.TYPE[neighbor_J])
    if mask_same_elem.any():
        dR_mskd_same = dR_mskd[mask_same_elem & tmp_mask]
        Ti_same_el = Ti[mask_same_elem[tmp_mask]]
        t1, dt1 = coul_same_elem_and_ang(Ti_same_el, dR_mskd_same)
        tmp1[mask_same_elem[tmp_mask]] -= t1
        dtmp1[mask_same_elem[tmp_mask]] -= dt1
    if (~mask_same_elem).any():
        dR_mskd_diff = dR_mskd[(~mask_same_elem) & tmp_mask]
        Ti_diff_el = Ti[~mask_same_elem[tmp_mask]]
        Tj_diff_el = Tj[~mask_same_elem[tmp_mask]]
        t1, dt1 = coul_diff_elem_and_ang(Ti_diff_el, Tj_diff_el, dR_mskd_diff)
        tmp1[~mask_same_elem[tmp_mask]] -= t1
        dtmp1[~mask_same_elem[tmp_mask]] -= dt1
    tmp1 *= KECONST
    dtmp1 *= KECONST
    idx_row = (structure.H_INDEX_START_U[neighbor_I[tmp_mask]] + 2) * CDIM
    idx_col = structure.H_INDEX_START_U[neighbor_J[tmp_mask]] + 2
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
