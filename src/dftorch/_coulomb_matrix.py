import torch
import math
import time


def CoulombMatrix_vectorized(
    Hubbard_U,
    TYPE,
    RX,
    RY,
    RZ,
    LBox,
    lattice_vecs,
    Nr_atoms,
    Coulomb_acc,
    nnRx,
    nnRy,
    nnRz,
    nnType,
    neighbor_I,
    neighbor_J,
    CALPHA,
    verbose=False,
):
    """
    Computes the real-space and reciprocal-space (k-space) Ewald-summed Coulomb matrix and its
    derivatives for a system of atoms with periodic boundary conditions.

    This function uses a vectorized GPU-accelerated implementation of the Ewald summation to
    compute the electron-electron Coulomb interaction matrix and its derivatives with respect to
    atomic coordinates.

    Parameters
    ----------
    RX, RY, RZ : torch.Tensor of shape (Nr_atoms,)
        Cartesian coordinates of atoms along x, y, and z directions.
    LBox : tuple of floats
        Simulation box dimensions (Lx, Ly, Lz).
    Hubbard_U : torch.Tensor of shape (Nr_atoms,)
        Hubbard U parameters for the atomic species.
    TYPE : torch.Tensor of shape (Nr_atoms,)
        Integer identifiers of the atomic element types.
    Nr_atoms : int
        Number of atoms in the system.
    HDIM : int
        Total dimension of the Hamiltonian matrix (used in downstream operations).
    Coulomb_acc : float
        Desired accuracy of the Ewald summation (e.g., 1e-6).
    TIMERATIO : float
        Empirical scaling parameter used in the Ewald parameter (CALPHA) estimation.
    nnRx, nnRy, nnRz : torch.Tensor of shape (Nr_atoms, max_neighbors)
        Cartesian coordinates of the neighbor atoms for each atom (including periodic images).
    nrnnlist : torch.Tensor of shape (Nr_atoms, 1)
        Number of neighbors for each atom.
    nnType : torch.Tensor of shape (Nr_atoms, max_neighbors)
        Type/index of each neighbor atom (refers back to original atoms, not images).
    H_INDEX_START, H_INDEX_END : torch.Tensor
        Index mapping to define start and end of each atom's block in the Hamiltonian matrix.

    Returns
    -------
    CC : torch.Tensor of shape (Nr_atoms, Nr_atoms)
        Coulomb interaction matrix computed using Ewald summation.
    dCC_dxyz : torch.Tensor of shape (3, Nr_atoms, Nr_atoms)
        Derivatives of the Coulomb matrix with respect to atomic positions (x, y, z).

    Notes
    -----
    - The Ewald summation is split into real-space and reciprocal-space (k-space) contributions.
    - The real-space contribution is computed using a vectorized routine that exploits the
      precomputed neighbor list and periodic images.
    - The k-space part is computed separately using a fast reciprocal space summation method.
    - The returned matrix `CC` may be optionally symmetrized depending on downstream use.
    - The `dCC_dxyz` tensor provides gradients for force calculations or geometry optimizations.
    - This routine is designed for GPU execution with PyTorch tensors.
    """

    print("CoulombMatrix_vectorized")
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
    CC_real, dCC_dxyz_real = Ewald_Real_Space_vectorized(
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
        CC_k, dCC_dR_k = Ewald_k_Space_vectorized(
            RX, RY, RZ, LBox, lattice_vecs, dq_J, Nr_atoms, Coulomb_acc, CALPHA, verbose
        )
        print("  Coulomb_k t {:.1f} s\n".format(time.perf_counter() - start_time1))

    CC = CC_real + CC_k
    dCC_dxyz = dCC_dxyz_real + dCC_dR_k

    return CC, -dCC_dxyz  # , CC_sr, -dCC_dxyz_sr


def Ewald_Real_Space_vectorized(
    Hubbard_U, TYPE, dR, dR_dxyz, nnType, neighbor_I, neighbor_J, CALPHA
):
    """
    This one is vectorized in a fashion of _slater_koster_pair.py.
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
def Ewald_k_Space_vectorized(
    RX,
    RY,
    RZ,
    LBox,
    lattice_vecs,
    DELTAQ,
    Nr_atoms,
    COULACC,
    CALPHA,
    verbose,
    do_vec=False,
):
    """
    Computes the reciprocal-space (k-space) contribution to the Coulomb interaction matrix
    and its derivatives using the Ewald summation method.

    Parameters
    ----------
    RX, RY, RZ : torch.Tensor
        Tensors of shape (Nr_atoms,) representing the x, y, and z coordinates of atomic positions.
    LBox : torch.Tensor
        Tensor of shape (3,) containing the simulation box lengths in Ångströms (assumed orthorhombic).
    DELTAQ : torch.Tensor
        Not used in this function, but typically expected to hold atomic charge differences.
    Nr_atoms : int
        Total number of atoms in the system.
    COULACC : float
        Desired accuracy of the Coulomb sum (e.g., 1e-6). Controls the reciprocal cutoff.
    TIMERATIO : float
        Volume-normalized parameter for balancing real and reciprocal-space contributions.
    do_vec : bool, optional
        If True, perform a fully vectorized computation of the k-space summation.
        Faster but more memory-intensive. Default is False (sequential).

    Returns
    -------
    COULOMBV : torch.Tensor
        (Nr_atoms, Nr_atoms) matrix of Coulomb interactions computed via reciprocal-space Ewald sum.
    dC_dR : torch.Tensor
        (3, Nr_atoms, Nr_atoms) tensor containing the derivatives of the Coulomb interaction
        matrix with respect to atomic positions (used for force computations).

    Notes
    -----
    - Uses an orthorhombic unit cell; general cells not currently supported.
    - K-space cutoff is determined automatically from the `COULACC` and `TIMERATIO` parameters.
    - A self-interaction correction is included in the returned Coulomb matrix.
    - Memory usage may become prohibitive for large systems if `do_vec=True`.

    References
    ----------
    - Ewald, P. P. (1921). Die Berechnung optischer und elektrostatischer Gitterpotentiale.
      Annalen der Physik, 369(3), 253–287.
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
def Ewald_Real_Space_vectorized_sr(
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


def coul_diff_elem_and_ang(Ti_diff_el, Tj_diff_el, dR_mskd_diff):
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


def coul_same_elem_and_ang(Ti_same_el, dR_mskd_same):
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
