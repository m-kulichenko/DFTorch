import torch
import math
import time

def CoulombMatrix_vectorized(RX, RY, RZ, LBox, Hubbard_U, Element_Type, Nr_atoms, HDIM,
                  Coulomb_acc, TIMERATIO, nnRx, nnRy, nnRz, nrnnlist, nnType,
                  H_INDEX_START, H_INDEX_END):
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
    Element_Type : torch.Tensor of shape (Nr_atoms,)
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


    print('CoulombMatrix_vectorized')
    #torch.cuda.synchronize()
    print('  Do Coulomb Real')
    start_time1 = time.perf_counter()
    
    COULVOL = LBox[0] * LBox[1] * LBox[2]
    SQRTX = math.sqrt(-math.log(Coulomb_acc))
    CALPHA = math.sqrt(math.pi) * ((TIMERATIO * Nr_atoms / (COULVOL ** 2)) ** (1.0 / 6.0))
    COULCUT = SQRTX / CALPHA
    if COULCUT > 50.0:
        COULCUT = 50.0
        CALPHA = SQRTX / COULCUT
    CALPHA2 = CALPHA * CALPHA
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)
    
    Ra = torch.stack((RX.unsqueeze(-1), RY.unsqueeze(-1), RZ.unsqueeze(-1)), dim=-1)
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)
    dR_dxyz = Rab/dR.unsqueeze(-1)
    dist_mask = (dR <= COULCUT)*(dR > 1e-12)
    
    ##################
    
    CC_real, dCC_dxyz_real = Ewald_Real_Space_vectorized(RX, RY, RZ, dR, dR_dxyz, dist_mask, 
                                                         LBox, Hubbard_U, Element_Type, Nr_atoms, HDIM,
                  Coulomb_acc, TIMERATIO, nnRx, nnRy, nnRz, nrnnlist, nnType,
                  H_INDEX_START, H_INDEX_END, CALPHA)
    ##################

    dq_J = torch.zeros(Nr_atoms, dtype=RX.dtype, device = RX.device)

    time_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
# this is a less vectorized option. may be more memory efficient?
#     for J in range(Nr_atoms):
#         torch.cuda.synchronize()
#         start_time2 = time.perf_counter()
#         #print(J)
#         dq_J[J] = 1.0
        
#         ## First, real space        
#         mask_pert_neigh_vec = nnType == J
        
#         # to avoid computing already computed. SImilar to for J in range(Nr_atoms): for I in range(J,Nr_atoms):
#         mask_pert_neigh_vec[0:J] = False
#         mask_to_match_IJ = torch.repeat_interleave(torch.arange(Nr_atoms, device = RX.device), torch.sum(mask_pert_neigh_vec, dim=-1))
        
#         torch.cuda.synchronize()
#         time_dict[5] += time.perf_counter() - start_time2
        
#         pot, dc_dxyz = Ewald_Real_Space_vectorized_less(J, dR[mask_pert_neigh_vec], dR_dxyz[mask_pert_neigh_vec],
#                   dist_mask[mask_pert_neigh_vec],
#                   LBox, dq_J, Hubbard_U, Element_Type, Nr_atoms, Coulomb_acc, TIMERATIO, nrnnlist, nnType[mask_pert_neigh_vec],
#                   mask_pert_neigh_vec.clone(), mask_to_match_IJ, CALPHA, CALPHA2, COULCUT, time_dict)

#         torch.cuda.synchronize()
#         start_time2 = time.perf_counter()
                
#         CC_real[:, J] = pot
#         dCC_dxyz_real[:, :, J] = dc_dxyz
#         dq_J[J] = 0.0
        
#         torch.cuda.synchronize()
#         time_dict[6] += time.perf_counter() - start_time2
        
        
#     print(CC_real_test - (CC_real + CC_real.T - torch.diag(torch.diagonal(CC_real))))
#     print(torch.max(abs(CC_real_test - CC_real - CC_real.T)))
#     print(torch.max(abs(dCC_dxyz_real_test - (dCC_dxyz_real - torch.transpose(dCC_dxyz_real, -1,-2) - torch.diag_embed(dCC_dxyz_real.diagonal(dim1=1, dim2=2)) )   )))

    print("  Coulomb_Real t {:.1f} s\n".format( time.perf_counter()-start_time1 ))

    
    ## Second, k-space
    start_time1 = time.perf_counter()
    print('  Doing Coulomb k')

    CC_k, dCC_dR_k = Ewald_k_Space_vectorized(RX, RY, RZ, LBox, dq_J, Nr_atoms, Coulomb_acc, TIMERATIO)
    print("  Coulomb_k t {:.1f} s\n".format( time.perf_counter()-start_time1 ))
    
    
    #CC = CC_real + CC_real.T - torch.diag(torch.diagonal(CC_real)) + CC_k # this is for a less vectorized code
    CC = CC_real + CC_k
    dCC_dxyz = dCC_dxyz_real + dCC_dR_k
    return CC, -dCC_dxyz


def Ewald_Real_Space_vectorized(RX, RY, RZ, dR, dR_dxyz, dist_mask, LBox, Hubbard_U, Element_Type, Nr_atoms, HDIM,
                  Coulomb_acc, TIMERATIO, nnRx, nnRy, nnRz, nrnnlist, nnType,
                  H_INDEX_START, H_INDEX_END, CALPHA):
    """
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
    Element_Type : torch.Tensor of shape (Nr_atoms,)
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
    #torch.cuda.synchronize()
    start_time1 = time.perf_counter()
    
    # Constants
    SQRTX = math.sqrt(-math.log(Coulomb_acc))
    COULCUT = min(SQRTX / CALPHA, 50.0)
    CALPHA2 = CALPHA ** 2
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = math.sqrt(math.pi)
    
    dR_dist_mskd = dR[dist_mask]
    MAGR2_dist_mskd = dR_dist_mskd **2

    # Pair indices
    i_atoms = torch.arange(Nr_atoms, device=RX.device).view(-1, 1)  # (N, 1)
    j_atoms = nnType  # (N, MAXNN)

    # Element and U type
    Ti = TFACT * Hubbard_U[i_atoms]  # (N, 1)
    Tj = TFACT * Hubbard_U[j_atoms]  # (N, MAXNN)

    same_elem_mask = (Element_Type[i_atoms] == Element_Type[j_atoms]) & dist_mask  # (N, MAXNN)
    diff_elem_mask = (~same_elem_mask) & dist_mask  # (N, MAXNN)
    
    mask_same_elem_type_inside_dist_mask = (Element_Type[i_atoms] == Element_Type[j_atoms])[dist_mask]
    
    MAGR2 = dR ** 2 + (~dist_mask) * 1e10

    # Initialize potential
    CA = torch.erfc(CALPHA * dR_dist_mskd) / dR_dist_mskd
    COULOMBV = CA.clone()
    
    CA += 2 * CALPHA * torch.exp(-CALPHA2 * MAGR2_dist_mskd) / SQRTPI
    dC_dR = -CA / dR_dist_mskd

    ## Same-element correction
    if mask_same_elem_type_inside_dist_mask.any():
        dR_dist_mskd_same = dR_dist_mskd[mask_same_elem_type_inside_dist_mask]
        MAGR2_dist_mskd_same = MAGR2_dist_mskd[mask_same_elem_type_inside_dist_mask]
        TI_same = Ti.expand_as(Tj)[dist_mask][mask_same_elem_type_inside_dist_mask]
        TI2 = TI_same ** 2
        TI3 = TI2 * TI_same
        
        SSB = TI3 / 48.0
        SSC = 3 * TI2 / 16.0
        SSD = 11 * TI_same / 16.0
        EXPTI = torch.exp(-TI_same * dR_dist_mskd_same)
        tmp = (SSB * MAGR2_dist_mskd_same + SSC * dR_dist_mskd_same + SSD + 1. / dR_dist_mskd_same)

        COULOMBV[mask_same_elem_type_inside_dist_mask] -= EXPTI * tmp
        dC_dR[mask_same_elem_type_inside_dist_mask] -= EXPTI * (
            (-TI_same) * tmp + (2 * SSB * dR_dist_mskd_same + SSC - 1. / MAGR2_dist_mskd_same)
        )

    ## Diff-element correction
    if (~mask_same_elem_type_inside_dist_mask).any():
        dR_dist_mskd_diff = dR_dist_mskd[~mask_same_elem_type_inside_dist_mask]
        MAGR2_dist_mskd_diff = MAGR2_dist_mskd[~mask_same_elem_type_inside_dist_mask]

        TI_diff = Ti.expand_as(Tj)[dist_mask][~mask_same_elem_type_inside_dist_mask]
        TJ_diff = Tj[dist_mask][~mask_same_elem_type_inside_dist_mask]
        TI2 = TI_diff ** 2
        TI4 = TI2 ** 2
        TI6 = TI4 * TI2
        TJ2 = TJ_diff ** 2
        TJ4 = TJ2 ** 2
        TJ6 = TJ4 * TJ2

        EXPTI = torch.exp(-TI_diff * dR_dist_mskd_diff)
        EXPTJ = torch.exp(-TJ_diff * dR_dist_mskd_diff)

        TI2MTJ2 = TI2 - TJ2
        TJ2MTI2 = -TI2MTJ2

        SB = TJ4 * TI_diff / (2 * TI2MTJ2 ** 2)
        SC = (TJ6 - 3 * TJ4 * TI2) / (TI2MTJ2 ** 3)
        SE = TI4 * TJ_diff / (2 * TJ2MTI2 ** 2)
        SF = (TI6 - 3 * TI4 * TJ2) / (TJ2MTI2 ** 3)
        
        COULOMBV_tmp1 = (SB - SC / dR_dist_mskd_diff)
        COULOMBV_tmp2 = (SE - SF / dR_dist_mskd_diff)
        COULOMBV[~mask_same_elem_type_inside_dist_mask] -= EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2

        dC_dR[~mask_same_elem_type_inside_dist_mask] -= EXPTI * ((-TI_diff)*COULOMBV_tmp1 + SC/MAGR2_dist_mskd_diff) + \
            EXPTJ * ((-TJ_diff)*COULOMBV_tmp2 + SF/MAGR2_dist_mskd_diff)

    # === Assembling full CC_real matrix === #
    COULOMBV *= KECONST
    dC_dR *= KECONST
    
    CC_real = torch.zeros((Nr_atoms, Nr_atoms), device=RX.device, dtype=RX.dtype)
    dCC_dxyz_real = torch.zeros((3, Nr_atoms, Nr_atoms), device=RX.device, dtype=RX.dtype)

    # Each i has MAXNN neighbors j
    i_idx = i_atoms.expand_as(j_atoms)[dist_mask]
    j_idx = j_atoms[dist_mask]
    CC_real.index_put_((i_idx, j_idx), COULOMBV, accumulate=True)

    # Derivatives    
    dC_dxyz = dC_dR.unsqueeze(-1) * dR_dxyz[dist_mask]  # (N, MAXNN, 3)
    for k in range(3):
        dCC_dxyz_real[k].index_put_((i_idx, j_idx), dC_dxyz[:, k], accumulate=True)

    return CC_real, dCC_dxyz_real

def Ewald_k_Space_vectorized(RX, RY, RZ, LBox, DELTAQ, Nr_atoms, COULACC, TIMERATIO, do_vec=False):
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

    COULVOL = LBox[0] * LBox[1] * LBox[2]
    SQRTX = math.sqrt(-math.log(COULACC))
    CALPHA = math.sqrt(math.pi) * ((TIMERATIO * Nr_atoms / (COULVOL ** 2)) ** (1/6))
    COULCUT = SQRTX / CALPHA

    if COULCUT > 50:
        COULCUT = 50
        CALPHA = SQRTX / COULCUT

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

    FCOUL = torch.zeros((3, Nr_atoms), dtype=RX.dtype, device=device)
    COULOMBV = torch.zeros((Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)
    
    dC_dR = torch.zeros((3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)
    
    if do_vec:
        start_time = time.perf_counter()
        # Create meshgrid of all combinations
        print('  init L,M,N,K')
        L_vals = torch.arange(0, LMAX + 1)
        M_vals = torch.arange(-MMAX, MMAX + 1)
        N_vals = torch.arange(-NMAX, NMAX + 1)
        L_vec, M_vec, N_vec = torch.meshgrid(L_vals, M_vals, N_vals, indexing='ij')

        L_vec = L_vec.flatten()
        M_vec = M_vec.flatten()
        N_vec = N_vec.flatten()

        mask = ~((L_vec == 0)*(M_vec < 0))  # exclude L==0 and M<0
        mask &= ~((L_vec == 0)*(M_vec == 0)*(N_vec < 1)) # exclude L==0, M==0, N<1
        L_vec = L_vec[mask]
        M_vec = M_vec[mask]
        N_vec = N_vec[mask]

        # Step 3: Stack into a (N, 3) tensor of LMN vectors
        LMN = torch.stack([L_vec, M_vec, N_vec], dim=1).to(dtype=RX.dtype, device=device)  # shape: (num_valid, 3)
        K_vectors = LMN @ RECIPVECS
        K2 = torch.sum(K_vectors**2, dim=1)

        exp_factor = torch.exp(-K2 / (4 * CALPHA2))
        prefactor = 8 * torch.pi * exp_factor / (COULVOL * K2)
        KEPREF = KECONST * prefactor

        dot = K_vectors[:,0] * RX.unsqueeze(-1) + K_vectors[:,1] * RY.unsqueeze(-1) + K_vectors[:,2] * RZ.unsqueeze(-1)
        sin_list = torch.sin(dot)
        cos_list = torch.cos(dot)

        COULOMBV += (KEPREF * (cos_list.unsqueeze(1) * cos_list.unsqueeze(0) + sin_list.unsqueeze(1) * sin_list.unsqueeze(0))).sum(-1)
        force_tmp =( KEPREF * (-cos_list.unsqueeze(1) * sin_list.unsqueeze(0) + sin_list.unsqueeze(1) * cos_list.unsqueeze(0)))
        dC_dR[0] += (force_tmp*K_vectors[:, 0]).sum(-1)
        dC_dR[1] += (force_tmp*K_vectors[:, 1]).sum(-1)
        dC_dR[2] += (force_tmp*K_vectors[:, 2]).sum(-1)
    else:
        start_time = time.perf_counter()
        print('   LMAX:', LMAX)
        for L in range(0, LMAX + 1):
            print('  ',L)
            MMIN = 0 if L == 0 else -MMAX
            for M in range(MMIN, MMAX + 1):
                NMIN = 1 if (L == 0 and M == 0) else -NMAX
                for N in range(NMIN, NMAX + 1):
                    #print('      ', N)
#                     Kx = L * RECIPVECS[0, 0] + M * RECIPVECS[1, 0] + N * RECIPVECS[2, 0]
#                     Ky = L * RECIPVECS[0, 1] + M * RECIPVECS[1, 1] + N * RECIPVECS[2, 1]
#                     Kz = L * RECIPVECS[0, 2] + M * RECIPVECS[1, 2] + N * RECIPVECS[2, 2]
#                     K = torch.tensor([Kx, Ky, Kz], dtype=RX.dtype, device=device)
#                     K2 = torch.dot(K, K)                

#                     if K2 <= KCUTOFF2:
#                         exp_factor = torch.exp(-K2 / (4 * CALPHA2))
#                         prefactor = 8 * torch.pi * exp_factor / (COULVOL * K2)
#                         KEPREF = KECONST * prefactor

#                         dot = K[0] * RX + K[1] * RY + K[2] * RZ
#                         sin_list = torch.sin(dot)
#                         cos_list = torch.cos(dot)

#                         COULOMBV += KEPREF * (torch.outer(cos_list, cos_list) + torch.outer(sin_list, sin_list))
#                         force_tmp = KEPREF * (-torch.outer(cos_list, sin_list) + torch.outer(sin_list, cos_list))
#                         dC_dR += force_tmp*K[:, None, None]
                        
                    kvec = L * RECIPVECS[:, 0] + M * RECIPVECS[:, 1] + N * RECIPVECS[:, 2]
                    K2 = torch.dot(kvec, kvec)
                    if K2 > KCUTOFF2:
                        continue

                    exp_factor = torch.exp(-K2 / (4 * CALPHA2))
                    prefactor = 8 * math.pi * exp_factor / (COULVOL * K2)
                    KEPREF = 14.3996437701414 * prefactor  # KECONST in eV·Å/e²

                    dot = torch.matmul(kvec.view(1, 3), torch.stack((RX, RY, RZ), dim=0)).squeeze(0)  # shape (N,)
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
    DELTAQ_vec = torch.eye(Nr_atoms, device = device)
    CORRFACT = 2 * KECONST * CALPHA / SQRTPI
    COULOMBV -= CORRFACT*DELTAQ_vec
    return COULOMBV, dC_dR

def Ewald_Real_Space_vectorized_less(pert_J, dR, dR_dxyz, dist_mask, LBox, DELTAQ, U, Element_Type, Nr_atoms, COULACC, TIMERATIO, nrnnlist, nnType, mask_pert_neigh_vec, mask_to_match_IJ, CALPHA, CALPHA2, COULCUT, time_dict):
    '''
    # this is a less vectorized option. may be more memory efficient?
    '''

    torch.cuda.synchronize()
    start_time1 = time.perf_counter()
    
    RELPERM = 1.0
    KECONST = 14.3996437701414 * RELPERM
    TFACT = 16.0 / (5.0 * KECONST)
    SQRTPI = torch.pi**0.5
 
    J = nnType
    
    # === Masking & Indexing === #
    torch.cuda.synchronize()
    time_dict[0] += time.perf_counter() - start_time1
    start_time1 = time.perf_counter()
    
    
    dR_dist_mskd = dR[dist_mask]
    mask_same_elem_type_inside_dist_mask = (Element_Type[mask_to_match_IJ[dist_mask]] == Element_Type[pert_J])
    MAGR2_dist_mskd = dR_dist_mskd **2
    
    TI = TFACT * U[mask_to_match_IJ][dist_mask]
    TJ = TFACT * U[J[dist_mask]]

    torch.cuda.synchronize()
    time_dict[1] += time.perf_counter() - start_time1
    start_time1 = time.perf_counter()
    
    # === Preallocate result === #
    CA = torch.erfc(CALPHA * dR_dist_mskd) / dR_dist_mskd
    COULOMBV = CA.clone()
    CA += 2 * CALPHA * torch.exp(-CALPHA2 * MAGR2_dist_mskd) / SQRTPI
    dC_dR = -CA / dR_dist_mskd
    
    torch.cuda.synchronize()
    time_dict[2] += time.perf_counter() - start_time1
    start_time1 = time.perf_counter()

    ## handle same elements ##
    if mask_same_elem_type_inside_dist_mask.any():
        dR_dist_mskd_same = dR_dist_mskd[mask_same_elem_type_inside_dist_mask]
        MAGR2_dist_mskd_same = MAGR2_dist_mskd[mask_same_elem_type_inside_dist_mask]
        TI_same = TI[mask_same_elem_type_inside_dist_mask]
        TI2 = TI_same ** 2
        TI3 = TI2 * TI_same
        
        SSB = TI3 / 48.0
        SSC = 3 * TI2 / 16.0
        SSD = 11 * TI_same / 16.0
        EXPTI = torch.exp(-TI_same * dR_dist_mskd_same)

        COULOMBV_tmp = (SSB * MAGR2_dist_mskd_same + SSC * dR_dist_mskd_same + SSD + 1. / dR_dist_mskd_same)

        COULOMBV[mask_same_elem_type_inside_dist_mask] -= EXPTI * COULOMBV_tmp
        dC_dR[mask_same_elem_type_inside_dist_mask] -= EXPTI*((-TI_same) * COULOMBV_tmp + \
            + (2*SSB*dR_dist_mskd_same + SSC - 1. / MAGR2_dist_mskd_same))
        
    
    ## handle diff elements ##
    if (~mask_same_elem_type_inside_dist_mask).any():

        dR_diff_elem = dR_dist_mskd[~mask_same_elem_type_inside_dist_mask]
        MAGR2_dist_mskd_diff = MAGR2_dist_mskd[~mask_same_elem_type_inside_dist_mask]
        
        TI_diff = TI[~mask_same_elem_type_inside_dist_mask]
        TI2 = TI_diff ** 2
        TI4 = TI2 ** 2
        TI6 = TI4 * TI2
        
        TJ_diff = TJ[~mask_same_elem_type_inside_dist_mask]
                
        TJ2 = TJ_diff**2
        TJ3 = TJ2 * TJ_diff
        TJ4 = TJ2 * TJ2
        TJ6 = TJ4 * TJ2
        
        EXPTI = torch.exp(-TI_diff * dR_diff_elem)
        
        EXPTJ = torch.exp(-TJ_diff * dR_diff_elem)
        
        TI2MTJ2 = TI2 - TJ2
        TJ2MTI2 = -TI2MTJ2

        SB = TJ4 * TI_diff / (2 * TI2MTJ2 * TI2MTJ2)
        SC = (TJ6 - 3 * TJ4 * TI2) / (TI2MTJ2 * TI2MTJ2 * TI2MTJ2)
        SE = TI4 * TJ_diff / (2 * TJ2MTI2 * TJ2MTI2)
        SF = (TI6 - 3 * TI4 * TJ2) / (TJ2MTI2 * TJ2MTI2 * TJ2MTI2)

        COULOMBV_tmp1 = (SB - SC / dR_diff_elem)
        COULOMBV_tmp2 = (SE - SF / dR_diff_elem)
        COULOMBV[~mask_same_elem_type_inside_dist_mask] -= EXPTI * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2

        dC_dR[~mask_same_elem_type_inside_dist_mask] -= EXPTI * ((-TI_diff)*COULOMBV_tmp1 + SC/MAGR2_dist_mskd_diff) + \
            EXPTJ * ((-TJ_diff)*COULOMBV_tmp2 + SF/MAGR2_dist_mskd_diff)
        
    # === Final Multiply === #
    COULOMBV *= KECONST
    dC_dR *= KECONST
    dC_dxyz = dC_dR*dR_dxyz[dist_mask].T
    
    torch.cuda.synchronize()
    time_dict[3] += time.perf_counter() - start_time1
    start_time1 = time.perf_counter()
    
    mask_pert_neigh_vec = mask_pert_neigh_vec.masked_scatter(mask_pert_neigh_vec, dist_mask)
    num_per_atom = mask_pert_neigh_vec.sum(dim=-1)
    COULOMBV_vec = torch.zeros(num_per_atom.shape, device = dR.device, dtype=dR.dtype)
    dC_dxyz_vec = torch.zeros(3, *num_per_atom.shape, device = dR.device, dtype=dR.dtype)
        
    # === Reduction === #
    # Precompute segment indices
    segment_ids = torch.arange(len(num_per_atom), device=dR.device, dtype=torch.int64).repeat_interleave(num_per_atom)
    # Vectorized sum over segments for COULOMBV
    COULOMBV_vec = COULOMBV_vec.scatter_add(0, segment_ids, COULOMBV)
    # Vectorized sum over segments for dC_dxyz
    dC_dxyz_vec = torch.zeros((3, len(num_per_atom)), dtype=dC_dxyz.dtype, device=dC_dxyz.device)
    dC_dxyz_vec.scatter_add_(1, segment_ids.expand(3, -1), dC_dxyz)
    
    torch.cuda.synchronize()
    time_dict[4] += time.perf_counter() - start_time1
        
    return COULOMBV_vec, dC_dxyz_vec


# def Ewald_k_Space_vectorized(J, RX, RY, RZ, LBox, DELTAQ, Nr_atoms, COULACC, TIMERATIO, do_vec=False):
#     device = RX.device
#     COULVOL = LBox[0] * LBox[1] * LBox[2]
#     SQRTX = math.sqrt(-math.log(COULACC))
#     CALPHA = math.sqrt(math.pi) * ((TIMERATIO * Nr_atoms / (COULVOL ** 2)) ** (1/6))
#     COULCUT = min(SQRTX / CALPHA, 50)
#     CALPHA = SQRTX / COULCUT
#     CALPHA2 = CALPHA * CALPHA
#     KCUTOFF = 2 * CALPHA * SQRTX
#     KCUTOFF2 = KCUTOFF * KCUTOFF

#     RECIPVECS = torch.zeros((3, 3), dtype=RX.dtype, device=device)
#     RECIPVECS[0, 0] = 2 * math.pi / LBox[0]
#     RECIPVECS[1, 1] = 2 * math.pi / LBox[1]
#     RECIPVECS[2, 2] = 2 * math.pi / LBox[2]

#     LMAX = int(KCUTOFF / RECIPVECS[0, 0])
#     MMAX = int(KCUTOFF / RECIPVECS[1, 1])
#     NMAX = int(KCUTOFF / RECIPVECS[2, 2])

#     KECONST = 14.3996437701414
#     SQRTPI = math.sqrt(math.pi)

#     FCOUL = torch.zeros((3, Nr_atoms), dtype=RX.dtype, device=device)
#     COULOMBV = torch.zeros((Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)
#     dC_dR = torch.zeros((3, Nr_atoms, Nr_atoms), dtype=RX.dtype, device=device)

#     print('  LMAX:', LMAX)
#     for L in range(0, LMAX + 1):
#         print('  ',L)
#         for M in range(-MMAX, MMAX + 1):
#             for N in range(-NMAX, NMAX + 1):
#                 if L == 0 and M < 0:
#                     continue
#                 if L == 0 and M == 0 and N < 1:
#                     continue

#                 kvec = L * RECIPVECS[:, 0] + M * RECIPVECS[:, 1] + N * RECIPVECS[:, 2]
#                 K2 = torch.dot(kvec, kvec)
#                 if K2 > KCUTOFF2:
#                     continue

#                 exp_factor = torch.exp(-K2 / (4 * CALPHA2))
#                 prefactor = 8 * math.pi * exp_factor / (COULVOL * K2)
#                 KEPREF = KECONST * prefactor

#                 dot = torch.matmul(kvec.view(1, 3), torch.stack((RX, RY, RZ), dim=0)).squeeze(0)
#                 sin_list = torch.sin(dot)
#                 cos_list = torch.cos(dot)

#                 sin_i = sin_list.view(-1, 1)
#                 sin_j = sin_list.view(1, -1)
#                 cos_i = cos_list.view(-1, 1)
#                 cos_j = cos_list.view(1, -1)

#                 symmetry_factor = 2.0 if L != 0 or M != 0 or N != 0 else 1.0

#                 COULOMBV += symmetry_factor * KEPREF * (cos_i * cos_j + sin_i * sin_j)

#                 force_term = symmetry_factor * KEPREF * (-cos_i * sin_j + sin_i * cos_j)
#                 dC_dR += force_term * kvec.view(3, 1, 1)

#     CORRFACT = 2 * KECONST * CALPHA / SQRTPI
#     COULOMBV -= CORRFACT * torch.eye(Nr_atoms, device=device)

#     return COULOMBV, dC_dR


# def CoulombMatrix_vectorized(RX, RY, RZ, LBox, Hubbard_U, Element_Type, Nr_atoms, HDIM,
#                   Coulomb_acc, TIMERATIO, nnRx, nnRy, nnRz, nrnnlist, nnType,
#                   H_INDEX_START, H_INDEX_END):
#     # Initialize charge deviation vector
#     dq_J = torch.zeros(Nr_atoms, dtype=torch.float64, device = RX.device)
#     #dq_J_vec = torch.ones(Nr_atoms, dtype=torch.float64, device = RX.device)

#     # Initialize Coulomb matrix
#     CC_real = torch.zeros((Nr_atoms, Nr_atoms), dtype=torch.float64, device = RX.device)
#     #CC_k = torch.zeros((Nr_atoms, Nr_atoms), dtype=torch.float64, device = RX.device)
    
#     dCC_dxyz_real = torch.zeros((3, Nr_atoms, Nr_atoms), dtype=torch.float64, device = RX.device)
    
#     dCC_dR_k = torch.zeros((Nr_atoms, Nr_atoms), dtype=torch.float64, device = RX.device)

#     Coulomb_Pot_k = 0.0
#     for J in range(Nr_atoms):
#         print(J)
#         dq_J[J] = 1.0
        
#         ## First, real space
#         Coulomb_Pot_Real = torch.zeros(Nr_atoms, dtype=torch.float64, device = RX.device)
#         dc_dxyz_Real = torch.zeros((3, Nr_atoms), dtype=torch.float64, device = RX.device)
        
#         mask_pert_neigh_vec = nnType == J
        
#         # to avoid computing already computed. SImilar to for J in range(Nr_atoms): for I in range(J,Nr_atoms):
#         mask_pert_neigh_vec[0:J] = False
#         mask_to_match_IJ = torch.repeat_interleave(torch.arange(Nr_atoms, device = RX.device), torch.sum(mask_pert_neigh_vec, dim=-1))
        
#         pot, dc_dxyz = Ewald_Real_Space_vectorized_MORE(J, RX[mask_to_match_IJ], RY[mask_to_match_IJ], RZ[mask_to_match_IJ], LBox, dq_J, Hubbard_U,
#                                           Element_Type, Nr_atoms, Coulomb_acc,
#                                           TIMERATIO, nnRx[mask_pert_neigh_vec], nnRy[mask_pert_neigh_vec], nnRz[mask_pert_neigh_vec], nrnnlist, nnType[mask_pert_neigh_vec], mask_pert_neigh_vec.clone(), mask_to_match_IJ)

#         print(Coulomb_Pot_Real.shape, J+1, pot.shape)
#         Coulomb_Pot_Real[J+1:] = pot
#         dc_dxyz_Real[:, J+1:] = dc_dxyz
                
#         #Coulomb_Pot_dq_J = Coulomb_Pot_Real + Coulomb_Pot_k
#         CC_real[:, J] = Coulomb_Pot_Real
#         dCC_dxyz_real[:, :, J] = dc_dxyz_Real
        
#         dq_J[J] = 0.0
        
#     ## Second, k-space
#     CC_k, dCC_dR_k = Ewald_k_Space_vectorized(J, RX, RY, RZ, LBox, dq_J, Nr_atoms, Coulomb_acc, TIMERATIO)
#     # Optional: symmetrize the Coulomb matrix
#     # CC = 0.5 * (CC + CC.T)
#     CC = CC_real + CC_real.T - torch.diag(torch.diagonal(CC_real)) + CC_k
#     dCC_dxyz = dCC_dxyz_real - torch.transpose(dCC_dxyz_real, -1,-2) - torch.diag_embed(dCC_dxyz_real.diagonal(dim1=1, dim2=2)) + dCC_dR_k
        
#     return CC, -dCC_dxyz

# def Ewald_Real_Space_vectorized_MORE(pert_J, RX, RY, RZ, LBox, DELTAQ, U, Element_Type, Nr_atoms, COULACC, TIMERATIO, nnRx, nnRy, nnRz, nrnnlist, nnType, mask_pert_neigh_vec, mask_to_match_IJ):

#     COULVOL = LBox[0] * LBox[1] * LBox[2]
#     SQRTX = math.sqrt(-math.log(COULACC))
#     CALPHA = math.sqrt(math.pi) * ((TIMERATIO * Nr_atoms / (COULVOL ** 2)) ** (1.0 / 6.0))
#     COULCUT = SQRTX / CALPHA
#     if COULCUT > 50.0:
#         COULCUT = 50.0
#         CALPHA = SQRTX / COULCUT
#     COULCUT2 = COULCUT * COULCUT
#     CALPHA2 = CALPHA * CALPHA

#     RELPERM = 1.0
#     KECONST = 14.3996437701414 * RELPERM
#     TFACT = 16.0 / (5.0 * KECONST)

#     SQRTPI = math.sqrt(math.pi)
#     FCOUL = torch.zeros(3, dtype=torch.float64, device = RX.device)
#     COULOMBV = 0.0

    
#     Ra = torch.stack((RX, RY, RZ), dim=-1)
    
#     ### vec
#     Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
#     Rab = Rb - Ra
    
#     J = nnType

#     dR = torch.norm(Rab, dim=-1)
#     Rab_X = Rab[:,0]
    
#     dR_dxyz = Rab/dR.unsqueeze(-1)
    
#     MAGR2 = dR * dR
    
#     dist_mask = (dR <= COULCUT)*(dR > 1e-12)
#     mask_diff_elem_type_global = (Element_Type[mask_to_match_IJ] != Element_Type[J])*dist_mask
#     mask_same_elem_type_global = (Element_Type[mask_to_match_IJ] == Element_Type[J])*dist_mask
#     mask_pert_neigh_vec = mask_pert_neigh_vec.masked_scatter(mask_pert_neigh_vec, dist_mask)
#     num_per_atom = torch.sum(mask_pert_neigh_vec, dim = -1)
#     num_per_atom = num_per_atom[num_per_atom!=0]
#     num_per_atom_end = torch.cumsum(num_per_atom,dim=0)
    
#     num_per_atom_start = torch.zeros(num_per_atom_end.shape, dtype=torch.int64)
#     num_per_atom_start[1:] = num_per_atom_end[:-1]

#     TI = TFACT * U[mask_to_match_IJ]
#     TI2 = TI * TI
#     TI3 = TI2 * TI
#     TI4 = TI2 * TI2
#     TI6 = TI4 * TI2

#     SSA = TI
#     SSB = TI3[mask_same_elem_type_global] / 48.0
#     SSC = 3 * TI2[mask_same_elem_type_global] / 16.0
#     SSD = 11 * TI[mask_same_elem_type_global] / 16.0
#     SSE = 1.0

    
#     TJ = TFACT * U[J[dist_mask]]

#     Z = abs(CALPHA * dR[dist_mask])
#     NUMREP_ERFC = torch.erfc(Z)
#     CA = NUMREP_ERFC / dR[dist_mask]
#     COULOMBV = DELTAQ[J[dist_mask]] * CA
    
#     CA += 2 * CALPHA * torch.exp(-CALPHA2 * MAGR2[dist_mask]) / SQRTPI
    
#     dC_dR = -CA / dR[dist_mask]
#     #FORCE = -KECONST * DELTAQ[I] * CA / dR[dist_mask]
#     EXPTI = torch.exp(-TI * dR)
    
#     ## handle same elements ##
    
#     mask_same_elem_type_inside_dist_mask = (Element_Type[mask_to_match_IJ[dist_mask]] == Element_Type[J[dist_mask]])
#     COULOMBV_tmp = (SSB * MAGR2[mask_same_elem_type_global] + SSC * dR[mask_same_elem_type_global] + SSD + SSE / dR[mask_same_elem_type_global])
    
#     COULOMBV[mask_same_elem_type_inside_dist_mask] -= EXPTI[mask_same_elem_type_global] * COULOMBV_tmp
        
#     dC_dR[mask_same_elem_type_inside_dist_mask] -= EXPTI[mask_same_elem_type_global]*((-TI[mask_same_elem_type_global]) * COULOMBV_tmp + \
#         + (2*SSB*dR[mask_same_elem_type_global] + SSC - SSE / MAGR2[mask_same_elem_type_global]))
#     ## end handle same elements ##
    
#     ## handle diff elements ##
    
#     SD = TJ[~mask_same_elem_type_inside_dist_mask]
    
#     dR_diff_elem = dR[mask_diff_elem_type_global]
    
#     TJ2 = SD**2
#     TJ3 = TJ2 * SD
#     TJ4 = TJ2 * TJ2
#     TJ6 = TJ4 * TJ2
#     EXPTJ = torch.exp(-SD * dR_diff_elem)
#     TI2MTJ2 = TI2[mask_diff_elem_type_global] - TJ2
#     TJ2MTI2 = -TI2MTJ2
    
#     SA = TI[mask_diff_elem_type_global]
#     SB = TJ4 * SA / (2 * TI2MTJ2 * TI2MTJ2)
#     SC = (TJ6 - 3 * TJ4 * TI2[mask_diff_elem_type_global]) / (TI2MTJ2 * TI2MTJ2 * TI2MTJ2)
#     SE = TI4[mask_diff_elem_type_global] * SD / (2 * TJ2MTI2 * TJ2MTI2)
#     SF = (TI6[mask_diff_elem_type_global] - 3 * TI4[mask_diff_elem_type_global] * TJ2) / (TJ2MTI2 * TJ2MTI2 * TJ2MTI2)
    
    
#     COULOMBV_tmp1 = (SB - SC / dR_diff_elem)
#     COULOMBV_tmp2 = (SE - SF / dR_diff_elem)
    
#     COULOMBV[~mask_same_elem_type_inside_dist_mask] -= EXPTI[mask_diff_elem_type_global] * COULOMBV_tmp1 + EXPTJ * COULOMBV_tmp2
    
#     dC_dR[~mask_same_elem_type_inside_dist_mask] -= EXPTI[mask_diff_elem_type_global] * ((-SA)*COULOMBV_tmp1 + SC/MAGR2[mask_diff_elem_type_global]) + \
#         EXPTJ * ((-SD)*COULOMBV_tmp2 + SF/MAGR2[mask_diff_elem_type_global])
    
#     ## handle diff elements ##
#     ###
#     COULOMBV *= KECONST
#     dC_dR *= KECONST
    
#     dC_dxyz = dC_dR*dR_dxyz[dist_mask].T
    
#     COULOMBV_vec = torch.zeros(num_per_atom.shape)
#     dC_dxyz_vec = torch.zeros(3, *num_per_atom.shape)
#     #print('v',COULOMBV)
#     for i in range(len(num_per_atom)):
#         COULOMBV_vec[i] = torch.sum(COULOMBV[num_per_atom_start[i]:num_per_atom_end[i]])
#         dC_dxyz_vec[:,i] = torch.sum(dC_dxyz[:,num_per_atom_start[i]:num_per_atom_end[i]], dim=1)
        
#     #COULOMBV = torch.sum(COULOMBV)
#     return COULOMBV_vec, dC_dxyz_vec

