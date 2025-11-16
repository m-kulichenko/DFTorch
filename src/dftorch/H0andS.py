import torch
import time
from .BondIntegral import *
from .SlaterKosterPair import Slater_Koster_Pair_vectorized, Slater_Koster_Pair_SKF_vectorized
from .AtomicDensityMatrix import AtomicDensityMatrix
from .Tools import ordered_pairs_from_TYPE

#@torch.compile
def H0_and_S_vectorized(TYPE, RX, RY, RZ, Nr_atoms, diagonal, H_INDEX_START, H_INDEX_END, Znuc,
                        nnRx, nnRy, nnRz, nnType,
                        const, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type,
                        R_orb, coeffs_tensor,
                        verbose=False):
    """
    Constructs the Hamiltonian (H0), overlap matrix (S), their derivatives, and other DFTB-related
    data using a vectorized Slater-Koster approach.

    Args:
        TYPE (torch.Tensor): Atomic types (N,).
        RX, RY, RZ (torch.Tensor): Atomic coordinates (N,).
        LBox (torch.Tensor): Simulation box dimensions (3,).
        Nr_atoms (int): Number of atoms.
        nrnnlist (torch.Tensor): Number of neighbors per atom (N, 1).
        nnRx, nnRy, nnRz (torch.Tensor): Neighbor coordinates (N, max_neigh).
        nnType (torch.Tensor): Neighbor types (N, max_neigh).
        const: Object with necessary constants (e.g., orbital count, masses, Slater-Koster parameters).

    Returns:
        H0 (torch.Tensor): Hamiltonian matrix (HDIM, HDIM).
        dH0 (torch.Tensor): Derivatives of Hamiltonian (3, HDIM, HDIM).
        S (torch.Tensor): Overlap matrix (HDIM, HDIM).
        dS (torch.Tensor): Derivatives of overlap matrix (3, HDIM, HDIM).
        D0 (torch.Tensor): Initial guess for density matrix (HDIM, HDIM).
        H_INDEX_START (torch.Tensor): Start index of each atom's orbitals in H0.
        H_INDEX_END (torch.Tensor): End index of each atom's orbitals in H0.
        TYPE (torch.Tensor): Atomic types (N,).
        Mnuc (torch.Tensor): Atomic masses (N,).
        Znuc (torch.Tensor): Effective nuclear charges (N,).
        Hubbard_U (torch.Tensor): Hubbard U parameters (N,).
        neighbor_I (torch.Tensor): Atom indices for neighbor pairs.
        neighbor_J (torch.Tensor): Neighbor indices for each atom in neighbor_I.
    """
    # Map atom type to properties
    # Support both str and int input
    if verbose: print('H0_and_S')
    start_time1 = time.perf_counter()
    start_time3 = time.perf_counter()
    
    if verbose: print('  Do H off-diag')
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
        
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)
    
    L = Rab_X/dR
    L_dx = (Rab_Y**2 + Rab_Z**2)/(dR**3)
    L_dy = -Rab_X*Rab_Y/(dR**3)
    L_dz = -Rab_X*Rab_Z/(dR**3)
    
    M = Rab_Y/dR
    M_dx = -Rab_Y*Rab_X/(dR**3)
    M_dy = (Rab_X**2 + Rab_Z**2)/(dR**3)
    M_dz = -Rab_Y*Rab_Z/(dR**3)

    N = Rab_Z/dR
    N_dx = -Rab_Z*Rab_X/(dR**3)
    N_dy = -Rab_Z*Rab_Y/(dR**3)
    N_dz = (Rab_X**2 + Rab_Y**2)/(dR**3)
    
    #HDIM = sum(non_hydro_mask)*4 + sum(hydro_mask)
    HDIM = len(diagonal)
    pair_mask_HH = (const.n_orb[TYPE[neighbor_I]] == 1)&(const.n_orb[TYPE[neighbor_J]] == 1)
    pair_mask_HX = (const.n_orb[TYPE[neighbor_I]] == 1)&(const.n_orb[TYPE[neighbor_J]] == 4)
    pair_mask_XH = (const.n_orb[TYPE[neighbor_I]] == 4)&(const.n_orb[TYPE[neighbor_J]] == 1)    
    pair_mask_XX = (const.n_orb[TYPE[neighbor_I]] == 4)&(const.n_orb[TYPE[neighbor_J]] == 4)

    pair_mask_HY = (const.n_orb[TYPE[neighbor_I]] == 1)&(const.n_orb[TYPE[neighbor_J]] == 9)
    pair_mask_XY = (const.n_orb[TYPE[neighbor_I]] == 4)&(const.n_orb[TYPE[neighbor_J]] == 9)
    pair_mask_YH = (const.n_orb[TYPE[neighbor_I]] == 9)&(const.n_orb[TYPE[neighbor_J]] == 1)
    pair_mask_YX = (const.n_orb[TYPE[neighbor_I]] == 9)&(const.n_orb[TYPE[neighbor_J]] == 4)
    pair_mask_YY = (const.n_orb[TYPE[neighbor_I]] == 9)&(const.n_orb[TYPE[neighbor_J]] == 9)

    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    L_mskd = L[nn_mask]
    M_mskd = M[nn_mask]
    N_mskd = N[nn_mask]
    
    L_dxyz = torch.stack((L_dx, L_dy, L_dz), dim=0)[:,nn_mask]
    M_dxyz = torch.stack((M_dx, M_dy, M_dz), dim=0)[:,nn_mask]
    N_dxyz = torch.stack((N_dx, N_dy, N_dz), dim=0)[:,nn_mask]

    dR_dxyz = torch.stack((Rab_X,Rab_Y,Rab_Z), dim=0)[:,nn_mask]/dR_mskd

    if verbose: print("  t <dR and pair mask> {:.1f} s\n".format( time.perf_counter()-start_time3 ))
    start_time4 = time.perf_counter()
    
    idx = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
    idx = torch.clamp(idx, 0, len(R_orb))
    dx = (dR_mskd - R_orb[idx])

    if verbose: print("  t <SKF> {:.1f} s\n".format( time.perf_counter()-start_time4 ))
    start_time5 = time.perf_counter()

    if verbose: print('  Do H and S')
    H0, dH0 = Slater_Koster_Pair_SKF_vectorized(HDIM, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                            pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                            pair_mask_HY, pair_mask_XY, pair_mask_YH, pair_mask_YX, pair_mask_YY,
                                            dx, idx, IJ_pair_type, JI_pair_type, coeffs_tensor,
                                            neighbor_I, neighbor_J, H_INDEX_START,0)
    
    H0 = H0.reshape(HDIM,HDIM)
    H0 = H0 + torch.diag(diagonal)
    dH0 = dH0.reshape(3,HDIM,HDIM)

    #### S PART ###      
    S, dS = Slater_Koster_Pair_SKF_vectorized(HDIM, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                            pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                            pair_mask_HY, pair_mask_XY, pair_mask_YH, pair_mask_YX, pair_mask_YY,
                                            dx, idx, IJ_pair_type, JI_pair_type, coeffs_tensor,
                                            neighbor_I, neighbor_J, H_INDEX_START,1)

    S = S.reshape(HDIM,HDIM)/27.21138625
    S = S + torch.eye(HDIM, device=S.device)
    dS = dS.reshape(3,HDIM,HDIM)/27.21138625

    if verbose: print("  t <H and S> {:.1f} s\n".format( time.perf_counter()-start_time5 ))
    start_time6 = time.perf_counter()
    D0 = AtomicDensityMatrix(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
    #D0_ = AtomicDensityMatrix_vectorized(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
    
    D0 = 0.5 * D0
    if verbose: print("  t <D0> {:.1f} s\n".format( time.perf_counter()-start_time6 ))
    print("H0_and_S t {:.1f} s\n".format( time.perf_counter()-start_time1 ))
    return D0, H0, dH0, S, dS




def H0_and_S_vectorized_OLD_FOR_POLY(TYPE, RX, RY, RZ, LBox, Nr_atoms, diagonal, H_INDEX_START, H_INDEX_END, Znuc,
                        nrnnlist, nnRx, nnRy, nnRz, nnType,
                        const, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type,
                        R_orb, coeffs_tensor,
                        verbose=False):
    """
    OLDER VERSION TO GENERATE BOND INTS FROM POLYNOMIAL DATA
    Constructs the Hamiltonian (H0), overlap matrix (S), their derivatives, and other DFTB-related
    data using a vectorized Slater-Koster approach.

    Args:
        TYPE (torch.Tensor): Atomic types (N,).
        RX, RY, RZ (torch.Tensor): Atomic coordinates (N,).
        LBox (torch.Tensor): Simulation box dimensions (3,).
        Nr_atoms (int): Number of atoms.
        nrnnlist (torch.Tensor): Number of neighbors per atom (N, 1).
        nnRx, nnRy, nnRz (torch.Tensor): Neighbor coordinates (N, max_neigh).
        nnType (torch.Tensor): Neighbor types (N, max_neigh).
        const: Object with necessary constants (e.g., orbital count, masses, Slater-Koster parameters).

    Returns:
        H0 (torch.Tensor): Hamiltonian matrix (HDIM, HDIM).
        dH0 (torch.Tensor): Derivatives of Hamiltonian (3, HDIM, HDIM).
        S (torch.Tensor): Overlap matrix (HDIM, HDIM).
        dS (torch.Tensor): Derivatives of overlap matrix (3, HDIM, HDIM).
        D0 (torch.Tensor): Initial guess for density matrix (HDIM, HDIM).
        H_INDEX_START (torch.Tensor): Start index of each atom's orbitals in H0.
        H_INDEX_END (torch.Tensor): End index of each atom's orbitals in H0.
        TYPE (torch.Tensor): Atomic types (N,).
        Mnuc (torch.Tensor): Atomic masses (N,).
        Znuc (torch.Tensor): Effective nuclear charges (N,).
        Hubbard_U (torch.Tensor): Hubbard U parameters (N,).
        neighbor_I (torch.Tensor): Atom indices for neighbor pairs.
        neighbor_J (torch.Tensor): Neighbor indices for each atom in neighbor_I.
    """
    # Map atom type to properties
    # Support both str and int input
    print('H0_and_S')
    start_time1 = time.perf_counter()
    
    n_orbitals_per_atom = const.n_orb[TYPE]
        
    # H_INDEX_START = torch.zeros(Nr_atoms, dtype=torch.int64, device=RX.device)
    # H_INDEX_START[1:] = torch.cumsum(n_orbitals_per_atom, dim=0)[:-1]
    # H_INDEX_END = H_INDEX_START + n_orbitals_per_atom - 1
    
    #Znuc = const.tore[TYPE]
            
    # === Vectorized neighbor type pair generation ===
    #max_neighbors = nnType.shape[-1]

    # Create mask for valid neighbors
    # neighbor_mask = torch.arange(max_neighbors, device=RX.device).unsqueeze(0) < nrnnlist
    # neighbor_J = nnType[neighbor_mask]
    # neighbor_I = torch.repeat_interleave(torch.arange(nrnnlist.squeeze(-1).shape[0], device=nrnnlist.device), nrnnlist.squeeze(-1))
    
    print('  Load H integral params')
    import os
    param_dir = os.path.join(os.path.dirname(__file__), 'params')
    fss_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'fss_sigma.csv'))
    fsp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'fsp_sigma.csv'))
    fps_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'fps_sigma.csv'))
    fpp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, os.path.join(param_dir, 'fpp_sigma.csv'))
    fpp_pi = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, os.path.join(param_dir, 'fpp_pi.csv'))
    
    
    
    
    # Es_dict = torch.as_tensor([ 0.00000,
    #    -6.492650686,                                                                                                                                                                                    0.0,
    #     0.0,       0.0,                                                                                                                     0.0,  -13.73881005,  -18.5565,  -23.9377,       0.0,       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -4.3392,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     ],
    #    dtype=RX.dtype, device=RX.device)
    
    # Ep_dict = torch.as_tensor([ 0.00000,
    #     0.0,                                                                                                                                                                                       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,   -5.28867445,   -7.0625,   -9.0035,       0.0,       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -0.7580,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     ],
    #    dtype=RX.dtype, device=RX.device)

    # Ed_dict = torch.as_tensor([ 0.00000,
    #     0.0,                                                                                                                                                                                       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,   -4.7987,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     ],
    #    dtype=RX.dtype, device=RX.device)
    
    # U_dict = torch.as_tensor([ 0.0,
    #     11.4151819,                                                                                                                                                                                   0.0,
    #     0.0,       0.0,                                                                                                                     0.0,   9.923997,   17.3729,   11.8761,       0.0,       0.0,
    #     0.0,       0.0,                                                                                                                     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,    6.2979,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
    #     ],
    #    dtype=RX.dtype, device=RX.device)
    
    # Hubbard_U = torch.zeros(Nr_atoms, dtype=RX.dtype, device = RX.device)
    # Hubbard_U = U_dict[TYPE]
    
    # print('  Do H diagonal')
    # hydro_mask = (const.n_orb[TYPE] == 1)  # True for hydrogen atoms (only s orbital)
    # non_hydro_mask = (const.n_orb[TYPE] == 4)
    # d_mask =  (const.n_orb[TYPE] == 9)

    # # Shell on-site energies per atom (pulled from your dicts)
    # EsA = Es_dict[TYPE]   # (Nr_atoms,)
    # EpA = Ep_dict[TYPE]   # (Nr_atoms,)
    # EdA = Ed_dict[TYPE]   # (Nr_atoms,)

    # # Which shells exist for each atom, based on your basis size:
    # # 1  -> H-like: s
    # # 4  -> main-group sp: s + 3*p
    # # 9  -> transition-metal spd: s + 3*p + 5*d
    # has_p = (const.n_orb[TYPE] >= 4)          # p present for 4 or 9
    # has_d = (const.n_orb[TYPE] == 9)          # d present only for 9 here
    # # (Optional: if you ever use sd-only (6 orbitals), set has_d |= (const.n_orb[TYPE] == 6)
    # #            and exclude p for that case.)

    # # Build a per-atom template in the standard AO order: [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]
    # # (All p orbitals get EpA; all d orbitals get EdA.)
    # template = torch.stack(
    #     (
    #         EsA,             # s
    #         EpA, EpA, EpA,   # p triplet
    #         EdA, EdA, EdA, EdA, EdA  # d quintet
    #     ),
    #     dim=1,  # shape: (Nr_atoms, 9)
    # )

    # # Per-atom mask telling which of the 9 positions are actually present
    # mask = torch.zeros_like(template, dtype=torch.bool)  # (Nr_atoms, 9)
    # mask[:, 0]    = True                                  # s always present
    # mask[:, 1:4]  = has_p.unsqueeze(1).expand(-1, 3)      # p block present?
    # mask[:, 4:9]  = has_d.unsqueeze(1).expand(-1, 5)      # d block present?

    #diagonal = template[mask]                              # 1-D tensor



    #diagonal = torch.repeat_interleave(Ep_dict[TYPE], repeats=n_orbitals_per_atom)
    #diagonal[H_INDEX_START] = Es_dict[TYPE]
    
    print('  Do H off-diag')
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    
    Rab_X = Rab_X - LBox[0] * torch.round(Rab_X / LBox[0])
    Rab_Y = Rab_Y - LBox[1] * torch.round(Rab_Y / LBox[1])
    Rab_Z = Rab_Z - LBox[2] * torch.round(Rab_Z / LBox[2])
    
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)
    
    L = Rab_X/dR
    L_dx = (Rab_Y**2 + Rab_Z**2)/(dR**3)
    L_dy = -Rab_X*Rab_Y/(dR**3)
    L_dz = -Rab_X*Rab_Z/(dR**3)
    
    M = Rab_Y/dR
    M_dx = -Rab_Y*Rab_X/(dR**3)
    M_dy = (Rab_X**2 + Rab_Z**2)/(dR**3)
    M_dz = -Rab_Y*Rab_Z/(dR**3)

    N = Rab_Z/dR
    N_dx = -Rab_Z*Rab_X/(dR**3)
    N_dy = -Rab_Z*Rab_Y/(dR**3)
    N_dz = (Rab_X**2 + Rab_Y**2)/(dR**3)
    
    #HDIM = sum(non_hydro_mask)*4 + sum(hydro_mask)
    HDIM = len(diagonal)
    H0 = torch.zeros((HDIM*HDIM), dtype=RX.dtype, device = RX.device)
    pair_mask_HH = (const.n_orb[TYPE[neighbor_I]] == 1)*(const.n_orb[TYPE[neighbor_J]] == 1)
    pair_mask_HX = (const.n_orb[TYPE[neighbor_I]] == 1)*(const.n_orb[TYPE[neighbor_J]] == 4)
    pair_mask_XH = (const.n_orb[TYPE[neighbor_I]] == 4)*(const.n_orb[TYPE[neighbor_J]] == 1)    
    pair_mask_XX = (const.n_orb[TYPE[neighbor_I]] == 4)*(const.n_orb[TYPE[neighbor_J]] == 4)

    pair_mask_HY = (const.n_orb[TYPE[neighbor_I]] == 1)*(const.n_orb[TYPE[neighbor_J]] == 9)
    pair_mask_YH = (const.n_orb[TYPE[neighbor_I]] == 9)*(const.n_orb[TYPE[neighbor_J]] == 1)
    pair_mask_XY = (const.n_orb[TYPE[neighbor_I]] == 4)*(const.n_orb[TYPE[neighbor_J]] == 9)
    pair_mask_YX = (const.n_orb[TYPE[neighbor_I]] == 9)*(const.n_orb[TYPE[neighbor_J]] == 4)
    pair_mask_YY = (const.n_orb[TYPE[neighbor_I]] == 9)*(const.n_orb[TYPE[neighbor_J]] == 9)

        
    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    L_mskd = L[nn_mask]
    M_mskd = M[nn_mask]
    N_mskd = N[nn_mask]
    
    L_dxyz = torch.stack((L_dx, L_dy, L_dz), dim=0)[:,nn_mask]
    M_dxyz = torch.stack((M_dx, M_dy, M_dz), dim=0)[:,nn_mask]
    N_dxyz = torch.stack((N_dx, N_dy, N_dz), dim=0)[:,nn_mask]

    dR_dxyz = torch.stack((Rab_X,Rab_Y,Rab_Z), dim=0)[:,nn_mask]/dR_mskd

    #################################
    ### Get tensors for SKF files ###
    pairs_tensor, pairs_list, label_list = ordered_pairs_from_TYPE(TYPE, const)
    pairs_tensor, pairs_list, label_list

    # Allocate padded tensors
    
    
    print('  Do H Slater-Koster')
    H0, dH0 = Slater_Koster_Pair_vectorized(H0, HDIM, dR_mskd, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                            pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                            fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi,
                                            neighbor_I, neighbor_J, nnType, H_INDEX_START, H_INDEX_END)
    

    H0 = H0.reshape(HDIM,HDIM)
    H0 = H0 + torch.diag(diagonal)
    dH0 = dH0.reshape(3,HDIM,HDIM)
    dH0 = dH0 #- torch.transpose(dH0, 1, 2)
    
    #### S PART ###
    print('  Load S integral params')
    param_dir = os.path.join(os.path.dirname(__file__), 'params')
    fss_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'S_fss_sigma.csv'))
    fsp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'S_fsp_sigma.csv'))
    fps_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'S_fps_sigma.csv'))
    fpp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'S_fpp_sigma.csv'))
    fpp_pi = LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, os.path.join(param_dir, 'S_fpp_pi.csv'))

    print('  Do S Slater-Koster')
    S = torch.zeros((HDIM*HDIM), dtype=RX.dtype, device = RX.device)
    S, dS = Slater_Koster_Pair_vectorized(S, HDIM, dR_mskd, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                          pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                          fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi,
                                          neighbor_I, neighbor_J, nnType, H_INDEX_START, H_INDEX_END)  
      
    S = S.reshape(HDIM,HDIM)
    S = S + torch.eye(HDIM, device=S.device)
    dS = dS.reshape(3,HDIM,HDIM)
    dS = dS #- torch.transpose(dS, 1, 2)


    D0 = AtomicDensityMatrix(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
#     D0_ = AtomicDensityMatrix_vectorized(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
    
#     print(D0-D0_)
#     print(torch.max(abs(D0-D0_)))
    
    D0 = 0.5 * D0
    
    print("  t {:.1f} s\n".format( time.perf_counter()-start_time1 ))



    return D0, H0, dH0, S, dS