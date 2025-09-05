import torch
import time
from .BondIntegral import *
from .SlaterKosterPair import Slater_Koster_Pair_vectorized
from .AtomicDensityMatrix import AtomicDensityMatrix, AtomicDensityMatrix_vectorized
def H0_and_S_vectorized(TYPE, RX, RY, RZ, LBox, Nr_atoms, nrnnlist, nnRx, nnRy, nnRz, nnType, const):
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
    print('H0_and_S')
    start_time1 = time.perf_counter()
    
    
    n_orbitals_per_atom = const.n_orb[TYPE]
        
    H_INDEX_START = torch.zeros(Nr_atoms, dtype=torch.int64, device=RX.device)
    H_INDEX_START[1:] = torch.cumsum(n_orbitals_per_atom, dim=0)[:-1]
    H_INDEX_END = H_INDEX_START + n_orbitals_per_atom - 1
    
    Mnuc = const.mass[TYPE]
    Znuc = const.tore[TYPE]
            
    # === Vectorized neighbor type pair generation ===
    max_neighbors = nnType.shape[-1]

    # Create mask for valid neighbors
    neighbor_mask = torch.arange(max_neighbors, device=RX.device).unsqueeze(0) < nrnnlist
    neighbor_J = nnType[neighbor_mask]
    neighbor_I = torch.repeat_interleave(torch.arange(nrnnlist.squeeze(-1).shape[0], device=nrnnlist.device), nrnnlist.squeeze(-1))
    
    print('  Load H integral params')
    fss_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'fss_sigma.csv')
    fsp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'fsp_sigma.csv')
    fps_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'fps_sigma.csv')
    fpp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'fpp_sigma.csv')
    fpp_pi = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'fpp_pi.csv')
    
    Es_dict = torch.as_tensor([ 0.00000,
       -6.4835,                                                                                    0.0,
       0.0,   0.0,                                    0.0,  -13.7199,  -18.5565,  -23.9377,  0.0,  0.0,],
       dtype=RX.dtype, device=RX.device             )
    
    Ep_dict = torch.as_tensor([ 0.00000,
       0.0,                                                                                     0.0,  
       0.0,   0.0,                                    0.0,  -5.2541,  -7.0625,  -9.0035,  0.0,  0.0,],
       dtype=RX.dtype, device=RX.device             )
    
    U_dict = torch.as_tensor([ 0.00000,
       12.054683,                                                                                    0.0,
       0.0,   0.0,                                    0.0,  14.240811,  17.3729,  11.876141,  0.0,  0.0,],
       dtype=RX.dtype, device=RX.device             )
    
    Hubbard_U = torch.zeros(Nr_atoms, dtype=RX.dtype, device = RX.device)
    Hubbard_U = U_dict[TYPE]
    
    print('  Do H diagonal')
    hydro_mask = TYPE==1
    non_hydro_mask = TYPE != 1
    diagonal = torch.repeat_interleave(Ep_dict[TYPE], repeats=n_orbitals_per_atom)
    diagonal[H_INDEX_START] = Es_dict[TYPE]
    
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
    
    HDIM = sum(non_hydro_mask)*4 + sum(hydro_mask)
    H0 = torch.zeros((HDIM*HDIM), dtype=RX.dtype, device = RX.device)
    pair_mask_HH = (TYPE[neighbor_I]==1)*(TYPE[neighbor_J]==1)
    pair_mask_HX = (TYPE[neighbor_I]==1)*(TYPE[neighbor_J]!=1)
    pair_mask_XH = (TYPE[neighbor_I]!=1)*(TYPE[neighbor_J]==1)    
    pair_mask_XX = (TYPE[neighbor_I]!=1)*(TYPE[neighbor_J]!=1)
        
    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    L_mskd = L[nn_mask]
    M_mskd = M[nn_mask]
    N_mskd = N[nn_mask]
    
    L_dxyz = torch.stack((L_dx, L_dy, L_dz), dim=0)[:,nn_mask]
    M_dxyz = torch.stack((M_dx, M_dy, M_dz), dim=0)[:,nn_mask]
    N_dxyz = torch.stack((N_dx, N_dy, N_dz), dim=0)[:,nn_mask]

    dR_dxyz = torch.stack((Rab_X,Rab_Y,Rab_Z), dim=0)[:,nn_mask]/dR_mskd
    
    print('  Do H Slater-Koster')
    H0, dH0 = Slater_Koster_Pair_vectorized(H0, HDIM, dR_mskd, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                            pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                            fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi,
                                            neighbor_I, neighbor_J, nnType, H_INDEX_START, H_INDEX_END)
    H0 = H0.reshape(HDIM,HDIM)
    H0 = H0 + torch.transpose(H0, 0, 1) + torch.diag(diagonal)
    
    dH0 = dH0.reshape(3,HDIM,HDIM)
    dH0 = dH0 - torch.transpose(dH0, 1, 2)

    #### S PART ###
    print('  Load S integral params')
    fss_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'S_fss_sigma.csv')
    fsp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'S_fsp_sigma.csv')
    fps_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'S_fps_sigma.csv')
    fpp_sigma = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'S_fpp_sigma.csv')
    fpp_pi = LoadBondIntegralParameters(neighbor_I, neighbor_J,TYPE, 'S_fpp_pi.csv')

    print('  Do S Slater-Koster')
    S = torch.zeros((HDIM*HDIM), dtype=RX.dtype, device = RX.device)
    S, dS = Slater_Koster_Pair_vectorized(S, HDIM, dR_mskd, dR_dxyz, L_mskd, M_mskd, N_mskd, L_dxyz, M_dxyz, N_dxyz,
                                          pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                          fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi,
                                          neighbor_I, neighbor_J, nnType, H_INDEX_START, H_INDEX_END)    
    S = S.reshape(HDIM,HDIM)
    S = S + torch.transpose(S, 0, 1) + torch.eye(HDIM, device=S.device)
    dS = dS.reshape(3,HDIM,HDIM)
    dS = dS - torch.transpose(dS, 1, 2)
    D0 = AtomicDensityMatrix(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
#     D0_ = AtomicDensityMatrix_vectorized(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc)
    
#     print(D0-D0_)
#     print(torch.max(abs(D0-D0_)))
    
    D0 = 0.5 * D0
    
    print("  t {:.1f} s\n".format( time.perf_counter()-start_time1 ))

    return H0, dH0, S, dS, D0, H_INDEX_START, H_INDEX_END, TYPE, Mnuc, Znuc, Hubbard_U, neighbor_I, neighbor_J