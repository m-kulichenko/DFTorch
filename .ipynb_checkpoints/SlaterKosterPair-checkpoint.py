import torch
from .BondIntegral import *

def Slater_Koster_Pair_vectorized(H0, HDIM, dR, dR_dxyz, L, M, N, L_dxyz, M_dxyz, N_dxyz,
                                  pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                  fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi,
                                  neighbor_I, neighbor_J, nnType, H_INDEX_START, H_INDEX_END):
    """
    Compute the Slater-Koster matrix elements and their derivatives for pairs of atoms 
    using vectorized operations for efficiency.

    This function evaluates the atomic block Hamiltonian matrix elements (H0) and their 
    spatial derivatives (dH0) based on the Slater-Koster parameterization for s and p orbitals.
    It handles four types of atom pairs: H-H, H-X, X-H, and X-X, where H represents hydrogen-like atoms,
    and X represents other atom types with s and p orbitals.

    Parameters
    ----------
    H0 : torch.Tensor
        Preallocated 1D tensor containing the Hamiltonian matrix elements, flattened.
        Shape: (HDIM * HDIM,)

    HDIM : int
        Dimension of the atomic orbital block (e.g., 4 for sp orbitals).

    dR : torch.Tensor
        Tensor of interatomic distances for all pairs. Shape: (num_pairs,)

    dR_dxyz : torch.Tensor
        Tensor of derivatives of the interatomic distances with respect to Cartesian coordinates.
        Shape: (3, num_pairs), where axis 0 corresponds to x,y,z components.

    L, M, N : torch.Tensor
        Direction cosines (components of the unit vector pointing from one atom to another).
        Shape: (num_pairs,)

    L_dxyz, M_dxyz, N_dxyz : torch.Tensor
        Derivatives of the direction cosines with respect to Cartesian coordinates.
        Shape: (3, num_pairs)

    pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX : torch.BoolTensor
        Boolean masks selecting pairs belonging to each type:
        - H-H pairs
        - H-X pairs
        - X-H pairs
        - X-X pairs

    fss_sigma, fsp_sigma, fps_sigma, fpp_sigma, fpp_pi : torch.Tensor
        Bond integral parameters for the Slater-Koster functions for each pair.
        Shape: (num_pairs,)

    neighbor_I, neighbor_J : torch.Tensor
        Indices of atoms in each pair.
        Shape: (num_pairs,)

    nnType : ?
        (Unused in this snippet — possibly the neighbor type or classification.)

    H_INDEX_START, H_INDEX_END : torch.Tensor or list
        Starting and ending indices in H0 corresponding to atomic orbitals of each atom.
        Used to place computed integrals into the correct positions in H0.

    Returns
    -------
    H0 : torch.Tensor
        Updated Hamiltonian matrix elements tensor with new values for the pairs processed.

    dH0 : torch.Tensor
        Derivatives of the Hamiltonian matrix elements with respect to Cartesian coordinates.
        Shape: (3, HDIM * HDIM)

    Notes
    -----
    - The function uses vectorized bond integral evaluations for improved computational efficiency.
    - The calculation covers both overlap and Hamiltonian matrix elements for s and p orbitals.
    - Direction cosine derivatives and bond integral derivatives are used to compute the gradients.
    - Periodic boundary conditions or lattice considerations are assumed handled externally.
    """
    # %%% Standard Slater-Koster sp-parameterization for an atomic block between a pair of atoms
    # %%% IDim, JDim: dimensions of the output block, e.g. 1 x 4 for H-O or 4 x 4 for O-O, or 4 x 1 for O-H
    # %%% Ra, Rb: are the vectors of the positions of the two atoms
    # %%% LBox: Periodic boundary conditions, i.e. length of box in x, y, z (cubic box only)
    # %%% Type_pair(1 or 2): Character of the type of each atom in the pair, e.g. 'H' for hydrogen of 'O' for oxygen
    # %%% fss_sigma, ... , fpp_pi: paramters for the bond integrals
    # %%% diagonal(1 or 2): atomic energies Es and Ep or diagonal elements of the overlap i.e. diagonal = 1
    
    dH0 = torch.zeros(3,HDIM*HDIM, dtype=H0.dtype, device=H0.device)
    
    # H-H    
    HSSS = bond_integral_vectorized(dR[pair_mask_HH], fss_sigma[pair_mask_HH])
    H0[ H_INDEX_START[neighbor_I[pair_mask_HH]]*HDIM + H_INDEX_START[neighbor_J[pair_mask_HH]] ] = HSSS
    
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_HH], fss_sigma[pair_mask_HH])        
    HSSS_dxyz = HSSS_dR * dR_dxyz[:, pair_mask_HH]
        
    dH0[:, H_INDEX_START[neighbor_I[pair_mask_HH]]*HDIM + H_INDEX_START[neighbor_J[pair_mask_HH]] ] = HSSS_dxyz
    #########

    # H-X
    idx_row = H_INDEX_START[neighbor_I[pair_mask_HX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_HX]]
    HSSS = bond_integral_vectorized(dR[pair_mask_HX], fss_sigma[pair_mask_HX])
    HSPS = bond_integral_vectorized(dR[pair_mask_HX], fsp_sigma[pair_mask_HX])
    
    H0[ idx_row*HDIM + idx_col ]   = HSSS
        
    H0[ idx_row*HDIM + idx_col +1] = L[pair_mask_HX]*HSPS
    H0[ idx_row*HDIM + idx_col +2] = M[pair_mask_HX]*HSPS
    H0[ idx_row*HDIM + idx_col +3] = N[pair_mask_HX]*HSPS
    
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_HX], fss_sigma[pair_mask_HX])
    HSSS_dxyz = HSSS_dR * dR_dxyz[:, pair_mask_HX]
    
    HSPS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_HX], fsp_sigma[pair_mask_HX])
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, pair_mask_HX]
        
    dH0[:, idx_row*HDIM + idx_col  ]  = HSSS_dxyz
    
    dH0[:, idx_row*HDIM + idx_col +1] = L[pair_mask_HX]*HSPS_dxyz + L_dxyz[:, pair_mask_HX]*HSPS
    dH0[:, idx_row*HDIM + idx_col +2] = M[pair_mask_HX]*HSPS_dxyz + M_dxyz[:, pair_mask_HX]*HSPS
    dH0[:, idx_row*HDIM + idx_col +3] = N[pair_mask_HX]*HSPS_dxyz + N_dxyz[:, pair_mask_HX]*HSPS
    #########

    # X-H
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XH]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XH]]
    HSSS = bond_integral_vectorized(dR[pair_mask_XH], fss_sigma[pair_mask_XH])
    HSPS = bond_integral_vectorized(dR[pair_mask_XH], fsp_sigma[pair_mask_XH])

    H0[ idx_row*HDIM + idx_col ]     = HSSS
        
    H0[ (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH]*HSPS
    H0[ (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH]*HSPS
    H0[ (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH]*HSPS
    
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_XH], fss_sigma[pair_mask_XH])
    HSSS_dxyz = HSSS_dR * dR_dxyz[:, pair_mask_XH]
    
    HSPS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_XH], fsp_sigma[pair_mask_XH])
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, pair_mask_XH]
    
    dH0[:, idx_row*HDIM + idx_col]      = HSSS_dxyz
    
    dH0[:, (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH]*HSPS_dxyz - L_dxyz[:,pair_mask_XH]*HSPS
    dH0[:, (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH]*HSPS_dxyz - M_dxyz[:,pair_mask_XH]*HSPS
    dH0[:, (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH]*HSPS_dxyz - N_dxyz[:,pair_mask_XH]*HSPS
    #########
    
    # X-X
    L_XX = L[pair_mask_XX]
    M_XX = M[pair_mask_XX]
    N_XX = N[pair_mask_XX]
    dR_XX = dR[pair_mask_XX]
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XX]]
    HSSS = bond_integral_vectorized(dR_XX, fss_sigma[pair_mask_XX])
    HSPS = bond_integral_vectorized(dR_XX, fsp_sigma[pair_mask_XX])
    HPSS = bond_integral_vectorized(dR_XX, fps_sigma[pair_mask_XX])
    HPPS = bond_integral_vectorized(dR_XX, fpp_sigma[pair_mask_XX])
    HPPP = bond_integral_vectorized(dR_XX, fpp_pi[pair_mask_XX])
    PPSMPP = HPPS - HPPP
    PXPX = HPPP + L_XX * L_XX * PPSMPP
    PXPY = L_XX * M_XX * PPSMPP
    PXPZ = L_XX * N_XX * PPSMPP
    PYPX = M_XX * L_XX * PPSMPP
    PYPY = HPPP + M_XX * M_XX * PPSMPP
    PYPZ = M_XX * N_XX * PPSMPP
    PZPX = N_XX * L_XX * PPSMPP
    PZPY = N_XX * M_XX * PPSMPP
    PZPZ = HPPP + N_XX * N_XX * PPSMPP

    ####
    H0[ idx_row*HDIM + idx_col ]       = HSSS
        
    H0[ idx_row*HDIM + idx_col +1]     = L_XX*HSPS
    H0[ idx_row*HDIM + idx_col +2]     = M_XX*HSPS
    H0[ idx_row*HDIM + idx_col +3]     = N_XX*HSPS
    
    ####
    H0[ (idx_row+1)*HDIM + idx_col ]   = -L_XX*HPSS

    H0[ (idx_row+1)*HDIM + idx_col +1] = PXPX
    H0[ (idx_row+1)*HDIM + idx_col +2] = PXPY
    H0[ (idx_row+1)*HDIM + idx_col +3] = PXPZ
    
    ####
    H0[ (idx_row+2)*HDIM + idx_col ]   = -M_XX*HPSS

    H0[ (idx_row+2)*HDIM + idx_col +1] = PYPX
    H0[ (idx_row+2)*HDIM + idx_col +2] = PYPY
    H0[ (idx_row+2)*HDIM + idx_col +3] = PYPZ
    
    ####
    H0[ (idx_row+3)*HDIM + idx_col ]   = -N_XX*HPSS

    H0[ (idx_row+3)*HDIM + idx_col +1] = PZPX
    H0[ (idx_row+3)*HDIM + idx_col +2] = PZPY
    H0[ (idx_row+3)*HDIM + idx_col +3] = PZPZ
    
    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, pair_mask_XX]
    L_dxyz_XX = L_dxyz[:,pair_mask_XX]
    M_dxyz_XX = M_dxyz[:,pair_mask_XX]
    N_dxyz_XX = N_dxyz[:,pair_mask_XX]
    
    HSSS_dR = bond_integral_with_grad_vectorized(dR_XX, fss_sigma[pair_mask_XX])
    HSSS_dxyz = HSSS_dR * dR_dxyz_XX

    HSPS_dR = bond_integral_with_grad_vectorized(dR_XX, fsp_sigma[pair_mask_XX])
    HSPS_dxyz = HSPS_dR * dR_dxyz_XX

    HPSS_dR = bond_integral_with_grad_vectorized(dR_XX, fps_sigma[pair_mask_XX])
    HPSS_dxyz = HPSS_dR * dR_dxyz_XX
    
    HPPS_dR = bond_integral_with_grad_vectorized(dR_XX, fpp_sigma[pair_mask_XX])
    HPPS_dxyz = HPPS_dR * dR_dxyz_XX

    HPPP_dR = bond_integral_with_grad_vectorized(dR_XX, fpp_pi[pair_mask_XX])
    HPPP_dxyz = HPPP_dR * dR_dxyz_XX

    PPSMPP_dxyz = HPPS_dxyz - HPPP_dxyz
    PXPX_dxyz = HPPP_dxyz + (L_XX**2) * PPSMPP_dxyz + 2*L_XX* L_dxyz_XX * PPSMPP
    PXPY_dxyz = L_XX * M_XX * PPSMPP_dxyz + L_dxyz_XX * M_XX * PPSMPP + L_XX * M_dxyz_XX * PPSMPP
    PXPZ_dxyz = L_XX * N_XX * PPSMPP_dxyz + L_dxyz_XX * N_XX * PPSMPP + L_XX * N_dxyz_XX * PPSMPP
    PYPX_dxyz = M_XX * L_XX * PPSMPP_dxyz + M_XX * L_dxyz_XX * PPSMPP + M_dxyz_XX * L_XX * PPSMPP
    PYPY_dxyz = HPPP_dxyz + (M_XX**2) * PPSMPP_dxyz + 2*M_XX*M_dxyz_XX * PPSMPP
    PYPZ_dxyz = M_XX * N_XX * PPSMPP_dxyz + M_dxyz_XX * N_XX * PPSMPP + M_XX * N_dxyz_XX * PPSMPP
    PZPX_dxyz = N_XX * L_XX * PPSMPP_dxyz + N_XX * L_dxyz_XX * PPSMPP + N_dxyz_XX * L_XX * PPSMPP
    PZPY_dxyz = N_XX * M_XX * PPSMPP_dxyz + N_XX * M_dxyz_XX * PPSMPP + N_dxyz_XX * M_XX * PPSMPP
    PZPZ_dxyz = HPPP_dxyz + (N_XX**2) * PPSMPP_dxyz + 2*N_XX*N_dxyz_XX * PPSMPP

    ####    
    dH0[:, idx_row*HDIM + idx_col ]       = HSSS_dxyz
        
    dH0[:, idx_row*HDIM + idx_col +1]     = L_XX*HSPS_dxyz + L_dxyz_XX*HSPS
    dH0[:, idx_row*HDIM + idx_col +2]     = M_XX*HSPS_dxyz + M_dxyz_XX*HSPS
    dH0[:, idx_row*HDIM + idx_col +3]     = N_XX*HSPS_dxyz + N_dxyz_XX*HSPS

    ####    
    dH0[:, (idx_row+1)*HDIM + idx_col ]   = -L_XX*HPSS_dxyz - L_dxyz_XX*HPSS
    
    dH0[:, (idx_row+1)*HDIM + idx_col +1] = PXPX_dxyz
    dH0[:, (idx_row+1)*HDIM + idx_col +2] = PXPY_dxyz
    dH0[:, (idx_row+1)*HDIM + idx_col +3] = PXPZ_dxyz
        
    ####    
    dH0[:, (idx_row+2)*HDIM + idx_col ]   = -M_XX*HPSS_dxyz - M_dxyz_XX*HPSS
    
    dH0[:, (idx_row+2)*HDIM + idx_col +1] = PYPX_dxyz
    dH0[:, (idx_row+2)*HDIM + idx_col +2] = PYPY_dxyz
    dH0[:, (idx_row+2)*HDIM + idx_col +3] = PYPZ_dxyz

    ####    
    dH0[:, (idx_row+3)*HDIM + idx_col ]   = -N_XX*HPSS_dxyz - N_dxyz_XX*HPSS
    
    dH0[:, (idx_row+3)*HDIM + idx_col +1] = PZPX_dxyz
    dH0[:, (idx_row+3)*HDIM + idx_col +2] = PZPY_dxyz
    dH0[:, (idx_row+3)*HDIM + idx_col +3] = PZPZ_dxyz
    #########
    return H0, dH0