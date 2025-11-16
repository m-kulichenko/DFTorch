import torch
from .BondIntegral import *

#@torch.compile(fullgraph=True, dynamic=True)  # optional extra flags
def Slater_Koster_Pair_SKF_vectorized(HDIM, dR_dxyz, L, M, N, L_dxyz, M_dxyz, N_dxyz,
                                  pair_mask_HH, pair_mask_HX, pair_mask_XH, pair_mask_XX,
                                  pair_mask_HY, pair_mask_XY, pair_mask_YH, pair_mask_YX, pair_mask_YY,
                                  dx, idx, IJ_pair_type, JI_pair_type, coeffs_tensor,
                                  neighbor_I, neighbor_J, H_INDEX_START, SH_shift):
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
    
    H0 = torch.zeros((HDIM*HDIM), dtype=dR_dxyz.dtype, device = dR_dxyz.device)
    dH0 = torch.zeros(3,HDIM*HDIM, dtype=H0.dtype, device=H0.device)
    
    #######
    coeffs_selected = coeffs_tensor[IJ_pair_type, idx, 9 + SH_shift*10]
    HSSS_all  = coeffs_selected[:,0] + coeffs_selected[:,1]*dx + coeffs_selected[:,2]*dx**2 + coeffs_selected[:,3]*dx**3
    H0.index_add_(0, H_INDEX_START[neighbor_I]*HDIM + H_INDEX_START[neighbor_J], HSSS_all)
    HSSS_dR  = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*dx + 3*coeffs_selected[:,3]*dx**2
    #######
    
    # H-H        
    ######### dH/dx
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0.index_add_(1, H_INDEX_START[neighbor_I]*HDIM + H_INDEX_START[neighbor_J], HSSS_dxyz)
    #########

    # H-X
    ###### HSPS_all
    tmp_mask = pair_mask_HX | pair_mask_XX | pair_mask_HY | pair_mask_YY | pair_mask_XY | pair_mask_YX
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 8 + SH_shift*10]
    HSPS_all = coeffs_selected[:,0] + coeffs_selected[:,1]*dx[tmp_mask] + coeffs_selected[:,2]*dx[tmp_mask]**2 + coeffs_selected[:,3]*dx[tmp_mask]**3

    H0.index_add_(0, idx_row*HDIM + idx_col +1, L[tmp_mask]*HSPS_all)
    H0.index_add_(0, idx_row*HDIM + idx_col +2, M[tmp_mask]*HSPS_all)
    H0.index_add_(0, idx_row*HDIM + idx_col +3, N[tmp_mask]*HSPS_all)
    
    ######### dH/dx    
    HSPS_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*dx[tmp_mask] + 3*coeffs_selected[:,3]*dx[tmp_mask]**2
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, tmp_mask]
            
    dH0.index_add_(1, idx_row*HDIM + idx_col +1, L[tmp_mask]*HSPS_dxyz + L_dxyz[:, tmp_mask]*HSPS_all)
    dH0.index_add_(1, idx_row*HDIM + idx_col +2, M[tmp_mask]*HSPS_dxyz + M_dxyz[:, tmp_mask]*HSPS_all)
    dH0.index_add_(1, idx_row*HDIM + idx_col +3, N[tmp_mask]*HSPS_dxyz + N_dxyz[:, tmp_mask]*HSPS_all)
    #########

    ### HPSS_all ###
    tmp_mask = pair_mask_XH | pair_mask_XX | pair_mask_YH | pair_mask_YY | pair_mask_XY | pair_mask_YX
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 8 + SH_shift*10]
    HPSS_all = coeffs_selected[:,0] + coeffs_selected[:,1]*dx[tmp_mask] + coeffs_selected[:,2]*dx[tmp_mask]**2 + coeffs_selected[:,3]*dx[tmp_mask]**3

    H0.index_add_(0, (idx_row +1)*HDIM + idx_col, -L[tmp_mask]*HPSS_all)
    H0.index_add_(0, (idx_row +2)*HDIM + idx_col, -M[tmp_mask]*HPSS_all)
    H0.index_add_(0, (idx_row +3)*HDIM + idx_col, -N[tmp_mask]*HPSS_all)

    ################
    ######### dH/dx    
    HPSS_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*dx[tmp_mask] + 3*coeffs_selected[:,3]*dx[tmp_mask]**2
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, tmp_mask]
        
    dH0.index_add_(1, (idx_row +1)*HDIM + idx_col, -L[tmp_mask]*HPSS_dxyz - L_dxyz[:,tmp_mask]*HPSS_all)
    dH0.index_add_(1, (idx_row +2)*HDIM + idx_col, -M[tmp_mask]*HPSS_dxyz - M_dxyz[:,tmp_mask]*HPSS_all)
    dH0.index_add_(1, (idx_row +3)*HDIM + idx_col, -N[tmp_mask]*HPSS_dxyz - N_dxyz[:,tmp_mask]*HPSS_all)
    #########
    
    # X-X
    tmp_mask = pair_mask_XX | pair_mask_YY | pair_mask_XY | pair_mask_YX
    L_XX = L[tmp_mask]
    M_XX = M[tmp_mask]
    N_XX = N[tmp_mask]
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 5 + SH_shift*10]
    HPPS = coeffs_selected[:,0] + coeffs_selected[:,1]*dx[tmp_mask] + coeffs_selected[:,2]*dx[tmp_mask]**2 + coeffs_selected[:,3]*dx[tmp_mask]**3
    HPPS_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*dx[tmp_mask] + 3*coeffs_selected[:,3]*dx[tmp_mask]**2
    
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 6 + SH_shift*10]
    HPPP = coeffs_selected[:,0] + coeffs_selected[:,1]*dx[tmp_mask] + coeffs_selected[:,2]*dx[tmp_mask]**2 + coeffs_selected[:,3]*dx[tmp_mask]**3
    HPPP_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*dx[tmp_mask] + 3*coeffs_selected[:,3]*dx[tmp_mask]**2

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

    H0.index_add_(0, (idx_row+1)*HDIM + idx_col +1, PXPX)
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col +2, PXPY)
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col +3, PXPZ)
    
    ####

    H0.index_add_(0, (idx_row+2)*HDIM + idx_col +1, PYPX)
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col +2, PYPY)
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col +3, PYPZ)
    
    ####

    H0.index_add_(0, (idx_row+3)*HDIM + idx_col +1, PZPX)
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col +2, PZPY)
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col +3, PZPZ)
    
    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, tmp_mask]
    L_dxyz_XX = L_dxyz[:,tmp_mask]
    M_dxyz_XX = M_dxyz[:,tmp_mask]
    N_dxyz_XX = N_dxyz[:,tmp_mask]
        
    HPPS_dxyz = HPPS_dR * dR_dxyz_XX
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
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col +1, PXPX_dxyz)
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col +2, PXPY_dxyz)
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col +3, PXPZ_dxyz)
        
    ####    
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col +1, PYPX_dxyz)
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col +2, PYPY_dxyz)
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col +3, PYPZ_dxyz)

    ####    
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col +1, PZPX_dxyz)
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col +2, PZPY_dxyz)
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col +3, PZPZ_dxyz)
    #########

    ### s-d
    tmp_mask = pair_mask_HY | pair_mask_XY | pair_mask_YY
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 7 + SH_shift*10]
    V_sd_sigma = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    H_S_XY = (3**0.5)*tmp_L*tmp_M*V_sd_sigma
    H_S_YZ = (3**0.5)*tmp_M*tmp_N*V_sd_sigma
    H_S_ZX = (3**0.5)*tmp_N*tmp_L*V_sd_sigma
    H_S_X2Y2 = 0.5*(3**0.5)*(tmp_L**2 - tmp_M**2)*V_sd_sigma
    H_S_Z2 = (tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_sd_sigma
    H0.index_add_(0, (idx_row)*HDIM + idx_col + 4, H_S_XY)
    H0.index_add_(0, (idx_row)*HDIM + idx_col + 5, H_S_YZ)
    H0.index_add_(0, (idx_row)*HDIM + idx_col + 6, H_S_ZX)
    H0.index_add_(0, (idx_row)*HDIM + idx_col + 7, H_S_X2Y2)
    H0.index_add_(0, (idx_row)*HDIM + idx_col + 8, H_S_Z2)
    # s-d/dx
    tmp_L_dxyz = L_dxyz[:,tmp_mask]
    tmp_M_dxyz = M_dxyz[:,tmp_mask]
    tmp_N_dxyz = N_dxyz[:,tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_sd_sigma_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    V_sd_sigma_dxyz = V_sd_sigma_dR * tmp_dR_dxyz
    H_S_XY_dxyz   = (3**0.5)*(tmp_L_dxyz*tmp_M*V_sd_sigma + tmp_L*tmp_M_dxyz*V_sd_sigma + tmp_L*tmp_M*V_sd_sigma_dxyz)
    H_S_YZ_dxyz   = (3**0.5)*(tmp_M_dxyz*tmp_N*V_sd_sigma + tmp_M*tmp_N_dxyz*V_sd_sigma + tmp_M*tmp_N*V_sd_sigma_dxyz)
    H_S_ZX_dxyz   = (3**0.5)*(tmp_N_dxyz*tmp_L*V_sd_sigma + tmp_N*tmp_L_dxyz*V_sd_sigma + tmp_N*tmp_L*V_sd_sigma_dxyz)
    H_S_X2Y2_dxyz = 0.5*(3**0.5)*((2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz)*V_sd_sigma + (tmp_L**2 - tmp_M**2)*V_sd_sigma_dxyz)
    H_S_Z2_dxyz   = (2*tmp_N*tmp_N_dxyz - 0.5*(2*tmp_L*tmp_L_dxyz + 2*tmp_M*tmp_M_dxyz))*V_sd_sigma + (tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_sd_sigma_dxyz
    dH0.index_add_(1, (idx_row)*HDIM + idx_col + 4, H_S_XY_dxyz  )
    dH0.index_add_(1, (idx_row)*HDIM + idx_col + 5, H_S_YZ_dxyz  )
    dH0.index_add_(1, (idx_row)*HDIM + idx_col + 6, H_S_ZX_dxyz  )
    dH0.index_add_(1, (idx_row)*HDIM + idx_col + 7, H_S_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row)*HDIM + idx_col + 8, H_S_Z2_dxyz  )

    ### p-d
    tmp_mask = pair_mask_XY | pair_mask_YY
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 3 + SH_shift*10]
    V_pd_sigma = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_pd_sigma_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 4 + SH_shift*10]
    V_pd_pi = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_pd_pi_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    H_X_XY   = (3**0.5)*tmp_L**2*tmp_M*V_pd_sigma + tmp_M*(1 - 2*tmp_L**2)*V_pd_pi
    H_X_YZ   = (3**0.5)*tmp_L*tmp_M*tmp_N*V_pd_sigma - 2*tmp_L*tmp_M*tmp_N*V_pd_pi
    H_X_ZX   = (3**0.5)*tmp_L**2*tmp_N*V_pd_sigma + tmp_N*(1 - 2*tmp_L**2)*V_pd_pi
    H_X_X2Y2 = 0.5*(3**0.5)*tmp_L*(tmp_L**2 - tmp_M**2)*V_pd_sigma + tmp_L*(1 - tmp_L**2 + tmp_M**2)*V_pd_pi
    H_X_Z2   = tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma - 3**0.5*tmp_L*tmp_N**2*V_pd_pi
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col + 4, H_X_XY  )
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col + 5, H_X_YZ  )
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col + 6, H_X_ZX  )
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col + 7, H_X_X2Y2)
    H0.index_add_(0, (idx_row+1)*HDIM + idx_col + 8, H_X_Z2  )
    H_Y_XY = (3**0.5)*tmp_M**2*tmp_L*V_pd_sigma + tmp_L*(1 - 2*tmp_M**2)*V_pd_pi                    
    H_Y_YZ = (3**0.5)*tmp_M**2*tmp_N*V_pd_sigma + tmp_N*(1 - 2*tmp_M**2)*V_pd_pi                    
    H_Y_ZX = (3**0.5)*tmp_L*tmp_M*tmp_N*V_pd_sigma - 2*tmp_L*tmp_M*tmp_N*V_pd_pi                            
    H_Y_X2Y2 = 0.5*(3**0.5)*tmp_M*(tmp_L**2 - tmp_M**2)*V_pd_sigma - tmp_M*(1 + tmp_L**2 - tmp_M**2)*V_pd_pi
    H_Y_Z2 = tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma - 3**0.5*tmp_M*tmp_N**2*V_pd_pi        
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col + 4, H_Y_XY  )
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col + 5, H_Y_YZ  )
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col + 6, H_Y_ZX  )
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col + 7, H_Y_X2Y2)
    H0.index_add_(0, (idx_row+2)*HDIM + idx_col + 8, H_Y_Z2  )
    H_Z_XY = (3**0.5)*tmp_L*tmp_M*tmp_N*V_pd_sigma - 2*tmp_L*tmp_M*tmp_N*V_pd_pi                             
    H_Z_YZ = (3**0.5)*tmp_N**2*tmp_M*V_pd_sigma + tmp_M*(1 - 2*tmp_N**2)*V_pd_pi                     
    H_Z_ZX = (3**0.5)*tmp_N**2*tmp_L*V_pd_sigma + tmp_L*(1 - 2*tmp_N**2)*V_pd_pi                     
    H_Z_X2Y2 = 0.5*(3**0.5)*tmp_N*(tmp_L**2 - tmp_M**2)*V_pd_sigma - tmp_N*(tmp_L**2 - tmp_M**2)*V_pd_pi     
    H_Z_Z2 = tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma + 3**0.5*tmp_N*(tmp_L**2 + tmp_M**2)*V_pd_pi
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col + 4, H_Z_XY  )
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col + 5, H_Z_YZ  )
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col + 6, H_Z_ZX  )
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col + 7, H_Z_X2Y2)
    H0.index_add_(0, (idx_row+3)*HDIM + idx_col + 8, H_Z_Z2  )
    # p-d/dx
    tmp_L_dxyz = L_dxyz[:,tmp_mask]
    tmp_M_dxyz = M_dxyz[:,tmp_mask]
    tmp_N_dxyz = N_dxyz[:,tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_pd_sigma_dxyz = V_pd_sigma_dR * tmp_dR_dxyz
    V_pd_pi_dxyz = V_pd_pi_dR * tmp_dR_dxyz

    H_X_XY_dxyz   = (3**0.5)*(2*tmp_L*tmp_L_dxyz*tmp_M*V_pd_sigma + tmp_L**2*tmp_M_dxyz*V_pd_sigma + tmp_L**2*tmp_M*V_pd_sigma_dxyz) +\
                    (tmp_M_dxyz*(1 - 2*tmp_L**2) - 4*tmp_L*tmp_L_dxyz*tmp_M)*V_pd_pi + tmp_M*(1 - 2*tmp_L**2)*V_pd_pi_dxyz    
    H_X_YZ_dxyz   = (3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_pd_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_pd_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_pd_sigma + tmp_L*tmp_M*tmp_N*V_pd_sigma_dxyz) -\
                    2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_pd_pi - 2*tmp_L*tmp_M*tmp_N*V_pd_pi_dxyz
    H_X_ZX_dxyz   = (3**0.5)*(2*tmp_L*tmp_L_dxyz*tmp_N*V_pd_sigma + tmp_L**2*tmp_N_dxyz*V_pd_sigma + tmp_L**2*tmp_N*V_pd_sigma_dxyz) +\
                    (tmp_N_dxyz*(1 - 2*tmp_L**2) - 4*tmp_L*tmp_L_dxyz*tmp_N)*V_pd_pi + tmp_N*(1 - 2*tmp_L**2)*V_pd_pi_dxyz
    H_X_X2Y2_dxyz = 0.5*(3**0.5)*((tmp_L_dxyz*(tmp_L**2 - tmp_M**2) + tmp_L*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_L*(tmp_L**2 - tmp_M**2)*V_pd_sigma_dxyz) +\
                    (tmp_L_dxyz*(1 - tmp_L**2 + tmp_M**2) - tmp_L*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_pi + tmp_L*(1 - tmp_L**2 + tmp_M**2)*V_pd_pi_dxyz
    H_X_Z2_dxyz   = (tmp_L_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_L*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma_dxyz -\
                    3**0.5*(tmp_L_dxyz*tmp_N**2 + 2*tmp_L*tmp_N*tmp_N_dxyz)*V_pd_pi - 3**0.5*tmp_L*tmp_N**2*V_pd_pi_dxyz
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col + 4, H_X_XY_dxyz  )
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col + 5, H_X_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col + 6, H_X_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col + 7, H_X_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+1)*HDIM + idx_col + 8, H_X_Z2_dxyz  )
    H_Y_XY_dxyz   = (3**0.5)*(2*tmp_M*tmp_M_dxyz*tmp_L*V_pd_sigma + tmp_M**2*tmp_L_dxyz*V_pd_sigma + tmp_M**2*tmp_L*V_pd_sigma_dxyz) +\
                    (tmp_L_dxyz*(1 - 2*tmp_M**2) - 4*tmp_M*tmp_M_dxyz*tmp_L)*V_pd_pi + tmp_L*(1 - 2*tmp_M**2)*V_pd_pi_dxyz    
    H_Y_YZ_dxyz   = (3**0.5)*(2*tmp_M*tmp_M_dxyz*tmp_N*V_pd_sigma + tmp_M**2*tmp_N_dxyz*V_pd_sigma + tmp_M**2*tmp_N*V_pd_sigma_dxyz) +\
                    (tmp_N_dxyz*(1 - 2*tmp_M**2) - 4*tmp_M*tmp_M_dxyz*tmp_N)*V_pd_pi + tmp_N*(1 - 2*tmp_M**2)*V_pd_pi_dxyz
    H_Y_ZX_dxyz   = (3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_pd_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_pd_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_pd_sigma + tmp_L*tmp_M*tmp_N*V_pd_sigma_dxyz) -\
                    2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_pd_pi - 2*tmp_L*tmp_M*tmp_N*V_pd_pi_dxyz
    H_Y_X2Y2_dxyz = 0.5*(3**0.5)*((tmp_M_dxyz*(tmp_L**2 - tmp_M**2) + tmp_M*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_M*(tmp_L**2 - tmp_M**2)*V_pd_sigma_dxyz) -\
                    (tmp_M_dxyz*(1 + tmp_L**2 - tmp_M**2) + tmp_M*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_pi - tmp_M*(1 + tmp_L**2 - tmp_M**2)*V_pd_pi_dxyz
    H_Y_Z2_dxyz   = (tmp_M_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_M*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma_dxyz -\
                    3**0.5*(tmp_M_dxyz*tmp_N**2 + 2*tmp_M*tmp_N*tmp_N_dxyz)*V_pd_pi - 3**0.5*tmp_M*tmp_N**2*V_pd_pi_dxyz
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col + 4, H_Y_XY_dxyz  )
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col + 5, H_Y_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col + 6, H_Y_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col + 7, H_Y_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+2)*HDIM + idx_col + 8, H_Y_Z2_dxyz  )
    H_Z_XY_dxyz   = (3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_pd_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_pd_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_pd_sigma + tmp_L*tmp_M*tmp_N*V_pd_sigma_dxyz) -\
                    2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_pd_pi - 2*tmp_L*tmp_M*tmp_N*V_pd_pi_dxyz
    H_Z_YZ_dxyz   = (3**0.5)*(2*tmp_N*tmp_N_dxyz*tmp_M*V_pd_sigma + tmp_N**2*tmp_M_dxyz*V_pd_sigma + tmp_N**2*tmp_M*V_pd_sigma_dxyz) +\
                    (tmp_M_dxyz*(1 - 2*tmp_N**2) - 4*tmp_N*tmp_N_dxyz*tmp_M)*V_pd_pi + tmp_M*(1 - 2*tmp_N**2)*V_pd_pi_dxyz        
    H_Z_ZX_dxyz   = (3**0.5)*(2*tmp_N*tmp_N_dxyz*tmp_L*V_pd_sigma + tmp_N**2*tmp_L_dxyz*V_pd_sigma + tmp_N**2*tmp_L*V_pd_sigma_dxyz) +\
                    (tmp_L_dxyz*(1 - 2*tmp_N**2) - 4*tmp_N*tmp_N_dxyz*tmp_L)*V_pd_pi + tmp_L*(1 - 2*tmp_N**2)*V_pd_pi_dxyz    
    H_Z_X2Y2_dxyz = 0.5*(3**0.5)*((tmp_N_dxyz*(tmp_L**2 - tmp_M**2) + tmp_N*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_N*(tmp_L**2 - tmp_M**2)*V_pd_sigma_dxyz) -\
                    (tmp_N_dxyz*(tmp_L**2 - tmp_M**2) + tmp_N*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_pd_pi - tmp_N*(tmp_L**2 - tmp_M**2)*V_pd_pi_dxyz    
    H_Z_Z2_dxyz   = (tmp_N_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_N*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_pd_sigma + tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_pd_sigma_dxyz +\
                    3**0.5*(tmp_N_dxyz*(tmp_L**2 + tmp_M**2) + 2*tmp_N*(tmp_L*tmp_L_dxyz + tmp_M*tmp_M_dxyz))*V_pd_pi + 3**0.5*tmp_N*(tmp_L**2 + tmp_M**2)*V_pd_pi_dxyz
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col + 4, H_Z_XY_dxyz  )
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col + 5, H_Z_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col + 6, H_Z_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col + 7, H_Z_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+3)*HDIM + idx_col + 8, H_Z_Z2_dxyz  )

    ### d-s
    tmp_mask = pair_mask_YH | pair_mask_YX | pair_mask_YY
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 7 + SH_shift*10]
    V_ds_sigma = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_ds_sigma_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    H_XY_S   =  (3**0.5)*tmp_L*tmp_M*V_ds_sigma                                                                                                                        
    H_YZ_S   = (3**0.5)*tmp_M*tmp_N*V_ds_sigma                                                                                                                         
    H_ZX_S   =  (3**0.5)*tmp_N*tmp_L*V_ds_sigma                                                                                                                        
    H_X2Y2_S =   0.5*(3**0.5)*(tmp_L**2 - tmp_M**2)*V_ds_sigma                                                                                                       
    H_Z2_S   =  (tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_ds_sigma    
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col, H_XY_S  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col, H_YZ_S  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col, H_ZX_S  )
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col, H_X2Y2_S)
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col, H_Z2_S  )        
    # d-s/dx
    tmp_L_dxyz = L_dxyz[:,tmp_mask]
    tmp_M_dxyz = M_dxyz[:,tmp_mask]
    tmp_N_dxyz = N_dxyz[:,tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_ds_sigma_dxyz = V_ds_sigma_dR * tmp_dR_dxyz
    H_XY_S_dxyz   = (3**0.5)*(tmp_L_dxyz*tmp_M*V_ds_sigma + tmp_L*tmp_M_dxyz*V_ds_sigma + tmp_L*tmp_M*V_ds_sigma_dxyz)
    H_YZ_S_dxyz   = (3**0.5)*(tmp_M_dxyz*tmp_N*V_ds_sigma + tmp_M*tmp_N_dxyz*V_ds_sigma + tmp_M*tmp_N*V_ds_sigma_dxyz)
    H_ZX_S_dxyz   = (3**0.5)*(tmp_N_dxyz*tmp_L*V_ds_sigma + tmp_N*tmp_L_dxyz*V_ds_sigma + tmp_N*tmp_L*V_ds_sigma_dxyz)
    H_X2Y2_S_dxyz = 0.5*(3**0.5)*((2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz)*V_ds_sigma + (tmp_L**2 - tmp_M**2)*V_ds_sigma_dxyz)
    H_Z2_S_dxyz   = (2*tmp_N*tmp_N_dxyz - (tmp_L*tmp_L_dxyz + tmp_M*tmp_M_dxyz))*V_ds_sigma + (tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_ds_sigma_dxyz
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col, H_XY_S_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col, H_YZ_S_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col, H_ZX_S_dxyz  )
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col, H_X2Y2_S_dxyz)
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col, H_Z2_S_dxyz  )
    
    ### d-p                                                                                                
    tmp_mask = pair_mask_YX | pair_mask_YY
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = JI_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 3 + SH_shift*10]
    V_dp_sigma = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_dp_sigma_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 4 + SH_shift*10]
    V_dp_pi = coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_dp_pi_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    H_XY_X = -((3**0.5)*tmp_L**2*tmp_M*V_dp_sigma + tmp_M*(1 - 2*tmp_L**2)*V_dp_pi)                                                                                       
    H_XY_Y = -((3**0.5)*tmp_M**2*tmp_L*V_dp_sigma + tmp_L*(1 - 2*tmp_M**2)*V_dp_pi)                                                                                       
    H_XY_Z = -((3**0.5)*tmp_L*tmp_M*tmp_N*V_dp_sigma - 2*tmp_L*tmp_M*tmp_N*V_dp_pi)      
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+1, H_XY_X  )
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+2, H_XY_Y  )
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+3, H_XY_Z  )
    H_YZ_X = -((3**0.5)*tmp_L*tmp_M*tmp_N*V_dp_sigma - 2*tmp_L*tmp_M*tmp_N*V_dp_pi)        
    H_YZ_Y = -((3**0.5)*tmp_M**2*tmp_N*V_dp_sigma + tmp_N*(1 - 2*tmp_M**2)*V_dp_pi)
    H_YZ_Z = -((3**0.5)*tmp_N**2*tmp_M*V_dp_sigma + tmp_M*(1 - 2*tmp_N**2)*V_dp_pi)
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+1, H_YZ_X  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+2, H_YZ_Y  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+3, H_YZ_Z  )
    H_ZX_X = -((3**0.5)*tmp_L**2*tmp_N*V_dp_sigma + tmp_N*(1 - 2*tmp_L**2)*V_dp_pi)
    H_ZX_Y = -((3**0.5)*tmp_L*tmp_M*tmp_N*V_dp_sigma - 2*tmp_L*tmp_M*tmp_N*V_dp_pi)        
    H_ZX_Z = -((3**0.5)*tmp_N**2*tmp_L*V_dp_sigma + tmp_L*(1 - 2*tmp_N**2)*V_dp_pi)    
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+1, H_ZX_X  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+2, H_ZX_Y  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+3, H_ZX_Z  )
    H_X2Y2_X = -(0.5*(3**0.5)*tmp_L*(tmp_L**2 - tmp_M**2)*V_dp_sigma + tmp_L*(1 - tmp_L**2 + tmp_M**2)*V_dp_pi)
    H_X2Y2_Y = -(0.5*(3**0.5)*tmp_M*(tmp_L**2 - tmp_M**2)*V_dp_sigma - tmp_M*(1 + tmp_L**2 - tmp_M**2)*V_dp_pi)       
    H_X2Y2_Z = -(0.5*(3**0.5)*tmp_N*(tmp_L**2 - tmp_M**2)*V_dp_sigma - tmp_N*(tmp_L**2 - tmp_M**2)*V_dp_pi)    
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+1, H_X2Y2_X)
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+2, H_X2Y2_Y)
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+3, H_X2Y2_Z)
    H_Z2_X = -(tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma - 3**0.5*tmp_L*tmp_N**2*V_dp_pi)         
    H_Z2_Y = -(tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma - 3**0.5*tmp_M*tmp_N**2*V_dp_pi)         
    H_Z2_Z = -(tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma + 3**0.5*tmp_N*(tmp_L**2 + tmp_M**2)*V_dp_pi)
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+1, H_Z2_X  )
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+2, H_Z2_Y  )
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+3, H_Z2_Z  )
    # d-p/dx
    tmp_L_dxyz = L_dxyz[:,tmp_mask]
    tmp_M_dxyz = M_dxyz[:,tmp_mask]
    tmp_N_dxyz = N_dxyz[:,tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_dp_sigma_dxyz = V_dp_sigma_dR * tmp_dR_dxyz
    V_dp_pi_dxyz = V_dp_pi_dR * tmp_dR_dxyz
    H_XY_X_dxyz   = -((3**0.5)*(2*tmp_L*tmp_L_dxyz*tmp_M*V_dp_sigma + tmp_L**2*tmp_M_dxyz*V_dp_sigma + tmp_L**2*tmp_M*V_dp_sigma_dxyz) +\
                      (tmp_M_dxyz*(1 - 2*tmp_L**2) - 4*tmp_L*tmp_L_dxyz*tmp_M)*V_dp_pi + tmp_M*(1 - 2*tmp_L**2)*V_dp_pi_dxyz)
    H_XY_Y_dxyz   = -((3**0.5)*(2*tmp_M*tmp_M_dxyz*tmp_L*V_dp_sigma + tmp_M**2*tmp_L_dxyz*V_dp_sigma + tmp_M**2*tmp_L*V_dp_sigma_dxyz) +\
                      (tmp_L_dxyz*(1 - 2*tmp_M**2) - 4*tmp_M*tmp_M_dxyz*tmp_L)*V_dp_pi + tmp_L*(1 - 2*tmp_M**2)*V_dp_pi_dxyz)
    H_XY_Z_dxyz   = -((3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_dp_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_dp_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_dp_sigma + tmp_L*tmp_M*tmp_N*V_dp_sigma_dxyz) -\
                      2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_dp_pi - 2*tmp_L*tmp_M*tmp_N*V_dp_pi_dxyz)
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+1, H_XY_X_dxyz  )
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+2, H_XY_Y_dxyz  )
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+3, H_XY_Z_dxyz  )
    H_YZ_X_dxyz   = -((3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_dp_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_dp_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_dp_sigma + tmp_L*tmp_M*tmp_N*V_dp_sigma_dxyz) -\
                      2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_dp_pi - 2*tmp_L*tmp_M*tmp_N*V_dp_pi_dxyz)
    H_YZ_Y_dxyz   = -((3**0.5)*(2*tmp_M*tmp_M_dxyz*tmp_N*V_dp_sigma + tmp_M**2*tmp_N_dxyz*V_dp_sigma + tmp_M**2*tmp_N*V_dp_sigma_dxyz) +\
                      (tmp_N_dxyz*(1 - 2*tmp_M**2) - 4*tmp_M*tmp_M_dxyz*tmp_N)*V_dp_pi + tmp_N*(1 - 2*tmp_M**2)*V_dp_pi_dxyz)
    H_YZ_Z_dxyz   = -((3**0.5)*(2*tmp_N*tmp_N_dxyz*tmp_M*V_dp_sigma + tmp_N**2*tmp_M_dxyz*V_dp_sigma + tmp_N**2*tmp_M*V_dp_sigma_dxyz) +\
                      (tmp_M_dxyz*(1 - 2*tmp_N**2) - 4*tmp_N*tmp_N_dxyz*tmp_M)*V_dp_pi + tmp_M*(1 - 2*tmp_N**2)*V_dp_pi_dxyz)
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+1, H_YZ_X_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+2, H_YZ_Y_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+3, H_YZ_Z_dxyz  )
    H_ZX_X_dxyz   = -((3**0.5)*(2*tmp_L*tmp_L_dxyz*tmp_N*V_dp_sigma + tmp_L**2*tmp_N_dxyz*V_dp_sigma + tmp_L**2*tmp_N*V_dp_sigma_dxyz) +\
                      (tmp_N_dxyz*(1 - 2*tmp_L**2) - 4*tmp_L*tmp_L_dxyz*tmp_N)*V_dp_pi + tmp_N*(1 - 2*tmp_L**2)*V_dp_pi_dxyz)    
    H_ZX_Y_dxyz   = -((3**0.5)*(tmp_L_dxyz*tmp_M*tmp_N*V_dp_sigma + tmp_L*tmp_M_dxyz*tmp_N*V_dp_sigma + tmp_L*tmp_M*tmp_N_dxyz*V_dp_sigma + tmp_L*tmp_M*tmp_N*V_dp_sigma_dxyz) -\
                      2*(tmp_L_dxyz*tmp_M*tmp_N + tmp_L*tmp_M_dxyz*tmp_N + tmp_L*tmp_M*tmp_N_dxyz)*V_dp_pi - 2*tmp_L*tmp_M*tmp_N*V_dp_pi_dxyz)    
    H_ZX_Z_dxyz   = -((3**0.5)*(2*tmp_N*tmp_N_dxyz*tmp_L*V_dp_sigma + tmp_N**2*tmp_L_dxyz*V_dp_sigma + tmp_N**2*tmp_L*V_dp_sigma_dxyz) +\
                      (tmp_L_dxyz*(1 - 2*tmp_N**2) - 4*tmp_N*tmp_N_dxyz*tmp_L)*V_dp_pi + tmp_L*(1 - 2*tmp_N**2)*V_dp_pi_dxyz)
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+1, H_ZX_X_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+2, H_ZX_Y_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+3, H_ZX_Z_dxyz  )
    H_X2Y2_X_dxyz = - (0.5*(3**0.5)*((tmp_L_dxyz*(tmp_L**2 - tmp_M**2) + tmp_L*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_L*(tmp_L**2 - tmp_M**2)*V_dp_sigma_dxyz) +\
                       (tmp_L_dxyz*(1 - tmp_L**2 + tmp_M**2) - tmp_L*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_pi + tmp_L*(1 - tmp_L**2 + tmp_M**2)*V_dp_pi_dxyz)
    H_X2Y2_Y_dxyz = - (0.5*(3**0.5)*((tmp_M_dxyz*(tmp_L**2 - tmp_M**2) + tmp_M*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_M*(tmp_L**2 - tmp_M**2)*V_dp_sigma_dxyz) -\
                       (tmp_M_dxyz*(1 + tmp_L**2 - tmp_M**2) + tmp_M*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_pi - tmp_M*(1 + tmp_L**2 - tmp_M**2)*V_dp_pi_dxyz)
    H_X2Y2_Z_dxyz = - (0.5*(3**0.5)*((tmp_N_dxyz*(tmp_L**2 - tmp_M**2) + tmp_N*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_N*(tmp_L**2 - tmp_M**2)*V_dp_sigma_dxyz) -\
                       (tmp_N_dxyz*(tmp_L**2 - tmp_M**2) + tmp_N*(2*tmp_L*tmp_L_dxyz - 2*tmp_M*tmp_M_dxyz))*V_dp_pi - tmp_N*(tmp_L**2 - tmp_M**2)*V_dp_pi_dxyz)
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+1, H_X2Y2_X_dxyz)
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+2, H_X2Y2_Y_dxyz)
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+3, H_X2Y2_Z_dxyz)
    H_Z2_X_dxyz   = -((tmp_L_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_L*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma_dxyz -\
                      3**0.5*(tmp_L_dxyz*tmp_N**2 + 2*tmp_L*tmp_N*tmp_N_dxyz)*V_dp_pi - 3**0.5*tmp_L*tmp_N**2*V_dp_pi_dxyz)         
    H_Z2_Y_dxyz   = -((tmp_M_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_M*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma_dxyz -\
                      3**0.5*(tmp_M_dxyz*tmp_N**2 + 2*tmp_M*tmp_N*tmp_N_dxyz)*V_dp_pi - 3**0.5*tmp_M*tmp_N**2*V_dp_pi_dxyz)
    H_Z2_Z_dxyz   = -((tmp_N_dxyz*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2)) + tmp_N*(2*tmp_N*tmp_N_dxyz - tmp_L*tmp_L_dxyz - tmp_M*tmp_M_dxyz))*V_dp_sigma + tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dp_sigma_dxyz +\
                      3**0.5*(tmp_N_dxyz*(tmp_L**2 + tmp_M**2) + 2*tmp_N*(tmp_L*tmp_L_dxyz + tmp_M*tmp_M_dxyz))*V_dp_pi + 3**0.5*tmp_N*(tmp_L**2 + tmp_M**2)*V_dp_pi_dxyz)
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+1, H_Z2_X_dxyz  )
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+2, H_Z2_Y_dxyz  )
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+3, H_Z2_Z_dxyz  )

    ### d-d
    tmp_mask = pair_mask_YY
    idx_row = H_INDEX_START[neighbor_I[tmp_mask]]
    idx_col = H_INDEX_START[neighbor_J[tmp_mask]]
    tmp_dx = dx[tmp_mask]
    tmp_L = L[tmp_mask]
    tmp_M = M[tmp_mask]
    tmp_N = N[tmp_mask]
    sel_IJ = IJ_pair_type[tmp_mask]
    sel_idx = idx[tmp_mask]
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 0 + SH_shift*10]
    V_dd_sigma =    coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_dd_sigma_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 1 + SH_shift*10]
    V_dd_pi =       coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_dd_pi_dR =    coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    coeffs_selected = coeffs_tensor[sel_IJ, sel_idx, 2 + SH_shift*10]
    V_dd_delta =    coeffs_selected[:,0] + coeffs_selected[:,1]*tmp_dx + coeffs_selected[:,2]*tmp_dx**2 + coeffs_selected[:,3]*tmp_dx**3
    V_dd_delta_dR = coeffs_selected[:,1] + 2*coeffs_selected[:,2]*tmp_dx + 3*coeffs_selected[:,3]*tmp_dx**2
    H_XY_XY   = 3*tmp_L**2*tmp_M**2*V_dd_sigma + (tmp_L**2 + tmp_M**2 - 4*tmp_L**2*tmp_M**2)*V_dd_pi + (tmp_N**2 + tmp_L**2*tmp_M**2)*V_dd_delta                             
    H_XY_YZ   = 3*tmp_L*tmp_M**2*tmp_N*V_dd_sigma + tmp_L*tmp_N*(1 - 4*tmp_M**2)*V_dd_pi + tmp_L*tmp_N*(tmp_M**2 - 1)*V_dd_delta                                             
    H_XY_ZX   = 3*tmp_L**2*tmp_M*tmp_N*V_dd_sigma + tmp_M*tmp_N*(1 - 4*tmp_L**2)*V_dd_pi + tmp_M*tmp_N*(tmp_L**2 - 1)*V_dd_delta                                             
    H_XY_X2Y2 = 1.5*tmp_L*tmp_M*(tmp_L**2 - tmp_M**2)*V_dd_sigma + 2*tmp_L*tmp_M*(tmp_M**2 - tmp_L**2)*V_dd_pi + 0.5*tmp_L*tmp_M*(tmp_L**2 - tmp_M**2)*V_dd_delta                      
    H_XY_Z2   = (3**0.5)*tmp_L*tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma - (3**0.5)*2*tmp_L*tmp_M*tmp_N**2*V_dd_pi + (3**0.5)*0.5*tmp_L*tmp_M*(1 + tmp_N**2)*V_dd_delta
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+4, H_XY_XY  )
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+5, H_XY_YZ  )
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+6, H_XY_ZX  )
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+7, H_XY_X2Y2)
    H0.index_add_(0, (idx_row+4)*HDIM + idx_col+8, H_XY_Z2  )        
    H_YZ_XY   = 3*tmp_M**2*tmp_N*tmp_L*V_dd_sigma + tmp_L*tmp_N*(1 - 4*tmp_M**2)*V_dd_pi + tmp_L*tmp_N*(tmp_M**2 - 1)*V_dd_delta                                                                   
    H_YZ_YZ   = 3*tmp_M**2*tmp_N**2*V_dd_sigma + (tmp_M**2 + tmp_N**2 - 4*tmp_M**2*tmp_N**2)*V_dd_pi + (tmp_L**2 + tmp_M**2*tmp_N**2)*V_dd_delta                                                   
    H_YZ_ZX   = 3*tmp_M*tmp_N**2*tmp_L*V_dd_sigma + tmp_L*tmp_M*(1 - 4*tmp_N**2)*V_dd_pi + tmp_L*tmp_M*(tmp_N**2 - 1)*V_dd_delta                                                                   
    H_YZ_X2Y2 = 1.5*tmp_M*tmp_N*(tmp_L**2 - tmp_M**2)*V_dd_sigma - tmp_M*tmp_N*(1 + 2*(tmp_L**2 - tmp_M**2))*V_dd_pi + tmp_M*tmp_N*(1 + 0.5*(tmp_L**2 - tmp_M**2))*V_dd_delta                                
    H_YZ_Z2   = (3**0.5)*tmp_M*tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_M*tmp_N*(tmp_L**2 + tmp_M**2 - tmp_N**2)*V_dd_pi - (3**0.5)*0.5*tmp_M*tmp_N*(tmp_L**2 + tmp_M**2)*V_dd_delta     
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+4, H_YZ_XY  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+5, H_YZ_YZ  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+6, H_YZ_ZX  )
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+7, H_YZ_X2Y2)
    H0.index_add_(0, (idx_row+5)*HDIM + idx_col+8, H_YZ_Z2  )
    H_ZX_XY   = 3*tmp_L**2*tmp_M*tmp_N*V_dd_sigma + tmp_M*tmp_N*(1 - 4*tmp_L**2)*V_dd_pi + tmp_M*tmp_N*(tmp_L**2 - 1)*V_dd_delta                                                                   
    H_ZX_YZ   = 3*tmp_M*tmp_N**2*tmp_L*V_dd_sigma + tmp_L*tmp_M*(1 - 4*tmp_N**2)*V_dd_pi + tmp_L*tmp_M*(tmp_N**2 - 1)*V_dd_delta                                                                   
    H_ZX_ZX   = 3*tmp_N**2*tmp_L**2*V_dd_sigma + (tmp_N**2 + tmp_L**2 - 4*tmp_N**2*tmp_L**2)*V_dd_pi + (tmp_M**2 + tmp_N**2*tmp_L**2)*V_dd_delta                                                   
    H_ZX_X2Y2 = 1.5*tmp_N*tmp_L*(tmp_L**2 - tmp_M**2)*V_dd_sigma + tmp_N*tmp_L*(1 - 2*(tmp_L**2 - tmp_M**2))*V_dd_pi - tmp_N*tmp_L*(1 - 0.5*(tmp_L**2 - tmp_M**2))*V_dd_delta                                
    H_ZX_Z2   = (3**0.5)*tmp_N*tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_N*tmp_L*(tmp_L**2 + tmp_M**2 - tmp_N**2)*V_dd_pi - (3**0.5)*0.5*tmp_N*tmp_L*(tmp_L**2 + tmp_M**2)*V_dd_delta
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+4, H_ZX_XY  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+5, H_ZX_YZ  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+6, H_ZX_ZX  )
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+7, H_ZX_X2Y2)
    H0.index_add_(0, (idx_row+6)*HDIM + idx_col+8, H_ZX_Z2  )
    H_X2Y2_XY   = 1.5*tmp_L*tmp_M*(tmp_L**2 - tmp_M**2)*V_dd_sigma + 2*tmp_L*tmp_M*(tmp_M**2 - tmp_L**2)*V_dd_pi + 0.5*tmp_L*tmp_M*(tmp_L**2 - tmp_M**2)*V_dd_delta                                            
    H_X2Y2_YZ   = 1.5*tmp_M*tmp_N*(tmp_L**2 - tmp_M**2)*V_dd_sigma - tmp_M*tmp_N*(1 + 2*(tmp_L**2 - tmp_M**2))*V_dd_pi + tmp_M*tmp_N*(1 + 0.5*(tmp_L**2 - tmp_M**2))*V_dd_delta                                
    H_X2Y2_ZX   = 1.5*tmp_N*tmp_L*(tmp_L**2 - tmp_M**2)*V_dd_sigma + tmp_N*tmp_L*(1 - 2*(tmp_L**2 - tmp_M**2))*V_dd_pi - tmp_N*tmp_L*(1 - 0.5*(tmp_L**2 - tmp_M**2))*V_dd_delta                                
    H_X2Y2_X2Y2 = 0.75*(tmp_L**2 - tmp_M**2)**2*V_dd_sigma + (tmp_L**2 + tmp_M**2 - (tmp_L**2 - tmp_M**2)**2)*V_dd_pi + (tmp_N**2 + 0.25*(tmp_L**2 - tmp_M**2)**2)*V_dd_delta                    
    H_X2Y2_Z2   = (3**0.5)*0.5*(tmp_L**2 - tmp_M**2)*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_N**2*(tmp_M**2 - tmp_L**2)*V_dd_pi + (3**0.5)*0.25*(1 + tmp_N**2)*(tmp_L**2 - tmp_M**2)*V_dd_delta   
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+4, H_X2Y2_XY  )
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+5, H_X2Y2_YZ  )
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+6, H_X2Y2_ZX  )
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+7, H_X2Y2_X2Y2)
    H0.index_add_(0, (idx_row+7)*HDIM + idx_col+8, H_X2Y2_Z2  )
    H_Z2_XY   = (3**0.5)*tmp_L*tmp_M*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma - (3**0.5)*2*tmp_L*tmp_M*tmp_N**2*V_dd_pi + (3**0.5)*0.5*tmp_L*tmp_M*(1 + tmp_N**2)*V_dd_delta                      
    H_Z2_YZ   = (3**0.5)*tmp_M*tmp_N*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_M*tmp_N*(tmp_L**2 + tmp_M**2 - tmp_N**2)*V_dd_pi - (3**0.5)*0.5*tmp_M*tmp_N*(tmp_L**2 + tmp_M**2)*V_dd_delta     
    H_Z2_ZX   = (3**0.5)*tmp_N*tmp_L*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_N*tmp_L*(tmp_L**2 + tmp_M**2 - tmp_N**2)*V_dd_pi - (3**0.5)*0.5*tmp_N*tmp_L*(tmp_L**2 + tmp_M**2)*V_dd_delta     
    H_Z2_X2Y2 = (3**0.5)*0.5*(tmp_L**2 - tmp_M**2)*(tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))*V_dd_sigma + (3**0.5)*tmp_N**2*(tmp_M**2 - tmp_L**2)*V_dd_pi + (3**0.5)*0.25*(1 + tmp_N**2)*(tmp_L**2 - tmp_M**2)*V_dd_delta   
    H_Z2_Z2   = (tmp_N**2 - 0.5*(tmp_L**2 + tmp_M**2))**2*V_dd_sigma + 3*tmp_N**2*(tmp_L**2 + tmp_M**2)*V_dd_pi + 0.75*(tmp_L**2 + tmp_M**2)**2*V_dd_delta                                     
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+4, H_Z2_XY  )
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+5, H_Z2_YZ  )
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+6, H_Z2_ZX  )
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+7, H_Z2_X2Y2)
    H0.index_add_(0, (idx_row+8)*HDIM + idx_col+8, H_Z2_Z2  )
    # d-d/dx
    tmp_L_dxyz = L_dxyz[:,tmp_mask]
    tmp_M_dxyz = M_dxyz[:,tmp_mask]
    tmp_N_dxyz = N_dxyz[:,tmp_mask]
    tmp_dR_dxyz = dR_dxyz[:, tmp_mask]
    V_dd_sigma_dxyz = V_dd_sigma_dR * tmp_dR_dxyz
    V_dd_pi_dxyz = V_dd_pi_dR * tmp_dR_dxyz
    V_dd_delta_dxyz = V_dd_delta_dR * tmp_dR_dxyz
    # t - time, m - minus, p - plus
    L_t_Ldx = tmp_L*tmp_L_dxyz
    M_t_Mdx = tmp_M*tmp_M_dxyz
    N_t_Ndx = tmp_N*tmp_N_dxyz
    L_t_M = tmp_L*tmp_M
    M_t_N = tmp_M*tmp_N
    N_t_L = tmp_N*tmp_L

    L2 = tmp_L**2
    M2 = tmp_M**2
    N2 = tmp_N**2


    H_XY_XY_dxyz   = 3*(2*L_t_Ldx*M2 + L2*2*M_t_Mdx)*V_dd_sigma + 3*L2*M2*V_dd_sigma_dxyz +\
                     ((2*L_t_Ldx + 2*M_t_Mdx) - 4*(2*L_t_Ldx*M2 + L2*2*M_t_Mdx))*V_dd_pi + (L2 + M2 - 4*L2*M2)*V_dd_pi_dxyz +\
                     (2*N_t_Ndx + 2*L_t_Ldx*M2 + L2*2*M_t_Mdx)*V_dd_delta + (N2 + L2*M2)*V_dd_delta_dxyz                          
    H_XY_YZ_dxyz   = 3*(tmp_L_dxyz*M2*tmp_N + tmp_L*2*M_t_Mdx*tmp_N + tmp_L*M2*tmp_N_dxyz)*V_dd_sigma + 3*tmp_L*M2*tmp_N*V_dd_sigma_dxyz +\
                     (tmp_L_dxyz*tmp_N + tmp_L*tmp_N_dxyz)*(1 - 4*M2)*V_dd_pi + tmp_L*tmp_N*(-8*M_t_Mdx)*V_dd_pi + tmp_L*tmp_N*(1 - 4*M2)*V_dd_pi_dxyz +\
                     (tmp_L_dxyz*tmp_N + tmp_L*tmp_N_dxyz)*(M2 - 1)*V_dd_delta + tmp_L*tmp_N*(2*M_t_Mdx)*V_dd_delta + tmp_L*tmp_N*(M2 - 1)*V_dd_delta_dxyz                                                 
    H_XY_ZX_dxyz   = 3*(2*L_t_Ldx*M_t_N + L2*tmp_M_dxyz*tmp_N + L2*tmp_M*tmp_N_dxyz)*V_dd_sigma + 3*L2*M_t_N*V_dd_sigma_dxyz +\
                     (tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 - 4*L2)*V_dd_pi + M_t_N*(-8*L_t_Ldx)*V_dd_pi + M_t_N*(1 - 4*L2)*V_dd_pi_dxyz +\
                     (tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 - 1)*V_dd_delta + M_t_N*(2*L_t_Ldx)*V_dd_delta + M_t_N*(L2 - 1)*V_dd_delta_dxyz                                      
    H_XY_X2Y2_dxyz = 1.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(L2 - M2) + L_t_M*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*L_t_M*(L2 - M2)*V_dd_sigma_dxyz +\
                     2*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(M2 - L2) + L_t_M*(2*M_t_Mdx - 2*L_t_Ldx))*V_dd_pi + 2*L_t_M*(M2 - L2)*V_dd_pi_dxyz +\
                     0.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(L2 - M2) + L_t_M*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_delta + 0.5*L_t_M*(L2 - M2)*V_dd_delta_dxyz
    H_XY_Z2_dxyz   = (3**0.5)*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(N2 - 0.5*(L2 + M2)) + L_t_M*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*L_t_M*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz -\
                     (3**0.5)*2*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*N2 + L_t_M*2*N_t_Ndx)*V_dd_pi - (3**0.5)*2*L_t_M*N2*V_dd_pi_dxyz +\
                     (3**0.5)*0.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(1 + N2) + L_t_M*2*N_t_Ndx)*V_dd_delta + (3**0.5)*0.5*L_t_M*(1 + N2)*V_dd_delta_dxyz
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+4, H_XY_XY_dxyz  )
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+5, H_XY_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+6, H_XY_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+7, H_XY_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+4)*HDIM + idx_col+8, H_XY_Z2_dxyz  )
    H_YZ_XY_dxyz   = 3*(2*M_t_Mdx*N_t_L + M2*tmp_N_dxyz*tmp_L + M2*tmp_N*tmp_L_dxyz)*V_dd_sigma + 3*M2*N_t_L*V_dd_sigma_dxyz +\
                     (tmp_L_dxyz*tmp_N + tmp_L*tmp_N_dxyz)*(1 - 4*M2)*V_dd_pi + tmp_L*tmp_N*(-8*M_t_Mdx)*V_dd_pi + tmp_L*tmp_N*(1 - 4*M2)*V_dd_pi_dxyz +\
                     (tmp_L_dxyz*tmp_N + tmp_L*tmp_N_dxyz)*(M2 - 1)*V_dd_delta + tmp_L*tmp_N*(2*M_t_Mdx)*V_dd_delta + tmp_L*tmp_N*(M2 - 1)*V_dd_delta_dxyz
    H_YZ_YZ_dxyz   = 3*(2*M_t_Mdx*N2 + M2*2*N_t_Ndx)*V_dd_sigma + 3*M2*N2*V_dd_sigma_dxyz +\
                     (2*M_t_Mdx + 2*N_t_Ndx - 8*(M_t_Mdx*N2 + M2*N_t_Ndx))*V_dd_pi + (M2 + N2 - 4*M2*N2)*V_dd_pi_dxyz +\
                     (2*L_t_Ldx + 2*M_t_Mdx*N2 + M2*2*N_t_Ndx)*V_dd_delta + (L2 + M2*N2)*V_dd_delta_dxyz
    H_YZ_ZX_dxyz   = 3*(tmp_M_dxyz*N2*tmp_L + tmp_M*2*N_t_Ndx*tmp_L + tmp_M*N2*tmp_L_dxyz)*V_dd_sigma + 3*tmp_M*N2*tmp_L*V_dd_sigma_dxyz +\
                     (tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(1 - 4*N2)*V_dd_pi + L_t_M*(-8*N_t_Ndx)*V_dd_pi + L_t_M*(1 - 4*N2)*V_dd_pi_dxyz +\
                     (tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(N2 - 1)*V_dd_delta + L_t_M*(2*N_t_Ndx)*V_dd_delta + L_t_M*(N2 - 1)*V_dd_delta_dxyz
    H_YZ_X2Y2_dxyz = 1.5*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 - M2) + M_t_N*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*M_t_N*(L2 - M2)*V_dd_sigma_dxyz +\
                    -((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 + 2*(L2 - M2)) + M_t_N*(4*L_t_Ldx - 4*M_t_Mdx))*V_dd_pi - M_t_N*(1 + 2*(L2 - M2))*V_dd_pi_dxyz +\
                     ((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 + 0.5*(L2 - M2)) + M_t_N*(L_t_Ldx - M_t_Mdx))*V_dd_delta + M_t_N*(1 + 0.5*(L2 - M2))*V_dd_delta_dxyz
    H_YZ_Z2_dxyz   = (3**0.5)*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(N2 - 0.5*(L2 + M2)) + M_t_N*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*M_t_N*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                     (3**0.5)*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 + M2 - N2) + M_t_N*(2*L_t_Ldx + 2*M_t_Mdx - 2*N_t_Ndx))*V_dd_pi + (3**0.5)*M_t_N*(L2 + M2 - N2)*V_dd_pi_dxyz +\
                    -(3**0.5)*0.5*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 + M2) + M_t_N*(2*L_t_Ldx + 2*M_t_Mdx))*V_dd_delta - (3**0.5)*0.5*M_t_N*(L2 + M2)*V_dd_delta_dxyz
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+4, H_YZ_XY_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+5, H_YZ_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+6, H_YZ_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+7, H_YZ_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+5)*HDIM + idx_col+8, H_YZ_Z2_dxyz  )
    H_ZX_XY_dxyz  = 3*(2*L_t_Ldx*M_t_N + L2*tmp_M_dxyz*tmp_N + L2*tmp_M*tmp_N_dxyz)*V_dd_sigma + 3*L2*M_t_N*V_dd_sigma_dxyz +\
                    (tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 - 4*L2)*V_dd_pi + M_t_N*(-8*L_t_Ldx)*V_dd_pi + M_t_N*(1 - 4*L2)*V_dd_pi_dxyz +\
                    (tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 - 1)*V_dd_delta + M_t_N*(2*L_t_Ldx)*V_dd_delta + M_t_N*(L2 - 1)*V_dd_delta_dxyz
    H_ZX_YZ_dxyz  = 3*(tmp_M_dxyz*N2*tmp_L + tmp_M*2*N_t_Ndx*tmp_L + tmp_M*N2*tmp_L_dxyz)*V_dd_sigma + 3*tmp_M*N2*tmp_L*V_dd_sigma_dxyz +\
                    (tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(1 - 4*N2)*V_dd_pi + L_t_M*(-8*N_t_Ndx)*V_dd_pi + L_t_M*(1 - 4*N2)*V_dd_pi_dxyz +\
                    (tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(N2 - 1)*V_dd_delta + L_t_M*(2*N_t_Ndx)*V_dd_delta + L_t_M*(N2 - 1)*V_dd_delta_dxyz
    H_ZX_ZX_dxyz  = 3*(2*N_t_Ndx*L2 + N2*2*L_t_Ldx)*V_dd_sigma + 3*N2*L2*V_dd_sigma_dxyz +\
                    (2*N_t_Ndx + 2*L_t_Ldx - 8*(N_t_Ndx*L2 + N2*L_t_Ldx))*V_dd_pi + (N2 + L2 - 4*N2*L2)*V_dd_pi_dxyz +\
                    (2*M_t_Mdx + 2*N_t_Ndx*L2 + N2*2*L_t_Ldx)*V_dd_delta + (M2 + N2*L2)*V_dd_delta_dxyz    
    H_ZX_X2Y2_dxyz = 1.5*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 - M2) + N_t_L*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*N_t_L*(L2 - M2)*V_dd_sigma_dxyz +\
                     ((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(1 - 2*(L2 - M2)) + N_t_L*(-4*(L_t_Ldx - M_t_Mdx)))*V_dd_pi + N_t_L*(1 - 2*(L2 - M2))*V_dd_pi_dxyz +\
                    -((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(1 - 0.5*(L2 - M2)) + N_t_L*(-L_t_Ldx + M_t_Mdx))*V_dd_delta - N_t_L*(1 - 0.5*(L2 - M2))*V_dd_delta_dxyz
    H_ZX_Z2_dxyz = (3**0.5)*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(N2 - 0.5*(L2 + M2)) + N_t_L*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*N_t_L*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                     (3**0.5)*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 + M2 - N2) + N_t_L*(2*L_t_Ldx + 2*M_t_Mdx - 2*N_t_Ndx))*V_dd_pi + (3**0.5)*N_t_L*(L2 + M2 - N2)*V_dd_pi_dxyz +\
                    -(3**0.5)*0.5*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 + M2) + N_t_L*(2*L_t_Ldx + 2*M_t_Mdx))*V_dd_delta - (3**0.5)*0.5*N_t_L*(L2 + M2)*V_dd_delta_dxyz
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+4, H_ZX_XY_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+5, H_ZX_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+6, H_ZX_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+7, H_ZX_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+6)*HDIM + idx_col+8, H_ZX_Z2_dxyz  )
    H_X2Y2_XY_dxyz = 1.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(L2 - M2) + L_t_M*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*L_t_M*(L2 - M2)*V_dd_sigma_dxyz +\
                     2*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(M2 - L2) + L_t_M*(2*M_t_Mdx - 2*L_t_Ldx))*V_dd_pi + 2*L_t_M*(M2 - L2)*V_dd_pi_dxyz +\
                     0.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(L2 - M2) + L_t_M*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_delta + 0.5*L_t_M*(L2 - M2)*V_dd_delta_dxyz
    H_X2Y2_YZ_dxyz = 1.5*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 - M2) + M_t_N*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*M_t_N*(L2 - M2)*V_dd_sigma_dxyz +\
                    -((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 + 2*(L2 - M2)) + M_t_N*(4*L_t_Ldx - 4*M_t_Mdx))*V_dd_pi - M_t_N*(1 + 2*(L2 - M2))*V_dd_pi_dxyz +\
                     ((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(1 + 0.5*(L2 - M2)) + M_t_N*(L_t_Ldx - M_t_Mdx))*V_dd_delta + M_t_N*(1 + 0.5*(L2 - M2))*V_dd_delta_dxyz   
    H_X2Y2_ZX_dxyz = 1.5*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 - M2) + N_t_L*(2*L_t_Ldx - 2*M_t_Mdx))*V_dd_sigma + 1.5*N_t_L*(L2 - M2)*V_dd_sigma_dxyz +\
                     ((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(1 - 2*(L2 - M2)) + N_t_L*(-4*(L_t_Ldx - M_t_Mdx)))*V_dd_pi + N_t_L*(1 - 2*(L2 - M2))*V_dd_pi_dxyz +\
                    -((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(1 - 0.5*(L2 - M2)) + N_t_L*(-L_t_Ldx + M_t_Mdx))*V_dd_delta - N_t_L*(1 - 0.5*(L2 - M2))*V_dd_delta_dxyz
    H_X2Y2_X2Y2_dxyz = 0.75*4*(L2 - M2)*(L_t_Ldx - M_t_Mdx )*V_dd_sigma + 0.75*(L2 - M2)**2*V_dd_sigma_dxyz +\
                       ( 2*L_t_Ldx + 2*M_t_Mdx - 4*(L2 - M2)*( L_t_Ldx - M_t_Mdx ) )*V_dd_pi + (L2 + M2 - (L2 - M2)**2)*V_dd_pi_dxyz +\
                       (2*N_t_Ndx + (L2 - M2)*(L_t_Ldx - M_t_Mdx) )*V_dd_delta + (N2 + 0.25*(L2 - M2)**2)*V_dd_delta_dxyz
    H_X2Y2_Z2_dxyz = (3**0.5)*0.5*( 2*(L_t_Ldx - M_t_Mdx)*(N2 - 0.5*(L2 + M2)) + (L2 - M2)*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)) )*V_dd_sigma + (3**0.5)*0.5*(L2 - M2)*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                      (3**0.5)*(2*N_t_Ndx*(M2 - L2) + N2*2*(M_t_Mdx - L_t_Ldx) )*V_dd_pi + (3**0.5)*N2*(M2 - L2)*V_dd_pi_dxyz +\
                      (3**0.5)*0.25*(2*N_t_Ndx*(L2 - M2) + (1 + N2)*2*(L_t_Ldx - M_t_Mdx) )*V_dd_delta + (3**0.5)*0.25*(1 + N2)*(L2 - M2)*V_dd_delta_dxyz  
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+4, H_X2Y2_XY_dxyz  )
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+5, H_X2Y2_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+6, H_X2Y2_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+7, H_X2Y2_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+7)*HDIM + idx_col+8, H_X2Y2_Z2_dxyz  )
    H_Z2_XY_dxyz = (3**0.5)*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(N2 - 0.5*(L2 + M2)) + L_t_M*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*L_t_M*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz -\
                     (3**0.5)*2*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*N2 + L_t_M*2*N_t_Ndx)*V_dd_pi - (3**0.5)*2*L_t_M*N2*V_dd_pi_dxyz +\
                     (3**0.5)*0.5*((tmp_L_dxyz*tmp_M + tmp_L*tmp_M_dxyz)*(1 + N2) + L_t_M*2*N_t_Ndx)*V_dd_delta + (3**0.5)*0.5*L_t_M*(1 + N2)*V_dd_delta_dxyz
    H_Z2_YZ_dxyz = (3**0.5)*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(N2 - 0.5*(L2 + M2)) + M_t_N*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*M_t_N*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                     (3**0.5)*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 + M2 - N2) + M_t_N*2*(L_t_Ldx + M_t_Mdx - N_t_Ndx))*V_dd_pi + (3**0.5)*M_t_N*(L2 + M2 - N2)*V_dd_pi_dxyz +\
                    -(3**0.5)*0.5*((tmp_M_dxyz*tmp_N + tmp_M*tmp_N_dxyz)*(L2 + M2) + M_t_N*2*(L_t_Ldx + M_t_Mdx))*V_dd_delta - (3**0.5)*0.5*M_t_N*(L2 + M2)*V_dd_delta_dxyz
    H_Z2_ZX_dxyz = (3**0.5)*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(N2 - 0.5*(L2 + M2)) + N_t_L*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)))*V_dd_sigma + (3**0.5)*N_t_L*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                     (3**0.5)*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 + M2 - N2) + N_t_L*2*(L_t_Ldx + M_t_Mdx - N_t_Ndx))*V_dd_pi + (3**0.5)*N_t_L*(L2 + M2 - N2)*V_dd_pi_dxyz +\
                    -(3**0.5)*0.5*((tmp_N_dxyz*tmp_L + tmp_N*tmp_L_dxyz)*(L2 + M2) + N_t_L*2*(L_t_Ldx + M_t_Mdx))*V_dd_delta - (3**0.5)*0.5*N_t_L*(L2 + M2)*V_dd_delta_dxyz     
    H_Z2_X2Y2_dxyz = (3**0.5)*0.5*( 2*(L_t_Ldx - M_t_Mdx)*(N2 - 0.5*(L2 + M2)) + (L2 - M2)*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx)) )*V_dd_sigma + (3**0.5)*0.5*(L2 - M2)*(N2 - 0.5*(L2 + M2))*V_dd_sigma_dxyz +\
                      (3**0.5)*(2*N_t_Ndx*(M2 - L2) + N2*2*(M_t_Mdx - L_t_Ldx) )*V_dd_pi + (3**0.5)*N2*(M2 - L2)*V_dd_pi_dxyz +\
                      (3**0.5)*0.25*(2*N_t_Ndx*(L2 - M2) + (1 + N2)*2*(L_t_Ldx - M_t_Mdx) )*V_dd_delta + (3**0.5)*0.25*(1 + N2)*(L2 - M2)*V_dd_delta_dxyz
    H_Z2_Z2_dxyz = 2*(N2 - 0.5*(L2 + M2))*(2*N_t_Ndx - (L_t_Ldx + M_t_Mdx))*V_dd_sigma + (N2 - 0.5*(L2 + M2))**2*V_dd_sigma_dxyz +\
                    3*(2*N_t_Ndx*(L2 + M2) + N2*2*(L_t_Ldx + M_t_Mdx))*V_dd_pi + 3*N2*(L2 + M2)*V_dd_pi_dxyz +\
                    0.75*2*(L2 + M2)*2*(L_t_Ldx + M_t_Mdx)*V_dd_delta + 0.75*(L2 + M2)**2*V_dd_delta_dxyz
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+4, H_Z2_XY_dxyz  )
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+5, H_Z2_YZ_dxyz  )
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+6, H_Z2_ZX_dxyz  )
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+7, H_Z2_X2Y2_dxyz)
    dH0.index_add_(1, (idx_row+8)*HDIM + idx_col+8, H_Z2_Z2_dxyz  )

    ''' 
    $ - from table
    O - derived via permutation

    S_XY = (3**0.5)*L*M*V_sd_sigma                                                                                                                         # $
    S_YZ = (3**0.5)*M*N*V_sd_sigma                                                                                                                         # O 
    S_ZX = (3**0.5)*N*L*V_sd_sigma                                                                                                                         # O
    S_X2Y2 = 0.5*(3**0.5)*(L**2 - M**2)*V_sd_sigma                                                                                                         # $
    S_Z2 = (N**2-0.5*(L**2 + M**2))*V_sd_sigma                                                                                                             # $

    X_XY = (3**0.5)*L**2*M*V_pd_sigma + M*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_YZ = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # $
    X_ZX = (3**0.5)*L**2*N*V_pd_sigma + N*(1 - 2*L**2)*V_pd_pi                                                                                             # $
    X_X2Y2 = 0.5*(3**0.5)*L*(L**2 - M**2)*V_pd_sigma + L*(1 - L**2 + M**2)*V_pd_pi                                                                         # $
    X_Z2 = L*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*L*N**2*V_pd_pi                                                                                 # $

    Y_XY = (3**0.5)*M**2*L*V_pd_sigma + L*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_YZ = (3**0.5)*M**2*N*V_pd_sigma + N*(1 - 2*M**2)*V_pd_pi                                                                                             # O
    Y_ZX = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Y_X2Y2 = 0.5*(3**0.5)*M*(L**2 - M**2)*V_pd_sigma - M*(1 + L**2 - M**2)*V_pd_pi                                                                         # $
    Y_Z2 = M*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma - 3**0.5*M*N**2*V_pd_pi                                                                                 # $

    Z_XY = (3**0.5)*L*M*N*V_pd_sigma - 2*L*M*N*V_pd_pi                                                                                                     # O
    Z_YZ = (3**0.5)*N**2*M*V_pd_sigma + M*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_ZX = (3**0.5)*N**2*L*V_pd_sigma + L*(1 - 2*N**2)*V_pd_pi                                                                                             # O
    Z_X2Y2 = 0.5*(3**0.5)*N*(L**2 - M**2)*V_pd_sigma - N*(L**2 - M**2)*V_pd_pi                                                                             # $
    Z_Z2 = N*(N**2 - 0.5*(L**2 + M**2))*V_pd_sigma + 3**0.5*N*(L**2 + M**2)*V_pd_pi                                                                        # $

    XY_S =  (3**0.5)*L*M*V_ds_sigma                                                                                                                        # O same as  S_XY
    YZ_S = (3**0.5)*M*N*V_ds_sigma                                                                                                                         # O same as  S_YZ
    ZX_S =  (3**0.5)*N*L*V_ds_sigma                                                                                                                        # O same as  S_ZX
    X2Y2_S =   0.5*(3**0.5)*(L**2 - M**2)*V_ds_sigma                                                                                                       # O same as  S_X2Y2
    Z2_S =  (N**2-0.5*(L**2 + M**2))*V_ds_sigma                                                                                                            # O same as  S_Z2

    XY_X = -((3**0.5)*L**2*M*V_dp_sigma + M*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_XY
    XY_Y = -((3**0.5)*M**2*L*V_dp_sigma + L*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_XY
    XY_Z = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Z_XY

    YZ_X = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -X_YZ
    YZ_Y = -((3**0.5)*M**2*N*V_dp_sigma + N*(1 - 2*M**2)*V_dp_pi)                                                                                          # O same as -Y_YZ
    YZ_Z = -((3**0.5)*N**2*M*V_dp_sigma + M*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_YZ

    ZX_X = -((3**0.5)*L**2*N*V_dp_sigma + N*(1 - 2*L**2)*V_dp_pi)                                                                                          # O same as -X_ZX
    ZX_Y = -((3**0.5)*L*M*N*V_dp_sigma - 2*L*M*N*V_dp_pi)                                                                                                  # O same as -Y_ZX
    ZX_Z = -((3**0.5)*N**2*L*V_dp_sigma + L*(1 - 2*N**2)*V_dp_pi)                                                                                          # O same as -Z_ZX    

    X2Y2_X = -(0.5*(3**0.5)*L*(L**2 - M**2)*V_dp_sigma + L*(1 - L**2 + M**2)*V_dp_pi)                                                                      # O same as -X_X2Y2
    X2Y2_Y = -(0.5*(3**0.5)*M*(L**2 - M**2)*V_dp_sigma - M*(1 + L**2 - M**2)*V_dp_pi)                                                                      # O same as -Y_X2Y2       
    X2Y2_Z = -(0.5*(3**0.5)*N*(L**2 - M**2)*V_dp_sigma - N*(L**2 - M**2)*V_dp_pi)                                                                          # O same as -Z_X2Y2

    Z2_X = -(L*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*L*N**2*V_dp_pi)                                                                              # O same as -X_Z2
    Z2_Y = -(M*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma - 3**0.5*M*N**2*V_dp_pi)                                                                              # O same as -Y_Z2
    Z2_Z = -(N*(N**2 - 0.5*(L**2 + M**2))*V_dp_sigma + 3**0.5*N*(L**2 + M**2)*V_dp_pi)                                                                     # O same as -Z_Z2


    XY_XY = 3*L**2*M**2*V_dd_sigma + (L**2 + M**2 - 4*L**2*M**2)*V_dd_pi + (N**2 + L**2*M**2)*V_dd_delta                                                   # $
    XY_YZ = 3*L*M**2*N*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # $
    XY_ZX = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # $
    XY_X2Y2 = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # $
    XY_Z2 = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # $

    YZ_XY = 3*M**2*N*L*V_dd_sigma + L*N*(1 - 4*M**2)*V_dd_pi + L*N*(M**2 - 1)*V_dd_delta                                                                   # O
    YZ_YZ = 3*M**2*N**2*V_dd_sigma + (M**2 + N**2 - 4*M**2*N**2)*V_dd_pi + (L**2 + M**2*N**2)*V_dd_delta                                                   # O
    YZ_ZX = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O
    YZ_X2Y2 = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # $
    YZ_Z2 = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # $

    ZX_XY = 3*L**2*M*N*V_dd_sigma + M*N*(1 - 4*L**2)*V_dd_pi + M*N*(L**2 - 1)*V_dd_delta                                                                   # O same as XY_ZX
    ZX_YZ = 3*M*N**2*L*V_dd_sigma + L*M*(1 - 4*N**2)*V_dd_pi + L*M*(N**2 - 1)*V_dd_delta                                                                   # O same as YZ_ZX
    ZX_ZX = 3*N**2*L**2*V_dd_sigma + (N**2 + L**2 - 4*N**2*L**2)*V_dd_pi + (M**2 + N**2*L**2)*V_dd_delta                                                   # $O
    ZX_X2Y2 = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # $
    ZX_Z2 = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # $

    X2Y2_XY = 1.5*L*M*(L**2 - M**2)*V_dd_sigma + 2*L*M*(M**2 - L**2)*V_dd_pi + 0.5*L*M*(L**2 - M**2)*V_dd_delta                                            # O same as  XY_X2Y2
    X2Y2_YZ = 1.5*M*N*(L**2 - M**2)*V_dd_sigma - M*N*(1 + 2*(L**2 - M**2))*V_dd_pi + M*N*(1 + 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  YZ_X2Y2
    X2Y2_ZX = 1.5*N*L*(L**2 - M**2)*V_dd_sigma + N*L*(1 - 2*(L**2 - M**2))*V_dd_pi - N*L*(1 - 0.5*(L**2 - M**2))*V_dd_delta                                # O same as  ZX_X2Y2
    X2Y2_X2Y2 = 0.75*(L**2 - M**2)**2*V_dd_sigma + (L**2 + M**2 - (L**2 - M**2)**2)*V_dd_pi + (N**2 + 0.25*(L**2 - M**2)**2)*V_dd_delta                    # $
    X2Y2_Z2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # $

    Z2_XY = (3**0.5)*L*M*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma - (3**0.5)*2*L*M*N**2*V_dd_pi + (3**0.5)*0.5*L*M*(1 + N**2)*V_dd_delta                      # O same as  XY_Z2
    Z2_YZ = (3**0.5)*M*N*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*M*N*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*M*N*(L**2 + M**2)*V_dd_delta     # O same as  YZ_Z2
    Z2_ZX = (3**0.5)*N*L*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N*L*(L**2 + M**2 - N**2)*V_dd_pi - (3**0.5)*0.5*N*L*(L**2 + M**2)*V_dd_delta     # O same as  ZX_Z2
    Z2_X2Y2 = (3**0.5)*0.5*(L**2 - M**2)*(N**2 - 0.5*(L**2 + M**2))*V_dd_sigma + (3**0.5)*N**2*(M**2 - L**2)*V_dd_pi + (3**0.5)*0.25*(1 + N**2)*(L**2 - M**2)*V_dd_delta   # O same as X2Y2_Z2
    Z2_Z2 = (N**2 - 0.5*(L**2 + M**2))**2*V_dd_sigma + 3*N**2*(L**2 + M**2)*V_dd_pi + 0.75*(L**2 + M**2)**2*V_dd_delta                                     # $


    '''
    return H0, dH0

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
    
    #######
    HSSS_all = bond_integral_vectorized(dR, fss_sigma)
    H0[ H_INDEX_START[neighbor_I]*HDIM + H_INDEX_START[neighbor_J] ] = HSSS_all

    #######
    
    # H-H        
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR, fss_sigma)        
    HSSS_dxyz = HSSS_dR * dR_dxyz
    dH0[:, H_INDEX_START[neighbor_I]*HDIM + H_INDEX_START[neighbor_J] ] = HSSS_dxyz
    #########
    


    # H-X
    ###### HSPS_all
    idx_row = H_INDEX_START[neighbor_I[pair_mask_HX + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_HX + pair_mask_XX]]
    HSPS_all = bond_integral_vectorized(dR[pair_mask_HX + pair_mask_XX], fsp_sigma[pair_mask_HX + pair_mask_XX])
    
    H0[ idx_row*HDIM + idx_col +1] = L[pair_mask_HX + pair_mask_XX]*HSPS_all
    H0[ idx_row*HDIM + idx_col +2] = M[pair_mask_HX + pair_mask_XX]*HSPS_all
    H0[ idx_row*HDIM + idx_col +3] = N[pair_mask_HX + pair_mask_XX]*HSPS_all
    ######### dH/dx    
    HSPS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_HX + pair_mask_XX], fsp_sigma[pair_mask_HX + pair_mask_XX])
    HSPS_dxyz = HSPS_dR * dR_dxyz[:, pair_mask_HX + pair_mask_XX]
            
    dH0[:, idx_row*HDIM + idx_col +1] = L[pair_mask_HX + pair_mask_XX]*HSPS_dxyz + L_dxyz[:, pair_mask_HX + pair_mask_XX]*HSPS_all
    dH0[:, idx_row*HDIM + idx_col +2] = M[pair_mask_HX + pair_mask_XX]*HSPS_dxyz + M_dxyz[:, pair_mask_HX + pair_mask_XX]*HSPS_all
    dH0[:, idx_row*HDIM + idx_col +3] = N[pair_mask_HX + pair_mask_XX]*HSPS_dxyz + N_dxyz[:, pair_mask_HX + pair_mask_XX]*HSPS_all
    #########

    ### HPSS_all ###
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XH + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XH + pair_mask_XX]]
    HPSS_all = bond_integral_vectorized(dR[pair_mask_XH + pair_mask_XX], fps_sigma[pair_mask_XH + pair_mask_XX])
    H0[ (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH + pair_mask_XX]*HPSS_all
    H0[ (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH + pair_mask_XX]*HPSS_all
    H0[ (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH + pair_mask_XX]*HPSS_all
    ################
    ######### dH/dx    
    HPSS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_XH + pair_mask_XX], fps_sigma[pair_mask_XH + pair_mask_XX])
    HPSS_dxyz = HPSS_dR * dR_dxyz[:, pair_mask_XH + pair_mask_XX]
        
    dH0[:, (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH + pair_mask_XX]*HPSS_dxyz - L_dxyz[:,pair_mask_XH + pair_mask_XX]*HPSS_all
    dH0[:, (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH + pair_mask_XX]*HPSS_dxyz - M_dxyz[:,pair_mask_XH + pair_mask_XX]*HPSS_all
    dH0[:, (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH + pair_mask_XX]*HPSS_dxyz - N_dxyz[:,pair_mask_XH + pair_mask_XX]*HPSS_all
    #########
    
    # X-X
    L_XX = L[pair_mask_XX]
    M_XX = M[pair_mask_XX]
    N_XX = N[pair_mask_XX]
    dR_XX = dR[pair_mask_XX]
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XX]]
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

    H0[ (idx_row+1)*HDIM + idx_col +1] = PXPX
    H0[ (idx_row+1)*HDIM + idx_col +2] = PXPY
    H0[ (idx_row+1)*HDIM + idx_col +3] = PXPZ
    
    ####

    H0[ (idx_row+2)*HDIM + idx_col +1] = PYPX
    H0[ (idx_row+2)*HDIM + idx_col +2] = PYPY
    H0[ (idx_row+2)*HDIM + idx_col +3] = PYPZ
    
    ####

    H0[ (idx_row+3)*HDIM + idx_col +1] = PZPX
    H0[ (idx_row+3)*HDIM + idx_col +2] = PZPY
    H0[ (idx_row+3)*HDIM + idx_col +3] = PZPZ
    
    ######### dH/dx
    dR_dxyz_XX = dR_dxyz[:, pair_mask_XX]
    L_dxyz_XX = L_dxyz[:,pair_mask_XX]
    M_dxyz_XX = M_dxyz[:,pair_mask_XX]
    N_dxyz_XX = N_dxyz[:,pair_mask_XX]
    
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

    dH0[:, (idx_row+1)*HDIM + idx_col +1] = PXPX_dxyz
    dH0[:, (idx_row+1)*HDIM + idx_col +2] = PXPY_dxyz
    dH0[:, (idx_row+1)*HDIM + idx_col +3] = PXPZ_dxyz
        
    ####    
    
    dH0[:, (idx_row+2)*HDIM + idx_col +1] = PYPX_dxyz
    dH0[:, (idx_row+2)*HDIM + idx_col +2] = PYPY_dxyz
    dH0[:, (idx_row+2)*HDIM + idx_col +3] = PYPZ_dxyz

    ####    
    
    dH0[:, (idx_row+3)*HDIM + idx_col +1] = PZPX_dxyz
    dH0[:, (idx_row+3)*HDIM + idx_col +2] = PZPY_dxyz
    dH0[:, (idx_row+3)*HDIM + idx_col +3] = PZPZ_dxyz
    #########
    return H0, dH0

def Slater_Koster_Pair_vectorized_OLD(H0, HDIM, dR, dR_dxyz, L, M, N, L_dxyz, M_dxyz, N_dxyz,
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
    
    #######
    HSSS_all = bond_integral_vectorized(dR, fss_sigma)
    H0[ H_INDEX_START[neighbor_I]*HDIM + H_INDEX_START[neighbor_J] ] = HSSS_all

    #######
    
    # H-H    
    #HSSS = bond_integral_vectorized(dR[pair_mask_HH], fss_sigma[pair_mask_HH])
    #H0[ H_INDEX_START[neighbor_I[pair_mask_HH]]*HDIM + H_INDEX_START[neighbor_J[pair_mask_HH]] ] = HSSS
    
    ######### dH/dx
    HSSS_dR = bond_integral_with_grad_vectorized(dR[pair_mask_HH], fss_sigma[pair_mask_HH])        
    HSSS_dxyz = HSSS_dR * dR_dxyz[:, pair_mask_HH]
        
    dH0[:, H_INDEX_START[neighbor_I[pair_mask_HH]]*HDIM + H_INDEX_START[neighbor_J[pair_mask_HH]] ] = HSSS_dxyz
    #########
    
    ###### HSPS_all
    idx_row = H_INDEX_START[neighbor_I[pair_mask_HX + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_HX + pair_mask_XX]]
    HSPS_all = bond_integral_vectorized(dR[pair_mask_HX + pair_mask_XX], fsp_sigma[pair_mask_HX + pair_mask_XX])
    
    H0[ idx_row*HDIM + idx_col +1] = L[pair_mask_HX + pair_mask_XX]*HSPS_all
    H0[ idx_row*HDIM + idx_col +2] = M[pair_mask_HX + pair_mask_XX]*HSPS_all
    H0[ idx_row*HDIM + idx_col +3] = N[pair_mask_HX + pair_mask_XX]*HSPS_all
    ########

    # H-X
    idx_row = H_INDEX_START[neighbor_I[pair_mask_HX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_HX]]
    #HSSS = bond_integral_vectorized(dR[pair_mask_HX], fss_sigma[pair_mask_HX])
    HSPS = bond_integral_vectorized(dR[pair_mask_HX], fsp_sigma[pair_mask_HX])
    
    #H0[ idx_row*HDIM + idx_col ]   = HSSS
    # H0[ idx_row*HDIM + idx_col +1] = L[pair_mask_HX]*HSPS
    # H0[ idx_row*HDIM + idx_col +2] = M[pair_mask_HX]*HSPS
    # H0[ idx_row*HDIM + idx_col +3] = N[pair_mask_HX]*HSPS
    
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

    
    ### HPSS_all ###
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XH + pair_mask_XX]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XH + pair_mask_XX]]
    HPSS_all = bond_integral_vectorized(dR[pair_mask_XH + pair_mask_XX], fps_sigma[pair_mask_XH + pair_mask_XX])
    H0[ (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH + pair_mask_XX]*HPSS_all
    H0[ (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH + pair_mask_XX]*HPSS_all
    H0[ (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH + pair_mask_XX]*HPSS_all
    ################

    # X-H
    ### for X-H and H-X integrals, atomic order does not matter. 
    idx_row = H_INDEX_START[neighbor_I[pair_mask_XH]]
    idx_col = H_INDEX_START[neighbor_J[pair_mask_XH]]
    #HSSS = bond_integral_vectorized(dR[pair_mask_XH], fss_sigma[pair_mask_XH])
    HSPS = bond_integral_vectorized(dR[pair_mask_XH], fsp_sigma[pair_mask_XH])
    HPSS = bond_integral_vectorized(dR[pair_mask_XH], fps_sigma[pair_mask_XH])

    ########
    # H0[ (idx_row +1)*HDIM + idx_col] = -L[pair_mask_XH]*HPSS
    # H0[ (idx_row +2)*HDIM + idx_col] = -M[pair_mask_XH]*HPSS
    # H0[ (idx_row +3)*HDIM + idx_col] = -N[pair_mask_XH]*HPSS
    ########
    
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
    #HSSS = bond_integral_vectorized(dR_XX, fss_sigma[pair_mask_XX])
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
    #H0[ idx_row*HDIM + idx_col ]       = HSSS
        
    # H0[ idx_row*HDIM + idx_col +1]     = L_XX*HSPS
    # H0[ idx_row*HDIM + idx_col +2]     = M_XX*HSPS
    # H0[ idx_row*HDIM + idx_col +3]     = N_XX*HSPS
    
    ####
    # H0[ (idx_row+1)*HDIM + idx_col ]   = -L_XX*HPSS
    # H0[ (idx_row+2)*HDIM + idx_col ]   = -M_XX*HPSS
    # H0[ (idx_row+3)*HDIM + idx_col ]   = -N_XX*HPSS

    ####

    H0[ (idx_row+1)*HDIM + idx_col +1] = PXPX
    H0[ (idx_row+1)*HDIM + idx_col +2] = PXPY
    H0[ (idx_row+1)*HDIM + idx_col +3] = PXPZ
    
    ####

    H0[ (idx_row+2)*HDIM + idx_col +1] = PYPX
    H0[ (idx_row+2)*HDIM + idx_col +2] = PYPY
    H0[ (idx_row+2)*HDIM + idx_col +3] = PYPZ
    
    ####

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