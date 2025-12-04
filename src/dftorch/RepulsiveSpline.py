import torch
from .nearestneighborlist import vectorized_nearestneighborlist, vectorized_nearestneighborlist_batch

def get_repulsion_energy(
    R_rep_tensor: torch.Tensor,
    rep_splines_tensor: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    LBox,
    Rcut: float,
    Nats: int,
    const,
    verbose: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the total repulsive energy and its Cartesian derivatives.

    Parameters
    ----------
    R_rep_tensor : (P, M) torch.Tensor
        Radial grid (Å) per pair type (P = number of pair types).
    rep_splines_tensor : (P, M, >=6) torch.Tensor
        Spline coefficients c0..c5 (Hartree) for 5th‑order local polynomial segments.
    TYPE : (Nats,) torch.Tensor
        Atom type indices (compatible with const.label).
    RX, RY, RZ : (Nats,) torch.Tensor
        Atomic Cartesian coordinates in Å.
    LBox : sequence or tensor length 3
        Box lengths (Å) for periodic boundaries.
    Rcut : float
        Cutoff (Å) used to build the neighbor list.
    Nats : int
        Number of atoms.
    const : object
        Contains chemical label mapping for pair typing.
    verbose : bool
        If True, neighbor list routine may emit timing info.

    Returns
    -------
    Vr : torch.Tensor (scalar, eV)
        Total repulsive energy.
    dVr : torch.Tensor, shape (3, Nats, Nats), eV/Å
        Antisymmetric matrix of pairwise force contributions:
        dVr[:, i, j] = +∂E/∂r_i from pair (i,j); dVr[:, j, i] = −dVr[:, i, j].

    Notes
    -----
    - Distances converted to Bohr internally (factor 0.52917721).
    - Energy converted Hartree → eV using 27.21138625.
    - Neighbor list uses minimum image (min_image_only=True).
    """
    _, _, nnRx, nnRy, nnRz,\
    nnType, _, _, neighbor_I, neighbor_J,\
    IJ_pair_type, _  = \
        vectorized_nearestneighborlist(TYPE, RX, RY, RZ, LBox, Rcut, Nats, const,
                                       upper_tri_only=True, remove_self_neigh=False, min_image_only=True,
                                       verbose=verbose);
    
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)
    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    # indices = torch.zeros((len(dR_mskd)), dtype=torch.int64, device=RX.device)
    # for i in range(len(dR_mskd)):
    #     idx = torch.searchsorted(R_rep_tensor[IJ_pair_type[i]], dR_mskd[i], right=True) - 1
    #     indices[i] = idx
    indices = torch.searchsorted(
        R_rep_tensor[IJ_pair_type], dR_mskd.unsqueeze(-1), right=True
    ).squeeze(-1) - 1
    # Optionally clamp to keep indices in bounds
    # K = R_rep_tensor.size(1)
    # indices = indices.clamp(min=0, max=K-1)
    
    
    dx = (dR_mskd - R_rep_tensor[IJ_pair_type, indices])/0.52917721
    Vr = rep_splines_tensor[IJ_pair_type,indices,0] + rep_splines_tensor[IJ_pair_type,indices,1]*dx + rep_splines_tensor[IJ_pair_type,indices,2]*dx**2 +\
            rep_splines_tensor[IJ_pair_type,indices,3]*dx**3 + rep_splines_tensor[IJ_pair_type,indices,4]*dx**4 + rep_splines_tensor[IJ_pair_type,indices,5]*dx**5
    
    Vr = Vr.sum()*27.21138625 # eV

    # gradients
    dR_dxyz = torch.stack((Rab_X,Rab_Y,Rab_Z), dim=0)[:,nn_mask]/dR_mskd
    dVr = torch.zeros((3,Nats*Nats), device=RX.device)
    ind_start = torch.arange(Nats, device=RX.device)
    # now, it's Ha/Bohr
    dVr_dR = rep_splines_tensor[IJ_pair_type, indices, 1] + 2*rep_splines_tensor[IJ_pair_type, indices, 2]*dx +\
            3*rep_splines_tensor[IJ_pair_type, indices, 3]*dx**2 + 4*rep_splines_tensor[IJ_pair_type, indices, 4]*dx**3 + 5*rep_splines_tensor[IJ_pair_type, indices, 5]*dx**4
    dVr.index_add_(1, ind_start[neighbor_I]*Nats + ind_start[neighbor_J], dVr_dR*dR_dxyz)
    # now, it's eV/A
    dVr = dVr.reshape(3,Nats,Nats)*27.21138625 /0.52917721
    dVr = dVr - torch.transpose(dVr, 1, 2)
    del nnRx, nnRy, nnRz,nnType, neighbor_I, neighbor_J, IJ_pair_type, _

    return Vr, dVr

def get_repulsion_energy_batch(
    R_rep_tensor: torch.Tensor,
    rep_splines_tensor: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    LBox,
    Rcut: float,
    Nats: int,
    const,
    verbose: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the total repulsive energy and its Cartesian derivatives.
    Notes
    -----
    - Distances converted to Bohr internally (factor 0.52917721).
    - Energy converted Hartree → eV using 27.21138625.
    - Neighbor list uses minimum image (min_image_only=True).
    """
    batch_size = RX.shape[0]
    _, _, nnRx, nnRy, nnRz,\
    nnType, _, _, neighbor_I, neighbor_J,\
    IJ_pair_type, _  = \
        vectorized_nearestneighborlist_batch(TYPE, RX, RY, RZ, LBox, Rcut, Nats, const,
                                       upper_tri_only=True, remove_self_neigh=False, min_image_only=True,
                                       verbose=verbose);
    
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)

    valid_pairs = (neighbor_I >= 0) & (neighbor_J >= 0) # $$$ maybe '& (neighbor_J >= 0)' is not necessary???
    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    safe_IJ = IJ_pair_type.clamp(min=0)

    R_rep_valid = R_rep_tensor[safe_IJ][valid_pairs]

    indices = torch.searchsorted(
        R_rep_valid, dR_mskd.unsqueeze(-1), right=True).squeeze(-1) - 1
    # Optionally clamp to keep indices in bounds
    # K = R_rep_tensor.size(1)
    # indices = indices.clamp(min=0, max=K-1)
    
    R_selected = torch.gather(R_rep_valid, 1, indices.unsqueeze(1)).squeeze(1)  # (P,)
    dx = (dR_mskd - R_selected)/0.52917721

    sel_IJ = IJ_pair_type[valid_pairs]

    Vr = torch.zeros(batch_size, device=RX.device)
    batch_ids = torch.arange(batch_size, device=RX.device).unsqueeze(1).expand_as(safe_IJ)[valid_pairs]

    coeffs = rep_splines_tensor[sel_IJ, indices]  # (P, 6)
    # pair_interactions = sum_{n=0..5} a_n * dx^n
    # Horner: ((((a5*dx + a4)*dx + a3)*dx + a2)*dx + a1)*dx + a0
    poly = coeffs[:, 5]
    poly = poly * dx + coeffs[:, 4]
    poly = poly * dx + coeffs[:, 3]
    poly = poly * dx + coeffs[:, 2]
    poly = poly * dx + coeffs[:, 1]
    pair_interactions = poly * dx + coeffs[:, 0]
    Vr.index_add_(0, batch_ids, pair_interactions*27.21138625)

    # gradients
    dR_dxyz = torch.stack((Rab_X,Rab_Y,Rab_Z), dim=0)[:,nn_mask]/dR_mskd
    dVr = torch.zeros((3,batch_size*Nats*Nats), device=RX.device)
    ind_start = torch.arange(Nats, device=RX.device)
    # now, it's Ha/Bohr    
    a = coeffs
    dpoly = 5.0 * a[:, 5]
    dpoly = dpoly * dx + 4.0 * a[:, 4]
    dpoly = dpoly * dx + 3.0 * a[:, 3]
    dpoly = dpoly * dx + 2.0 * a[:, 2]
    dVr_dR = dpoly * dx + a[:, 1]           # Ha/Bohr
    dVr.index_add_(1,
                   ind_start[neighbor_I[valid_pairs]]*Nats + ind_start[neighbor_J[valid_pairs]] + batch_ids*Nats*Nats,
                   dVr_dR*dR_dxyz)
    # now, it's eV/A
    dVr = dVr.view(3, batch_size, Nats, Nats)*27.21138625 /0.52917721
    dVr = dVr.permute(1, 0, 2, 3).contiguous() 
    dVr = dVr - torch.transpose(dVr, 2, 3)
    del nnRx, nnRy, nnRz,nnType, neighbor_I, neighbor_J, IJ_pair_type, _
    return Vr, dVr