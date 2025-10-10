import torch
from .nearestneighborlist import vectorized_nearestneighborlist
def get_repulsion_energy(R_rep_tensor, rep_splines_tensor,
                         TYPE, RX, RY, RZ, LBox, Rcut, Nats, const,
                         verbose
                         ):

    nrnnlist_no_self, nndist_no_self, nnRx, nnRy, nnRz,\
    nnType, nnStruct_no_self, nrnnStruct_no_self, neighbor_I, neighbor_J,\
    IJ_pair_type, JI_pair_type  = \
        vectorized_nearestneighborlist(TYPE, RX, RY, RZ, LBox, Rcut, Nats, const,
                                       upper_tri_only=True, remove_self_neigh=False, min_image_only=True,
                                       verbose=verbose);

    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)

    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)

    nn_mask = nnType!=-1 # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    indices = torch.zeros((len(dR_mskd)), dtype=torch.int64, device=RX.device)
    for i in range(len(dR_mskd)):
        idx = torch.searchsorted(R_rep_tensor[IJ_pair_type[i]], dR_mskd[i], right=True) - 1
        indices[i] = idx
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

    return Vr, dVr
