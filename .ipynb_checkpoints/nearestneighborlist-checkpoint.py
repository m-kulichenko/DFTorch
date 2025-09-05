import torch
def vectorized_nearestneighborlist(Rx, Ry, Rz, LBox, Rcut, N, upper_tri_only=True, remove_self_neigh=False):
    """
    Computes the neighbor list for a set of atoms using periodic boundary conditions (PBC) 
    and a cutoff radius, in a fully vectorized manner.

    Parameters
    ----------
    Rx, Ry, Rz : torch.Tensor of shape (N,)
        Cartesian coordinates of atoms in the x, y, and z directions.
    LBox : tuple or list of float
        Box dimensions (Lx, Ly, Lz) for periodic boundary conditions.
    Rcut : float
        Cutoff radius for determining neighbors.
    N : int
        Number of atoms.
    upper_tri_only : bool, optional (default=True)
        If True, only returns neighbors J > I to reduce redundancy.
    remove_self_neigh : bool, optional (default=False)
        If True, excludes self-neighbors (i.e., atom I does not neighbor itself).

    Returns
    -------
    nrnnlist : torch.Tensor of shape (N, 1)
        Number of neighbors within Rcut for each atom (including periodic images).
    nndist : torch.Tensor of shape (N, max_neighbors)
        Pairwise distances between atom I and its J-th neighbor.
    nnRx, nnRy, nnRz : torch.Tensor of shape (N, max_neighbors)
        Coordinates of each neighbor J of atom I (including image translations).
        Default values are padded with large values (e.g., 10000) to indicate unused slots.
    nnType : torch.Tensor of shape (N, max_neighbors)
        Index of the original atom corresponding to each neighbor, before applying periodic image shifts.
    nnStruct : torch.Tensor of shape (N, max_neighbors)
        Neighbor indices corresponding to atoms within the original simulation box (i.e., no periodic shift).
    nrnnStruct : torch.Tensor of shape (N, 1)
        Number of neighbors for each atom within the original simulation box only (no images).

    Notes
    -----
    - Periodic boundary conditions are applied by replicating the simulation box in a 3x3x3 supercell.
    - This function is vectorized and GPU-accelerated using PyTorch for high performance.
    - Neighbor lists are zero-padded to match the maximum neighbor count across all atoms.
    - When `upper_tri_only=True`, only unique pairs (i < j) are returned to avoid double-counting.
    - When `remove_self_neigh=True`, atoms do not include themselves as neighbors.
    """
    # % Rx, Ry, Rz are the coordinates of atoms
    # % LBox dimensions of peridic BC
    # % N number of atoms
    # % nrnnlist(I): number of atoms within distance of Rcut from atom I including atoms in the skin
    # % nndist(I,J): distance between atom I(in box) and J (including atoms in the skin)
    # % nnRx(I,J): x-coordinte of neighbor J to I within RCut (including atoms in the skin)
    # % nnRy(I,J): y-coordinte of neighbor J to I within RCut (including atoms in the skin)
    # % nnRz(I,J): z-coordinte of neighbor J to I within RCut (including atoms in the skin)
    # % nnType(I,J): The neighbor J of I corresponds to some translated atom number in the box that we need to keep track of
    # % nnStruct(I,J): The neigbors J to I within Rcut that are all within the box (not in the skin).
    # % nrnnStruct(I): Number of neigbors to I within Rcut that are all within the box (not in the skin).

    Lx, Ly, Lz = LBox
    R = torch.stack((Rx, Ry, Rz), dim=1)  # (N, 3)

    shifts = torch.tensor([
        [i, j, k] for i in [-1, 0, 1]
                  for j in [-1, 0, 1]
                  for k in [-1, 0, 1]
    ], dtype=Rx.dtype, device=R.device)  # (27, 3)
    
    box = torch.tensor([Lx, Ly, Lz], dtype=Rx.dtype, device=R.device)

    R_translated = R.unsqueeze(1) + shifts.unsqueeze(0) * box.view(1, 1, 3)  # (N, 27, 3)
    
    # diff = R.view(N, 1, 1, 3) - R_translated.view(1, N, 27, 3)  # (N, N, 27, 3)
    # dist = torch.norm(diff, dim=-1)  # (N, N, 27)
    
    A = R.view(N, 3)                          # shape (N, 3)
    B = R_translated.view(N * 27, 3)         # shape (N*27, 3)

    A_norm_sq = A.pow(2).sum(1, keepdim=True)         # (N, 1)
    B_norm_sq = B.pow(2).sum(1, keepdim=True).T       # (1, N*27)
    cross_term = 2*A @ B.T                              # (N, N*27)


    # Compute squared distances
    dists_sq = A_norm_sq + B_norm_sq - cross_term  # (N, N*27)
    dists = dists_sq.clamp(min=1e-12).sqrt()           # avoid sqrt(neg)

    # Reshape back to (N, N, 27)
    dist = dists.view(N, N, 27)


    neighbor_mask = dist < Rcut
    
    # remove self-neighbors
    if remove_self_neigh:
        idx = torch.arange(N, device=neighbor_mask.device)
        neighbor_mask[idx, idx, :] = False


    i_idx, j_idx, s_idx = neighbor_mask.nonzero(as_tuple=True)
    if upper_tri_only:
        valid_mask = j_idx > i_idx
        i_idx = i_idx[valid_mask]
        j_idx = j_idx[valid_mask]
        s_idx = s_idx[valid_mask]

    num_neighbors = torch.bincount(i_idx, minlength=N)
    max_neighbors = num_neighbors.max().item()

    nndist = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device)
    nnRx = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device) + 10000.0 # so zero padded neighs (-1) are far
    nnRy = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device)
    nnRz = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device)
    nnType = torch.full((N, max_neighbors), -1, dtype=torch.int64, device=R.device)
    nnStruct = torch.full((N, max_neighbors), 0, dtype=torch.int64, device=R.device)
    nrnnStruct = torch.zeros((N,), dtype=torch.int64, device=R.device)
    nrnnlist = num_neighbors

    dist_vals = dist[i_idx, j_idx, s_idx]
    neighbor_pos = R_translated[j_idx, s_idx]  # (nnz, 3)

    # Fill neighbor data
    sort_idx = torch.argsort(i_idx)
    i_idx_sorted = i_idx[sort_idx]
    j_idx_sorted = j_idx[sort_idx]
    s_idx_sorted = s_idx[sort_idx]
    dist_vals_sorted = dist_vals[sort_idx]
    #neighbor_pos_sorted = neighbor_pos[sort_idx]
    
    idx_counts = torch.bincount(i_idx_sorted, minlength=N)
    offsets = torch.cat([torch.tensor([0], device=R.device), idx_counts.cumsum(0)[:-1]])
    idx_map = torch.arange(len(i_idx_sorted), device=R.device)
    local_idx = idx_map - offsets[i_idx_sorted]

    nndist[i_idx_sorted, local_idx] = dist_vals
    nnRx[i_idx_sorted, local_idx] = neighbor_pos[:, 0]
    nnRy[i_idx_sorted, local_idx] = neighbor_pos[:, 1]
    nnRz[i_idx_sorted, local_idx] = neighbor_pos[:, 2]
    #nnType[i_idx_sorted, local_idx] = j_idx_sorted
    nnType[i_idx_sorted, local_idx] = j_idx

    is_in_box = (shifts[s_idx_sorted] == 0).all(dim=1)

    #nnStruct[i_idx_sorted[is_in_box], local_idx[is_in_box]] = j_idx_sorted[is_in_box]
    #nnStruct[i_idx_sorted,local_idx] = j_idx_sorted
    nnStruct[i_idx_sorted,local_idx] = j_idx
    #nrnnStruct = torch.bincount(i_idx_sorted[is_in_box], minlength=N)
    nrnnStruct = torch.bincount(i_idx_sorted, minlength=N)

    return nrnnlist.view(-1, 1), nndist, nnRx, nnRy, nnRz, nnType, nnStruct, nrnnStruct.view(-1, 1)
