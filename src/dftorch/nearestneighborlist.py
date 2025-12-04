import torch
import time
from typing import Any, Union, Tuple, Optional, List

from .Tools import ordered_pairs_from_TYPE

@torch.compile(dynamic=False)
def vectorized_nearestneighborlist(
    TYPE: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    LBox: Union[torch.Tensor, Tuple[float, float, float]],
    Rcut: float,
    N: int,
    const,
    upper_tri_only: bool = True,
    remove_self_neigh: bool = False,
    min_image_only: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a periodic (3x3x3 images) neighbor list in a fully vectorized manner.

    Parameters
    ----------
    TYPE : torch.Tensor, shape (N,)
        Integer type index per atom; must align with const.label for pair typing.
    Rx, Ry, Rz : torch.Tensor, each (N,)
        Cartesian coordinates of atoms along x, y, z (same device/dtype).
    LBox : torch.Tensor or tuple(float, float, float)
        Periodic box lengths (Lx, Ly, Lz). If tensor, should be on the same device.
    Rcut : float
        Cutoff radius. Pairs with min-image distance < Rcut are kept.
    N : int
        Number of atoms (must match the length of Rx, Ry, Rz, TYPE).
    const : Any
        Container providing chemical labels (const.label) used for pair typing.
    upper_tri_only : bool, optional (default=True)
        Keep only unique pairs with j > i to avoid double counting.
    remove_self_neigh : bool, optional (default=False)
        Exclude self-pairs (i == j).
    min_image_only : bool, optional (default=False)
        Keep only the closest periodic image for each (i, j).
    verbose : bool, optional (default=False)
        Print timing information.

    Returns
    -------
    nrnnlist : torch.Tensor, shape (N, 1)
        Count of neighbors per atom after filtering.
    nndist : torch.Tensor, shape (N, Kmax)
        Distances to neighbors; padded with a large sentinel for unused slots.
    nnRx, nnRy, nnRz : torch.Tensor, each (N, Kmax)
        Coordinates of neighbor positions (including periodic shifts).
    nnType : torch.Tensor, shape (N, Kmax)
        Index j of the original atom for each neighbor entry.
    nnStruct : torch.Tensor, shape (N, Kmax)
        Index of the neighbor within the original simulation box (no shift).
    nrnnStruct : torch.Tensor, shape (N, 1)
        Number of neighbors per atom within the original box.
    neighbor_I, neighbor_J : torch.Tensor, shape (K_total,)
        Flattened lists of i and j indices for all kept neighbor pairs.
    IJ_pair_type, JI_pair_type : torch.Tensor, shape (K_total,)
        Pair-type indices for ordered pairs (i, j) and (j, i), aligned with const.label.

    Notes
    -----
    - Fully vectorized; no Python loops over atoms.
    - Uses large sentinels (e.g., 10000, 17320.5) for padded entries.
    - All outputs are on the same device/dtype as the inputs.
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

    start_time1 = time.perf_counter()
    Lx, Ly, Lz = LBox
    R = torch.stack((Rx, Ry, Rz), dim=1)  # (N, 3)

    #shift = [-2, -1, 0, 1, 2]
    shift = [-1, 0, 1]
    #shift = [0]
    shifts = torch.tensor([
        [i, j, k] for i in shift
                  for j in shift
                  for k in shift
    ], dtype=Rx.dtype, device=R.device)  # (27, 3)
    
    box = torch.tensor([Lx, Ly, Lz], dtype=Rx.dtype, device=R.device)

    R_translated = R.unsqueeze(1) + shifts.unsqueeze(0) * box.view(1, 1, 3)  # (N, 27, 3)
    
    diff = R.view(N, 1, 1, 3) - R_translated.view(1, N, len(shift)**3, 3)  # (N, N, 27, 3)
    dist = torch.norm(diff, dim=-1)  # (N, N, 27)
    del diff

    # mask minimum distance images
    if min_image_only:
        idx = dist.argmin(dim=2, keepdim=True)          # (N, N, 1)
        mask_min_image = torch.zeros_like(dist, dtype=torch.bool) # (N, N, 27)
        mask_min_image.scatter_(2, idx, True)                  # one-hot at argmin
    else:
        mask_min_image = torch.tensor([True], dtype=torch.bool, device=R.device)

    neighbor_mask = (dist < Rcut) * (dist > 1e-4) * mask_min_image
    
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

    nndist = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device) + 17320.5
    # so zero padded neighs (-1) are far
    nnRx = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device) + 10000.0
    nnRy = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device) + 10000.0
    nnRz = torch.zeros((N, max_neighbors), dtype=Rx.dtype, device=R.device) + 10000.0
    nnType = torch.full((N, max_neighbors), -1, dtype=torch.int64, device=R.device)
    nnStruct = torch.full((N, max_neighbors), -1, dtype=torch.int64, device=R.device)
    nrnnStruct = torch.zeros((N,), dtype=torch.int64, device=R.device)
    nrnnlist = num_neighbors.view(-1, 1)

    dist_vals = dist[i_idx, j_idx, s_idx]
    del dist
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

    nnStruct[i_idx_sorted,local_idx] = j_idx
    nrnnStruct = torch.bincount(i_idx_sorted, minlength=N)

    # === Vectorized neighbor type pair generation ===
    max_neighbors = nnType.shape[-1]

    # Create mask for valid neighbors
    neighbor_mask = torch.arange(max_neighbors, device=Rx.device).unsqueeze(0) < nrnnlist
    neighbor_J = nnType[neighbor_mask]
    neighbor_I = torch.repeat_interleave(torch.arange(nrnnlist.squeeze(-1).shape[0], device=nrnnlist.device), nrnnlist.squeeze(-1))
    del neighbor_mask, dist_vals

    ### Get tensors for SKF files ###
    _, _, label_list = ordered_pairs_from_TYPE(TYPE)

    pair_type_dict = {}

    for i in range(len(label_list)):
        pair_type_dict[label_list[i]] = i

    # Build a 2D lookup table once (no function), then index it
    labels = [s.strip() for s in const.label.tolist()]            # fix spaces like ' P', 'V ', etc.
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    Z = len(labels)
    pair_lookup = torch.full((Z, Z), -1, dtype=torch.long, device=Rx.device)
    for k, v in pair_type_dict.items():                           # keys like "C-H"
        a, b = k.split('-')
        ai = label_to_idx[a]
        bi = label_to_idx[b]
        pair_lookup[ai, bi] = int(v)
        # If the mapping is symmetric and reverse might be missing, also do:
        # pair_lookup[bi, ai] = int(v)
    ti = TYPE[neighbor_I].long()
    tj = TYPE[neighbor_J].long()
    IJ_pair_type = pair_lookup[ti, tj]        # shape: (len(neighbor_I),)
    JI_pair_type = pair_lookup[tj, ti]
    if verbose: print("  t <neighbor list> {:.1f} s\n".format( time.perf_counter()-start_time1 ))

    return nrnnlist, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, nrnnStruct.view(-1, 1),\
           neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type

@torch.compile(dynamic=False)
def vectorized_nearestneighborlist_batch(TYPE, Rx, Ry, Rz, LBox, Rcut, N, const,
                                         upper_tri_only=True, remove_self_neigh=False,
                                         min_image_only=False, verbose=False):
    """
    Batched version of vectorized_nearestneighborlist.
    TYPE, Rx, Ry, Rz have shape (B, N). LBox can be (3,) or (B,3).
    Returns (nrnnlist, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, nrnnStruct,
             neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type)
    neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type have shape (B, Kmax), padded with -1.
    """
    start_time1 = time.perf_counter()
    B = TYPE.shape[0]
    device = Rx.device
    dtype = Rx.dtype

    R = torch.stack((Rx, Ry, Rz), dim=-1)               # (B, N, 3)

    shift = [-1, 0, 1]
    shifts = torch.tensor([[i, j, k] for i in shift for j in shift for k in shift],
                          dtype=dtype, device=device)   # (27,3)
    S = shifts.shape[0]

    # Normalize LBox to shape (B,3); LBox always tensor
    lb = LBox.to(device=device, dtype=dtype)
    if lb.dim() == 1:
        assert lb.numel() == 3, "LBox 1D must have length 3"
        box = lb.view(1, 3).expand(B, 3)
    elif lb.dim() == 2:
        if lb.shape == (1, 3):
            box = lb.expand(B, 3)
        else:
            assert lb.shape == (B, 3), f"Expected LBox shape (B,3), got {tuple(lb.shape)}"
            box = lb
    else:
        raise ValueError("LBox tensor must be 1D (3,) or 2D (B,3)")

    # Use per-batch box
    R_translated = R.unsqueeze(2) + shifts.view(1, 1, S, 3) * box.view(B, 1, 1, 3)   # (B,N,S,3)

    # Pairwise differences: (B,N,N,S,3)
    diff = R.unsqueeze(2).unsqueeze(3) - R_translated.unsqueeze(1)                   # (B,N,N,S,3)
    dist = torch.norm(diff, dim=-1)                                                  # (B,N,N,S)
    del diff

    if min_image_only:
        # Keep only minimum image per (B,i,j)
        idx_min = dist.argmin(dim=3, keepdim=True)                                   # (B,N,N,1)
        mask_min = torch.zeros_like(dist, dtype=torch.bool)
        mask_min.scatter_(3, idx_min, True)
    else:
        mask_min = torch.tensor(True, device=device)

    neighbor_mask = (dist < Rcut) & (dist > 1e-4) & mask_min                        # (B,N,N,S)

    if remove_self_neigh:
        diag = torch.arange(N, device=device)
        neighbor_mask[:, diag, diag, :] = False

    # Deterministic flat enumeration (row-major): (B,N,N,S) -> flat
    flat_mask = neighbor_mask.view(-1)
    flat_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)  # (K,)
    S = shifts.shape[0]
    # Decode indices
    s_idx = flat_idx % S
    tmp = flat_idx // S
    j_idx = tmp % N
    tmp //= N
    i_idx = tmp % N
    b_idx = tmp // N

    if upper_tri_only:
        keep = j_idx > i_idx
        b_idx, i_idx, j_idx, s_idx = b_idx[keep], i_idx[keep], j_idx[keep], s_idx[keep]

    # Distances for kept pairs
    dist_vals = dist[b_idx, i_idx, j_idx, s_idx]

    # Counts per (B,i)
    flat_bi = b_idx * N + i_idx
    counts_flat = torch.bincount(flat_bi, minlength=B * N).view(B, N)
    max_neighbors = int(counts_flat.max().item())

    # Allocate outputs (sentinel values as in non-batch)
    nndist   = torch.zeros((B, N, max_neighbors), dtype=dtype, device=device) + 17320.5
    nnRx     = torch.zeros((B, N, max_neighbors), dtype=dtype, device=device) + 10000.0
    nnRy     = torch.zeros((B, N, max_neighbors), dtype=dtype, device=device) + 10000.0
    nnRz     = torch.zeros((B, N, max_neighbors), dtype=dtype, device=device) + 10000.0
    nnType   = torch.full((B, N, max_neighbors), -1, dtype=torch.long, device=device)
    nnStruct = torch.full((B, N, max_neighbors), -1, dtype=torch.long, device=device)
    nrnnlist = counts_flat.unsqueeze(-1)

    neighbor_pos = R_translated[b_idx, j_idx, s_idx]
    del dist

    # Sort only by (b,i) to mimic single-structure logic (i grouping)
    order_key = b_idx * N + i_idx
    sort_idx = torch.argsort(order_key, stable=True)
    b_s = b_idx[sort_idx]; i_s = i_idx[sort_idx]; j_s = j_idx[sort_idx]; s_s = s_idx[sort_idx]
    dist_s = dist_vals[sort_idx]; pos_s = neighbor_pos[sort_idx]

    # Local index within each (b,i) group preserving intra-group (j,s) order (s fastest)
    flat_group = b_s * N + i_s
    group_counts = torch.bincount(flat_group, minlength=B * N)
    group_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                               group_counts.cumsum(0)[:-1]])
    seq = torch.arange(flat_group.shape[0], device=device)
    local_idx = seq - group_offsets[flat_group]

    # Scatter filled data
    nndist[b_s, i_s, local_idx] = dist_s
    nnRx[b_s, i_s, local_idx] = pos_s[:, 0]
    nnRy[b_s, i_s, local_idx] = pos_s[:, 1]
    nnRz[b_s, i_s, local_idx] = pos_s[:, 2]
    nnType[b_s, i_s, local_idx] = j_s
    nnStruct[b_s, i_s, local_idx] = j_s
    nrnnStruct = counts_flat.unsqueeze(-1)

    # Build 2D neighbor pair lists (same ordering)
    batch_counts = torch.bincount(b_s, minlength=B)
    max_K = int(batch_counts.max().item())
    if max_K == 0:
        neighbor_I_2d = torch.empty((B, 0), dtype=torch.long, device=device)
        neighbor_J_2d = torch.empty((B, 0), dtype=torch.long, device=device)
        IJ_pair_type_2d = torch.empty((B, 0), dtype=torch.long, device=device)
        JI_pair_type_2d = torch.empty((B, 0), dtype=torch.long, device=device)
        if verbose:
            print(f"  t <batched neighbor list> {time.perf_counter()-start_time1:.3f} s  (B={B}, N={N})")
        return (nrnnlist, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, nrnnStruct,
                neighbor_I_2d, neighbor_J_2d, IJ_pair_type_2d, JI_pair_type_2d)

    neighbor_I_2d = torch.full((B, max_K), -1, dtype=torch.long, device=device)
    neighbor_J_2d = torch.full((B, max_K), -1, dtype=torch.long, device=device)
    batch_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                               batch_counts.cumsum(0)[:-1]])
    seqK = torch.arange(b_s.shape[0], device=device)
    k_in_batch = seqK - batch_offsets[b_s]
    neighbor_I_2d[b_s, k_in_batch] = i_s
    neighbor_J_2d[b_s, k_in_batch] = j_s

    # Pair typing (unchanged)
    TYPE_flat = TYPE.view(-1)
    _, _, label_list = ordered_pairs_from_TYPE(TYPE_flat)
    pair_type_dict = {label_list[i]: i for i in range(len(label_list))}
    labels = [s.strip() for s in const.label.tolist()]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    Z = len(labels)
    pair_lookup = torch.full((Z, Z), -1, dtype=torch.long, device=device)
    if pair_type_dict:
        keys = list(pair_type_dict.keys())
        splits = [k.split('-') for k in keys]
        ai = torch.tensor([label_to_idx[a] for a, _ in splits], device=device)
        bi = torch.tensor([label_to_idx[b] for _, b in splits], device=device)
        vals = torch.tensor([pair_type_dict[k] for k in keys], device=device)
        pair_lookup[ai, bi] = vals

    ti_vals = TYPE[b_s, i_s]
    tj_vals = TYPE[b_s, j_s]
    IJ_pair_type_2d = torch.full((B, max_K), -1, dtype=torch.long, device=device)
    JI_pair_type_2d = torch.full((B, max_K), -1, dtype=torch.long, device=device)
    IJ_pair_type_2d[b_s, k_in_batch] = pair_lookup[ti_vals.long(), tj_vals.long()]
    JI_pair_type_2d[b_s, k_in_batch] = pair_lookup[tj_vals.long(), ti_vals.long()]

    if verbose:
        print(f"  t <batched neighbor list> {time.perf_counter()-start_time1:.3f} s  (B={B}, N={N})")

    return (nrnnlist, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, nrnnStruct,
            neighbor_I_2d, neighbor_J_2d, IJ_pair_type_2d, JI_pair_type_2d)