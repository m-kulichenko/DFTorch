from __future__ import annotations
import time
import torch
import torch.distributed as dist
import numpy as np

from dftorch.ewald_pme.neighbor_list import NeighborState
from dftorch._tools import (
    calculate_dist_dips,
    fractional_matrix_power_symm,
    ordered_pairs_from_TYPE,
)
from dftorch import Structure, Constants

from dftorch._h0ands import H0_and_S_vectorized

from dftorch.ewald_pme import calculate_PME_ewald
from sedacs.graph import (
    compute_added,
    compute_removed,
    update_graph,
    symmetrize_graph,
    get_initial_graph,
)
from sedacs.graph_partition import get_coreHaloIndices, graph_partition

from dftorch._nearestneighborlist import vectorized_nearestneighborlist


def get_nl(structure, dftorch_params):
    positions = torch.stack(
        (structure.RX, structure.RY, structure.RZ),
    )
    nbr_state = NeighborState(
        positions,
        structure.cell,
        None,
        dftorch_params["COULOMB_CUTOFF"],
        is_dense=True,
        buffer=2.0,
        use_triton=False,
    )
    disps, dists, nl = calculate_dist_dips(
        positions, nbr_state, dftorch_params["graph_cutoff"]
    )

    if dftorch_params["graph_cutoff"] < dftorch_params["COULOMB_CUTOFF"]:
        nl = torch.where(
            (dists > dftorch_params["graph_cutoff"]) | (dists == 0.0), -1, nl
        )
        nl = nl.sort(dim=1, descending=True)[0]
        nl = nl[:, : torch.max(torch.sum(nl != -1, dim=1))]

    elif dftorch_params["graph_cutoff"] > dftorch_params["COULOMB_CUTOFF"]:
        raise ValueError(
            "Coulomb cutoff must be greater than or equal to graph cutoff for this implementation."
        )

    # num_neighbors = torch.sum(nl != -1, dim=1)
    # nl = torch.cat((num_neighbors.unsqueeze(1), nl), dim=1)
    nl = nl.cpu().numpy()
    return nbr_state, disps, dists, nl


def pack_lol_int(lol, dtype):
    lengths = torch.tensor([len(x) for x in lol], dtype=dtype)
    offsets = torch.zeros((lengths.numel() + 1,), dtype=dtype)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    flat = torch.tensor([v for sub in lol for v in sub], dtype=dtype)
    return flat, offsets


def unpack_lol_int(flat, offsets):
    out = []
    for i in range(offsets.numel() - 1):
        a = int(offsets[i].item())
        b = int(offsets[i + 1].item())
        out.append(flat[a:b].tolist())
    return out


def bcast_1d_int(t, dtype, device, src=0):
    rank = dist.get_rank()
    if rank == src:
        t = t.to(device=device, dtype=dtype)
        n = torch.tensor([t.numel()], dtype=dtype, device=device)
    else:
        n = torch.empty((1,), dtype=dtype, device=device)
    dist.broadcast(n, src)
    n = int(n.item())
    if rank != src:
        t = torch.empty((n,), dtype=dtype, device=device)
    dist.broadcast(t, src)
    return t


def prepare_initial_graph_data(
    structure,
    dftorch_params,
    nparts,
    n_jumps,
    max_deg,
    int_dtype,
    device,
):
    rank = dist.get_rank()

    if rank == 0:
        positions = torch.stack((structure.RX, structure.RY, structure.RZ))

        nbr_state = NeighborState(
            positions,
            structure.lattice_vecs,
            None,
            dftorch_params["COULOMB_CUTOFF"],
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nl = calculate_dist_dips(
            positions, nbr_state, dftorch_params["COULOMB_CUTOFF"]
        )

        nl_init = torch.where(
            (dists > dftorch_params["graph_cutoff"]) | (dists == 0.0), -1, nl
        )
        nl_init = nl_init.sort(dim=1, descending=True)[0]
        nl_init = nl_init[:, : torch.max(torch.sum(nl_init != -1, dim=1))]

        full_graph, _ = get_initial_graph(
            positions.T.cpu().numpy(),
            nl_init.cpu().numpy(),
            dftorch_params["graph_cutoff"],
            max_deg,
            structure.cell.diag().cpu().numpy(),
        )
        full_graph = symmetrize_graph(full_graph)

        tick = time.perf_counter()
        parts = graph_partition(
            structure,
            structure,
            full_graph,
            "Metis",
            nparts,
            positions.T.cpu().numpy(),
            verb=False,
        )
        print(f"Graph partitioning time: {time.perf_counter() - tick:.1f} s")

        parts_core_halo = []
        num_cores = []
        print("\nCore and halos indices for every part:")
        for part in parts:
            core_halo, n_core, _ = get_coreHaloIndices(part, full_graph, n_jumps)
            parts_core_halo.append(core_halo)
            num_cores.append(n_core)

            print(
                "Rank",
                dist.get_rank(),
                "has core and core-halo size:",
                n_core,
                len(core_halo),
            )

        num_cores = torch.tensor(num_cores, dtype=int_dtype, device=device)
        full_graph = torch.tensor(full_graph, dtype=int_dtype, device=device)
        flat, offsets = pack_lol_int(parts_core_halo, int_dtype)
        nl_shape = torch.tensor(list(nl.shape), dtype=int_dtype, device=device)
        full_graph_shape = torch.tensor(
            list(full_graph.shape), dtype=int_dtype, device=device
        )
    else:
        nbr_state = None
        disps = None
        dists = None
        flat = None
        offsets = None
        num_cores = torch.empty((nparts,), dtype=int_dtype, device=device)
        nl_shape = torch.empty((2,), dtype=int_dtype, device=device)
        full_graph_shape = torch.empty((2,), dtype=int_dtype, device=device)

    dist.broadcast(num_cores, 0)
    dist.broadcast(nl_shape, 0)
    dist.broadcast(full_graph_shape, 0)

    if rank != 0:
        nl = torch.empty(
            tuple(int(x) for x in nl_shape.tolist()), dtype=int_dtype, device=device
        )
        full_graph = torch.empty(
            tuple(int(x) for x in full_graph_shape.tolist()),
            dtype=int_dtype,
            device=device,
        )

    dist.broadcast(nl, 0)
    dist.broadcast(full_graph, 0)

    flat = bcast_1d_int(flat, int_dtype, device=device, src=0)
    offsets = bcast_1d_int(offsets, int_dtype, device=device, src=0)
    parts_core_halo = unpack_lol_int(flat.cpu(), offsets.cpu())

    return nbr_state, disps, dists, nl, full_graph, parts_core_halo, num_cores


def gather_1d_to_rank0(x: torch.Tensor, device, src: int = 0) -> torch.Tensor:
    """
    Gather variable-length 1D tensors from all ranks to rank 0.
    Returns a single flattened 1D tensor on rank 0, and an empty tensor on other ranks.
    """
    if x.ndim != 1:
        x = x.reshape(-1)

    rank = dist.get_rank()
    world = dist.get_world_size()

    # 1) gather lengths
    n_local = torch.tensor([x.numel()], dtype=torch.int64, device=device)
    n_all = [torch.empty((1,), dtype=torch.int64, device=device) for _ in range(world)]
    dist.all_gather(n_all, n_local)
    sizes = torch.tensor(
        [int(t.item()) for t in n_all], dtype=torch.int64, device=device
    )
    max_n = int(sizes.max().item())

    # 2) pad to max length
    x_pad = torch.empty((max_n,), dtype=x.dtype, device=device)
    x_pad.zero_()
    if x.numel() > 0:
        x_pad[: x.numel()] = x

    # 3) gather padded
    gathered = [
        torch.empty((max_n,), dtype=x.dtype, device=device) for _ in range(world)
    ]
    dist.all_gather(gathered, x_pad)

    if rank != src:
        return torch.empty((0,), dtype=x.dtype, device=device)

    # 4) unpad + flatten
    parts = [g[: int(sizes[i].item())] for i, g in enumerate(gathered)]
    return torch.cat(parts, dim=0)


def _pad_graph_for_symmetrize(graph: np.ndarray) -> np.ndarray:
    """
    Pad a graph so sedacs.graph.symmetrize_graph can add inbound-only neighbors
    without exceeding the current width.

    symmetrize_graph uses the graph width as a hard maxDeg and raises before it
    can resize. Here we compute a safe upper bound for the post-symmetrized
    degree of each node: out_degree + in_degree.
    """
    nnodes = graph.shape[0]
    max_deg = graph.shape[1] - 1
    indeg = np.zeros((nnodes,), dtype=np.int64)

    for i in range(nnodes):
        deg = int(graph[i, 0])
        if deg == 0:
            continue
        indeg[graph[i, 1 : 1 + deg]] += 1

    required_deg = graph[:, 0].astype(np.int64) + indeg
    padded_max_deg = int(required_deg.max())
    if padded_max_deg <= max_deg:
        return graph

    graph_padded = np.full((nnodes, padded_max_deg + 1), -1, dtype=graph.dtype)
    graph_padded[:, : graph.shape[1]] = graph
    return graph_padded


def _graph_to_dense_adjacency(graph: np.ndarray) -> np.ndarray:
    adj = np.zeros((graph.shape[0], graph.shape[0]), dtype=np.uint8)
    for i in range(graph.shape[0]):
        deg = int(graph[i, 0])
        if deg == 0:
            continue
        adj[i, graph[i, 1 : 1 + deg]] = 1
    return adj


def symmetrize_graph_safe(graph: np.ndarray) -> np.ndarray:
    graph = np.ascontiguousarray(graph)
    try:
        return symmetrize_graph(graph)
    except ValueError as exc:
        if "larger degree than original maxDeg" not in str(exc):
            raise

        padded_graph = _pad_graph_for_symmetrize(graph)
        if padded_graph.shape[1] == graph.shape[1]:
            raise
        return symmetrize_graph(padded_graph)


def graph_diff_and_update(
    prev_graph,
    graph_on_rank,
    parts_on_rank,
    maxToAddRemove=100,
    device=None,
    add_only=False,
):
    """
    torch.distributed mirror of sedacs graph_diff_and_update (graph.py).
    Uses the same compute_added / compute_removed / update_graph @njit kernels,
    replacing comm.Allreduce with dist.all_reduce.
    """
    # Normalise to contiguous CPU int64 numpy (required by @njit kernels)
    if torch.is_tensor(prev_graph):
        prev_graph = prev_graph.cpu().numpy().astype(np.int64)
    else:
        prev_graph = np.ascontiguousarray(prev_graph, dtype=np.int64)
    if torch.is_tensor(graph_on_rank):
        graph_on_rank = graph_on_rank.cpu().numpy().astype(np.int64)
    else:
        graph_on_rank = np.ascontiguousarray(graph_on_rank, dtype=np.int64)
    if torch.is_tensor(parts_on_rank):
        parts_on_rank = parts_on_rank.cpu().numpy().astype(np.int64)
    else:
        parts_on_rank = np.asarray(parts_on_rank, dtype=np.int64)

    nnodes = prev_graph.shape[0]

    # Determine device for dist.all_reduce (NCCL requires GPU tensors)
    if device is None:
        if dist.get_backend() == "nccl":
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

    G_added = np.zeros((nnodes, maxToAddRemove + 1), dtype=prev_graph.dtype)
    G_removed = np.zeros((nnodes, maxToAddRemove + 1), dtype=prev_graph.dtype)

    if parts_on_rank.size > 0:
        local_graph = prev_graph[parts_on_rank]
        local_graph_new = graph_on_rank[parts_on_rank].copy()

        G_added[parts_on_rank, 1:], G_added[parts_on_rank, 0] = compute_added(
            local_graph[:, 1:],
            local_graph_new[:, 1:],
            local_graph[:, 0],
            local_graph_new[:, 0],
            nnodes,
            maxToAddRemove=maxToAddRemove,
        )
        if not add_only:
            G_removed[parts_on_rank, 1:], G_removed[parts_on_rank, 0] = compute_removed(
                local_graph[:, 1:],
                local_graph_new[:, 1:],
                local_graph[:, 0],
                local_graph_new[:, 0],
                nnodes,
                maxToAddRemove=maxToAddRemove,
            )

    local_overflow = int(
        G_added[:, 0].max() > maxToAddRemove
        or (not add_only and G_removed[:, 0].max() > maxToAddRemove)
    )
    overflow_t = torch.tensor([local_overflow], dtype=torch.int64, device=device)
    dist.all_reduce(overflow_t, op=dist.ReduceOp.SUM)

    if overflow_t.item() > 0:
        # Fallback: OR-reduce full adjacency across ranks
        adj = _graph_to_dense_adjacency(graph_on_rank)
        adj_t = torch.from_numpy(adj).to(device)
        dist.all_reduce(adj_t, op=dist.ReduceOp.MAX)
        adj = adj_t.cpu().numpy().astype(bool)
        # Allocate wide enough to hold the maximum degree after union + symmetrization
        # headroom: extra maxToAddRemove columns so symmetrize_graph won't overflow
        max_adj_deg = int(adj.sum(axis=1).max())
        out_width = max(prev_graph.shape[1], max_adj_deg + 1 + maxToAddRemove)
        G_out = np.full((nnodes, out_width), -1, dtype=prev_graph.dtype)
        G_out[:, 0] = adj.sum(axis=1)
        for i in range(nnodes):
            nbrs_i = np.where(adj[i])[0]
            G_out[i, 1 : 1 + nbrs_i.size] = nbrs_i
        return torch.from_numpy(_pad_graph_for_symmetrize(G_out))

    # All-reduce deltas (SUM is correct: parts partition all N nodes, no conflicts)
    G_added_t = torch.from_numpy(G_added).to(device)
    G_removed_t = torch.from_numpy(G_removed).to(device)
    dist.all_reduce(G_added_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(G_removed_t, op=dist.ReduceOp.SUM)
    G_added = G_added_t.cpu().numpy().astype(prev_graph.dtype)
    G_removed = G_removed_t.cpu().numpy().astype(prev_graph.dtype)

    # Clamp nadd per node so cnt + nadd never exceeds maxDeg in update_graph.
    # (compute_added has an off-by-one: k > maxToAddRemove fires at k=101, giving
    #  N_added = 101 which would cause an OOB write inside update_graph.)
    maxDeg = prev_graph.shape[1] - 1
    headroom = np.maximum(0, maxDeg - prev_graph[:, 0]).astype(prev_graph.dtype)
    G_added[:, 0] = np.minimum(G_added[:, 0], np.minimum(headroom, maxToAddRemove))
    if not add_only:
        G_removed[:, 0] = np.minimum(G_removed[:, 0], maxToAddRemove)

    # Apply delta — same @njit kernel as the mpi4py version
    updated_graph = update_graph(
        prev_graph[:, 1:],
        prev_graph[:, 0],
        G_removed[:, 1:],
        G_removed[:, 0],
        G_added[:, 1:],
        G_added[:, 0],
    )

    return torch.from_numpy(_pad_graph_for_symmetrize(updated_graph))


def get_ij(TYPE, TYPE_all, nnType, nrnnlist, const):
    # === Vectorized neighbor type pair generation ===
    max_neighbors = nnType.shape[-1]

    # Create mask for valid neighbors
    neighbor_mask = (
        torch.arange(max_neighbors, device=nnType.device).unsqueeze(0) < nrnnlist
    )
    neighbor_J = nnType[neighbor_mask]
    neighbor_I = torch.repeat_interleave(
        torch.arange(nrnnlist.squeeze(-1).shape[0], device=nnType.device),
        nrnnlist.squeeze(-1),
    )
    del neighbor_mask

    ### Get tensors for SKF files ###
    _, _, label_list = ordered_pairs_from_TYPE(TYPE_all)

    pair_type_dict = {}

    for i in range(len(label_list)):
        pair_type_dict[label_list[i]] = i

    # Build a 2D lookup table once (no function), then index it
    labels = [
        s.strip() for s in const.label.tolist()
    ]  # fix spaces like ' P', 'V ', etc.
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    Z = len(labels)
    pair_lookup = torch.full((Z, Z), -1, dtype=torch.long, device=nnType.device)
    for k, v in pair_type_dict.items():  # keys like "C-H"
        a, b = k.split("-")
        ai = label_to_idx[a]
        bi = label_to_idx[b]
        pair_lookup[ai, bi] = int(v)
        # If the mapping is symmetric and reverse might be missing, also do:
        # pair_lookup[bi, ai] = int(v)
    ti = TYPE[neighbor_I].long()
    tj = TYPE[neighbor_J].long()
    IJ_pair_type = pair_lookup[ti, tj]  # shape: (len(neighbor_I),)
    JI_pair_type = pair_lookup[tj, ti]

    return neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type


def get_subsy_on_rank(structure, dftorch_params, q, ch, core_size, CoulPot, device):

    ch_structure = Structure(
        dftorch_params,
        structure.const,
        device=device,
        species=structure.TYPE[ch].unsqueeze(0),
        coordinates=structure.coordinates.T[ch].unsqueeze(0),
        ignore_spin=True,
    )

    ch_structure.core_ao_slice = ch_structure.n_orbitals_per_atom[:core_size].sum()
    ch_structure.ch = ch
    ch_structure.core_size = core_size
    ch_structure.KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
        ch_structure.Nats, device=device
    )  # Initial mixing coefficient for linear mixing

    positions = torch.stack(
        (ch_structure.RX, ch_structure.RY, ch_structure.RZ),
    )
    nbr_state = NeighborState(
        positions,
        ch_structure.cell,
        None,
        dftorch_params["RCUT_ELECTRONIC"],
        is_dense=True,
        buffer=0.0,
        use_triton=False,
    )
    disps_sub, dists_sub, nl_sub = calculate_dist_dips(
        positions, nbr_state, dftorch_params["RCUT_ELECTRONIC"]
    )
    num_neighbors = torch.sum(nl_sub != -1, dim=1)
    nl_sub = torch.cat((num_neighbors.unsqueeze(1), nl_sub), dim=1)

    nnRx, nnRy, nnRz = ch_structure.coordinates[:, :, None] + disps_sub
    nnType = nl_sub[:, 1:]
    neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type = get_ij(
        ch_structure.TYPE,
        structure.TYPE,
        nnType,
        nl_sub[:, 0].unsqueeze(1),
        structure.const,
    )

    ch_structure.H0, ch_structure.dH0, ch_structure.S, ch_structure.dS = (
        H0_and_S_vectorized(
            ch_structure.TYPE,
            ch_structure.RX,
            ch_structure.RY,
            ch_structure.RZ,
            ch_structure.diagonal,
            ch_structure.H_INDEX_START,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            ch_structure.const,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
            ch_structure.const.R_orb,
            ch_structure.const.coeffs_tensor,
            verbose=False,
        )
    )

    del nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type
    ch_structure.Z = fractional_matrix_power_symm(ch_structure.S, -0.5)

    ch_structure.atom_ids = torch.repeat_interleave(
        torch.arange(len(ch_structure.n_orbitals_per_atom), device=device),
        ch_structure.n_orbitals_per_atom,
    )  # Generate atom index for each orbital
    Hubbard_U_gathered = ch_structure.Hubbard_U[ch_structure.atom_ids]
    ch_structure.hindex = torch.cat(
        (ch_structure.H_INDEX_START, ch_structure.H_INDEX_END[-1:] + 1), dim=0
    )

    if ch_structure.e_field.abs().sum() > 0.0:
        Hdipole = torch.diag(
            -ch_structure.RX[ch_structure.atom_ids] * ch_structure.e_field[0]
            - ch_structure.RY[ch_structure.atom_ids] * ch_structure.e_field[1]
            - ch_structure.RZ[ch_structure.atom_ids] * ch_structure.e_field[2]
        )
        Hdipole = 0.5 * Hdipole @ ch_structure.S + 0.5 * ch_structure.S @ Hdipole
    else:
        Hdipole = 0.0
    ch_structure.H0 = ch_structure.H0 + Hdipole

    if q is not None:
        Hcoul_diag = (
            Hubbard_U_gathered * q[ch_structure.atom_ids]
            + CoulPot[ch][ch_structure.atom_ids]
        )
        Hcoul = 0.5 * (
            Hcoul_diag.unsqueeze(1) * ch_structure.S
            + ch_structure.S * Hcoul_diag.unsqueeze(0)
        )
    else:
        Hcoul = 0.0
    ch_structure.H = ch_structure.H0 + Hcoul

    return ch_structure


def get_evals_dvals(Z, H, core_ao_slice):
    e_vals, Q = torch.linalg.eigh(Z.T @ H @ Z)
    d_vals = torch.sum(Q[:core_ao_slice, :] ** 2, dim=0)

    return e_vals, d_vals, Q


def calc_q_on_rank(ch_structure, atom_ids, S, Z, KK, Q, e_vals, q, mu0):

    kB = 8.61739e-5  # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
    beta = 1.0 / (kB * ch_structure.Te)

    q_old = q.clone()
    f = 1.0 / (torch.exp(beta * (e_vals - mu0)) + 1)
    Dorth = (Q * f.unsqueeze(-2)) @ Q.transpose(-2, -1)
    D = Z @ Dorth @ Z.T
    DS = 2.0 * (D * S.T).sum(dim=1)
    q = -1.0 * ch_structure.Znuc
    q.scatter_add_(
        0, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    Res = q - q_old
    K0Res = KK @ Res

    return q, D, f, K0Res


def get_energy_on_rank(per_part_data, per_part_D, device):

    eps = 1e-10
    kB = 8.61739e-5  # eV/K
    e_band0 = torch.tensor(0.0, device=device)
    s_entropy = torch.tensor(0.0, device=device)
    for ch_structure, D in zip(per_part_data, per_part_D):
        # Band energy
        e_band0 = (
            e_band0
            + 2
            * (
                ch_structure.H0[: ch_structure.core_ao_slice]
                * D[: ch_structure.core_ao_slice]
            ).sum()
        )

        # Entropy
        f_tmp = ch_structure.f[: ch_structure.core_ao_slice]
        mask = (f_tmp > eps) & (f_tmp < 1 - eps)
        f_safe = f_tmp.clamp(eps, 1 - eps)  # avoid log(0)
        term = f_safe * torch.log(f_safe) + (1 - f_safe) * torch.log(1 - f_safe)
        term = term * mask
        s_entropy = s_entropy + (-kB * term.sum())

    return e_band0, s_entropy


def get_forces_on_rank(structure, per_part_data, per_part_D, q_global, dq_p1, device):
    f_tot = torch.zeros((3, structure.Nats), device=device)
    for ch_structure, D in zip(per_part_data, per_part_D):
        ### FScoul
        CoulPot = dq_p1[ch_structure.ch]
        FScoul = torch.zeros((3, ch_structure.Nats), device=device)
        factor = (ch_structure.Hubbard_U * q_global[ch_structure.ch] + CoulPot) * 2
        dS_times_D = D * ch_structure.dS * factor[ch_structure.atom_ids].unsqueeze(-1)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
        FScoul.scatter_add_(
            1, ch_structure.atom_ids.expand(3, -1), dDS_XYZ_row_sum
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FScoul.scatter_add_(1, ch_structure.atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

        ### Fband0
        Fband0 = torch.zeros((3, ch_structure.Nats), device=device)
        TMP = 4 * (ch_structure.dH0 * D.T).sum(dim=-1)
        Fband0.scatter_add_(
            1, ch_structure.atom_ids.expand(3, -1), TMP
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        ### Pulay forces
        SIHD = 4 * ch_structure.Z @ ch_structure.Z.T @ ch_structure.H @ D
        FPulay = torch.zeros((3, ch_structure.Nats), device=device)
        TMP = -(ch_structure.dS * SIHD.T).sum(dim=-1)
        FPulay.scatter_add_(
            1, ch_structure.atom_ids.expand(3, -1), TMP
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        ### Fdipole = q_i * E
        if structure.e_field.abs().sum() > 0.0:
            Fdipole = q_global[ch_structure.ch].unsqueeze(0) * structure.e_field.view(
                3, 1
            )
        else:
            Fdipole = torch.zeros_like(Fband0)

        ### FSdipole
        D0 = torch.diag(ch_structure.D0)
        dotRE = (
            ch_structure.RX * ch_structure.e_field[0]
            + ch_structure.RY * ch_structure.e_field[1]
            + ch_structure.RZ * ch_structure.e_field[2]
        )
        FSdipole = torch.zeros((3, ch_structure.Nats), device=device)
        D_diff = D - D0
        tmp2 = -2 * (D_diff.unsqueeze(0) * ch_structure.dS.transpose(1, 2)).sum(dim=-1)
        FSdipole.scatter_add_(1, ch_structure.atom_ids.expand(3, -1), tmp2)
        FSdipole *= dotRE

        n_orb = ch_structure.dS.shape[1]
        a = ch_structure.dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
        outs_by_atom = torch.zeros((3, n_orb, ch_structure.Nats), device=device)
        outs_by_atom = outs_by_atom.index_add(2, ch_structure.atom_ids, a)
        new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[ch_structure.atom_ids]
        FSdipole -= 2 * new_fs

        f_tot[:, ch_structure.ch[: ch_structure.core_size]] = (
            FScoul[:, : ch_structure.core_size]
            + Fband0[:, : ch_structure.core_size]
            + FPulay[:, : ch_structure.core_size]
            + Fdipole[:, : ch_structure.core_size]
            + FSdipole[:, : ch_structure.core_size]
        )

    return f_tot


def kernel_global(
    structure,
    positions_global,
    nbr_inds_global,
    disps_global,
    dists_global,
    CALPHA_global,
    PME_data_global,
    dftorch_params,
    FelTol,
    K0Res_global,
    per_part_data,
    mu0,
    device,
):
    krylov_maxrank = dftorch_params.get("KRYLOV_MAXRANK", 20)
    if dist.get_rank() == 0:
        vi = torch.zeros(structure.Nats, krylov_maxrank, device=device)
        fi = torch.zeros(structure.Nats, krylov_maxrank, device=device)
    else:
        vi = None
        fi = None
    dr = K0Res_global.clone()
    krylov_rank = 0
    Fel = torch.tensor(float("inf"), device=device)

    while (krylov_rank < krylov_maxrank) and (Fel > FelTol):
        # Normalize current direction
        if dist.get_rank() == 0:
            torch.cuda.synchronize()
            norm_dr = torch.norm(dr)
            if norm_dr < 1e-9:
                print("zero norm_dr")
                break
            vi[:, krylov_rank] = dr / norm_dr

            # Modified Gram-Schmidt against previous vi
            if krylov_rank > 0:
                Vprev = vi[:, :krylov_rank]  # (Nats, krylov_rank)
                vi[:, krylov_rank] = vi[:, krylov_rank] - Vprev @ (
                    Vprev.T @ vi[:, krylov_rank]
                )

            norm_vi = torch.norm(vi[:, krylov_rank])
            if norm_vi < 1e-9:
                print("zero norm_vi")
                break
            vi[:, krylov_rank] = vi[:, krylov_rank] / norm_vi
            v = vi[:, krylov_rank].clone()  # current search direction

            # dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
            # PME Coulomb case
            _, _, d_CoulPot = calculate_PME_ewald(
                positions_global.detach().clone(),
                v,
                structure.cell,
                nbr_inds_global,
                disps_global,
                dists_global,
                CALPHA_global,
                dftorch_params["COULOMB_CUTOFF"],
                PME_data_global,
                hubbard_u=structure.Hubbard_U,
                atomtypes=structure.TYPE,
                screening=1,
                calculate_forces=0,
                calculate_dq=1,
                h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
                h5_params=dftorch_params.get("H5_PARAMS", None),
            )
        else:
            v = torch.empty((structure.Nats,), device=device)
            d_CoulPot = torch.empty((structure.Nats,), device=device)
        dist.broadcast(v, 0)
        dist.broadcast(d_CoulPot, 0)

        if dist.get_rank() == 0:
            torch.cuda.synchronize()

        dq_global = torch.zeros(structure.Nats, device=device)
        trP1 = torch.tensor(0.0, device=device)
        dqdmu_global = torch.zeros(structure.Nats, device=device)
        trdPdMu = torch.tensor(0.0, device=device)
        dr = torch.zeros(structure.Nats, device=device)
        for ch_structure in per_part_data:
            #### calc_dq
            d_Hcoul_diag = (
                ch_structure.Hubbard_U[ch_structure.atom_ids]
                * v[ch_structure.ch][ch_structure.atom_ids]
                + d_CoulPot[ch_structure.ch][ch_structure.atom_ids]
            )
            d_Hcoul = 0.5 * (
                d_Hcoul_diag.unsqueeze(1) * ch_structure.S
                + ch_structure.S * d_Hcoul_diag.unsqueeze(0)
            )
            H1_orth = ch_structure.Z.T @ d_Hcoul @ ch_structure.Z

            ### fermi_prt
            kB = 8.61739e-5  # eV/K
            beta = 1 / (kB * structure.Te)

            QH1Q = ch_structure.Q.T @ H1_orth @ ch_structure.Q
            fe = 1.0 / (torch.exp(beta * (ch_structure.e_vals - mu0)) + 1.0)
            ei = ch_structure.e_vals[:, None]
            ej = ch_structure.e_vals[None, :]
            de = ei - ej  # (N,N)

            # Susceptibility kernel χ_ij = (f_i - f_j)/(e_i - e_j), with diagonal limit -β f(1-f)
            chi = torch.empty_like(de)
            off = de.abs() > 1e-12
            chi[off] = (fe[:, None] - fe[None, :])[off] / de[off]
            diag = -beta * fe * (1.0 - fe)  # (N,)
            chi[~off] = diag.expand_as(de)[~off]

            # Response in eigenbasis
            X = chi * QH1Q

            # Enforce particle conservation via μ1
            # dN_dmu = diag.sum()                              # d tr(D)/dμ = -β ∑ f(1-f)
            # if torch.abs(dN_dmu) > 1e-15:
            #     mu1 = X.diagonal().sum() / dN_dmu
            #     X = X - torch.diag_embed(diag) * mu1

            # Numerically stable μ1: if |dN_dmu| is small, skip the correction.
            # mask = (torch.abs(dN_dmu) > 1e-12).to(H1_orth.dtype)
            # mu1 = (X.diagonal().sum() / (dN_dmu + (1.0 - mask))) * mask
            # X = X - torch.diag_embed(diag) * mu1

            # Instead, we will calculate a global correction
            dpdmu = -diag

            # D0 = Q @ torch.diag_embed(fe) @ Q.T
            D1 = ch_structure.Q @ X @ ch_structure.Q.T
            ### end fermi_prt

            D1 = ch_structure.Z @ D1 @ ch_structure.Z.T
            D1S = 2 * (D1 * ch_structure.S.T).sum(dim=1)

            # dq (atomic) from AO response
            dq = torch.zeros(ch_structure.Nats, device=device)
            dq.scatter_add_(0, ch_structure.atom_ids, D1S)
            del D1, D1S
            #### end calc_dq

            # Here we compute the charges response (q1) from P1 and we store it on
            # a vector q1 that stores all the previous q1s from past iranks iterations
            # We also compute the partial trace contribution (trP1) from this mpi
            # execution and the current part (partIndex).
            # Collect the charge response from the core region
            dq_global[ch_structure.ch[: ch_structure.core_size]] = dq[
                : ch_structure.core_size
            ]
            # Add up the partial trace contribution from the core region
            trP1 = trP1 + torch.sum(dq[: ch_structure.core_size])

            # Here we compute the charges response (dqdmu) from dPdMu and we store
            # them on a matrix dqdmu that stores all the previous dqdmus from past
            # irank iterations.
            # We also compute the partial trace contribution (trdPdMu) from this node
            # and the current part (partIndex).

            ### replaced this with code below ###
            # dPdMuAO = torch.zeros((ch_structure.HDIM, ch_structure.HDIM), device=device)
            # dPdMuAO.diagonal().copy_(dpdmu)
            # dPdMuAO = ch_structure.Q @ dPdMuAO @ ch_structure.Q.T
            # dPdMuAO = ch_structure.Z @ dPdMuAO @ ch_structure.Z.T

            # dPdMuAOS = 2 * torch.diag(dPdMuAO @ ch_structure.S)
            # dq = torch.zeros(ch_structure.Nats, device=device)
            # dq.scatter_add_(0, ch_structure.atom_ids, dPdMuAOS)

            # dqdmu_global[ch_structure.ch[:ch_structure.core_size]] = dq[:ch_structure.core_size]
            # trdPdMu = trdPdMu + torch.sum(dq[:ch_structure.core_size])

            # ---- dPdMu: avoid HDIMxHDIM zero matrix + diagonal fill ----
            # dPdMu_orth = Q @ diag(dpdmu) @ Q.T  === (Q * dpdmu[None,:]) @ Q.T
            dPdMu_orth = (ch_structure.Q * dpdmu.unsqueeze(0)) @ ch_structure.Q.T

            dPdMu = ch_structure.Z @ (dPdMu_orth @ ch_structure.Z.T)
            dPdMuAOS = 2.0 * (dPdMu * ch_structure.S.T).sum(dim=1)

            dq2 = torch.zeros(ch_structure.Nats, device=device)
            dq2.scatter_add_(0, ch_structure.atom_ids, dPdMuAOS)

            dqdmu_global[ch_structure.ch[: ch_structure.core_size]] = dq2[
                : ch_structure.core_size
            ]
            trdPdMu = trdPdMu + dq2[: ch_structure.core_size].sum()

        dist.all_reduce(dq_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(trP1, op=dist.ReduceOp.SUM)
        dist.all_reduce(dqdmu_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(trdPdMu, op=dist.ReduceOp.SUM)

        # Compute the response to the chemical potential (mu1) and adjust q1
        mu1_Global = -trP1 / trdPdMu if abs(trdPdMu) > 1e-12 else 0.0
        dq_global = dq_global + mu1_Global * dqdmu_global

        if dist.get_rank() == 0:
            torch.cuda.synchronize()

        # New residual (df/dlambda), preconditioned
        for ch_structure in per_part_data:
            dr[ch_structure.ch[: ch_structure.core_size]] = (
                dq_global[ch_structure.ch[: ch_structure.core_size]]
                - v[ch_structure.ch[: ch_structure.core_size]]
            )
            dr[ch_structure.ch[: ch_structure.core_size]] = (
                ch_structure.KK[: ch_structure.core_size, : ch_structure.core_size]
                @ dr[ch_structure.ch[: ch_structure.core_size]]
            )
        dist.all_reduce(dr, op=dist.ReduceOp.SUM)

        rank_m = krylov_rank + 1
        if dist.get_rank() == 0:
            # Store fi column
            fi[:, krylov_rank] = dr

            # Small overlap O and RHS (vectorized)
            F_small = fi[:, :rank_m]  # (Nats, r)
            O = F_small.T @ F_small  # (r, r)  # noqa: E741
            rhs = F_small.T @ K0Res_global  # (r,)

            # Solve O Y = rhs (stable) instead of explicit inverse
            Y = torch.linalg.solve(O, rhs)  # (r,)

            # Residual norm in the subspace
            Fel = torch.norm(F_small @ Y - K0Res_global)
            print("  Krylov rank: {:}, Fel = {:.7f}".format(krylov_rank, Fel.item()))
        else:
            Fel = torch.tensor(0.0, device=device)  # scalar, same shape as rank 0
        dist.broadcast(Fel, 0)
        krylov_rank += 1

        if dist.get_rank() == 0:
            torch.cuda.synchronize()

    if dist.get_rank() == 0:
        K0Res_global = vi[:, :rank_m] @ Y

    dist.broadcast(K0Res_global, 0)

    return K0Res_global


def repulsion(
    R_rep_tensor: torch.Tensor,
    rep_splines_tensor: torch.Tensor,
    close_exp_tensor: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell,
    Rcut: float,
    Nats: int,
    const,
    verbose: bool,
    compute_stress: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Compute the total repulsive energy and its Cartesian derivatives.

    Parameters
    ----------
    R_rep_tensor : (P, M) torch.Tensor
        Radial grid (Å) per pair type (P = number of pair types).
    rep_splines_tensor : (P, M, >=6) torch.Tensor
        Spline coefficients c0..c5 (Hartree) for 5th‑order local polynomial segments.
    close_exp_tensor : (P, 3) torch.Tensor
        Close exponential parameters for each pair type.
    TYPE : (Nats,) torch.Tensor
        Atom type indices (compatible with const.label).
    RX, RY, RZ : (Nats,) torch.Tensor
        Atomic Cartesian coordinates in Å.
    cell : sequence or tensor
        Periodic cell specification:
        - shape (3,) for orthorhombic box lengths
        - shape (3,3) for a triclinic cell matrix
    Rcut : float
        Cutoff (Å) used to build the neighbor list.
    Nats : int
        Number of atoms.
    const : object
        Contains chemical label mapping for pair typing.
    verbose : bool
        If True, neighbor list routine may emit timing info.
    compute_stress : bool
        If True, also compute and return the analytical pair-virial
        stress tensor (requires ``cell`` to be a (3,3) matrix).

    Returns
    -------
    Vr : torch.Tensor (scalar, eV)
        Total repulsive energy.
    dVr : torch.Tensor, shape (3, Nats, Nats), eV/Å
        Antisymmetric matrix of pairwise force contributions:
        dVr[:, i, j] = +∂E/∂r_i from pair (i,j); dVr[:, j, i] = −dVr[:, i, j].
    sigma_rep : torch.Tensor, shape (3, 3), eV/ų (only when ``compute_stress=True``)
        Symmetrised repulsive virial stress tensor:
        σ_{αβ} = (1/V) Σ_{ij} (dV/dR) R_{ij,α} R_{ij,β} / R_{ij}.

    Notes
    -----
    - Distances converted to Bohr internally (factor 0.52917721).
    - energy converted Hartree → eV using 27.21138625.
    - Neighbor list uses minimum image (min_image_only=True).
    """
    _, _, nnRx, nnRy, nnRz, nnType, _, _, neighbor_I, neighbor_J, IJ_pair_type, _ = (
        vectorized_nearestneighborlist(
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            Rcut,
            Nats,
            const,
            min_image_only=True,
            verbose=verbose,
        )
    )
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)
    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    # ── per-pair starting distance ────────────────────────────────────
    R0 = R_rep_tensor[IJ_pair_type, 0]  # (P,) first spline knot
    use_exp = dR_mskd < R0  # (P,) bool mask
    del R0

    # ── spline indices (only needed for spline region) ────────────────
    indices = (
        torch.searchsorted(
            R_rep_tensor[IJ_pair_type], dR_mskd.unsqueeze(-1), right=True
        ).squeeze(-1)
        - 1
    )
    # Optionally clamp to keep indices in bounds
    # K = R_rep_tensor.size(1)
    # indices = indices.clamp(min=0, max=K-1)

    dx = (dR_mskd - R_rep_tensor[IJ_pair_type, indices]) / 0.52917721

    # ── spline energy ─────────────────────────────────────────────────
    Vr = (
        rep_splines_tensor[IJ_pair_type, indices, 0]
        + rep_splines_tensor[IJ_pair_type, indices, 1] * dx
        + rep_splines_tensor[IJ_pair_type, indices, 2] * dx**2
        + rep_splines_tensor[IJ_pair_type, indices, 3] * dx**3
        + rep_splines_tensor[IJ_pair_type, indices, 4] * dx**4
        + rep_splines_tensor[IJ_pair_type, indices, 5] * dx**5
    )

    # ── exponential energy  E = exp(a0*r + a1) + a2 ──────────────────
    Vr_exp = (
        torch.exp(
            -close_exp_tensor[IJ_pair_type, 0] * dR_mskd / 0.52917721
            + close_exp_tensor[IJ_pair_type, 1]
        )
        + close_exp_tensor[IJ_pair_type, 2]
    )
    # ── select per pair ───────────────────────────────────────────────
    Vr = torch.where(use_exp, Vr_exp, Vr)
    Vr = Vr.sum() * 27.21138625  # eV
    del Vr_exp

    # ── spline gradient  dV/dR (Ha/Bohr) ─────────────────────────────
    # now, it's Ha/Bohr
    dVr_dR = (
        rep_splines_tensor[IJ_pair_type, indices, 1]
        + 2 * rep_splines_tensor[IJ_pair_type, indices, 2] * dx
        + 3 * rep_splines_tensor[IJ_pair_type, indices, 3] * dx**2
        + 4 * rep_splines_tensor[IJ_pair_type, indices, 4] * dx**3
        + 5 * rep_splines_tensor[IJ_pair_type, indices, 5] * dx**4
    )
    # ── exponential gradient  dV/dR (Ha/Bohr) ────────────────────────
    dVr_exp_dR = -close_exp_tensor[IJ_pair_type, 0] * torch.exp(
        -close_exp_tensor[IJ_pair_type, 0] * dR_mskd / 0.52917721
        + close_exp_tensor[IJ_pair_type, 1]
    )

    # ── select per pair ───────────────────────────────────────────────
    dVr_dR = torch.where(use_exp, dVr_exp_dR, dVr_dR)
    del dVr_exp_dR

    # ── stress tensor (pair virial) ──────────────────────────────────
    sigma_rep = None
    if compute_stress and cell is not None:
        Rab = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1)[nn_mask]  # (P, 3)
        dVr_dR_eV = dVr_dR * 27.21138625 / 0.52917721  # eV/Å
        Vcell = torch.abs(torch.det(cell))
        sigma_rep = (
            dVr_dR_eV[:, None, None]
            * (Rab.unsqueeze(-1) * Rab.unsqueeze(-2))
            / dR_mskd[:, None, None]
        ).sum(dim=0) / Vcell
        sigma_rep = 0.5 * (sigma_rep + sigma_rep.T)

    # ── accumulate forces ───────────────────
    dR_dxyz = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=0)[:, nn_mask] / dR_mskd
    dVr_dR_times_dR_dxyz = dVr_dR * dR_dxyz
    del dVr_dR, dR_dxyz

    # Do this O(N) equivalent:
    scale = 27.21138625 / 0.52917721
    f_rep = torch.zeros((3, Nats), device=RX.device)
    scaled = dVr_dR_times_dR_dxyz * scale
    f_rep.index_add_(1, neighbor_I, scaled)
    f_rep.index_add_(1, neighbor_J, -scaled)

    del nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, _

    if sigma_rep is not None:
        return Vr, f_rep, sigma_rep
    return Vr, f_rep, None

def prepare_structure(dftorch_params, device):
    ### Prepare DFTB+DFTorch parameters, structure, neighbor list, and graph partitioning ###
    const = Constants(dftorch_params).to(device)
    structure = Structure(dftorch_params, const, device=device)
    dftorch_params["CELL"] = structure.cell
    structure.SpecClustNN = 10
    structure.interface = 'dftorch'
    return structure, dftorch_params