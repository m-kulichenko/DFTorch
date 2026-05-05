import torch
import torch.distributed as dist
import time
from dftorch.ewald_pme import (
    calculate_PME_ewald,
    init_PME_data,
    calculate_alpha_and_num_grids,
)

from sedacs.chemical_potential import get_mu

from sedacs.graph_partition import get_coreHaloIndices
from sedacs.graph import (
    collect_graph_from_rho,
    adaptive_halo_expansion,
    symmetrize_graph,
)

from . import (
    get_energy_on_rank,
    gather_1d_to_rank0,
    graph_diff_and_update,
    kernel_global,
    get_forces_on_rank,
    get_subsy_on_rank,
    get_evals_dvals,
    calc_q_on_rank,
    repulsion,
)


def _get_parts_on_rank(ch, core_size, device):
    parts_on_rank = []
    for row_indices, local_core_size in zip(ch, core_size):
        n_core = int(local_core_size.item())
        parts_on_rank.append(
            torch.as_tensor(row_indices[:n_core], dtype=torch.long, device=device)
        )

    if len(parts_on_rank) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    return torch.unique(torch.cat(parts_on_rank), sorted=False)


def scf(
    structure,
    dftorch_params,
    fullGraph,
    ch,
    core_size,
    nbr_state,
    disps_global,
    dists_global,
    nl,
    works_per_rank,
    n_jumps,
    g_thresh,
    max_deg,
    device,
):

    timing = {}
    per_part_data = []
    per_part_D = []
    mu0 = torch.tensor(-5.5, device=device)
    if dist.get_rank() == 0:
        print("Initial mu", mu0)
    graphOnRank = None
    q_global = torch.zeros(structure.Nats, device=device)

    if dist.get_rank() == 0:
        positions_global = torch.stack(
            (structure.RX, structure.RY, structure.RZ),
        )
        CALPHA_global, grid_dimensions = calculate_alpha_and_num_grids(
            structure.cell.cpu().numpy(),
            dftorch_params["cutoff"],
            dftorch_params["Coulomb_acc"],
        )
        PME_data_global = init_PME_data(
            grid_dimensions,
            structure.cell,
            CALPHA_global,
            dftorch_params["PME_order"],
        )
        # nbr_state = NeighborState(positions_global, structure.cell, None, dftorch_params['cutoff'], is_dense=True, buffer=0.0, use_triton=False)
        # disps_global, dists_global, nbr_inds_global = calculate_dist_dips(positions_global, nbr_state, dftorch_params['cutoff'])
    else:
        positions_global, CALPHA_global, PME_data_global = [None] * 3

    fullGraph = symmetrize_graph(fullGraph.cpu().numpy())

    scf_error = torch.tensor(float("inf"), device=device)
    scf_iter = 0
    while (scf_error > dftorch_params["SCF_TOL"]) and (
        scf_iter < dftorch_params["SCF_MAX_ITER"]
    ):
        TIC = time.time()
        timing = {}
        scf_iter += 1
        if dist.get_rank() == 0:
            print("\nSCF iteration", scf_iter)

        tic = time.time()
        for i in range(works_per_rank):
            ch[i], core_size[i], nh = get_coreHaloIndices(
                ch[i][: core_size[i]], fullGraph, n_jumps
            )
            # print(
            #     "Rank",
            #     dist.get_rank(),
            #     "has core and core-halo size:",
            #     core_size[i].item(),
            #     len(ch[i]),
            # )
        timing["get_coreHaloIndices"] = time.time() - tic
        ch_sizes = [len(ch[i]) for i in range(works_per_rank)]
        ch_stats = torch.tensor(
            [max(ch_sizes), min(ch_sizes), sum(ch_sizes), works_per_rank],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(ch_stats[:1], op=dist.ReduceOp.MAX)
        dist.all_reduce(ch_stats[1:2], op=dist.ReduceOp.MIN)
        dist.all_reduce(ch_stats[2:], op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            g_max, g_min, g_sum, g_cnt = ch_stats.tolist()
            print(
                f"  CH sizes (all ranks): max={int(g_max)}, min={int(g_min)}, avg={g_sum / g_cnt:.1f}"
            )

        q_global_old = q_global.clone()
        per_part_data, per_part_D, mu0, K0Res_global, graphOnRank = scf_step(
            scf_iter,
            structure,
            positions_global,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            PME_data_global,
            dftorch_params,
            q_global,
            mu0,
            ch,
            core_size,
            works_per_rank,
            g_thresh,
            max_deg,
            device,
            timing,
        )

        q_global = q_global - K0Res_global

        if dist.get_rank() == 0:
            scf_error = torch.norm(q_global - q_global_old)
            print("scf error = {:.9f}".format(scf_error.item()))

        dist.broadcast(scf_error, 0)

        """
        The data flow:
        graphOnRank  (numpy, from collect_graph_from_rho)
            ↓
        graph_diff_and_update(fullGraph, graphOnRank, partsOnRank, ...)
            → njit compute_added/removed → dist.all_reduce → njit update_graph
            → returns torch.Tensor (CPU, int64)
            ↓  .numpy()
        fullGraph  (numpy int64)
            ↓
        symmetrize_graph(fullGraph)   # @njit, expects numpy
            ↓
        fullGraph  (numpy, same shape as before)
        """

        tic = time.time()
        partsOnRank = _get_parts_on_rank(ch, core_size, device)
        fullGraph = graph_diff_and_update(
            fullGraph,
            graphOnRank,
            partsOnRank,
            maxToAddRemove=100,
            device=device,
            add_only=True,
        ).numpy()
        timing["Graph_diff_update"] = time.time() - tic

        tic = time.time()
        fullGraph = symmetrize_graph(fullGraph)
        timing["Graph_symmetrize"] = time.time() - tic

        # tic = time.time()
        # partsOnRank = _get_parts_on_rank(ch, core_size, device)
        # fullGraph = graph_diff_and_update(
        #     fullGraph,
        #     graphOnRank_tensor,
        #     partsOnRank,
        #     maxToAddRemove=100,
        #     device=device,
        # ).cpu().numpy()
        # timing["Graph_all_reduce"] = time.time() - tic

        # tic = time.time()
        # fullGraph = symmetrize_graph(fullGraph)
        # timing["Graph_symmetrize"] = time.time() - tic
        timing["TOTAL"] = time.time() - TIC
        if dist.get_rank() == 0:
            print(timing)
        if dist.get_rank() == 0:
            print("TOTAL:", timing["TOTAL"])

    print("Doing Band Energy...")
    e_band0, s_entropy = get_energy_on_rank(per_part_data, per_part_D, device)
    dist.all_reduce(e_band0, op=dist.ReduceOp.SUM)
    dist.all_reduce(s_entropy, op=dist.ReduceOp.SUM)
    e_entropy = -2 * structure.Te * s_entropy

    # Final global calcs: nuclear repulsion, PME,
    if dist.get_rank() == 0:
        # nuclear repulsion

        print("Doing Repulsion Energy...")
        e_repulsion, f_rep, stress_repulsion = repulsion(
            structure.const.R_rep_tensor,
            structure.const.rep_splines_tensor,
            structure.const.close_exp_tensor,
            structure.TYPE,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.cell,
            4.0,
            structure.Nats,  # repulsive_rcut
            structure.const,
            verbose=False,
            compute_stress=False,
        )

        print("Doing Coulomb Energy...")
        ewald_e1, f_coul, dq_p1 = calculate_PME_ewald(
            positions_global.detach().clone(),
            q_global,
            structure.cell,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            dftorch_params["cutoff"],
            PME_data_global,
            hubbard_u=structure.Hubbard_U,
            atomtypes=structure.TYPE,
            screening=1,
            calculate_forces=1,
            calculate_dq=1,
            h_damp_exp=dftorch_params.get("h_damp_exp", None),
            h5_params=dftorch_params.get("h5_params", None),
        )

        # Coulomb energy
        e_coul = ewald_e1 + 0.5 * torch.sum(q_global**2 * structure.Hubbard_U)

        Ha = 0.0367493  # Ha -> eV
        print(f"{'Energy Components':─^52}")
        print(f"  {'E_band':22s} {e_band0 * Ha:12.6f} Ha ({e_band0:12.6f} eV)")
        print(f"  {'E_coulomb':22s} {e_coul * Ha:12.6f} Ha ({e_coul:12.6f} eV)")
        print(
            f"  {'E_repulsion':22s} {e_repulsion * Ha:12.6f} Ha ({e_repulsion:12.6f} eV)"
        )
        print(
            f"  {'E_entropy (-TS)':22s} {e_entropy * Ha:12.6f} Ha ({e_entropy:12.6f} eV)"
        )
        # print(f"  {'net spin':22s} {structure1.net_spin_sr.sum().item():12.4f}")
        # print(f"  {'E_spin':22s} {structure1.e_spin.item()*Ha:12.6f} Ha  ({structure1.e_spin.item():12.6f} eV)")
        print(f"{'─' * 52}")

        # Dipole energy
        if structure.e_field.abs().sum() > 0.0:
            Efield_term = (
                structure.RX * structure.e_field[0]
                + structure.RY * structure.e_field[1]
                + structure.RZ * structure.e_field[2]
            )
            e_dipole = -torch.sum(q_global * Efield_term, dim=-1)
        else:
            e_dipole = 0.0

        e_tot = e_band0 + e_coul + e_dipole + e_entropy + e_repulsion
        print(f"  {'E_total (F=E-TS)':22s} {e_tot * Ha:12.6f} Ha ({e_tot:12.6f} eV)")

    else:
        dq_p1 = torch.empty((structure.Nats,), device=device)
    dist.broadcast(dq_p1, 0)

    ### FORCES ###
    f_tot = get_forces_on_rank(
        structure, per_part_data, per_part_D, q_global, dq_p1, device
    )  # on rank.
    dist.all_reduce(f_tot, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        f_tot = f_tot + f_rep + f_coul
        torch.save(f_tot.cpu(), "f_tot.pt")
        structure.q = q_global
        structure.e_tot = e_tot
        structure.f_tot = f_tot
    else:
        structure.q = torch.empty((structure.Nats,), device=device)
        structure.f_tot = torch.empty((3, structure.Nats), device=device)
    dist.broadcast(structure.q, 0)
    dist.broadcast(structure.f_tot, 0)

    return mu0, fullGraph


def scf_step(
    scf_iter,
    structure,
    positions_global,
    nl,
    disps_global,
    dists_global,
    CALPHA_global,
    PME_data_global,
    dftorch_params,
    q_global,
    mu0,
    ch,
    core_size,
    works_per_rank,
    g_thresh,
    max_deg,
    device,
    timing,
):

    tic = time.time()
    if dist.get_rank() == 0:
        ewald_e1, forces1, CoulPot = calculate_PME_ewald(
            positions_global.detach().clone(),
            q_global,
            structure.cell,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            dftorch_params["cutoff"],
            PME_data_global,
            hubbard_u=structure.Hubbard_U,
            atomtypes=structure.TYPE,
            screening=1,
            calculate_forces=0,
            calculate_dq=1,
            h_damp_exp=dftorch_params.get("h_damp_exp", None),
            h5_params=dftorch_params.get("h5_params", None),
        )
    else:
        CoulPot = torch.empty((structure.Nats,), device=device)
    timing["PME"] = time.time() - tic
    tic = time.time()
    dist.broadcast(CoulPot, 0)
    timing["Broadcast"] = time.time() - tic

    tic = time.time()
    per_part_data = []  # store (ch_structure, hindex, atom_ids, S, Z, KK, Q, e_vals, d_vals)
    e_vals_on_rank = torch.empty((0,), device=device)
    d_vals_on_rank = torch.empty((0,), device=device)
    for i in range(works_per_rank):
        ch_structure = get_subsy_on_rank(
            structure,
            dftorch_params,
            q_global[ch[i]],
            ch[i],
            core_size[i],
            CoulPot,
            device,
        )
        ch_structure.e_vals, ch_structure.d_vals, ch_structure.Q = get_evals_dvals(
            ch_structure.Z, ch_structure.H, ch_structure.core_ao_slice
        )

        per_part_data.append((ch_structure))
        e_vals_on_rank = torch.cat((e_vals_on_rank, ch_structure.e_vals), dim=0)
        d_vals_on_rank = torch.cat((d_vals_on_rank, ch_structure.d_vals), dim=0)
    timing["Local_evals"] = time.time() - tic

    tic = time.time()
    e_vals_all = gather_1d_to_rank0(e_vals_on_rank, device=device, src=0)
    d_vals_all = gather_1d_to_rank0(d_vals_on_rank, device=device, src=0)
    timing["Gather_eval"] = time.time() - tic

    tic = time.time()
    if dist.get_rank() == 0:
        mu0 = get_mu(
            mu0.cpu().numpy().item(),
            e_vals_all.cpu().numpy(),
            structure.Te,
            structure.Nocc,
            dvals=d_vals_all.cpu().numpy(),
            verb=False,
        )
        mu0 = torch.tensor(mu0, device=device)
    else:
        mu0 = torch.tensor(0.0, device=device)
    dist.broadcast(mu0, 0)
    timing["mu0"] = time.time() - tic

    tic = time.time()
    per_part_D = []
    K0Res_global = torch.zeros(structure.Nats, device=device)
    graphOnRank = None
    for ch_structure in per_part_data:
        q, D, f, K0Res = calc_q_on_rank(
            ch_structure,
            ch_structure.atom_ids,
            ch_structure.S,
            ch_structure.Z,
            ch_structure.KK,
            ch_structure.Q,
            ch_structure.e_vals,
            q_global[ch_structure.ch],
            mu0,
        )
        per_part_D.append(D)
        ch_structure.f = f

        graphOnRank = adaptive_halo_expansion(
            graphOnRank,
            D.cpu(),
            g_thresh,  # gthresh
            structure.Nats,
            max_deg,  # maxDeg
            ch_structure.ch,
            ch_structure.ch[: ch_structure.core_size],
            ch_structure.hindex.cpu(),
            structure.coordinates.T.cpu().numpy(),
            structure.cell.cpu().numpy(),
            nl.cpu().numpy(),
            alpha=0.7,
        )

        graphOnRank = collect_graph_from_rho(
            graphOnRank,
            D.cpu(),
            g_thresh,
            structure.Nats,
            max_deg,
            ch_structure.ch,
            ch_structure.core_size,
            ch_structure.hindex.cpu(),
        )

        K0Res_global[ch_structure.ch[: ch_structure.core_size]] = K0Res[
            : ch_structure.core_size
        ]

    timing["K0Res_Local"] = time.time() - tic

    tic = time.time()
    dist.all_reduce(K0Res_global, op=dist.ReduceOp.SUM)
    timing["K0Res_all_reduce"] = time.time() - tic

    #####
    tic = time.time()
    if scf_iter > dftorch_params["KRYLOV_START"]:
        K0Res_global = kernel_global(
            structure,
            positions_global,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            PME_data_global,
            dftorch_params,
            dftorch_params["KRYLOV_TOL"],
            K0Res_global,
            per_part_data,
            mu0,
            device,
        )
    timing["K0Res_krylov"] = time.time() - tic

    return per_part_data, per_part_D, mu0, K0Res_global, graphOnRank
