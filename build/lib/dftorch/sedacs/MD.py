import time

import numpy as np
import torch
import torch.distributed as dist
from sedacs.chemical_potential import get_mu
from sedacs.graph import (
    adaptive_halo_expansion,
    collect_graph_from_rho,
)
from sedacs.graph_partition import (
    get_coreHaloIndices,
    graph_partition,
)

from dftorch._cell import wrap_positions
from dftorch._io import write_pdb_frame, write_velocity_trajectory, write_XYZ_trajectory
from dftorch._tools import calculate_dist_dips
from dftorch.ewald_pme import (
    calculate_alpha_and_num_grids,
    calculate_PME_ewald,
    init_PME_data,
)
from dftorch.ewald_pme.neighbor_list import NeighborState
from dftorch.MD import MDXL, initialize_velocities

from . import (
    bcast_1d_int,
    calc_q_on_rank,
    gather_1d_to_rank0,
    get_energy_on_rank,
    get_evals_dvals,
    get_forces_on_rank,
    get_subsy_on_rank,
    graph_diff_and_update,
    kernel_global,
    pack_lol_int,
    repulsion,
    symmetrize_graph_safe,
    unpack_lol_int,
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


def _full_graph_to_numpy(full_graph):
    if isinstance(full_graph, torch.Tensor):
        return full_graph.detach().cpu().numpy()
    return full_graph


def _repartition_local_parts(
    structure,
    full_graph,
    n_jumps,
    works_per_rank,
    int_dtype,
    device,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_parts = works_per_rank * world_size

    if rank == 0:
        graph_np = _full_graph_to_numpy(full_graph)
        coords_np = (
            torch.stack((structure.RX, structure.RY, structure.RZ)).T.cpu().numpy()
        )
        parts = graph_partition(
            structure,
            structure,
            graph_np,
            "Metis",
            num_parts,
            coords_np,
            verb=False,
        )

        parts_core_halo = []
        num_cores = []
        for part in parts:
            core_halo, n_core, _ = get_coreHaloIndices(part, graph_np, n_jumps)
            parts_core_halo.append(core_halo)
            num_cores.append(n_core)

        num_cores = torch.tensor(num_cores, dtype=int_dtype, device=device)
        flat, offsets = pack_lol_int(parts_core_halo, int_dtype)
    else:
        num_cores = torch.empty((num_parts,), dtype=int_dtype, device=device)
        flat = None
        offsets = None

    dist.broadcast(num_cores, 0)
    flat = bcast_1d_int(flat, int_dtype, device=device, src=0)
    offsets = bcast_1d_int(offsets, int_dtype, device=device, src=0)
    parts_core_halo = unpack_lol_int(flat.cpu(), offsets.cpu())

    start = rank * works_per_rank
    end = start + works_per_rank
    return parts_core_halo[start:end], num_cores[start:end]


class MDXL_Graph(MDXL):
    def __init__(self, const, temperature_K, n_jumps, g_thresh, max_deg, int_dtype):
        super().__init__(None, const, temperature_K)
        self.n_jumps = n_jumps
        self.g_thresh = g_thresh
        self.max_deg = max_deg
        self.int_dtype = int_dtype
        self.rank_nbr_state = None

    def run(
        self,
        structure,
        dftorch_params,
        num_steps,
        dt,
        mu0,
        fullGraph,
        nbr_state,
        ch,
        core_size,
        works_per_rank,
        device,
        dump_interval=1,
        traj_filename="md_trj.xyz",
    ):
        if dist.get_rank() == 0 and self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities(
                structure,
                temperature_K=self.temperature_K,
                remove_com=True,
                rescale_to_T=True,
                remove_angmom=True,
            )
        else:
            self.VX = torch.empty_like(structure.RX)
            self.VY = torch.empty_like(structure.RY)
            self.VZ = torch.empty_like(structure.RZ)
        dist.broadcast(self.VX, 0)
        dist.broadcast(self.VY, 0)
        dist.broadcast(self.VZ, 0)
        q = structure.q.clone()
        if self.n is None:
            self.n = q
            self.n_0 = q
            self.n_1 = q
            self.n_2 = q
            self.n_3 = q
            self.n_4 = q
            self.n_5 = q
        if self.K0Res is None:
            self.K0Res = 0.0  # structure.KK@(q-self.n)

        if dftorch_params["COUL_METHOD"] == "PME":
            self.CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
                structure.cell.cpu().numpy(),
                dftorch_params["COULOMB_CUTOFF"],
                dftorch_params.get("COULOMB_ACC", 1e-5),
            )
            self.PME_data = init_PME_data(
                grid_dimensions,
                structure.cell,
                self.CALPHA,
                dftorch_params.get("PME_ORDER", 4),
            )
        else:
            self.CALPHA = None
            self.PME_data = None

        if self.E_array is None:
            self.E_array = torch.empty((0,), device=structure.device)
            self.T_array = torch.empty((0,), device=structure.device)
            self.Ek_array = torch.empty((0,), device=structure.device)
            self.Ep_array = torch.empty((0,), device=structure.device)
            self.Res_array = torch.empty((0,), device=structure.device)

        if dist.get_rank() == 0:
            self.EPOT = structure.e_tot

        for md_step in range(num_steps):
            if dist.get_rank() == 0:
                print("########## Step = {:} ##########".format(md_step))
            if md_step > 0 and md_step % 100 == 0:
                if dist.get_rank() == 0:
                    print("Graph repartition")
                ch, core_size = _repartition_local_parts(
                    structure,
                    fullGraph,
                    self.n_jumps,
                    works_per_rank,
                    self.int_dtype,
                    device,
                )

            for i in range(works_per_rank):
                ch[i], core_size[i], nh = get_coreHaloIndices(
                    ch[i][: core_size[i]], fullGraph, self.n_jumps
                )
                # print("Rank", dist.get_rank(), "has core and core-halo size:", core_size[i].item(), len(ch[i]))

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
                    f"CH sizes (all ranks): max={int(g_max)}, min={int(g_min)}, avg={g_sum / g_cnt:.1f}"
                )

            mu0, graphOnRank = self.step(
                structure,
                dftorch_params,
                md_step,
                dt,
                dump_interval,
                traj_filename,
                mu0,
                nbr_state,
                ch,
                core_size,
                works_per_rank,
                device,
            )

            parts_on_rank = _get_parts_on_rank(ch, core_size, device)
            fullGraph = graph_diff_and_update(
                fullGraph,
                graphOnRank,
                parts_on_rank,
                maxToAddRemove=100,
                device=device,
                add_only=False,
            ).numpy()
            fullGraph = symmetrize_graph_safe(fullGraph)

    def step(
        self,
        structure,
        dftorch_params,
        md_step,
        dt,
        dump_interval,
        traj_filename,
        mu0,
        nbr_state,
        ch,
        core_size,
        works_per_rank,
        device,
    ):
        if dist.get_rank() == 0:
            if self.cuda_sync:
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            self.EKIN = (
                0.5
                * self.MVV2KE
                * torch.sum(structure.Mnuc * (self.VX**2 + self.VY**2 + self.VZ**2))
            )  # Kinetic energy in eV (MVV2KE: unit conversion)
            Temperature = (
                (2 / 3) * self.KE2T * self.EKIN / structure.Nats
            )  # Statistical temperature in Kelvin
            Energ = (
                self.EKIN + self.EPOT
            )  # Total Energy in eV, Total energy fluctuations Propto dt^2
            # Time = md_step * dt
            ResErr = torch.norm(structure.q - self.n) / (
                structure.Nats**0.5
            )  # ResErr Propto dt^2

            self.E_array = torch.cat((self.E_array, Energ.detach().unsqueeze(0)), dim=0)
            self.T_array = torch.cat(
                (self.T_array, Temperature.detach().unsqueeze(0)), dim=0
            )
            self.Ek_array = torch.cat(
                (self.Ek_array, self.EKIN.detach().unsqueeze(0)), dim=0
            )
            self.Ep_array = torch.cat(
                (self.Ep_array, self.EPOT.detach().unsqueeze(0)), dim=0
            )
            self.Res_array = torch.cat(
                (self.Res_array, ResErr.detach().unsqueeze(0)), dim=0
            )

            comm_string = f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, Res = {ResErr:.6f}, mu = {mu0:.4f} eV"
            if md_step % dump_interval == 0:
                write_XYZ_trajectory(
                    traj_filename, structure, comm_string, step=md_step
                )

                write_pdb_frame(
                    traj_filename + ".pdb",
                    structure,
                    structure.cell,
                    step=md_step,
                    comment=comm_string,
                    mode="a",
                )

                q_dump = structure.q
                n_dump = self.n
                write_velocity_trajectory(
                    traj_filename + ".vel",
                    structure,
                    self.VX,
                    self.VY,
                    self.VZ,
                    charges=q_dump,
                    n_charges=n_dump,
                    comment=comm_string,
                    step=md_step,
                )

            self.VX = (
                self.VX
                + 0.5 * dt * (self.F2V * structure.f_tot[0] / structure.Mnuc)
                - self.fric * self.VX
            )  # First 1/2 of Leapfrog step
            self.VY = (
                self.VY
                + 0.5 * dt * (self.F2V * structure.f_tot[1] / structure.Mnuc)
                - self.fric * self.VY
            )  # F2V: Unit conversion
            self.VZ = (
                self.VZ
                + 0.5 * dt * (self.F2V * structure.f_tot[2] / structure.Mnuc)
                - self.fric * self.VZ
            )  # -c*V c>0 => Fricition

            if self.langevin_enabled:
                self._langevin_kick(structure, dt)

            if structure.cell is not None:
                R = torch.stack(
                    (
                        structure.RX + dt * self.VX,
                        structure.RY + dt * self.VY,
                        structure.RZ + dt * self.VZ,
                    ),
                    dim=-1,
                )
                R = wrap_positions(R, structure.cell, structure.cell_inv)
                structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)
            else:
                structure.RX = structure.RX + dt * self.VX
                structure.RY = structure.RY + dt * self.VY
                structure.RZ = structure.RZ + dt * self.VZ

            structure.coordinates = torch.stack(
                (structure.RX, structure.RY, structure.RZ),
            ).contiguous()

            if self.cuda_sync:
                torch.cuda.synchronize()
            tic2_1 = time.perf_counter()

            self.n = (
                2 * self.n_0
                - self.n_1
                - self.kappa * self.K0Res
                + self.alpha
                * (
                    self.C0 * self.n_0
                    + self.C1 * self.n_1
                    + self.C2 * self.n_2
                    + self.C3 * self.n_3
                    + self.C4 * self.n_4
                    + self.C5 * self.n_5
                )
            )
            self.n_5 = self.n_4
            self.n_4 = self.n_3
            self.n_3 = self.n_2
            self.n_2 = self.n_1
            self.n_1 = self.n_0
            self.n_0 = self.n

            positions = torch.stack(
                (structure.RX, structure.RY, structure.RZ)
            ).contiguous()
            nbr_state.update(positions)
            disps, dists, nl = calculate_dist_dips(
                positions, nbr_state, dftorch_params["COULOMB_CUTOFF"]
            )
            ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                positions.detach().clone(),
                self.n,
                structure.cell,
                nl,
                disps,
                dists,
                self.CALPHA,
                dftorch_params["COULOMB_CUTOFF"],
                self.PME_data,
                hubbard_u=structure.Hubbard_U,
                atomtypes=structure.TYPE,
                screening=1,
                calculate_forces=1,
                calculate_dq=1,
                h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
                h5_params=dftorch_params.get("H5_PARAMS", None),
            )

            if self.cuda_sync:
                torch.cuda.synchronize()
            print("PME: {:.3f} s".format(time.perf_counter() - tic2_1))

        else:
            positions = None
            disps = None
            dists = None
            self.n = torch.empty((structure.Nats,), device=device)
            CoulPot = torch.empty((structure.Nats,), device=device)

            structure.coordinates = torch.empty(
                (3, structure.Nats),
                dtype=structure.RX.dtype,
                device=device,
            )

        tic2_1 = time.perf_counter()

        dist.broadcast(self.n, 0)
        dist.broadcast(CoulPot, 0)

        # dist.broadcast(structure.RX, 0)
        # dist.broadcast(structure.RY, 0)
        # dist.broadcast(structure.RZ, 0)

        dist.broadcast(structure.coordinates, 0)
        structure.RX, structure.RY, structure.RZ = structure.coordinates.unbind(dim=0)

        if dist.get_rank() != 0:
            positions = structure.coordinates.contiguous()
            if self.rank_nbr_state is None:
                self.rank_nbr_state = NeighborState(
                    positions,
                    structure.lattice_vecs,
                    None,
                    dftorch_params["COULOMB_CUTOFF"],
                    is_dense=True,
                    buffer=0.0,
                    use_triton=False,
                )
            else:
                self.rank_nbr_state.update(positions)
            _, _, nl = calculate_dist_dips(
                positions,
                self.rank_nbr_state,
                dftorch_params["COULOMB_CUTOFF"],
            )

        per_part_data = []  # store (ch_structure, hindex, atom_ids, S, Z, KK, Q, e_vals, d_vals)
        e_list = []
        d_list = []
        # e_vals_on_rank = torch.empty((0,), device=device)
        # d_vals_on_rank = torch.empty((0,), device=device)
        for i in range(works_per_rank):
            ch_structure = get_subsy_on_rank(
                structure,
                dftorch_params,
                self.n[ch[i]],
                ch[i],
                core_size[i],
                CoulPot,
                device,
            )
            ch_structure.e_vals, ch_structure.d_vals, ch_structure.Q = get_evals_dvals(
                ch_structure.Z, ch_structure.H, ch_structure.core_ao_slice
            )

            per_part_data.append((ch_structure))
            e_list.append(ch_structure.e_vals)
            d_list.append(ch_structure.d_vals)
        e_vals_on_rank = torch.cat(e_list)
        d_vals_on_rank = torch.cat(d_list)

        e_vals_all = gather_1d_to_rank0(e_vals_on_rank, device=device, src=0)
        d_vals_all = gather_1d_to_rank0(d_vals_on_rank, device=device, src=0)

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
            if self.cuda_sync:
                torch.cuda.synchronize()
            print("mu0: {:.3f} s".format(time.perf_counter() - tic2_1))
        else:
            mu0 = torch.tensor(0.0, device=device)
        dist.broadcast(mu0, 0)

        tic2_1 = time.perf_counter()

        per_part_D = []
        q_global = torch.zeros(structure.Nats, device=device)
        self.K0Res = torch.zeros(structure.Nats, device=device)
        graphOnRank = np.full((structure.Nats, self.max_deg + 1), -1, dtype=np.int64)
        graphOnRank[:, 0] = 0
        graph_coords_np = structure.coordinates.T.cpu().numpy()
        graph_cell_np = structure.cell.cpu().numpy()
        graph_nl_np = nl.cpu().numpy()
        for ch_structure in per_part_data:
            q, D, f, K0Res = calc_q_on_rank(
                ch_structure,
                ch_structure.atom_ids,
                ch_structure.S,
                ch_structure.Z,
                ch_structure.KK,
                ch_structure.Q,
                ch_structure.e_vals,
                self.n[ch_structure.ch],
                mu0,
            )

            q_global[ch_structure.ch[: ch_structure.core_size]] = q[
                : ch_structure.core_size
            ]
            per_part_D.append(D)
            ch_structure.q = q
            ch_structure.f = f
            D_cpu = D.cpu()
            hindex_cpu = ch_structure.hindex.cpu()

            graphOnRank = adaptive_halo_expansion(
                graphOnRank,
                D_cpu,
                self.g_thresh,
                structure.Nats,
                self.max_deg,  # maxDeg
                ch_structure.ch,
                ch_structure.ch[: ch_structure.core_size],
                hindex_cpu,
                graph_coords_np,
                graph_cell_np,
                graph_nl_np,
                alpha=0.7,
            )

            graphOnRank = collect_graph_from_rho(
                graphOnRank,
                D_cpu,
                self.g_thresh,
                structure.Nats,
                self.max_deg,
                ch_structure.ch,
                ch_structure.core_size,
                hindex_cpu,
            )

            self.K0Res[ch_structure.ch[: ch_structure.core_size]] = K0Res[
                : ch_structure.core_size
            ]

        dist.all_reduce(q_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.K0Res, op=dist.ReduceOp.SUM)

        if self.cuda_sync:
            torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic2_1 = time.perf_counter()

        if self.NoRank:
            1
        else:
            self.K0Res = kernel_global(
                structure,
                positions,
                nl,
                disps,
                dists,
                self.CALPHA,
                self.PME_data,
                dftorch_params,
                dftorch_params.get("KRYLOV_TOL_MD", 1e-2),
                self.K0Res,
                per_part_data,
                mu0,
                device,
            )

        if self.cuda_sync:
            torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print("KER: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic2_1 = time.perf_counter()

        ### ENERGY ###
        e_band0, s_entropy = get_energy_on_rank(per_part_data, per_part_D, device)
        dist.all_reduce(e_band0, op=dist.ReduceOp.SUM)
        dist.all_reduce(s_entropy, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            # nuclear repulsion
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

            # Coulomb energy
            e_coul = 0.5 * (2 * q_global - self.n) @ CoulPot + 0.5 * torch.sum(
                (2.0 * q_global - self.n) * structure.Hubbard_U * self.n
            )

            # Dipole energy
            if structure.e_field.abs().sum() > 0.0:
                efield_term = (
                    structure.RX * structure.e_field[0]
                    + structure.RY * structure.e_field[1]
                    + structure.RZ * structure.e_field[2]
                )
                e_dipole = -torch.sum(q_global * efield_term, dim=-1)
            else:
                e_dipole = 0.0

            e_entropy = -2 * structure.Te * s_entropy

            e_tot = e_band0 + e_coul + e_dipole + e_entropy + e_repulsion

            f_coul = forces1 * (2 * q_global / self.n - 1.0)
        else:
            e_tot = torch.tensor(0.0, device=device)
        dist.broadcast(e_tot, 0)
        structure.e_tot = e_tot
        self.EPOT = structure.e_tot

        ### FORCES ###
        # FScoul + Fband0 + FPulay + Fdipole + FSdipole
        # !!! Fdipole will use n, but q should be used
        f_tot = get_forces_on_rank(
            structure, per_part_data, per_part_D, self.n, CoulPot, device
        )  # on rank.
        dist.all_reduce(f_tot, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            f_tot = f_tot + f_rep + f_coul
            structure.q = q_global
            structure.f_tot = f_tot

            self.VX = (
                self.VX
                + 0.5 * dt * (self.F2V * structure.f_tot[0] / structure.Mnuc)
                - self.fric * self.VX
            )  # Integrate second 1/2 of leapfrog step
            self.VY = (
                self.VY
                + 0.5 * dt * (self.F2V * structure.f_tot[1] / structure.Mnuc)
                - self.fric * self.VY
            )  # - c*V  c > 0 => friction
            self.VZ = (
                self.VZ
                + 0.5 * dt * (self.F2V * structure.f_tot[2] / structure.Mnuc)
                - self.fric * self.VZ
            )

            if self.langevin_enabled:
                self._langevin_kick(structure, dt)

            if self.cuda_sync:
                torch.cuda.synchronize()
            print("F AND E: {:.3f} s".format(time.perf_counter() - tic2_1))
            tic2_1 = time.perf_counter()

            print(
                comm_string
                + ", t = {:.1f} s".format(
                    time.perf_counter() - start_time,
                )
            )

        return mu0, graphOnRank
