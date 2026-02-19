import os
import argparse
import time
from dftorch.H0andS import H0_and_S_vectorized
import torch
import torch.distributed as dist

import warnings
import logging
# to disable torchdynamo completely. Faster for smaller systems and single-point calculations.
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # hard-disable capture
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # disables torch.compile / Inductor globally


import numpy as np

from dftorch.Constants import Constants
from dftorch.Structure import Structure
from dftorch.MD import MDXL, initialize_velocities
from dftorch.RepulsiveSpline import get_repulsion_energy

from sedacs.graph import add_graphs, graph_diff_and_update, collect_graph_from_rho, adaptive_halo_expansion, symmetrize_graph, get_initial_graph
from sedacs.chemical_potential import get_mu
from sedacs.graph_partition import get_coreHaloIndices, graph_partition, get_coreHaloIndicesPYSEQM

from dftorch.Tools import calculate_dist_dips, fractional_matrix_power_symm, ordered_pairs_from_TYPE
#from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from dftorch.ewald_pme import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from dftorch.ewald_pme.neighbor_list import NeighborState
from dftorch.nearestneighborlist import vectorized_nearestneighborlist

from dftorch.io import write_XYZ_trajectory

from dftorch.sedacs import get_energy_on_rank, get_ch, pack_lol_int64, unpack_lol_int64, bcast_1d_int64,\
    get_ij, gather_1d_to_rank0, kernel_global, get_energy_on_rank, get_forces_on_rank, get_nlget_subsy_on_rank, \
    get_evals_dvals, calc_q_on_rank


### Configure torch and torch.compile ###
# Silence warnings and module logs
# warnings.filterwarnings("ignore")
# os.environ["TORCH_LOGS"] = ""               # disable PT2 logging
# os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
# os.environ["TORCHDYNAMO_VERBOSE"] = "0"
logging.getLogger("torch.fx").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)
# Enable dynamic shape capture for dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# default data type
torch.set_default_dtype(torch.float64)

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

MAX_DEG = 600
GTHRESH = 0.0002
NJUMPS = 1

def prepare_structure(device):
    ### Prepare DFTB+DFTorch parameters, structure, neighbor list, and graph partitioning ###
    dftorch_params = {
    'coul_method': 'PME', # 'FULL' for full coulomb matrix, 'PME' for PME method
    'Coulomb_acc': 5e-5,   # Coulomb accuracy for full coulomb calcs or t_err for PME
    'cutoff': 10.0,        # Coulomb cutoff
    'h0_cutoff': 8.0,        # Coulomb cutoff
    'graph_cutoff': 5.0,        # Graph cutoff
    'PME_order': 4,        # Ignored for FULL coulomb method
    'SCF_MAX_ITER': 100,    # Maximum number of SCF iterations
    'SCF_TOL': 1e-6,       # SCF convergence tolerance on density matrix
    'SCF_ALPHA': 0.5,      # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
    'KRYLOV_MAXRANK': 10,  # Maximum Krylov subspace rank
    'KRYLOV_TOL': 1e-6,    # Krylov subspace convergence tolerance in SCF
    'KRYLOV_TOL_MD': 1e-4, # Krylov subspace convergence tolerance in MD SCF
    'KRYLOV_START': 3,     # Number of initial SCF iterations before starting Krylov acceleration
                }

    # filename = 'COORD_far.xyz'            # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    # LBox = torch.tensor([25.0, 25.0, 25.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.

    # filename = 'COORD_8WATER.xyz'            # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    # LBox = torch.tensor([30.0, 30.0, 30.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.

    filename = 'water_30.xyz'            # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    LBox = torch.tensor([35.0, 35.0, 35.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.


    const = Constants(filename,
                    #'/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/ptbp/complete_set',
                    #'C:\\000_MyFiles\\Programs\\DFTorch\\tests\\sk_orig\\ptbp\\complete_set\\',
                    '/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/mio-1-1/mio-1-1/',
                    magnetic_hubbard_ldep=False
                    ).to(device)
    structure = Structure(filename, LBox, const, charge=0, Te=5000.0, device=device)
    structure.SpecClustNN = 20
    structure.interface = 'dftorch'
    return structure, dftorch_params


def run(backend, structure, dftorch_params, fullGraph, ch, core_size, nbr_state, disps_global, dists_global, nl, works_per_rank, device):
        
    per_part_data = []  # store (ch_structure, hindex, atom_ids, S, Z, KK, Q, e_vals, d_vals)
    e_vals_on_rank = torch.empty((0,), device=device)
    d_vals_on_rank = torch.empty((0,), device=device)
    for i in range(works_per_rank):
        ch_structure = get_subsy_on_rank(structure, dftorch_params, None, ch[i], core_size[i], None, device)
        ch_structure.e_vals, ch_structure.d_vals, ch_structure.Q = get_evals_dvals(ch_structure.Z, ch_structure.H, ch_structure.core_ao_slice)

        per_part_data.append((ch_structure))
        e_vals_on_rank = torch.cat((e_vals_on_rank, ch_structure.e_vals), dim=0)
        d_vals_on_rank = torch.cat((d_vals_on_rank, ch_structure.d_vals), dim=0)

    e_vals_all = gather_1d_to_rank0(e_vals_on_rank, device=device, src=0)
    d_vals_all = gather_1d_to_rank0(d_vals_on_rank, device=device, src=0)

    if dist.get_rank() == 0:
        mu0 = get_mu(-0.9, e_vals_all.cpu().numpy(), structure.Te, structure.Nocc, dvals = d_vals_all.cpu().numpy(), verb=False)
        mu0 = torch.tensor(mu0, device=device)
        print("Initial mu", mu0)
    else:
        mu0 = torch.tensor(0.0, device=device)
    dist.broadcast(mu0, 0)
    
    graphOnRank = None
    q_global = torch.zeros(structure.Nats, device=device)
    for (ch_structure) in per_part_data:
        q, D, _, _ = calc_q_on_rank(ch_structure, ch_structure.atom_ids, ch_structure.S, ch_structure.Z, ch_structure.KK, ch_structure.Q, ch_structure.e_vals, ch_structure.Znuc, mu0)
        q_global[ch_structure.ch[:ch_structure.core_size]] = q[:ch_structure.core_size]
        
        graphOnRank = adaptive_halo_expansion(
            graphOnRank,
            D.cpu(),
            GTHRESH, # gthresh
            structure.Nats,
            MAX_DEG, # maxDeg
            ch_structure.ch,
            ch_structure.ch[:ch_structure.core_size],
            ch_structure.hindex.cpu(),
            structure.coordinates.T.cpu().numpy(),
            structure.lattice_vecs.cpu().numpy(),
            nl.cpu().numpy(),
            alpha=0.7)

        graphOnRank = collect_graph_from_rho(
            graphOnRank,
            D.cpu(),
            GTHRESH,
            structure.Nats,
            MAX_DEG, # maxDeg
            ch_structure.ch,
            ch_structure.core_size,
            ch_structure.hindex.cpu())

    dist.all_reduce(q_global, op=dist.ReduceOp.SUM)

    graphOnRank_tensor = torch.tensor(graphOnRank, dtype=torch.int64, device=device) # fullGraphHalo
    dist.all_reduce(graphOnRank_tensor, op=dist.ReduceOp.SUM)

    fullGraph = add_graphs(graphOnRank_tensor.cpu().numpy(), fullGraph)
    fullGraph = symmetrize_graph(fullGraph)

    if dist.get_rank() == 0:
        positions_global = torch.stack((structure.RX, structure.RY, structure.RZ), )
        CALPHA_global, grid_dimensions = calculate_alpha_and_num_grids(structure.lattice_vecs.cpu().numpy(), dftorch_params['cutoff'], dftorch_params['Coulomb_acc'])
        PME_data_global = init_PME_data(grid_dimensions, structure.lattice_vecs, CALPHA_global, dftorch_params['PME_order'])
        #nbr_state = NeighborState(positions_global, structure.lattice_vecs, None, dftorch_params['cutoff'], is_dense=True, buffer=0.0, use_triton=False)
        #disps_global, dists_global, nbr_inds_global = calculate_dist_dips(positions_global, nbr_state, dftorch_params['cutoff'])
    else:
        positions_global, CALPHA_global, PME_data_global = [None]*3

    scf_error = torch.tensor(float('inf'), device=device)
    scf_iter = 0
    while (scf_error > dftorch_params['SCF_TOL']) and (scf_iter < dftorch_params['SCF_MAX_ITER']):
        scf_iter += 1
        if dist.get_rank() == 0:
            print("\nSCF iteration", scf_iter)

        for i in range(works_per_rank):
            ch[i], core_size[i], nh = get_coreHaloIndices(ch[i][:core_size[i]], fullGraph, NJUMPS)
            per_part_data[i].ch = ch[i]
            per_part_data[i].core_size = core_size[i]
            print('Rank', dist.get_rank(), 'has core and core-halo size:', core_size[i], len(ch[i]))

        q_global_old = q_global.clone()
        per_part_data, per_part_D, mu0, K0Res_global, graphOnRank = \
            scf_step(scf_iter, structure, positions_global, nl, disps_global, dists_global, CALPHA_global, PME_data_global, dftorch_params, \
                     q_global, mu0, ch, core_size, works_per_rank, graphOnRank, device)
        
        q_global = q_global - K0Res_global

        if dist.get_rank() == 0:
            scf_error = torch.norm(q_global - q_global_old)
            print("SCF error:", scf_error.item())
        dist.barrier()
        dist.broadcast(scf_error, 0)
                
        graphOnRank_tensor = torch.tensor(graphOnRank, dtype=torch.int64, device=device) # fullGraphHalo
        dist.all_reduce(graphOnRank_tensor, op=dist.ReduceOp.SUM)
        fullGraph = add_graphs(graphOnRank_tensor.cpu().numpy(), fullGraph)
        fullGraph = symmetrize_graph(fullGraph)

    
    e_band0, s_entropy = get_energy_on_rank(per_part_data, per_part_D, device)
    dist.all_reduce(e_band0, op=dist.ReduceOp.SUM)
    dist.all_reduce(s_entropy, op=dist.ReduceOp.SUM)
    e_entropy = -2 * structure.Te * s_entropy

    # Final global calcs: nuclear repulsion, PME, 
    if dist.get_rank() == 0:

        # nuclear repulsion
        e_repulsion, dVr = get_repulsion_energy(
            structure.const.R_rep_tensor, structure.const.rep_splines_tensor,
            structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
            6.0, structure.Nats, # repulsive_rcut
            structure.const, verbose=False)
        f_rep = dVr.sum(dim=2)
        
        ewald_e1, f_coul, dq_p1 =  calculate_PME_ewald(
            positions_global.detach().clone(),
            q_global,
            structure.lattice_vecs,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            dftorch_params['cutoff'],
            PME_data_global,
            hubbard_u = structure.Hubbard_U,
            atomtypes = structure.TYPE,
            screening = 1,
            calculate_forces=1,
            calculate_dq=1,)
        
        # Coulomb energy
        e_coul = ewald_e1 + 0.5 * torch.sum(q_global**2 * structure.Hubbard_U)

        print("Final e_entropy:", e_entropy.item())
        print("Final e_band0:", e_band0.item())
        print("Final e_repulsion:", e_repulsion.item())
        print("Final e_coul:", e_coul.item())

        # Dipole energy
        if structure.e_field.abs().sum() > 0.0:
            Efield_term = structure.RX * structure.e_field[0] + structure.RY * structure.e_field[1] + structure.RZ * structure.e_field[2]
            e_dipole = -torch.sum(q_global * Efield_term, dim=-1)
        else:
            e_dipole = 0.0

        e_tot = e_band0 + e_coul + e_dipole + e_entropy + e_repulsion
        print("Final e_tot:", e_tot.item())

    else:
        dq_p1 = torch.empty((structure.Nats,), device=device)
    dist.broadcast(dq_p1, 0)

    ### FORCES ###
    f_tot = get_forces_on_rank(structure, per_part_data, per_part_D, q_global, dq_p1, device) # on rank. 
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


def scf_step(scf_iter, structure, positions_global, nl, disps_global, dists_global, CALPHA_global, PME_data_global, dftorch_params,
             q_global, mu0, ch, core_size, works_per_rank, graphOnRank, device):

    if dist.get_rank() == 0:
        ewald_e1, forces1, CoulPot =  calculate_PME_ewald(
            positions_global.detach().clone(),
            q_global,
            structure.lattice_vecs,
            nl,
            disps_global,
            dists_global,
            CALPHA_global,
            dftorch_params['cutoff'],
            PME_data_global,
            hubbard_u = structure.Hubbard_U,
            atomtypes = structure.TYPE,
            screening = 1,
            calculate_forces=0,
            calculate_dq=1,)
    else:
        CoulPot = torch.empty((structure.Nats,), device=device)
    dist.broadcast(CoulPot, 0)

    per_part_data = []  # store (ch_structure, hindex, atom_ids, S, Z, KK, Q, e_vals, d_vals)
    e_vals_on_rank = torch.empty((0,), device=device)
    d_vals_on_rank = torch.empty((0,), device=device)
    for i in range(works_per_rank):
        ch_structure = get_subsy_on_rank(structure, dftorch_params, q_global[ch[i]], ch[i], core_size[i], CoulPot, device)
        ch_structure.e_vals, ch_structure.d_vals, ch_structure.Q = get_evals_dvals(ch_structure.Z, ch_structure.H, ch_structure.core_ao_slice)

        per_part_data.append((ch_structure))
        e_vals_on_rank = torch.cat((e_vals_on_rank, ch_structure.e_vals), dim=0)
        d_vals_on_rank = torch.cat((d_vals_on_rank, ch_structure.d_vals), dim=0)

    e_vals_all = gather_1d_to_rank0(e_vals_on_rank, device=device, src=0)
    d_vals_all = gather_1d_to_rank0(d_vals_on_rank, device=device, src=0)

    if dist.get_rank() == 0:
        mu0 = get_mu(mu0.cpu().numpy(), e_vals_all.cpu().numpy(), structure.Te, structure.Nocc, dvals = d_vals_all.cpu().numpy(), verb=False)
        mu0 = torch.tensor(mu0, device=device)
    else:
        mu0 = torch.tensor(0.0, device=device)
    dist.broadcast(mu0, 0)
    
    per_part_D = []
    K0Res_global = torch.zeros(structure.Nats, device=device)
    for ch_structure in per_part_data:
        q, D, f, K0Res = calc_q_on_rank(ch_structure, ch_structure.atom_ids, ch_structure.S, ch_structure.Z, ch_structure.KK, ch_structure.Q, ch_structure.e_vals, q_global[ch_structure.ch], mu0)
        per_part_D.append(D)
        ch_structure.f = f
        
        graphOnRank = adaptive_halo_expansion(
            graphOnRank,
            D.cpu(),
            GTHRESH, # gthresh
            structure.Nats,
            MAX_DEG, # maxDeg
            ch_structure.ch,
            ch_structure.ch[:ch_structure.core_size],
            ch_structure.hindex.cpu(),
            structure.coordinates.T.cpu().numpy(),
            structure.lattice_vecs.cpu().numpy(),
            nl.cpu().numpy(),
            alpha=0.7)

        graphOnRank = collect_graph_from_rho(
            graphOnRank,
            D.cpu(),
            GTHRESH,
            structure.Nats,
            MAX_DEG,
            ch_structure.ch,
            ch_structure.core_size,
            ch_structure.hindex.cpu())

        K0Res_global[ch_structure.ch[:ch_structure.core_size]] = K0Res[:ch_structure.core_size]

    dist.all_reduce(K0Res_global, op=dist.ReduceOp.SUM)
    
    #####
    if scf_iter > dftorch_params['KRYLOV_START']:
        K0Res_global = kernel_global(structure, positions_global, nl, disps_global, dists_global, CALPHA_global, PME_data_global, dftorch_params,
                  dftorch_params['KRYLOV_TOL'], K0Res_global, per_part_data, mu0, device)
        
    return per_part_data, per_part_D, mu0, K0Res_global, graphOnRank


class MDXL_Graph(MDXL):
    def __init__(self, const, temperature_K):
        super().__init__(None, const, temperature_K)
        # self.nl = nl
        # self.fullGraph = fullGraph

    def run(self, structure, dftorch_params, num_steps, dt,
            mu0, fullGraph, nbr_state, ch, core_size, works_per_rank, device,
            dump_interval=1, traj_filename='md_trj.xyz'):
        if dist.get_rank() == 0 and self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities(
                structure, temperature_K = self.temperature_K,
                remove_com=True, rescale_to_T=True, remove_angmom=True)
        else:
            self.VX = torch.empty_like(structure.RX)
            self.VY = torch.empty_like(structure.RY)
            self.VZ = torch.empty_like(structure.RZ)
        dist.broadcast(self.VX, 0)
        dist.broadcast(self.VY, 0)
        dist.broadcast(self.VZ, 0)
        q = structure.q.clone();
        if self.n is None:
            self.n = q; self.n_0 = q; self.n_1 = q; self.n_2 = q; self.n_3 = q; self.n_4 = q; self.n_5 = q;
        if self.K0Res is None:
            self.K0Res = 0.0 # structure.KK@(q-self.n)

        if dftorch_params['coul_method'] == 'PME':
            self.CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
                structure.lattice_vecs.cpu().numpy(), dftorch_params['cutoff'], dftorch_params['Coulomb_acc'])
            self.PME_data = init_PME_data(grid_dimensions, structure.lattice_vecs, self.CALPHA, dftorch_params['PME_order'])
        else:
            self.CALPHA = None
            self.PME_data = None

        if self.E_array is None:
            self.E_array   = torch.empty((0, ), device=structure.device)
            self.T_array   = torch.empty((0, ), device=structure.device)
            self.Ek_array  = torch.empty((0, ), device=structure.device)
            self.Ep_array  = torch.empty((0, ), device=structure.device)
            self.Res_array = torch.empty((0, ), device=structure.device)
        
        if dist.get_rank() == 0:
            self.EPOT = structure.e_tot
            
        for md_step in range(num_steps):
            for i in range(works_per_rank):
                ch[i], core_size[i], nh = get_coreHaloIndices(ch[i][:core_size[i]], fullGraph, NJUMPS)
                print('Rank', dist.get_rank(), 'has core and core-halo size:', core_size[i], len(ch[i]))

            mu0, graphOnRank = self.step(structure, dftorch_params, md_step, dt, dump_interval, traj_filename,
                      mu0, nbr_state, ch, core_size, works_per_rank, device)
            
            graphOnRank_tensor = torch.tensor(graphOnRank, dtype=torch.int64, device=device) # fullGraphHalo
            dist.all_reduce(graphOnRank_tensor, op=dist.ReduceOp.SUM)
            #fullGraph = graph_diff_and_update(graphOnRank_tensor.cpu().numpy(), fullGraph)
            #fullGraph = graph_diff_and_update(prevGraph, graphOnRank_tensor.cpu().numpy(), partsOnRank, comm)
            fullGraph = symmetrize_graph(graphOnRank_tensor.cpu().numpy())
            


    def step(self, structure, dftorch_params, md_step, dt, dump_interval, traj_filename,
             mu0, nbr_state, ch, core_size, works_per_rank, device):
        if dist.get_rank() == 0:
            if self.cuda_sync: torch.cuda.synchronize()
            start_time = time.perf_counter()
            print("########## Step = {:} ##########".format(md_step, ))
        
            self.EKIN = 0.5*self.MVV2KE*torch.sum(structure.Mnuc*(self.VX**2+self.VY**2+self.VZ**2))      # Kinetic energy in eV (MVV2KE: unit conversion)
            Temperature = (2/3)*self.KE2T*self.EKIN/structure.Nats              # Statistical temperature in Kelvin
            Energ = self.EKIN + self.EPOT;                                # Total Energy in eV, Total energy fluctuations Propto dt^2
            Time = md_step*dt;
            ResErr = torch.norm(structure.q - self.n)/(structure.Nats**0.5)                      # ResErr Propto dt^2

            self.E_array   = torch.cat((self.E_array,   Energ.detach().unsqueeze(0)), dim=0)
            self.T_array   = torch.cat((self.T_array,   Temperature.detach().unsqueeze(0)), dim=0)
            self.Ek_array  = torch.cat((self.Ek_array,  self.EKIN.detach().unsqueeze(0)), dim=0)
            self.Ep_array  = torch.cat((self.Ep_array,  self.EPOT.detach().unsqueeze(0)), dim=0)
            self.Res_array = torch.cat((self.Res_array, ResErr.detach().unsqueeze(0)), dim=0)

            if md_step%dump_interval == 0:
                comm_string = f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, Res = {ResErr:.6f}, mu = {mu0:.4f} eV\n"
                write_XYZ_trajectory(traj_filename, structure, comm_string, step=md_step)
            self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[0]/structure.Mnuc) - self.fric*self.VX;      # First 1/2 of Leapfrog step
            self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[1]/structure.Mnuc) - self.fric*self.VY;      # F2V: Unit conversion
            self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[2]/structure.Mnuc) - self.fric*self.VZ;      # -c*V c>0 => Fricition
            if structure.LBox is not None:
                structure.RX = (structure.RX + dt*self.VX) % structure.LBox[0]
                structure.RY = (structure.RY + dt*self.VY) % structure.LBox[1]
                structure.RZ = (structure.RZ + dt*self.VZ) % structure.LBox[2]
            else:
                structure.RX = (structure.RX + dt*self.VX)
                structure.RY = (structure.RY + dt*self.VY)
                structure.RZ = (structure.RZ + dt*self.VZ)
            structure.coordinates = torch.stack((structure.RX, structure.RY, structure.RZ), )

            if self.cuda_sync: torch.cuda.synchronize()
            tic2_1 = time.perf_counter()

            self.n = 2*self.n_0 - self.n_1 - self.kappa*self.K0Res + \
                self.alpha*(self.C0*self.n_0 + self.C1*self.n_1 + self.C2*self.n_2 + self.C3*self.n_3 + self.C4*self.n_4 + self.C5*self.n_5)
            self.n_5 = self.n_4; self.n_4 = self.n_3; self.n_3 = self.n_2; self.n_2 = self.n_1; self.n_1 = self.n_0; self.n_0 = self.n

            positions = torch.stack((structure.RX, structure.RY, structure.RZ))
            nbr_state.update(positions)
            disps, dists, nl = calculate_dist_dips(positions, nbr_state, dftorch_params['cutoff'])
            nl_shape = torch.tensor(list(nl.shape), dtype=torch.int64, device=device)

            ewald_e1, forces1, CoulPot =  calculate_PME_ewald(
                            positions.detach().clone(),
                            self.n,
                            structure.lattice_vecs,
                            nl,
                            disps,
                            dists,
                            self.CALPHA,
                            dftorch_params['cutoff'],
                            self.PME_data,
                            hubbard_u = structure.Hubbard_U,
                            atomtypes = structure.TYPE,
                            screening = 1,
                            calculate_forces=1,
                            calculate_dq=1,)
            
            if self.cuda_sync: torch.cuda.synchronize()
            print("PME: {:.3f} s".format(time.perf_counter()-tic2_1))
            
        else:
            positions = None
            disps = None; dists = None;
            nl_shape = torch.empty((2,), dtype=torch.int64, device=device)
            self.n = torch.empty((structure.Nats,), device=device)
            CoulPot = torch.empty((structure.Nats,), device=device)

        tic2_1 = time.perf_counter()

        dist.broadcast(nl_shape, 0)
        dist.broadcast(self.n, 0)
        dist.broadcast(CoulPot, 0)
        dist.broadcast(structure.RX, 0)
        dist.broadcast(structure.RY, 0)
        dist.broadcast(structure.RZ, 0)
        dist.broadcast(structure.coordinates, 0)

        if WORLD_RANK != 0:
            nl = torch.empty(tuple(int(x) for x in nl_shape.tolist()), dtype=torch.int64, device=device)

        dist.broadcast(nl, 0)

        per_part_data = []  # store (ch_structure, hindex, atom_ids, S, Z, KK, Q, e_vals, d_vals)
        e_list = []
        d_list = []
        # e_vals_on_rank = torch.empty((0,), device=device)
        # d_vals_on_rank = torch.empty((0,), device=device)
        for i in range(works_per_rank):
            ch_structure = get_subsy_on_rank(structure, dftorch_params, self.n[ch[i]], ch[i], core_size[i], CoulPot, device)
            ch_structure.e_vals, ch_structure.d_vals, ch_structure.Q = \
                get_evals_dvals(ch_structure.Z, ch_structure.H, ch_structure.core_ao_slice)

            per_part_data.append((ch_structure))
            e_list.append(ch_structure.e_vals)
            d_list.append(ch_structure.d_vals)
        e_vals_on_rank = torch.cat(e_list)
        d_vals_on_rank = torch.cat(d_list)
        
        e_vals_all = gather_1d_to_rank0(e_vals_on_rank, device=device, src=0)
        d_vals_all = gather_1d_to_rank0(d_vals_on_rank, device=device, src=0)

        if dist.get_rank() == 0:
            mu0 = get_mu(mu0.cpu().numpy(), e_vals_all.cpu().numpy(), structure.Te, structure.Nocc, dvals = d_vals_all.cpu().numpy(), verb=False)
            mu0 = torch.tensor(mu0, device=device)
            if self.cuda_sync: torch.cuda.synchronize()
            print("mu0: {:.3f} s".format(time.perf_counter()-tic2_1))
        else:
            mu0 = torch.tensor(0.0, device=device)
        dist.broadcast(mu0, 0)

        tic2_1 = time.perf_counter()

        per_part_D = []
        q_global = torch.zeros(structure.Nats, device=device)
        self.K0Res = torch.zeros(structure.Nats, device=device)
        graphOnRank = None
        for ch_structure in per_part_data:
            q, D, f, K0Res = calc_q_on_rank(ch_structure, ch_structure.atom_ids, ch_structure.S, ch_structure.Z,
                                        ch_structure.KK, ch_structure.Q, ch_structure.e_vals, self.n[ch_structure.ch], mu0)
            
            q_global[ch_structure.ch[:ch_structure.core_size]] = q[:ch_structure.core_size]
            per_part_D.append(D)
            ch_structure.q = q
            ch_structure.f = f
            
            graphOnRank = adaptive_halo_expansion(
                graphOnRank,
                D.cpu(),
                GTHRESH, # gthresh
                structure.Nats,
                MAX_DEG, # maxDeg
                ch_structure.ch,
                ch_structure.ch[:ch_structure.core_size],
                ch_structure.hindex.cpu(),
                structure.coordinates.T.cpu().numpy(),
                structure.lattice_vecs.cpu().numpy(),
                nl.cpu().numpy(),
                alpha=0.7)

            graphOnRank = collect_graph_from_rho(
                graphOnRank,
                D.cpu(),
                GTHRESH,
                structure.Nats,
                MAX_DEG,
                ch_structure.ch,
                ch_structure.core_size,
                ch_structure.hindex.cpu())

            self.K0Res[ch_structure.ch[:ch_structure.core_size]] = K0Res[:ch_structure.core_size]
        
        dist.all_reduce(q_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.K0Res, op=dist.ReduceOp.SUM)

        if self.cuda_sync: torch.cuda.synchronize()
        if dist.get_rank() == 0: print("H1: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic2_1 = time.perf_counter()

        if self.NoRank:
            1
        else:
            self.K0Res = kernel_global(structure, positions, nl, disps, dists, self.CALPHA, self.PME_data, dftorch_params,
                    dftorch_params['KRYLOV_TOL_MD'], self.K0Res, per_part_data, mu0, device)
        
        if self.cuda_sync: torch.cuda.synchronize()
        if dist.get_rank() == 0: print("KER: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic2_1 = time.perf_counter()

        ### ENERGY ###
        e_band0, s_entropy = get_energy_on_rank(per_part_data, per_part_D, device)
        dist.all_reduce(e_band0, op=dist.ReduceOp.SUM)
        dist.all_reduce(s_entropy, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            # nuclear repulsion
            e_repulsion, dVr = get_repulsion_energy(
                structure.const.R_rep_tensor, structure.const.rep_splines_tensor,
                structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
                6.0, structure.Nats, # repulsive_rcut
                structure.const, verbose=False)
            f_rep = dVr.sum(dim=2)

            # Coulomb energy
            e_coul = 0.5 * (2*q_global-self.n) @ CoulPot + 0.5 * torch.sum((2.0*q_global - self.n) * structure.Hubbard_U * self.n)
            
            # Dipole energy
            if structure.e_field.abs().sum() > 0.0:
                efield_term = structure.RX * structure.e_field[0] + structure.RY * structure.e_field[1] + structure.RZ * structure.e_field[2]
                e_dipole = -torch.sum(q_global * efield_term, dim=-1)
            else:
                e_dipole = 0.0

            e_entropy = -2 * structure.Te * s_entropy

            e_tot = e_band0 + e_coul + e_dipole + e_entropy + e_repulsion

            f_coul = forces1 * (2*q_global/self.n - 1.0)
        else:
            e_tot = torch.tensor(0.0, device=device)
        dist.broadcast(e_tot, 0)
        structure.e_tot = e_tot
        self.EPOT = structure.e_tot
        
        ### FORCES ###
        # FScoul + Fband0 + FPulay + Fdipole + FSdipole
        # !!! Fdipole will use n, but q should be used
        f_tot = get_forces_on_rank(structure, per_part_data, per_part_D, self.n, CoulPot, device) # on rank. 
        dist.all_reduce(f_tot, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            f_tot = f_tot + f_rep + f_coul
            structure.q = q_global
            structure.f_tot = f_tot

            self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[0]/structure.Mnuc) - self.fric*self.VX;      # Integrate second 1/2 of leapfrog step
            self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[1]/structure.Mnuc) - self.fric*self.VY;      # - c*V  c > 0 => friction
            self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[2]/structure.Mnuc) - self.fric*self.VZ;

            if self.cuda_sync: torch.cuda.synchronize()
            print("F AND E: {:.3f} s".format(time.perf_counter()-tic2_1))
            tic2_1 = time.perf_counter()

            print("ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}, t = {:.1f} s".format(Energ, self.EPOT.item(), self.EKIN.item(),  Temperature.item(), ResErr.item(), time.perf_counter()-start_time ))
            print(torch.cuda.memory_allocated() / 1e9, 'GB\n')
            print()

        return mu0, graphOnRank

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    # choose device for collectives
    if backend == "nccl":
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")

    nparts = 8
    structure, dftorch_params = prepare_structure(device)
    if dftorch_params['cutoff']  < dftorch_params['graph_cutoff']:
        raise ValueError("Coulomb cutoff must be greater than or equal to graph cutoff for this implementation.")

    if WORLD_RANK == 0:
        positions = torch.stack((structure.RX, structure.RY, structure.RZ), )

        nbr_state = NeighborState(positions, structure.lattice_vecs, None, dftorch_params['cutoff'], is_dense=True, buffer=1.0, use_triton=False)
        disps, dists, nl = calculate_dist_dips(positions, nbr_state, dftorch_params['cutoff'])

        if dftorch_params['graph_cutoff'] < dftorch_params['cutoff']:
            nl_init = torch.where(
                        (dists > dftorch_params['graph_cutoff']) | (dists == 0.0), -1, nl
                        )
            nl_init = nl_init.sort(dim=1, descending=True)[0]
            nl_init = nl_init[:, : torch.max(torch.sum(nl_init != -1, dim=1))]

        elif dftorch_params['graph_cutoff']  > dftorch_params['cutoff']:
            raise ValueError("Coulomb cutoff must be greater than or equal to graph cutoff for this implementation.")

        # num_neighbors = torch.sum(nl != -1, dim=1)
        # nl = torch.cat((num_neighbors.unsqueeze(1), nl), dim=1)

        #nl = nl.cpu().numpy()


        fullGraph, eweights = get_initial_graph(positions.T.cpu().numpy(), nl_init.cpu().numpy(), dftorch_params['graph_cutoff'], MAX_DEG, structure.LBox.cpu().numpy())
        parts  = graph_partition(structure, structure, fullGraph, "SpectralClustering", nparts, positions.T.cpu().numpy(), verb=False)
        
        partsCoreHalo = []
        numCores = []
        print("\nCore and halos indices for every part:")
        for i in range(nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, NJUMPS)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)

        numCores = torch.tensor(numCores, dtype=torch.int64, device=device)
        #nl = torch.tensor(nl, dtype=torch.int64, device=device)
        fullGraph = torch.tensor(fullGraph, dtype=torch.int64, device=device)
        flat, offsets = pack_lol_int64(partsCoreHalo)
        #disps_shape = torch.tensor(list(disps.shape), dtype=torch.int64, device=device)
        #dists_shape = torch.tensor(list(dists.shape), dtype=torch.int64, device=device)
        nl_shape = torch.tensor(list(nl.shape), dtype=torch.int64, device=device)
        fullGraph_shape = torch.tensor(list(fullGraph.shape), dtype=torch.int64, device=device)
    else:
        partsCoreHalo = None
        numCores = torch.empty((nparts,), dtype=torch.int64, device=device)
        nbr_state = None
        disps, dists = None, None
        flat, offsets = None, None
        #disps_shape = torch.empty((3,), dtype=torch.int64, device=device)
        #dists_shape = torch.empty((3,), dtype=torch.int64, device=device)
        nl_shape = torch.empty((2,), dtype=torch.int64, device=device)
        fullGraph_shape = torch.empty((2,), dtype=torch.int64, device=device)
    dist.broadcast(numCores, 0)
    #dist.broadcast(disps_shape, 0)
    #dist.broadcast(dists_shape, 0)
    dist.broadcast(nl_shape, 0)
    dist.broadcast(fullGraph_shape, 0)

    if WORLD_RANK != 0:
        #disps = torch.empty(tuple(int(x) for x in disps_shape.tolist()), device=device)
        nl = torch.empty(tuple(int(x) for x in nl_shape.tolist()), dtype=torch.int64, device=device)
        fullGraph = torch.empty(tuple(int(x) for x in fullGraph_shape.tolist()), dtype=torch.int64, device=device)

    #dist.broadcast(disps, 0)
    dist.broadcast(nl, 0)
    dist.broadcast(fullGraph, 0)

    #nl = nl.cpu()

    flat = bcast_1d_int64(flat, device=device, src=0)
    offsets = bcast_1d_int64(offsets, device=device, src=0)
    partsCoreHalo = unpack_lol_int64(flat.cpu(), offsets.cpu())

    works_per_rank = nparts // WORLD_SIZE
    if nparts % WORLD_SIZE != 0:
        raise ValueError("nparts must be divisible by WORLD_SIZE.")
    if WORLD_SIZE > nparts:
        raise ValueError("WORLD_SIZE must be less than or equal to nparts.")
    
    cur_rank = dist.get_rank()
    start = cur_rank * works_per_rank
    end = start + works_per_rank
    mu0, fullGraph = run(backend, structure, dftorch_params, fullGraph,
        partsCoreHalo[start:end], numCores[start:end], nbr_state, disps, dists, nl, works_per_rank, device)
    
    torch.manual_seed(0)
    temperature_K = torch.tensor(500.0, device=structure.device)
    mdDriver = MDXL_Graph(structure.const, temperature_K=temperature_K)
    # Set number of steps, time step (fs), dump interval and trajectory filename
    mdDriver.run(structure, dftorch_params, num_steps=100, dt=0.3,
                 mu0=mu0, fullGraph=fullGraph, nbr_state=nbr_state, ch=partsCoreHalo[start:end], core_size=numCores[start:end],
                 works_per_rank=works_per_rank, device=device,
                 dump_interval=5, traj_filename='md_trj.xyz')

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    start_time1 = time.perf_counter()
    try:
        init_processes(backend=args.backend)
    finally:
        # avoid NCCL resource leak warning on exit
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    print("  t TOTAL {:.1f} s\n".format( time.perf_counter()-start_time1 ))
