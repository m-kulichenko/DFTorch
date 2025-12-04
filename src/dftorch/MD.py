from .ESDriver import ESDriverBatch
from .Energy import EnergyShadow
from .ForcesBatch import forces_shadow_batch
from .Forces import forces_shadow, forces_shadow_pme
from .Kernel_Fermi import Kernel_Fermi
from .XLTools import calc_q, kernel_update_lr, calc_q_batch, kernel_update_lr_batch
import torch
#from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from .ewald_pme import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
#from sedacs.neighbor_list import NeighborState, calculate_displacement
from .ewald_pme.neighbor_list import NeighborState
from typing import Any, Tuple, Optional
import time
from .io import write_XYZ_trajectory
from dftorch.Tools import calculate_dist_dips


class MDXL:
    def __init__(self, es_driver: ESDriverBatch, const,
                 temperature_K: float):
        
        self.NoRank = False
        self.do_full_kernel = False
        self.const = const
        self.fric = 0.0;
        self.F2V = 0.01602176487/1.660548782;
        self.MVV2KE = 166.0538782/1.602176487;
        self.KE2T = 1/0.000086173435;
        self.C0 = -6; self.C1 = 14; self.C2 = -8; self.C3 = -3; self.C4 = 4; self.C5 = -1; # Coefficients for modified Verlet integration
        self.kappa = 1.82; self.alpha = 0.018;                         # Coefficients for modified Verlet integration

        self.es_driver = es_driver
        self.temperature_K = temperature_K
        self.n = None; self.n_0 = None; self.n_1 = None; self.n_2 = None; self.n_3 = None; self.n_4 = None; self.n_5 = None;
        self.VX, self.VY, self.VZ = None, None, None
        self.K0Res     = None
        self.E_array   = None
        self.T_array   = None
        self.Ek_array  = None
        self.Ep_array  = None
        self.Res_array = None

        self.cuda_sync = True

    def run(self, structure, dftorch_params, num_steps, dt, dump_interval=1, traj_filename='md_trj.xyz'):

        if self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities(
                structure, temperature_K = self.temperature_K,
                remove_com=True, rescale_to_T=True, remove_angmom=True)
        q = structure.q.clone();
        if self.n is None:
            self.n = q; self.n_0 = q; self.n_1 = q; self.n_2 = q; self.n_3 = q; self.n_4 = q; self.n_5 = q;
        if self.K0Res is None:
            self.K0Res = structure.KK@(q-self.n)

        # Generate atom index for each orbital
        self.atom_ids = torch.repeat_interleave(torch.arange(len(structure.n_orbitals_per_atom), device=structure.device),
                                                structure.n_orbitals_per_atom)
        self.Hubbard_U_gathered = structure.Hubbard_U[self.atom_ids]
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
        
        self.EPOT = structure.e_tot
        for md_step in range(num_steps):
            self.step(structure, dftorch_params, md_step, dt, dump_interval, traj_filename)


    def step(self, structure, dftorch_params, md_step, dt, dump_interval, traj_filename):
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
            comm_string = f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, Res = {ResErr:.6f}, mu = {structure.mu0:.4f} eV\n"
            write_XYZ_trajectory(traj_filename, structure, comm_string, step=md_step)
        self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[0]/structure.Mnuc) - self.fric*self.VX;      # First 1/2 of Leapfrog step
        self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[1]/structure.Mnuc) - self.fric*self.VY;      # F2V: Unit conversion
        self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[2]/structure.Mnuc) - self.fric*self.VZ;      # -c*V c>0 => Fricition
        # update positions and translate coordinates if go beyond box. Apply periodic boundary conditions
        structure.RX = (structure.RX + dt*self.VX) % structure.LBox[0]
        structure.RY = (structure.RY + dt*self.VY) % structure.LBox[1]
        structure.RZ = (structure.RZ + dt*self.VZ) % structure.LBox[2]

        if self.cuda_sync: torch.cuda.synchronize()
        tic2_1 = time.perf_counter()

        self.es_driver(structure, self.const, do_scf=False)
        self.n = 2*self.n_0 - self.n_1 - self.kappa*self.K0Res + \
            self.alpha*(self.C0*self.n_0 + self.C1*self.n_1 + self.C2*self.n_2 + self.C3*self.n_3 + self.C4*self.n_4 + self.C5*self.n_5)
        self.n_5 = self.n_4; self.n_4 = self.n_3; self.n_3 = self.n_2; self.n_2 = self.n_1; self.n_1 = self.n_0; self.n_0 = self.n

        if self.cuda_sync: torch.cuda.synchronize()
        print("H0: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic2_1 = time.perf_counter()

        if dftorch_params['coul_method'] == 'PME':
            nbr_state = NeighborState(
                torch.stack((structure.RX, structure.RY, structure.RZ)), structure.lattice_vecs, None,
                dftorch_params['cutoff'], is_dense=True, buffer=0.0, use_triton=False)
            disps, dists, nbr_inds = calculate_dist_dips(
                torch.stack((structure.RX, structure.RY, structure.RZ)), nbr_state, dftorch_params['cutoff'])

            _, forces1, CoulPot =  calculate_PME_ewald(torch.stack((structure.RX, structure.RY, structure.RZ)),
                            self.n,
                            structure.lattice_vecs,
                            nbr_inds,
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
        else:
            CoulPot = structure.C @ self.n
            nbr_inds = None
            disps = None
            dists = None

        structure.q, structure.H, structure.Hcoul, structure.D, Dorth, \
        Q, e, structure.f, structure.mu0 = calc_q(
            structure.H0, self.Hubbard_U_gathered, self.n[self.atom_ids], CoulPot[self.atom_ids],
            structure.S, structure.Z, structure.Te, structure.Nocc, structure.Znuc, self.atom_ids)

        if self.cuda_sync: torch.cuda.synchronize()
        print("H1: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic3 = time.perf_counter()

        # Update Kernel
        Res = structure.q - self.n
        if md_step%100000 == 0 and self.do_full_kernel:
            KK,_ = Kernel_Fermi(structure, structure.mu0,structure.Te,structure.Nats,structure.H,C,S,Z,Q,e)
            self.K0Res = KK@Res;
        elif self.NoRank:
            self.K0Res = -dftorch_params['SCF_ALPHA']*Res
        else: # Preconditioned Low-Rank Krylov SCF acceleration
            self.K0Res = kernel_update_lr(
                structure.RX, structure.RY, structure.RZ, structure.lattice_vecs, structure.TYPE,
                structure.Nats, structure.Hubbard_U, dftorch_params, dftorch_params['KRYLOV_TOL_MD'],
                structure.KK.clone(), Res, structure.q, structure.S, structure.Z, self.PME_data,
                self.atom_ids, Q, e, structure.mu0, structure.Te, structure.C,
                nbr_inds, disps, dists, self.CALPHA)
            
        if self.cuda_sync: torch.cuda.synchronize()
        print("KER: {:.3f} s".format(time.perf_counter()-tic3))
        tic4 = time.perf_counter()

        if dftorch_params['coul_method'] == 'PME':
            structure.e_elec_tot, structure.e_band0, structure.e_coul, structure.e_dipole, \
            structure.e_entropy, structure.s_ent = \
            EnergyShadow(
                structure.H0, structure.Hubbard_U, structure.e_field,
                structure.D0, None, CoulPot, structure.D, structure.q, self.n,
                structure.RX, structure.RY, structure.RZ, structure.f, structure.Te)

            # no f_coul in PME forces_shadow_pme. Done in calculate_PME_ewald
            structure.f_tot, _, structure.f_band0, \
            structure.f_dipole, structure.f_pulay, \
            structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
            forces_shadow_pme(
                structure.H, structure.Z, CoulPot, structure.D, structure.D0, structure.dH0, structure.dS,
                structure.dVr, structure.e_field, structure.Hubbard_U, structure.q, self.n,
                structure.RX, structure.RY, structure.RZ, structure.Nats, self.const, structure.TYPE)
            structure.f_coul = forces1 * (2*structure.q/self.n - 1.0)
            structure.f_tot = structure.f_tot + structure.f_coul
        else:
            structure.e_elec_tot, structure.e_band0, structure.e_coul, structure.e_dipole, \
            structure.e_entropy, structure.s_ent = \
            EnergyShadow(
                structure.H0, structure.Hubbard_U, structure.e_field,
                structure.D0, structure.C, None, structure.D, structure.q, self.n,
                structure.RX, structure.RY, structure.RZ, structure.f, structure.Te)
                        
            structure.f_tot, structure.f_coul, structure.f_band0, \
            structure.f_dipole, structure.f_pulay, \
            structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
            forces_shadow(
                structure.H, structure.Z, structure.C, structure.D, structure.D0, structure.dH0, structure.dS,
                structure.dCC, structure.dVr, structure.e_field, structure.Hubbard_U, structure.q, self.n,
                structure.RX, structure.RY, structure.RZ, structure.Nats, self.const, structure.TYPE)
        
        structure.e_tot = structure.e_elec_tot + structure.e_repulsion
        self.EPOT = structure.e_tot

        self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[0]/structure.Mnuc) - self.fric*self.VX;      # Integrate second 1/2 of leapfrog step
        self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[1]/structure.Mnuc) - self.fric*self.VY;      # - c*V  c > 0 => friction
        self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[2]/structure.Mnuc) - self.fric*self.VZ;

        if self.cuda_sync: torch.cuda.synchronize()
        print("F AND E: {:.3f} s".format(time.perf_counter()-tic4))

        print("ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}, t = {:.1f} s".format(Energ, self.EPOT.item(), self.EKIN.item(),  Temperature.item(), ResErr.item(), time.perf_counter()-start_time ))
        print(torch.cuda.memory_allocated() / 1e9, 'GB\n')
        print()

class MDXLBatch:
    def __init__(self, es_driver: ESDriverBatch, const,
                 temperature_K: float):
        
        self.NoRank = False
        self.do_full_kernel = False
        self.const = const
        self.fric = 0.0;
        self.F2V = 0.01602176487/1.660548782;
        self.MVV2KE = 166.0538782/1.602176487;
        self.KE2T = 1/0.000086173435;
        self.C0 = -6; self.C1 = 14; self.C2 = -8; self.C3 = -3; self.C4 = 4; self.C5 = -1; # Coefficients for modified Verlet integration
        self.kappa = 1.82; self.alpha = 0.018;                         # Coefficients for modified Verlet integration

        self.es_driver = es_driver
        self.temperature_K = temperature_K
        self.n = None; self.n_0 = None; self.n_1 = None; self.n_2 = None; self.n_3 = None; self.n_4 = None; self.n_5 = None;
        self.VX, self.VY, self.VZ = None, None, None
        self.K0Res     = None
        self.E_array   = None
        self.T_array   = None
        self.Ek_array  = None
        self.Ep_array  = None
        self.Res_array = None

        self.cuda_sync = True

    def run(self, structure, dftorch_params, num_steps, dt, dump_interval=1, traj_filename='md_trj.xyz'):

        if self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities_batch(
                structure, temperature_K=self.temperature_K,
                remove_com=True, rescale_to_T=True, remove_angmom=True)
        q = structure.q.clone();
        if self.n is None:
            self.n = q; self.n_0 = q; self.n_1 = q; self.n_2 = q; self.n_3 = q; self.n_4 = q; self.n_5 = q;
        if self.K0Res is None:
            self.K0Res = torch.matmul(structure.KK, (q - self.n).unsqueeze(-1)).squeeze(-1)

        # Generate atom index for each orbital
        counts = structure.n_orbitals_per_atom          # shape (B, N)
        cum_counts = torch.cumsum(counts, dim=1)        # cumulative sums per batch
        total_orbs = int(cum_counts[0, -1].item())
        r = torch.arange(total_orbs, device=counts.device).expand(counts.size(0), -1)  # (B, total_orbs)
        # For each orbital position r[b,k], find first atom index whose cumulative count exceeds r[b,k]
        self.atom_ids = (r.unsqueeze(2) < cum_counts.unsqueeze(1)).int().argmax(dim=2)      # (B, total_orbs)
        self.Hubbard_U_gathered = structure.Hubbard_U.gather(1, self.atom_ids)
        self.PME_data = None
        self.nbr_inds = None
        self.disps = None
        self.dists = None
        self.CALPHA = None

        if self.E_array is None:
            self.E_array   = torch.empty((0, structure.batch_size), device=structure.device)
            self.T_array   = torch.empty((0, structure.batch_size), device=structure.device)
            self.Ek_array  = torch.empty((0, structure.batch_size), device=structure.device)
            self.Ep_array  = torch.empty((0, structure.batch_size), device=structure.device)
            self.Res_array = torch.empty((0, structure.batch_size), device=structure.device)
        
        self.EPOT = structure.e_tot
        for md_step in range(num_steps):
            self.step(structure, dftorch_params, md_step, dt, dump_interval, traj_filename)


    def step(self, structure, dftorch_params, md_step, dt, dump_interval, traj_filename):
        if self.cuda_sync: torch.cuda.synchronize()
        start_time = time.perf_counter()
        print("########## Step = {:} ##########".format(md_step, ))
        
        self.EKIN = 0.5*self.MVV2KE*torch.sum(structure.Mnuc*(self.VX**2+self.VY**2+self.VZ**2),dim=1)      # Kinetic energy in eV (MVV2KE: unit conversion)
        Temperature = (2/3)*self.KE2T*self.EKIN/structure.Nats              # Statistical temperature in Kelvin
        Energ = self.EKIN + self.EPOT;                                # Total Energy in eV, Total energy fluctuations Propto dt^2
        Time = md_step*dt;
        ResErr = torch.norm(structure.q - self.n, dim=1)/(structure.Nats**0.5)                      # ResErr Propto dt^2

        self.E_array   = torch.cat((self.E_array,   Energ.detach().unsqueeze(0)), dim=0)
        self.T_array   = torch.cat((self.T_array,   Temperature.detach().unsqueeze(0)), dim=0)
        self.Ek_array  = torch.cat((self.Ek_array,  self.EKIN.detach().unsqueeze(0)), dim=0)
        self.Ep_array  = torch.cat((self.Ep_array,  self.EPOT.detach().unsqueeze(0)), dim=0)
        self.Res_array = torch.cat((self.Res_array, ResErr.detach().unsqueeze(0)), dim=0)

        if md_step%dump_interval == 100000:
            comm_string = f"Etot = {Energ:.6f} eV, Epot = {EPOT:.6f} eV, Ekin = {EKIN:.6f} eV, T = {Temperature:.2f} K, Res = {ResErr:.6f}, mu = {structure.mu0:.4f} eV\n"
            write_XYZ_trajectory(traj_filename, structure, comm_string, step=md_step)
        self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[:, 0]/structure.Mnuc) - self.fric*self.VX;      # First 1/2 of Leapfrog step
        self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[:, 1]/structure.Mnuc) - self.fric*self.VY;      # F2V: Unit conversion
        self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[:, 2]/structure.Mnuc) - self.fric*self.VZ;      # -c*V c>0 => Fricition
        # update positions and translate coordinates if go beyond box. Apply periodic boundary conditions
        structure.RX = (structure.RX + dt*self.VX) % structure.LBox[:,0].unsqueeze(-1); 
        structure.RY = (structure.RY + dt*self.VY) % structure.LBox[:,1].unsqueeze(-1);
        structure.RZ = (structure.RZ + dt*self.VZ) % structure.LBox[:,2].unsqueeze(-1);

        if self.cuda_sync: torch.cuda.synchronize()
        tic2_1 = time.perf_counter()

        self.es_driver(structure, self.const, do_scf=False)
        self.n = 2*self.n_0 - self.n_1 - self.kappa*self.K0Res + \
            self.alpha*(self.C0*self.n_0 + self.C1*self.n_1 + self.C2*self.n_2 + self.C3*self.n_3 + self.C4*self.n_4 + self.C5*self.n_5)
        self.n_5 = self.n_4; self.n_4 = self.n_3; self.n_3 = self.n_2; self.n_2 = self.n_1; self.n_1 = self.n_0; self.n_0 = self.n

        if self.cuda_sync: torch.cuda.synchronize()
        print("H0: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic2_1 = time.perf_counter()

        CoulPot = torch.matmul(structure.C, self.n.unsqueeze(-1)).squeeze(-1)
        structure.q, structure.H, structure.Hcoul, structure.D, Dorth, \
        Q, e, structure.f, structure.mu0 = calc_q_batch(
            structure.H0, self.Hubbard_U_gathered, self.n.gather(1, self.atom_ids),
            CoulPot.gather(1, self.atom_ids),
            structure.S, structure.Z, structure.Te, structure.Nocc, structure.Znuc, self.atom_ids)

        if self.cuda_sync: torch.cuda.synchronize()
        print("H1: {:.3f} s".format(time.perf_counter()-tic2_1))
        tic3 = time.perf_counter()

        # Update Kernel
        Res = structure.q - self.n
        if md_step%10000 == 0 and self.do_full_kernel:
            KK,_ = Kernel_Fermi(structure, structure.mu0,structure.Te,structure.Nats,structure.H,C,S,Z,Q,e)
            self.K0Res = KK@Res;
        elif self.NoRank:
            self.K0Res = -dftorch_params['SCF_ALPHA']*Res
        else: # Preconditioned Low-Rank Krylov SCF acceleration
            self.K0Res = kernel_update_lr_batch(
                structure.Nats, self.Hubbard_U_gathered, dftorch_params, dftorch_params['KRYLOV_TOL_MD'],
                structure.KK.clone(), Res, structure.q, structure.S, structure.Z, self.PME_data,
                self.atom_ids, Q, e, structure.mu0, structure.Te, structure.C,
                self.nbr_inds, self.disps, self.dists, self.CALPHA)
            
        if self.cuda_sync: torch.cuda.synchronize()
        print("KER: {:.3f} s".format(time.perf_counter()-tic3))
        tic4 = time.perf_counter()

        structure.e_elec_tot, structure.e_band0, structure.e_coul, structure.e_dipole, \
        structure.e_entropy, structure.s_ent = EnergyShadow(
            structure.H0, structure.Hubbard_U, structure.e_field,
            structure.D0, structure.C, None, structure.D, structure.q, self.n,
            structure.RX, structure.RY, structure.RZ, structure.f, structure.Te)
        
        structure.e_tot = structure.e_elec_tot + structure.e_repulsion
        self.EPOT = structure.e_tot
        
        structure.f_tot, structure.f_coul, structure.f_band0, \
        structure.f_dipole, structure.f_pulay, \
        structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
        forces_shadow_batch(
            structure.H, structure.Z, structure.C, structure.D, structure.D0, structure.dH0, structure.dS, structure.dCC, structure.dVr,
            structure.e_field, structure.Hubbard_U, structure.q, self.n,
            structure.RX, structure.RY, structure.RZ, structure.Nats, self.const, structure.TYPE)
        
        self.VX = self.VX + 0.5*dt*(self.F2V*structure.f_tot[:,0]/structure.Mnuc) - self.fric*self.VX;      # Integrate second 1/2 of leapfrog step
        self.VY = self.VY + 0.5*dt*(self.F2V*structure.f_tot[:,1]/structure.Mnuc) - self.fric*self.VY;      # - c*V  c > 0 => friction
        self.VZ = self.VZ + 0.5*dt*(self.F2V*structure.f_tot[:,2]/structure.Mnuc) - self.fric*self.VZ;

        if self.cuda_sync: torch.cuda.synchronize()
        print("F AND E: {:.3f} s".format(time.perf_counter()-tic4))

        for b in range(structure.batch_size):
            print(
                "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}, t = {:.1f} s".format(
                    Energ[b].item(),
                    self.EPOT[b].item(),
                    self.EKIN[b].item(),
                    Temperature[b].item(),
                    ResErr[b].item(),
                    time.perf_counter() - start_time,)
                )
        print(torch.cuda.memory_allocated() / 1e9, 'GB\n')
        print()


def initialize_velocities(
    structure: Any,
    temperature_K: float,
    masses_amu: Optional[torch.Tensor] = None,
    remove_com: bool = True,
    rescale_to_T: bool = True,
    remove_angmom: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize atomic velocities from a Maxwell–Boltzmann distribution at temperature_K.
    - Units: velocities in Å/fs; masses in amu; resulting kinetic energy EKIN in eV via
      EKIN = 0.5 * MVV2KE * sum_i m_i |v_i|^2.
    - Optionally enforces zero total linear momentum and zero net angular momentum.
    - Optionally rescales to match the exact target temperature after constraints.

    Parameters
    ----------
    structure : object
        Must provide RX, RY, RZ (positions, tensors with device/dtype), Mnuc (amu), Nats (int).
    temperature_K : float
        Target temperature in Kelvin.
    masses_amu : torch.Tensor, optional
        Shape (N,). Atomic masses in amu. Defaults to structure.Mnuc.
    remove_com : bool
        If True, subtract center-of-mass velocity (zero total momentum).
    rescale_to_T : bool
        If True, rescale velocities post-constraints to match temperature_K.
    remove_angmom : bool
        If True, remove net angular momentum about the center of mass.
    generator : torch.Generator, optional
        RNG for reproducible sampling.

    Returns
    -------
    VX, VY, VZ : torch.Tensor
        Each shape (N,), velocities in Å/fs on the same device/dtype as positions.
    """
    device = structure.RX.device
    dtype = structure.RX.dtype
    N = structure.Nats

    if masses_amu is None:
        masses_amu = structure.Mnuc.to(device=device, dtype=dtype)
    else:
        masses_amu = masses_amu.to(device=device, dtype=dtype)

    # Positive-mass mask
    mpos = masses_amu > 0
    npos = int(mpos.sum().item())

    # Physical constants
    kB_eV_per_K = 8.617333262145e-5  # eV/K
    amu_kg = 1.66053906660e-27
    ang2m = 1e-10
    fs2s = 1e-15
    eC = 1.602176634e-19
    MVV2KE = (amu_kg * (ang2m / fs2s) ** 2) / eC  # ≈ 103.642691 eV per (amu * (Å/fs)^2)

    # Sampling stddev (0 for zero-mass)
    kT = torch.as_tensor(kB_eV_per_K * temperature_K, device=device, dtype=dtype)
    inv_m = torch.where(mpos, 1.0 / masses_amu, torch.zeros_like(masses_amu))
    std = torch.sqrt(torch.clamp(kT * inv_m / MVV2KE, min=0)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = torch.randn((N, 3), device=device, dtype=dtype, generator=g) * std[:, None]

    # Remove center-of-mass velocity
    if remove_com and npos > 1:
        Mtot = masses_amu[mpos].sum()
        v_cm = (masses_amu[:, None] * V).sum(dim=0) / Mtot
        V = V - v_cm[None, :]

    # Optionally remove net angular momentum (about COM, without changing coordinates)
    if remove_angmom and npos > 1:
        # Positions
        R = torch.stack((structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)), dim=1)  # (N,3)
        Mtot = masses_amu[mpos].sum()
        r_com = (masses_amu[:, None] * R).sum(dim=0) / Mtot
        r_rel = R - r_com[None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:, None] * torch.cross(r_rel, V, dim=1))[mpos].sum(dim=0)  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device)
        r2 = (r_rel[mpos] * r_rel[mpos]).sum(dim=1)  # (npos,)
        I = (masses_amu[mpos][:, None, None] * (r2[:, None, None] * eye3 - r_rel[mpos][:, :, None] * r_rel[mpos][:, None, :])).sum(dim=0)  # (3,3)

        # Solve I * omega = L; use pinv for robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.linalg.pinv(I) @ L  # (3,)

        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if npos > 0:
            deltaV = torch.cross(r_rel[mpos], omega.expand_as(r_rel[mpos]), dim=1)  # (npos,3)
            V[mpos] = V[mpos] + deltaV

    # Rescale to target temperature using DOF of massive atoms minus constraints
    if rescale_to_T and npos > 0:
        dof = 3 * npos
        if remove_com and npos > 1:
            dof -= 3
        if remove_angmom and npos > 2:
            dof -= 3
        if dof > 0:
            EKIN = 0.5 * MVV2KE * (masses_amu[:, None] * (V * V)).sum()
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if T_cur > 0:
                scale = torch.sqrt(torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur)
                V = V * scale

    return V[:, 0].contiguous(), V[:, 1].contiguous(), V[:, 2].contiguous()


def initialize_velocities_batch(
    structure: Any,
    temperature_K: torch.Tensor,
    masses_amu: Optional[torch.Tensor] = None,
    remove_com: bool = True,
    rescale_to_T: bool = True,
    remove_angmom: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    

    batch_size = structure.RX.shape[0]
    device = structure.RX.device
    dtype = structure.RX.dtype
    N = structure.Nats

    if len(temperature_K) == 1 and batch_size > 1 :
        temperature_K = temperature_K.expand(batch_size)


    masses_amu = structure.Mnuc.to(device=device, dtype=dtype)
    mpos = masses_amu > 0
    npos = mpos.sum(-1).to(int)
    # Physical constants
    kB_eV_per_K = 8.617333262145e-5  # eV/K
    amu_kg = 1.66053906660e-27
    ang2m = 1e-10
    fs2s = 1e-15
    eC = 1.602176634e-19
    MVV2KE = (amu_kg * (ang2m / fs2s) ** 2) / eC  # ≈ 103.642691 eV per (amu * (Å/fs)^2)

    # Sampling stddev (0 for zero-mass)
    kT = torch.as_tensor(kB_eV_per_K * temperature_K, device=device, dtype=dtype)
    inv_m = torch.where(mpos, 1.0 / masses_amu, torch.zeros_like(masses_amu))
    std = torch.sqrt(torch.clamp(kT.unsqueeze(-1) * inv_m / MVV2KE, min=0)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = torch.randn((batch_size, N, 3), device=device, dtype=dtype, generator=g) * std[:,:, None]

    # Remove center-of-mass velocity
    if remove_com and (npos > 1).any():
        Mtot = masses_amu.sum(-1)
        v_cm = (masses_amu[:,:, None] * V).sum(dim=1) / Mtot.unsqueeze(-1)
        V = V - v_cm[:,None, :]

    # Optionally remove net angular momentum (about COM, without changing coordinates)
    if remove_angmom and (npos > 1).any():
        # Positions
        R = torch.stack((structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)), dim=2)  # (N,3)
        Mtot = masses_amu.sum(-1)
        r_com = (masses_amu[:,:, None] * R).sum(dim=1) / Mtot.unsqueeze(-1)
        r_rel = R - r_com[:, None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:,:, None] * torch.cross(r_rel, V, dim=2)).sum(dim=1)  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device) * torch.ones((batch_size,3,3), dtype=dtype, device=device)
        r2 = (r_rel * r_rel).sum(dim=2)  # (npos,)
        I = (masses_amu[:,:, None, None] * (r2[:,:, None, None] * eye3.unsqueeze(1) - r_rel[:, :, :, None] * r_rel[:, :, None, :])).sum(dim=1)  # (3,3)

        # Solve I * omega = L; use pinv for robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.bmm(torch.linalg.pinv(I), L.unsqueeze(-1)).squeeze(-1)
        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if (npos > 1).any():
            deltaV = torch.cross(r_rel, omega[:, None, :].expand_as(r_rel), dim=2)  # (npos,3)
            V = V + deltaV
    if rescale_to_T and (npos > 0).any():
        dof = 3 * npos
        if remove_com and (npos > 1).any():
            dof -= 3
        if remove_angmom and (npos > 2).any():
            dof -= 3
        if (dof > 0).any():
            EKIN = 0.5 * MVV2KE * (masses_amu[:,:, None]  * (V * V)).sum(dim=(1,2))
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if (T_cur > 0).any():
                scale = torch.sqrt(torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur)
                V = V * scale.unsqueeze(-1).unsqueeze(-1)

    return V[:, :, 0].contiguous(), V[:, :, 1].contiguous(), V[:, :, 2].contiguous()

