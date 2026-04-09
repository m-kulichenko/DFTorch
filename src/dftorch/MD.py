from ._cell import wrap_positions
from .ESDriver import ESDriverBatch
from ._energy import energy_shadow
from ._forces_batch import forces_shadow_batch
from ._forces import forces_shadow, forces_shadow_pme, forces_spin

# from ._kernel_fermi import _kernel_fermi
from ._xl_tools import (
    kernel_update_lr,
    kernel_update_lr_os,
    kernel_update_lr_batch,
    calc_q,
    calc_q_os,
    calc_q_batch,
)
from ._spin import get_h_spin, get_spin_energy_shadow
from ._stress import get_total_stress_analytical
from ._gbsa import create_gbsa, GBSABatch

import torch

from typing import Any, Tuple, Optional
import time
from ._io import write_XYZ_trajectory, write_pdb_frame
from dftorch._tools import calculate_dist_dips


class MDXL:
    """Extended-Lagrangian Born–Oppenheimer MD for_ closed- and open-shell systems.

    The class auto-detects the spin mode from the ``structure`` passed to
    :meth:`run` (open-shell ↔ ``structure.Nocc`` is a 2-element tensor) and
    adjusts the charge variable, kernel, and energy/force evaluation
    accordingly.
    """

    def __init__(self, es_driver: ESDriverBatch, const, temperature_K: float):

        self.NoRank = False
        self.do_full_kernel = False
        self.const = const
        self.fric = 0.0
        self.F2V = 0.01602176487 / 1.660548782
        self.MVV2KE = 166.0538782 / 1.602176487
        self.KE2T = 1 / 0.000086173435
        self.C0 = -6
        self.C1 = 14
        self.C2 = -8
        self.C3 = -3
        self.C4 = 4
        self.C5 = -1  # Coefficients for_ modified Verlet integration
        self.kappa = 1.82
        self.alpha = 0.018  # Coefficients for_ modified Verlet integration

        # Langevin thermostat parameters
        self.langevin_gamma = 0.0  # friction coefficient in 1/fs (0 = off)
        self.langevin_enabled = False

        # Berendsen barostat parameters
        self.barostat_enabled = False
        self.target_pressure = 0.0  # target pressure in eV/Å³
        self.barostat_tau = 100.0  # coupling time in fs
        self.barostat_isotropic = True  # True = isotropic, False = anisotropic
        self.barostat_compressibility = (
            4.57e-5  # compressibility in 1/GPa (water default)
        )
        self.P_array = None  # pressure history
        self.V_array = None  # volume history

        self.es_driver = es_driver
        self.temperature_K = temperature_K
        self.n = None
        self.n_0 = None
        self.n_1 = None
        self.n_2 = None
        self.n_3 = None
        self.n_4 = None
        self.n_5 = None
        self.VX, self.VY, self.VZ = None, None, None
        self.K0Res = None
        self.E_array = None
        self.T_array = None
        self.Ek_array = None
        self.Ep_array = None
        self.Res_array = None

        self.cuda_sync = False

        # These are set in run() once the spin mode is known
        self._os: bool = False  # open-shell flag
        self.atom_ids_sr = None  # shell-resolved atom ids (os only)
        self.shell_to_atom = None  # shell→atom mapping     (os only)

    def enable_langevin(self, gamma: float):
        """
        Enable Langevin thermostat.

        Parameters
        ----------
        gamma : float
            Friction coefficient in 1/fs. Typical values: 0.001–0.1 1/fs.
            Higher = stronger coupling to bath, less NVE-like.
        """
        self.langevin_gamma = gamma
        self.langevin_enabled = True

    def _langevin_kick(self, structure, dt):
        """
        Apply Langevin friction + random force as a velocity correction.
        Called once per half-step (splitting scheme).

        The impulse for_ half-step dt/2:
          v <- v * c1 + c2 * xi / sqrt(M)
        where
          c1 = exp(-gamma * dt/2)
          c2 = sqrt((1 - c1^2) * kB * T / MVV2KE)
        """
        kB_eV = 8.617333262145e-5  # eV/K
        T = (
            self.temperature_K.item()
            if hasattr(self.temperature_K, "item")
            else float(self.temperature_K)
        )

        dt_half = 0.5 * dt
        c1 = torch.exp(
            torch.tensor(
                -self.langevin_gamma * dt_half,
                dtype=structure.RX.dtype,
                device=structure.RX.device,
            )
        )
        # noise magnitude: sqrt(kB*T / (M * MVV2KE)) * sqrt(1 - c1^2)
        noise_scale = torch.sqrt(
            (1.0 - c1**2) * kB_eV * T / (structure.Mnuc * self.MVV2KE)
        )  # shape (N,)

        xi_x = torch.randn_like(self.VX)
        xi_y = torch.randn_like(self.VY)
        xi_z = torch.randn_like(self.VZ)

        self.VX = c1 * self.VX + noise_scale * xi_x
        self.VY = c1 * self.VY + noise_scale * xi_y
        self.VZ = c1 * self.VZ + noise_scale * xi_z

    def enable_barostat(
        self,
        target_pressure: float = 0.0,
        tau: float = 100.0,
        isotropic: bool = True,
        compressibility: float = 4.57e-5,
    ):
        """
        Enable Berendsen barostat for_ NPT ensemble.

        Parameters
        ----------
        target_pressure : float
            Target external pressure in GPa.
            0.0 = ambient / zero pressure.
        tau : float
            Pressure coupling time constant in fs.
            Typical: 100–1000 fs. Smaller = stronger coupling.
        isotropic : bool
            If True, uniform scaling along all three axes (hydrostatic).
            If False, allow independent scaling per axis (anisotropic).
        compressibility : float
            Isothermal compressibility in **1/bar** (the standard MD
            convention used e.g. by GROMACS).
            Default 4.57×10⁻⁵  1/bar  (liquid water at 300 K).
            For solids, ∼1–5×10⁻⁶  1/bar.
        """
        self.barostat_enabled = True
        # 1 GPa = 1/(160.21766208) eV/Å³  →  1 eV/Å³ = 160.21766208 GPa
        # 1 GPa = 1e4 bar  →  1 eV/Å³ = 160.21766208e4 bar
        # β in 1/bar  →  β in Å³/eV = β_bar × 160.21766208e4
        GPa_per_eVA3 = 160.21766208
        bar_per_eVA3 = GPa_per_eVA3 * 1e4
        self.target_pressure = target_pressure / GPa_per_eVA3  # GPa → eV/Å³
        self.barostat_tau = tau
        self.barostat_isotropic = isotropic
        # Convert compressibility from 1/bar → Å³/eV for_ internal use
        self.barostat_compressibility = compressibility * bar_per_eVA3  # Å³/eV

    def _compute_stress_tensor(self, structure, dftorch_params):
        """
        Compute the potential-energy stress tensor using analytical formulas.

        In XL-BOMD the shadow energy depends on both the SCF charges *q*
        (from diagonalisation) and the extrapolated charges *n* (from the
        extended Lagrangian).  We pass *n* to the stress routine so that the
        SCC-overlap and Coulomb contributions use the shadow expressions
        consistent with ``forces_shadow`` / ``energy_shadow``.

        Returns the 3×3 stress tensor in eV/Å³.
        """
        # Atom-resolved extrapolated charges for_ the shadow stress.
        # CS: self.n is already atom-resolved (Nats,).
        # OS: self.n is shell-resolved (2, Nshells); sum over spin then
        #     scatter to atoms.
        if self._os:
            n_atom = torch.zeros_like(structure.RX)
            n_atom.scatter_add_(0, self.shell_to_atom, self.n.sum(dim=0))
        else:
            n_atom = self.n

        stress_dict = get_total_stress_analytical(
            structure,
            self.const,
            repulsive_rcut=self.es_driver.repulsive_rcut,
            dftorch_params=dftorch_params,
            verbose=False,
            n=n_atom,
        )
        return stress_dict["total"]  # (3, 3) in eV/Å³

    def _compute_kinetic_stress(self, structure):
        """
        Compute the kinetic (ideal-gas) contribution to the stress tensor.

        σ^kin_αβ = (1/V) Σ_i  m_i v_iα v_iβ   (in eV/Å³)
        """
        V = torch.abs(torch.det(structure.cell))
        vel = torch.stack((self.VX, self.VY, self.VZ), dim=-1)  # (N, 3)
        mass = structure.Mnuc  # (N,) in amu
        # m_i v_i⊗v_i  summed → amu·Å²/fs²
        # 1 amu·Å²/fs² = 2 * MVV2KE eV  (MVV2KE converts ½mv² → eV)
        mvv = torch.einsum("i,ia,ib->ab", mass, vel, vel)  # (3,3)
        sigma_kin = (2.0 * self.MVV2KE * mvv) / V
        return sigma_kin  # eV/Å³

    def _barostat_scale(self, structure, dt):
        """
        Apply Berendsen barostat scaling to cell and positions.

        Scaling factor:
          μ = [1 - (β dt / (3 τ_P)) (P_target - P_inst)]^{1/3}

        Isotropic: scalar μ applied uniformly.
        Anisotropic: independent μ_α per axis from diagonal P_αα.
        """
        sigma_pot = self._compute_stress_tensor(structure, self._dftorch_params)
        sigma_kin = self._compute_kinetic_stress(structure)

        # Pressure tensor: P = σ_kin - σ_pot
        P_inst = sigma_kin - sigma_pot  # (3, 3) eV/Å³

        P_scalar = torch.trace(P_inst) / 3.0
        GPa_per_eVA3 = 160.21766208

        # Store for_ logging (GPa)
        self._P_inst_GPa = P_scalar * GPa_per_eVA3
        self._P_tensor = P_inst * GPa_per_eVA3

        # Berendsen prefactor: β dt / (3 τ_P)
        # β is in Å³/eV (converted from 1/bar in enable_barostat),
        # dP is in eV/Å³  →  β·dP is dimensionless  ✓
        prefactor = self.barostat_compressibility * dt / (3.0 * self.barostat_tau)

        if self.barostat_isotropic:
            dP = self.target_pressure - P_scalar  # eV/Å³
            mu = (1.0 - prefactor * dP) ** (1.0 / 3.0)
            mu_t = torch.tensor(
                mu, dtype=structure.cell.dtype, device=structure.cell.device
            )

            structure.cell = structure.cell * mu_t
            structure.cell_inv = torch.linalg.inv(structure.cell)

            structure.RX = structure.RX * mu_t
            structure.RY = structure.RY * mu_t
            structure.RZ = structure.RZ * mu_t
        else:
            # Anisotropic: per-axis diagonal scaling (eV/Å³)
            diag_P = torch.diagonal(P_inst)  # (3,) eV/Å³
            dP = self.target_pressure - diag_P
            mu_diag = (1.0 - prefactor * dP) ** (1.0 / 3.0)

            structure.cell = structure.cell * mu_diag.unsqueeze(-1)
            structure.cell_inv = torch.linalg.inv(structure.cell)

            structure.RX = structure.RX * mu_diag[0]
            structure.RY = structure.RY * mu_diag[1]
            structure.RZ = structure.RZ * mu_diag[2]

        # Wrap positions after rescaling
        R = torch.stack((structure.RX, structure.RY, structure.RZ), dim=-1)
        R = wrap_positions(R, structure.cell, structure.cell_inv)
        structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)

        # Rebuild PME data for_ the new cell (if using PME Coulomb)
        if self._dftorch_params["coul_method"] == "PME":
            from .ewald_pme import (
                init_PME_data,
                calculate_alpha_and_num_grids,
            )

            self.CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
                structure.cell.detach().cpu().numpy(),
                self._dftorch_params["cutoff"],
                self._dftorch_params["Coulomb_acc"],
            )
            self.PME_data = init_PME_data(
                grid_dimensions,
                structure.cell,
                self.CALPHA,
                self._dftorch_params["PME_order"],
            )

    def run(
        self,
        structure,
        dftorch_params,
        num_steps,
        dt,
        dump_interval=1,
        traj_filename="md_trj.xyz",
    ):
        # Detect open-shell from structure.Nocc:
        # CS: int or 1-element tensor, OS: 2-element tensor [Nocc_alpha, Nocc_beta]
        self._os = (
            isinstance(structure.Nocc, torch.Tensor) and structure.Nocc.numel() == 2
        )

        # Store params for_ barostat (needs them during step)
        self._dftorch_params = dftorch_params

        if self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities(
                structure,
                temperature_K=self.temperature_K,
                remove_com=True,
                rescale_to_T=True,
                remove_angmom=True,
            )

        # Propagated charge variable: atom-resolved q (CS) or shell-resolved
        # q_spin_sr of shape (2, Nshells) (OS).
        if self._os:
            q = structure.q_spin_sr.clone()
        else:
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
            if self._os:
                self.K0Res = torch.bmm(
                    structure.KK, (q - self.n).unsqueeze(-1)
                ).squeeze(-1)
            else:
                self.K0Res = structure.KK @ (q - self.n)

        # Generate atom index for_ each orbital
        self.atom_ids = torch.repeat_interleave(
            torch.arange(len(structure.n_orbitals_per_atom), device=structure.device),
            structure.n_orbitals_per_atom,
        )
        self.Hubbard_U_gathered = structure.Hubbard_U[self.atom_ids]
        if structure.dU_dq is not None:
            self.dU_dq_gathered = structure.dU_dq[self.atom_ids]
        else:
            self.dU_dq_gathered = None

        # Open-shell-specific index arrays
        if self._os:
            self.atom_ids_sr = torch.repeat_interleave(
                torch.arange(len(structure.shell_types), device=structure.device),
                self.const.shell_dim[structure.shell_types],
            )
            self.shell_to_atom = torch.repeat_interleave(
                torch.arange(len(structure.TYPE), device=structure.device),
                structure.n_shells_per_atom,
            )

        if dftorch_params["coul_method"] == "PME":
            from .ewald_pme import (
                init_PME_data,
                calculate_alpha_and_num_grids,
            )

            self.CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
                structure.cell.cpu().numpy(),
                dftorch_params["cutoff"],
                dftorch_params["Coulomb_acc"],
            )
            self.PME_data = init_PME_data(
                grid_dimensions,
                structure.cell,
                self.CALPHA,
                dftorch_params["PME_order"],
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

        if self.barostat_enabled and self.P_array is None:
            self.P_array = torch.empty((0,), device=structure.device)
            self.V_array = torch.empty((0,), device=structure.device)
            self._P_inst_GPa = torch.tensor(0.0, device=structure.device)
            self._P_tensor = torch.zeros(3, 3, device=structure.device)

        self.EPOT = structure.e_tot
        for md_step in range(num_steps):
            self.step(
                structure, dftorch_params, md_step, dt, dump_interval, traj_filename
            )

    def step(
        self, structure, dftorch_params, md_step, dt, dump_interval, traj_filename
    ):
        if self.cuda_sync:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        print(
            "########## Step = {:} ##########".format(
                md_step,
            )
        )

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

        q_current = structure.q_spin_sr if self._os else structure.q
        ResErr = torch.norm(q_current - self.n) / (
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

        if md_step % dump_interval == 0:
            if self._os:
                comm_string = (
                    f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, "
                    f"Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, "
                    f"NS = {structure.net_spin_sr.sum().item():.4f}, "
                    f"Res = {ResErr:.6f}"
                )
            else:
                comm_string = (
                    f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, "
                    f"Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, "
                    f"Res = {ResErr:.6f}, mu = {structure.mu0:.4f} eV"
                )
            write_XYZ_trajectory(traj_filename, structure, comm_string, step=md_step)

            write_pdb_frame(
                traj_filename + ".pdb",
                structure,
                structure.cell,
                step=md_step,
                etot=Energ,
                temp=Temperature,
                mode="a",
            )

        # ── First half velocity Verlet ───────────────────────────────────
        self.VX = (
            self.VX
            + 0.5 * dt * (self.F2V * structure.f_tot[0] / structure.Mnuc)
            - self.fric * self.VX
        )
        self.VY = (
            self.VY
            + 0.5 * dt * (self.F2V * structure.f_tot[1] / structure.Mnuc)
            - self.fric * self.VY
        )
        self.VZ = (
            self.VZ
            + 0.5 * dt * (self.F2V * structure.f_tot[2] / structure.Mnuc)
            - self.fric * self.VZ
        )

        # ── Langevin half-kick (before position update) ──────────────────
        if self.langevin_enabled:
            self._langevin_kick(structure, dt)

        # ── Position update + PBC wrapping ───────────────────────────────
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

        if self.cuda_sync:
            torch.cuda.synchronize()
        tic2_1 = time.perf_counter()

        # ── H0 + S build & charge extrapolation ─────────────────────────
        if self._os and ResErr > 0.05:
            # OS divergence guard: fall back to full SCF
            self.es_driver(structure, self.const, do_scf=True)
            self.n = structure.q_spin_sr.clone()
            self.n_5 = self.n
            self.n_4 = self.n
            self.n_3 = self.n
            self.n_2 = self.n
            self.n_1 = self.n
            self.n_0 = self.n
        else:
            self.es_driver(structure, self.const, do_scf=False)
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

        n_spin_atom = torch.zeros_like(structure.RX.unsqueeze(0).expand(2, -1))
        n_spin_atom.scatter_add_(
            1, self.shell_to_atom.unsqueeze(0).expand(2, -1), self.n
        )  # atom-resolved

        n_tot_atom = torch.zeros_like(structure.RX)
        n_tot_atom.scatter_add_(
            0, self.shell_to_atom, self.n.sum(dim=0)
        )  # atom-resolved
        n_net_spin_sr = self.n[0] - self.n[1]
        # ── OS: compute atom-resolved charges from shell-resolved n ──────
        if self._os:
            n_spin_atom = torch.zeros_like(structure.RX.unsqueeze(0).expand(2, -1))
            n_spin_atom.scatter_add_(
                1, self.shell_to_atom.unsqueeze(0).expand(2, -1), self.n
            )
            n_tot_atom = torch.zeros_like(structure.RX)
            n_tot_atom.scatter_add_(0, self.shell_to_atom, self.n.sum(dim=0))
            n_net_spin_sr = self.n[0] - self.n[1]

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("H0: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic2_1 = time.perf_counter()

        # ── Coulomb potential ────────────────────────────────────────────
        if dftorch_params["coul_method"] == "PME":
            from .ewald_pme import (
                calculate_PME_ewald,
            )
            from .ewald_pme.neighbor_list import NeighborState

            nbr_state = NeighborState(
                torch.stack((structure.RX, structure.RY, structure.RZ)),
                structure.cell,
                None,
                dftorch_params["cutoff"],
                is_dense=True,
                buffer=0.0,
                use_triton=False,
            )
            disps, dists, nbr_inds = calculate_dist_dips(
                torch.stack((structure.RX, structure.RY, structure.RZ)),
                nbr_state,
                dftorch_params["cutoff"],
            )

            pme_charge = n_tot_atom if self._os else self.n
            _, forces1, CoulPot = calculate_PME_ewald(
                torch.stack((structure.RX, structure.RY, structure.RZ)),
                pme_charge,
                structure.cell,
                nbr_inds,
                disps,
                dists,
                self.CALPHA,
                dftorch_params["cutoff"],
                self.PME_data,
                hubbard_u=structure.Hubbard_U,
                atomtypes=structure.TYPE,
                screening=1,
                calculate_forces=1,
                calculate_dq=1,
                h_damp_exp=dftorch_params.get("h_damp_exp", None),
                h5_params=dftorch_params.get("h5_params", None),
            )
        else:
            if self._os:
                CoulPot = structure.C @ n_tot_atom
            else:
                CoulPot = structure.C @ self.n
            nbr_inds = None
            disps = None
            dists = None

        H_spin = get_h_spin(
            structure.TYPE,
            n_net_spin_sr,
            self.const.w,
            structure.n_shells_per_atom,
            structure.shell_types,
        )
        (
            structure.q_spin_sr,
            structure.H,
            structure.Hcoul,
            structure.D,
            structure.Dorth,
            Q,
            e,
            structure.f,
            structure.mu0,
        ) = calc_q_os(
            structure.H0,
            H_spin,
            self.Hubbard_U_gathered,
            n_tot_atom[self.atom_ids],
            CoulPot[self.atom_ids],
            structure.S,
            structure.Z,
            structure.Te,
            structure.Nocc,
            structure.Znuc,
            self.atom_ids,
            self.atom_ids_sr,
            structure.el_per_shell,
            self.dU_dq_gathered,
            dftorch_params.get("SHARED_MU", False),
            dftorch_params["DELTA_SCF"],
            dftorch_params,
        )
        # ── GBSA implicit solvation: rebuild for_ new geometry ────────────
        if dftorch_params.get("solvent_param_file", None) is not None:
            structure.gbsa = create_gbsa(
                structure,
                structure.device,
                param_file=dftorch_params.get("solvent_param_file", None),
                solvation_model=dftorch_params.get("solvation_model", "gbsa"),
            )
            # Born shift uses extrapolated charges n (shadow Hamiltonian)
            solv_n = n_tot_atom if self._os else self.n
            solv_shift = structure.gbsa.get_shadow_shifts(solv_n)
            CoulPot = CoulPot + solv_shift
        else:
            structure.gbsa = None
            solv_shift = None

        # ── Full off-diagonal DFTB3: add third-order shift to CoulPot ────
        if structure.thirdorder is not None:
            to_n = n_tot_atom if self._os else self.n
            CoulPot = CoulPot + structure.thirdorder.get_shifts(to_n)

        # ── Diagonalise & compute charges ────────────────────────────────
        if self._os:
            H_spin = get_h_spin(
                structure.TYPE,
                n_net_spin_sr,
                self.const.w,
                structure.n_shells_per_atom,
                structure.shell_types,
            )
            (
                structure.q_spin_sr,
                structure.H,
                structure.Hcoul,
                structure.D,
                structure.Dorth,
                Q,
                e,
                structure.f,
                structure.mu0,
            ) = calc_q_os(
                structure.H0,
                H_spin,
                self.Hubbard_U_gathered,
                n_tot_atom[self.atom_ids],
                CoulPot[self.atom_ids],
                structure.S,
                structure.Z,
                structure.Te,
                structure.Nocc,
                structure.Znuc,
                self.atom_ids,
                self.atom_ids_sr,
                structure.el_per_shell,
                self.dU_dq_gathered if structure.thirdorder is None else None,
                dftorch_params.get("SHARED_MU", False),
            )

            structure.net_spin_sr = structure.q_spin_sr[0] - structure.q_spin_sr[1]
            q_spin_atom = torch.zeros_like(structure.RX.unsqueeze(0).expand(2, -1))
            q_spin_atom.scatter_add_(
                1,
                self.shell_to_atom.unsqueeze(0).expand(2, -1),
                structure.q_spin_sr,
            )
            q_tot_atom = torch.zeros_like(structure.RX)
            q_tot_atom.scatter_add_(
                0, self.shell_to_atom, structure.q_spin_sr.sum(dim=0)
            )
        else:
            (
                structure.q,
                structure.H,
                structure.Hcoul,
                structure.D,
                Dorth,
                Q,
                e,
                structure.f,
                structure.mu0,
            ) = calc_q(
                structure.H0,
                self.Hubbard_U_gathered,
                self.n[self.atom_ids],
                CoulPot[self.atom_ids],
                structure.S,
                structure.Z,
                structure.Te,
                structure.Nocc,
                structure.Znuc,
                self.atom_ids,
                self.dU_dq_gathered if structure.thirdorder is None else None,
            )

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic3 = time.perf_counter()

        # ── Kernel update ────────────────────────────────────────────────
        Res = (structure.q_spin_sr if self._os else structure.q) - self.n
        if md_step % 100000 == 0 and self.do_full_kernel:
            pass
        elif self.NoRank:
            self.K0Res = -dftorch_params["SCF_ALPHA"] * Res
        elif self._os:
            self.K0Res = kernel_update_lr_os(
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                structure.TYPE,
                structure.Nats,
                structure.Hubbard_U,
                dftorch_params,
                dftorch_params["KRYLOV_TOL_MD"],
                structure.KK.clone(),
                Res,
                structure.q_spin_sr,
                structure.S,
                structure.Z,
                self.PME_data,
                self.atom_ids,
                self.atom_ids_sr,
                Q,
                e,
                structure.mu0,
                structure.Te,
                self.const.w,
                structure.n_shells_per_atom,
                structure.shell_types,
                structure.C,
                nbr_inds,
                disps,
                dists,
                self.CALPHA,
                structure.dU_dq if structure.thirdorder is None else None,
                thirdorder=structure.thirdorder,
            )
        else:
            self.K0Res = kernel_update_lr(
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                structure.TYPE,
                structure.Nats,
                structure.Hubbard_U,
                dftorch_params,
                dftorch_params["KRYLOV_TOL_MD"],
                structure.KK.clone(),
                Res,
                structure.q,
                structure.S,
                structure.Z,
                self.PME_data,
                self.atom_ids,
                Q,
                e,
                structure.mu0,
                structure.Te,
                structure.C,
                nbr_inds,
                disps,
                dists,
                self.CALPHA,
                structure.dU_dq if structure.thirdorder is None else None,
                thirdorder=structure.thirdorder,
            )

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("KER: {:.3f} s".format(time.perf_counter() - tic3))
        tic4 = time.perf_counter()

        # ── Spin energy (OS only) ────────────────────────────────────────
        if self._os:
            structure.e_spin = get_spin_energy_shadow(
                structure.TYPE,
                structure.net_spin_sr,
                n_net_spin_sr,
                self.const.w,
                structure.n_shells_per_atom,
            )

        # Atom-resolved charges used for_ energy / forces
        if self._os:
            q_e = q_tot_atom  # SCF charges
            n_e = n_tot_atom  # extrapolated charges
        else:
            q_e = structure.q
            n_e = self.n

        # ── Energy + Forces ──────────────────────────────────────────────
        if dftorch_params["coul_method"] == "PME":
            (
                structure.e_elec_tot,
                structure.e_band0,
                structure.e_coul,
                structure.e_dipole,
                structure.e_entropy,
                structure.s_ent,
            ) = energy_shadow(
                structure.H0,
                structure.Hubbard_U,
                structure.e_field,
                structure.D0,
                None,
                CoulPot,
                structure.D,
                q_e,
                n_e,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.f,
                structure.Te,
                structure.dU_dq if structure.thirdorder is None else None,
                thirdorder=structure.thirdorder,
            )

            # no f_coul in PME forces_shadow_pme. Done in calculate_PME_ewald
            (
                structure.f_tot,
                _,
                structure.f_band0,
                structure.f_dipole,
                structure.f_pulay,
                structure.f_s_coul,
                structure.f_s_dipole,
                structure.f_rep,
            ) = forces_shadow_pme(
                structure.H,
                structure.Z,
                CoulPot,
                structure.D,
                structure.D0,
                structure.dH0,
                structure.dS,
                structure.dVr,
                structure.e_field,
                structure.Hubbard_U,
                q_e,
                n_e,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.Nats,
                self.const,
                structure.TYPE,
                structure.dU_dq if structure.thirdorder is None else None,
                solvation_shift=solv_shift,
                thirdorder_shift=(
                    structure.thirdorder.get_shifts(n_e)
                    if structure.thirdorder is not None
                    else None
                ),
            )
            structure.f_coul = forces1 * (2 * q_e / n_e - 1.0)
            structure.f_tot = structure.f_tot + structure.f_coul
        else:
            (
                structure.e_elec_tot,
                structure.e_band0,
                structure.e_coul,
                structure.e_dipole,
                structure.e_entropy,
                structure.s_ent,
            ) = energy_shadow(
                structure.H0,
                structure.Hubbard_U,
                structure.e_field,
                structure.D0,
                structure.C,
                None,
                structure.D,
                q_e,
                n_e,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.f,
                structure.Te,
                structure.dU_dq if structure.thirdorder is None else None,
                thirdorder=structure.thirdorder,
            )

            (
                structure.f_tot,
                structure.f_coul,
                structure.f_band0,
                structure.f_dipole,
                structure.f_pulay,
                structure.f_s_coul,
                structure.f_s_dipole,
                structure.f_rep,
            ) = forces_shadow(
                structure.H,
                structure.Z,
                structure.C,
                structure.D,
                structure.D0,
                structure.dH0,
                structure.dS,
                structure.dCC,
                structure.dVr,
                structure.e_field,
                structure.Hubbard_U,
                q_e,
                n_e,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.Nats,
                self.const,
                structure.TYPE,
                structure.dU_dq if structure.thirdorder is None else None,
                solvation_shift=solv_shift,
                thirdorder_shift=(
                    structure.thirdorder.get_shifts(n_e)
                    if structure.thirdorder is not None
                    else None
                ),
            )

        # ── Spin forces (OS only) ───────────────────────────────────────
        if self._os:
            structure.f_spin = forces_spin(
                structure.D,
                structure.dS,
                n_spin_atom,
                structure.Nats,
                self.const,
                structure.TYPE,
            )
            structure.f_tot = structure.f_tot + structure.f_spin
            structure.e_tot = (
                structure.e_elec_tot + structure.e_repulsion + structure.e_spin
            )
        else:
            structure.e_tot = structure.e_elec_tot + structure.e_repulsion

        # ── GBSA shadow solvation energy + gradients ─────────────────────
        if structure.gbsa is not None:
            e_gb, e_sasa = structure.gbsa.get_shadow_energies(q_e, n_e)
            structure.e_gb = e_gb
            structure.e_sasa = e_sasa
            structure.e_tot = structure.e_tot + e_gb + e_sasa

            structure.f_gbsa_sasa = structure.gbsa.get_shadow_sasa_gradients(
                q_e, n_e
            ).T  # (3, N)
            structure.f_gbsa_born = structure.gbsa.get_shadow_born_gradients(
                q_e, n_e
            ).T  # (3, N)
            structure.f_gbsa = structure.f_gbsa_sasa + structure.f_gbsa_born
            structure.f_tot = structure.f_tot + structure.f_gbsa

        # ── Full off-diagonal DFTB3: gradient from dΓ³/dr ───────────────
        if structure.thirdorder is not None:
            grad_dc = structure.thirdorder.get_gradient_dc_xlbomd(n_e, q_e)
            structure.f_tot = structure.f_tot + grad_dc

        # ── D3(BJ) dispersion energy + forces ───────────────────────────
        if structure.dftd3 is not None:
            coords_ang = torch.stack([structure.RX, structure.RY, structure.RZ], dim=1)
            structure.e_d3 = structure.dftd3.get_energy(coords_ang)
            structure.e_tot = structure.e_tot + structure.e_d3
            structure.f_d3 = structure.dftd3.get_forces(coords_ang)
            structure.f_tot = structure.f_tot + structure.f_d3
        else:
            structure.e_d3 = 0.0

        self.EPOT = structure.e_tot

        # ── Second half velocity Verlet ──────────────────────────────────
        self.VX = (
            self.VX
            + 0.5 * dt * (self.F2V * structure.f_tot[0] / structure.Mnuc)
            - self.fric * self.VX
        )
        self.VY = (
            self.VY
            + 0.5 * dt * (self.F2V * structure.f_tot[1] / structure.Mnuc)
            - self.fric * self.VY
        )
        self.VZ = (
            self.VZ
            + 0.5 * dt * (self.F2V * structure.f_tot[2] / structure.Mnuc)
            - self.fric * self.VZ
        )

        # ── Langevin half-kick (after force update) ──────────────────────
        if self.langevin_enabled:
            self._langevin_kick(structure, dt)

        # ── Berendsen barostat: rescale cell + positions ─────────────────
        if self.barostat_enabled:
            self._barostat_scale(structure, dt)
            V = torch.abs(torch.det(structure.cell))
            self.P_array = torch.cat(
                (self.P_array, self._P_inst_GPa.detach().unsqueeze(0)), dim=0
            )
            self.V_array = torch.cat((self.V_array, V.detach().unsqueeze(0)), dim=0)

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("F AND E: {:.3f} s".format(time.perf_counter() - tic4))

        if self.barostat_enabled:
            V = torch.abs(torch.det(structure.cell))
            P_str = f", P = {self._P_inst_GPa.item():.4f} GPa, V = {V.item():.2f} Å³"
        else:
            P_str = ""

        if self._os:
            print(
                "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f},  NS = {:.4f}, ResErr = {:.6f}{}, t = {:.1f} s".format(
                    Energ,
                    self.EPOT.item(),
                    self.EKIN.item(),
                    Temperature.item(),
                    structure.net_spin_sr.sum().item(),
                    ResErr.item(),
                    P_str,
                    time.perf_counter() - start_time,
                )
            )
        else:
            print(
                "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}{}, t = {:.1f} s".format(
                    Energ,
                    self.EPOT.item(),
                    self.EKIN.item(),
                    Temperature.item(),
                    ResErr.item(),
                    P_str,
                    time.perf_counter() - start_time,
                )
            )
        print(torch.cuda.memory_allocated() / 1e9, "GB\n")
        print()


# Backward-compatible alias — existing code that uses MDXLOS will keep working.
MDXLOS = MDXL


class MDXLBatch:
    def __init__(self, es_driver: ESDriverBatch, const, temperature_K: float):

        self.NoRank = False
        self.do_full_kernel = False
        self.const = const
        self.fric = 0.0
        self.F2V = 0.01602176487 / 1.660548782
        self.MVV2KE = 166.0538782 / 1.602176487
        self.KE2T = 1 / 0.000086173435
        self.C0 = -6
        self.C1 = 14
        self.C2 = -8
        self.C3 = -3
        self.C4 = 4
        self.C5 = -1  # Coefficients for_ modified Verlet integration
        self.kappa = 1.82
        self.alpha = 0.018  # Coefficients for_ modified Verlet integration

        # Langevin thermostat parameters
        self.langevin_gamma = 0.0  # friction coefficient in 1/fs (0 = off)
        self.langevin_enabled = False

        # Berendsen barostat parameters
        self.barostat_enabled = False
        self.target_pressure = 0.0  # target pressure in eV/Å³
        self.barostat_tau = 100.0  # coupling time in fs
        self.barostat_isotropic = True  # True = isotropic, False = anisotropic
        self.barostat_compressibility = (
            4.57e-5  # compressibility in 1/GPa (water default)
        )
        self.P_array = None  # pressure history
        self.V_array = None  # volume history

        self.es_driver = es_driver
        self.temperature_K = temperature_K
        self.n = None
        self.n_0 = None
        self.n_1 = None
        self.n_2 = None
        self.n_3 = None
        self.n_4 = None
        self.n_5 = None
        self.VX, self.VY, self.VZ = None, None, None
        self.K0Res = None
        self.E_array = None
        self.T_array = None
        self.Ek_array = None
        self.Ep_array = None
        self.Res_array = None

        self.cuda_sync = True

    def enable_langevin(self, gamma: float):
        """
        Enable Langevin thermostat.

        Parameters
        ----------
        gamma : float
            Friction coefficient in 1/fs. Typical values: 0.001–0.1 1/fs.
            Higher = stronger coupling to bath, less NVE-like.
        """
        self.langevin_gamma = gamma
        self.langevin_enabled = True

    def _langevin_kick(self, structure, dt):
        """
        Apply Langevin friction + random force as a velocity correction (batched).
        Called once per half-step (splitting scheme).

        The impulse for_ half-step dt/2:
          v <- v * c1 + c2 * xi / sqrt(M)
        where
          c1 = exp(-gamma * dt/2)
          c2 = sqrt((1 - c1^2) * kB * T / MVV2KE)
        """
        kB_eV = 8.617333262145e-5  # eV/K
        # temperature_K is (B,) tensor for batched MD
        T = self.temperature_K  # (B,)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=structure.RX.dtype, device=structure.RX.device)
        if T.dim() == 0:
            T = T.unsqueeze(0).expand(structure.batch_size)

        dt_half = 0.5 * dt
        c1 = torch.exp(
            torch.tensor(
                -self.langevin_gamma * dt_half,
                dtype=structure.RX.dtype,
                device=structure.RX.device,
            )
        )
        # noise magnitude: sqrt(kB*T / (M * MVV2KE)) * sqrt(1 - c1^2)
        # T is (B,), Mnuc is (B, N) → noise_scale is (B, N)
        noise_scale = torch.sqrt(
            (1.0 - c1**2) * kB_eV * T.unsqueeze(-1) / (structure.Mnuc * self.MVV2KE)
        )

        xi_x = torch.randn_like(self.VX)
        xi_y = torch.randn_like(self.VY)
        xi_z = torch.randn_like(self.VZ)

        self.VX = c1 * self.VX + noise_scale * xi_x
        self.VY = c1 * self.VY + noise_scale * xi_y
        self.VZ = c1 * self.VZ + noise_scale * xi_z

    def enable_barostat(
        self,
        target_pressure: float = 0.0,
        tau: float = 100.0,
        isotropic: bool = True,
        compressibility: float = 4.57e-5,
    ):
        """
        Enable Berendsen barostat for_ NPT ensemble (batched).

        Parameters
        ----------
        target_pressure : float
            Target external pressure in GPa.
            0.0 = ambient / zero pressure.
        tau : float
            Pressure coupling time constant in fs.
            Typical: 100–1000 fs. Smaller = stronger coupling.
        isotropic : bool
            If True, uniform scaling along all three axes (hydrostatic).
            If False, allow independent scaling per axis (anisotropic).
        compressibility : float
            Isothermal compressibility in **1/bar** (the standard MD
            convention used e.g. by GROMACS).
            Default 4.57×10⁻⁵  1/bar  (liquid water at 300 K).
            For solids, ∼1–5×10⁻⁶  1/bar.
        """
        self.barostat_enabled = True
        GPa_per_eVA3 = 160.21766208
        bar_per_eVA3 = GPa_per_eVA3 * 1e4
        self.target_pressure = target_pressure / GPa_per_eVA3  # GPa → eV/Å³
        self.barostat_tau = tau
        self.barostat_isotropic = isotropic
        self.barostat_compressibility = compressibility * bar_per_eVA3  # Å³/eV

    def _compute_kinetic_stress_batch(self, structure):
        """
        Compute the kinetic (ideal-gas) contribution to the stress tensor (batched).

        σ^kin_αβ = (1/V) Σ_i  m_i v_iα v_iβ   (in eV/Å³)

        Returns (B, 3, 3) tensor.
        """
        V = torch.abs(torch.det(structure.cell))  # (B,)
        vel = torch.stack((self.VX, self.VY, self.VZ), dim=-1)  # (B, N, 3)
        mass = structure.Mnuc  # (B, N) in amu
        # m_i v_i⊗v_i summed → (B, 3, 3) in amu·Å²/fs²
        mvv = torch.einsum("bi,bia,bib->bab", mass, vel, vel)  # (B, 3, 3)
        sigma_kin = (2.0 * self.MVV2KE * mvv) / V.unsqueeze(-1).unsqueeze(-1)
        return sigma_kin  # (B, 3, 3) eV/Å³

    def _compute_virial_stress_batch(self, structure):
        """
        Compute the virial contribution to the stress tensor (batched).

        σ^vir_αβ = -(1/V) Σ_i  r_iα f_iβ   (in eV/Å³)

        Returns (B, 3, 3) tensor.
        """
        V = torch.abs(torch.det(structure.cell))  # (B,)
        # positions: (B, N)  forces: (B, 3, N)
        R = torch.stack((structure.RX, structure.RY, structure.RZ), dim=-1)  # (B, N, 3)
        F = structure.f_tot.permute(0, 2, 1)  # (B, N, 3)
        # virial = -sum_i r_i ⊗ f_i
        virial = -torch.einsum("bia,bib->bab", R, F)  # (B, 3, 3)
        sigma_vir = virial / V.unsqueeze(-1).unsqueeze(-1)
        return sigma_vir  # (B, 3, 3) eV/Å³

    def _barostat_scale(self, structure, dt):
        """
        Apply Berendsen barostat scaling to cell and positions (batched).

        Scaling factor:
          μ = [1 - (β dt / (3 τ_P)) (P_target - P_inst)]^{1/3}

        Isotropic: scalar μ applied uniformly per batch element.
        Anisotropic: independent μ_α per axis from diagonal P_αα.
        """
        sigma_kin = self._compute_kinetic_stress_batch(structure)  # (B, 3, 3)
        sigma_vir = self._compute_virial_stress_batch(structure)  # (B, 3, 3)

        # Pressure tensor: P = σ_kin + σ_vir
        P_inst = sigma_kin + sigma_vir  # (B, 3, 3) eV/Å³

        # Scalar pressure = trace / 3
        P_scalar = torch.diagonal(P_inst, dim1=-2, dim2=-1).sum(dim=-1) / 3.0  # (B,)
        GPa_per_eVA3 = 160.21766208

        # Store for_ logging (GPa)
        self._P_inst_GPa = P_scalar * GPa_per_eVA3  # (B,)

        # Berendsen prefactor: β dt / (3 τ_P)
        prefactor = self.barostat_compressibility * dt / (3.0 * self.barostat_tau)

        if self.barostat_isotropic:
            dP = self.target_pressure - P_scalar  # (B,)
            mu = (1.0 - prefactor * dP) ** (1.0 / 3.0)  # (B,)

            # Scale cell: (B, 3, 3) * (B, 1, 1)
            structure.cell = structure.cell * mu.unsqueeze(-1).unsqueeze(-1)
            structure.cell_inv = torch.linalg.inv(structure.cell)

            # Scale positions: (B, N) * (B, 1)
            mu_expand = mu.unsqueeze(-1)
            structure.RX = structure.RX * mu_expand
            structure.RY = structure.RY * mu_expand
            structure.RZ = structure.RZ * mu_expand
        else:
            # Anisotropic: per-axis diagonal scaling (B, 3)
            diag_P = torch.diagonal(P_inst, dim1=-2, dim2=-1)  # (B, 3)
            dP = self.target_pressure - diag_P  # (B, 3)
            mu_diag = (1.0 - prefactor * dP) ** (1.0 / 3.0)  # (B, 3)

            # Scale cell: (B, 3, 3) * (B, 3, 1)
            structure.cell = structure.cell * mu_diag.unsqueeze(-1)
            structure.cell_inv = torch.linalg.inv(structure.cell)

            structure.RX = structure.RX * mu_diag[:, 0:1]
            structure.RY = structure.RY * mu_diag[:, 1:2]
            structure.RZ = structure.RZ * mu_diag[:, 2:3]

        # Wrap positions after rescaling
        R = torch.stack((structure.RX, structure.RY, structure.RZ), dim=-1)
        R = wrap_positions(R, structure.cell, structure.cell_inv)
        structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)

    def run(
        self,
        structure,
        dftorch_params,
        num_steps,
        dt,
        dump_interval=1,
        traj_filename="md_trj.xyz",
    ):

        # Store params for_ barostat (needs them during step)
        self._dftorch_params = dftorch_params

        if self.VX is None:
            self.VX, self.VY, self.VZ = initialize_velocities_batch(
                structure,
                temperature_K=self.temperature_K,
                remove_com=True,
                rescale_to_T=True,
                remove_angmom=True,
            )
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
            self.K0Res = torch.matmul(structure.KK, (q - self.n).unsqueeze(-1)).squeeze(
                -1
            )

        # Generate atom index for_ each orbital
        counts = structure.n_orbitals_per_atom  # shape (B, N)
        cum_counts = torch.cumsum(counts, dim=1)  # cumulative sums per batch
        total_orbs = int(cum_counts[0, -1].item())
        r = torch.arange(total_orbs, device=counts.device).expand(
            counts.size(0), -1
        )  # (B, total_orbs)
        # For each orbital position r[b,k], find first atom index whose cumulative count exceeds r[b,k]
        self.atom_ids = (
            (r.unsqueeze(2) < cum_counts.unsqueeze(1)).int().argmax(dim=2)
        )  # (B, total_orbs)
        self.Hubbard_U_gathered = structure.Hubbard_U.gather(1, self.atom_ids)
        if structure.dU_dq is not None:
            self.dU_dq_gathered = structure.dU_dq.gather(1, self.atom_ids)
        else:
            self.dU_dq_gathered = None

        self.PME_data = None
        self.nbr_inds = None
        self.disps = None
        self.dists = None
        self.CALPHA = None

        if self.E_array is None:
            self.E_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self.T_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self.Ek_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self.Ep_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self.Res_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )

        if self.barostat_enabled and self.P_array is None:
            self.P_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self.V_array = torch.empty(
                (0, structure.batch_size), device=structure.device
            )
            self._P_inst_GPa = torch.zeros(
                structure.batch_size, device=structure.device
            )

        self.EPOT = structure.e_tot
        for md_step in range(num_steps):
            self.step(
                structure, dftorch_params, md_step, dt, dump_interval, traj_filename
            )

    def step(
        self, structure, dftorch_params, md_step, dt, dump_interval, traj_filename
    ):
        if self.cuda_sync:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        print(
            "########## Step = {:} ##########".format(
                md_step,
            )
        )

        self.EKIN = (
            0.5
            * self.MVV2KE
            * torch.sum(structure.Mnuc * (self.VX**2 + self.VY**2 + self.VZ**2), dim=1)
        )  # Kinetic energy in eV (MVV2KE: unit conversion)
        Temperature = (
            (2 / 3) * self.KE2T * self.EKIN / structure.Nats
        )  # Statistical temperature in Kelvin
        Energ = (
            self.EKIN + self.EPOT
        )  # Total Energy in eV, Total energy fluctuations Propto dt^2
        # Time = md_step * dt
        ResErr = torch.norm(structure.q - self.n, dim=1) / (
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

        if md_step % dump_interval == 100000:
            comm_string = f"Etot = {Energ:.6f} eV, Epot = {self.EPOT:.6f} eV, Ekin = {self.EKIN:.6f} eV, T = {Temperature:.2f} K, Res = {ResErr:.6f}, mu = {structure.mu0:.4f} eV\n"
            write_XYZ_trajectory(traj_filename, structure, comm_string, step=md_step)
        self.VX = (
            self.VX
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 0] / structure.Mnuc)
            - self.fric * self.VX
        )  # First 1/2 of Leapfrog step
        self.VY = (
            self.VY
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 1] / structure.Mnuc)
            - self.fric * self.VY
        )  # F2V: Unit conversion
        self.VZ = (
            self.VZ
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 2] / structure.Mnuc)
            - self.fric * self.VZ
        )  # -c*V c>0 => Fricition
        # update positions and translate coordinates if go beyond box. Apply periodic boundary conditions
        if structure.cell is not None:
            R = torch.stack(
                (
                    structure.RX + dt * self.VX,
                    structure.RY + dt * self.VY,
                    structure.RZ + dt * self.VZ,
                ),
                dim=-1,
            )  # (B, N, 3)
            R = wrap_positions(R, structure.cell, structure.cell_inv)
            structure.RX, structure.RY, structure.RZ = R.unbind(dim=-1)
        else:
            structure.RX = structure.RX + dt * self.VX
            structure.RY = structure.RY + dt * self.VY
            structure.RZ = structure.RZ + dt * self.VZ

        if self.cuda_sync:
            torch.cuda.synchronize()
        tic2_1 = time.perf_counter()

        self.es_driver(structure, self.const, do_scf=False)
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

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("H0: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic2_1 = time.perf_counter()

        CoulPot = torch.matmul(structure.C, self.n.unsqueeze(-1)).squeeze(-1)

        # ── GBSA implicit solvation: rebuild for_ new geometry ────────────
        if dftorch_params.get("solvent_param_file", None) is not None:
            _gbsa_list = []
            for b in range(structure.batch_size):

                class _StructProxy:
                    pass

                proxy = _StructProxy()
                proxy.RX = structure.RX[b]
                proxy.RY = structure.RY[b]
                proxy.RZ = structure.RZ[b]
                proxy.TYPE = structure.TYPE[b]
                _gbsa_list.append(
                    create_gbsa(
                        proxy,
                        structure.device,
                        param_file=dftorch_params.get("solvent_param_file", None),
                        solvation_model=dftorch_params.get("solvation_model", "gbsa"),
                    )
                )
            structure.gbsa_list = _gbsa_list
            structure.gbsa_batch = GBSABatch(_gbsa_list)
            solv_shift = structure.gbsa_batch.get_shadow_shifts(self.n)  # (B, N)
            CoulPot = CoulPot + solv_shift
        else:
            solv_shift = None

        # ── Full off-diagonal DFTB3: add third-order shift to CoulPot ────
        if (
            hasattr(structure, "thirdorder_list")
            and structure.thirdorder_list is not None
        ):
            Nats = structure.Nats
            dev = structure.device
            for b in range(structure.batch_size):
                rxb, ryb, rzb = structure.RX[b], structure.RY[b], structure.RZ[b]
                dx = rxb.unsqueeze(0) - rxb.unsqueeze(1)
                dy = ryb.unsqueeze(0) - ryb.unsqueeze(1)
                dz = rzb.unsqueeze(0) - rzb.unsqueeze(1)
                dr = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)
                mask = (dr < 50.0) & (~torch.eye(Nats, dtype=torch.bool, device=dev))
                ni = torch.arange(Nats, device=dev).unsqueeze(1).expand(-1, Nats)[mask]
                nj = torch.arange(Nats, device=dev).unsqueeze(0).expand(Nats, -1)[mask]
                dR_m = dr[mask]
                dxyz = torch.stack([dx[mask], dy[mask], dz[mask]], dim=-1)
                dR_dxyz_m = dxyz / dR_m.unsqueeze(-1)
                structure.thirdorder_list[b].update_coords(
                    rxb, ryb, rzb, None, ni, nj, dR_m, dR_dxyz_m
                )
            structure.thirdorder_batch.refresh()
            thirdorder_shift = structure.thirdorder_batch.get_shifts(self.n)  # (B, N)
            CoulPot = CoulPot + thirdorder_shift
        else:
            thirdorder_shift = None

        (
            structure.q,
            structure.H,
            structure.Hcoul,
            structure.D,
            Dorth,
            Q,
            e,
            structure.f,
            structure.mu0,
        ) = calc_q_batch(
            structure.H0,
            self.Hubbard_U_gathered,
            self.n.gather(1, self.atom_ids),
            CoulPot.gather(1, self.atom_ids),
            structure.S,
            structure.Z,
            structure.Te,
            structure.Nocc,
            structure.Znuc,
            self.atom_ids,
            self.dU_dq_gathered if thirdorder_shift is None else None,
        )

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("H1: {:.3f} s".format(time.perf_counter() - tic2_1))
        tic3 = time.perf_counter()

        # Update Kernel
        Res = structure.q - self.n
        if md_step % 10000 == 0 and self.do_full_kernel:
            1
            # KK, _ = _kernel_fermi(
            #     structure,
            #     structure.mu0,
            #     structure.Te,
            #     structure.Nats,
            #     structure.H,
            #     C,
            #     S,
            #     Z,
            #     Q,
            #     e,
            # )
            # self.K0Res = KK @ Res
        elif self.NoRank:
            self.K0Res = -dftorch_params["SCF_ALPHA"] * Res
        else:  # Preconditioned Low-Rank Krylov _scf acceleration
            self.K0Res = kernel_update_lr_batch(
                structure.Nats,
                self.Hubbard_U_gathered,
                dftorch_params,
                dftorch_params["KRYLOV_TOL_MD"],
                structure.KK.clone(),
                Res,
                structure.q,
                structure.S,
                structure.Z,
                self.PME_data,
                self.atom_ids,
                Q,
                e,
                structure.mu0,
                structure.Te,
                structure.C,
                self.nbr_inds,
                self.disps,
                self.dists,
                self.CALPHA,
                self.dU_dq_gathered if thirdorder_shift is None else None,
                gbsa=structure.gbsa_batch if hasattr(structure, "gbsa_batch") else None,
                thirdorder=structure.thirdorder_batch
                if hasattr(structure, "thirdorder_batch")
                else None,
            )

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("KER: {:.3f} s".format(time.perf_counter() - tic3))
        tic4 = time.perf_counter()

        (
            structure.e_elec_tot,
            structure.e_band0,
            structure.e_coul,
            structure.e_dipole,
            structure.e_entropy,
            structure.s_ent,
        ) = energy_shadow(
            structure.H0,
            structure.Hubbard_U,
            structure.e_field,
            structure.D0,
            structure.C,
            None,
            structure.D,
            structure.q,
            self.n,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.f,
            structure.Te,
            structure.dU_dq if thirdorder_shift is None else None,
        )

        structure.e_tot = structure.e_elec_tot + structure.e_repulsion

        # ── Full off-diagonal DFTB3: add per-structure thirdorder energy ─
        if (
            hasattr(structure, "thirdorder_batch")
            and structure.thirdorder_batch is not None
        ):
            structure.e_thirdorder = structure.thirdorder_batch.get_energy_xlbomd(
                self.n, structure.q
            )  # (B,)
            structure.e_tot = structure.e_tot + structure.e_thirdorder

        # ── GBSA shadow solvation energy (vectorised) ────────────────────
        if hasattr(structure, "gbsa_batch") and structure.gbsa_batch is not None:
            structure.e_gb, structure.e_sasa = structure.gbsa_batch.get_shadow_energies(
                structure.q, self.n
            )  # each (B,)
            structure.e_tot = structure.e_tot + structure.e_gb + structure.e_sasa

        # ── D3(BJ) dispersion energy (vectorised) ──────────────────────
        if hasattr(structure, "dftd3") and structure.dftd3 is not None:
            coords_batch = torch.stack(
                [structure.RX, structure.RY, structure.RZ], dim=2
            )  # (B, N, 3)
            structure.e_d3 = structure.dftd3.get_energy_batch(coords_batch)  # (B,)
            structure.e_tot = structure.e_tot + structure.e_d3

        self.EPOT = structure.e_tot

        (
            structure.f_tot,
            structure.f_coul,
            structure.f_band0,
            structure.f_dipole,
            structure.f_pulay,
            structure.f_s_coul,
            structure.f_s_dipole,
            structure.f_rep,
        ) = forces_shadow_batch(
            structure.H,
            structure.Z,
            structure.C,
            structure.D,
            structure.D0,
            structure.dH0,
            structure.dS,
            structure.dCC,
            structure.dVr,
            structure.e_field,
            structure.Hubbard_U,
            structure.q,
            self.n,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.Nats,
            self.const,
            structure.TYPE,
            structure.dU_dq if thirdorder_shift is None else None,
            solvation_shift=solv_shift,
            thirdorder_shift=thirdorder_shift,
        )

        # ── GBSA shadow solvation gradients (vectorised) ──────────────────
        if hasattr(structure, "gbsa_batch") and structure.gbsa_batch is not None:
            f_sasa = structure.gbsa_batch.get_shadow_sasa_gradients(
                structure.q, self.n
            ).permute(0, 2, 1)  # (B,N,3) → (B,3,N)
            f_born = structure.gbsa_batch.get_shadow_born_gradients(
                structure.q, self.n
            ).permute(0, 2, 1)  # (B,N,3) → (B,3,N)
            structure.f_gbsa = f_sasa + f_born  # (B,3,N)
            structure.f_tot = structure.f_tot + structure.f_gbsa

        # ── Full off-diagonal DFTB3: gradient from dΓ³/dr (vectorised) ──
        if (
            hasattr(structure, "thirdorder_batch")
            and structure.thirdorder_batch is not None
        ):
            structure.f_thirdorder = structure.thirdorder_batch.get_gradient_dc_xlbomd(
                self.n, structure.q
            )  # (B, 3, N)
            structure.f_tot = structure.f_tot + structure.f_thirdorder

        # ── D3(BJ) dispersion forces (vectorised) ────────────────────────
        if hasattr(structure, "dftd3") and structure.dftd3 is not None:
            coords_batch = torch.stack(
                [structure.RX, structure.RY, structure.RZ], dim=2
            )  # (B, N, 3)
            structure.f_d3 = structure.dftd3.get_forces_batch(coords_batch)  # (B,3,N)
            structure.f_tot = structure.f_tot + structure.f_d3

        self.VX = (
            self.VX
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 0] / structure.Mnuc)
            - self.fric * self.VX
        )  # Integrate second 1/2 of leapfrog step
        self.VY = (
            self.VY
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 1] / structure.Mnuc)
            - self.fric * self.VY
        )  # - c*V  c > 0 => friction
        self.VZ = (
            self.VZ
            + 0.5 * dt * (self.F2V * structure.f_tot[:, 2] / structure.Mnuc)
            - self.fric * self.VZ
        )

        # ── Langevin half-kick (after force update) ──────────────────────
        if self.langevin_enabled:
            self._langevin_kick(structure, dt)

        # ── Berendsen barostat: rescale cell + positions ─────────────────
        if self.barostat_enabled:
            self._barostat_scale(structure, dt)
            V = torch.abs(torch.det(structure.cell))  # (B,)
            self.P_array = torch.cat(
                (self.P_array, self._P_inst_GPa.detach().unsqueeze(0)), dim=0
            )
            self.V_array = torch.cat((self.V_array, V.detach().unsqueeze(0)), dim=0)

        if self.cuda_sync:
            torch.cuda.synchronize()
        print("F AND E: {:.3f} s".format(time.perf_counter() - tic4))

        if self.barostat_enabled:
            V = torch.abs(torch.det(structure.cell))  # (B,)

        for b in range(structure.batch_size):
            P_str = ""
            if self.barostat_enabled:
                P_str = f", P = {self._P_inst_GPa[b].item():.4f} GPa, V = {V[b].item():.2f} Å³"
            print(
                "ETOT = {:.8f}, EPOT = {:.8f}, EKIN = {:.8f}, T = {:.8f}, ResErr = {:.6f}{}, t = {:.1f} s".format(
                    Energ[b].item(),
                    self.EPOT[b].item(),
                    self.EKIN[b].item(),
                    Temperature[b].item(),
                    ResErr[b].item(),
                    P_str,
                    time.perf_counter() - start_time,
                )
            )
        print(torch.cuda.memory_allocated() / 1e9, "GB\n")
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
        RNG for_ reproducible sampling.

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

    # Sampling stddev (0 for_ zero-mass)
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
        R = torch.stack(
            (structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)),
            dim=1,
        )  # (N,3)
        Mtot = masses_amu[mpos].sum()
        r_com = (masses_amu[:, None] * R).sum(dim=0) / Mtot
        r_rel = R - r_com[None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:, None] * torch.cross(r_rel, V, dim=1))[mpos].sum(
            dim=0
        )  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device)
        r2 = (r_rel[mpos] * r_rel[mpos]).sum(dim=1)  # (npos,)
        I = (  # noqa: E741
            masses_amu[mpos][:, None, None]
            * (
                r2[:, None, None] * eye3
                - r_rel[mpos][:, :, None] * r_rel[mpos][:, None, :]
            )
        ).sum(dim=0)  # (3,3)

        # Solve I * omega = L; use pinv for_ robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.linalg.pinv(I) @ L  # (3,)

        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if npos > 0:
            deltaV = torch.cross(
                r_rel[mpos], omega.expand_as(r_rel[mpos]), dim=1
            )  # (npos,3)
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
                scale = torch.sqrt(
                    torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur
                )
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

    if len(temperature_K) == 1 and batch_size > 1:
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

    # Sampling stddev (0 for_ zero-mass)
    kT = torch.as_tensor(kB_eV_per_K * temperature_K, device=device, dtype=dtype)
    inv_m = torch.where(mpos, 1.0 / masses_amu, torch.zeros_like(masses_amu))
    std = torch.sqrt(torch.clamp(kT.unsqueeze(-1) * inv_m / MVV2KE, min=0)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = (
        torch.randn((batch_size, N, 3), device=device, dtype=dtype, generator=g)
        * std[:, :, None]
    )

    # Remove center-of-mass velocity
    if remove_com and (npos > 1).any():
        Mtot = masses_amu.sum(-1)
        v_cm = (masses_amu[:, :, None] * V).sum(dim=1) / Mtot.unsqueeze(-1)
        V = V - v_cm[:, None, :]

    # Optionally remove net angular momentum (about COM, without changing coordinates)
    if remove_angmom and (npos > 1).any():
        # Positions
        R = torch.stack(
            (structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)),
            dim=2,
        )  # (N,3)
        Mtot = masses_amu.sum(-1)
        r_com = (masses_amu[:, :, None] * R).sum(dim=1) / Mtot.unsqueeze(-1)
        r_rel = R - r_com[:, None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:, :, None] * torch.cross(r_rel, V, dim=2)).sum(dim=1)  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device) * torch.ones(
            (batch_size, 3, 3), dtype=dtype, device=device
        )
        r2 = (r_rel * r_rel).sum(dim=2)  # (npos,)
        I = (  # noqa: E741
            masses_amu[:, :, None, None]
            * (
                r2[:, :, None, None] * eye3.unsqueeze(1)
                - r_rel[:, :, :, None] * r_rel[:, :, None, :]
            )
        ).sum(dim=1)  # (3,3)

        # Solve I * omega = L; use pinv for_ robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.bmm(torch.linalg.pinv(I), L.unsqueeze(-1)).squeeze(-1)
        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if (npos > 1).any():
            deltaV = torch.cross(
                r_rel, omega[:, None, :].expand_as(r_rel), dim=2
            )  # (npos,3)
            V = V + deltaV
    if rescale_to_T and (npos > 0).any():
        dof = 3 * npos
        if remove_com and (npos > 1).any():
            dof -= 3
        if remove_angmom and (npos > 2).any():
            dof -= 3
        if (dof > 0).any():
            EKIN = 0.5 * MVV2KE * (masses_amu[:, :, None] * (V * V)).sum(dim=(1, 2))
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if (T_cur > 0).any():
                scale = torch.sqrt(
                    torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur
                )
                V = V * scale.unsqueeze(-1).unsqueeze(-1)

    return V[:, :, 0].contiguous(), V[:, :, 1].contiguous(), V[:, :, 2].contiguous()


__all__ = ["MDXL", "MDXLBatch", "MDXLOS"]
