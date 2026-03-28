import torch
from ._h0ands import H0_and_S_vectorized, H0_and_S_vectorized_batch
from ._repulsive_spline import get_repulsion_energy, get_repulsion_energy_batch
from ._nearestneighborlist import (
    vectorized_nearestneighborlist,
    vectorized_nearestneighborlist_batch,
)
from ._tools import fractional_matrix_power_symm
from ._scf import scf_x_os, SCFx, SCFx_batch
from ._energy import energy
from ._forces import Forces, forces_spin, Forces_PME
from ._forces_batch import forces_batch
from ._coulomb_matrix import coulomb_matrix_vectorized
from dftorch._coulomb_matrix_batch import coulomb_matrix_vectorized_batch
from dftorch._spin import get_spin_energy, get_h_spin
from ._stress import get_total_stress_analytical
from ._gbsa import create_gbsa
from ._dftd3 import create_dftd3
from ._thirdorder import create_thirdorder


import math


class ESDriver(torch.nn.Module):
    def __init__(
        self,
        dftorch_params,
        electronic_rcut: float,
        repulsive_rcut: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dftorch_params = dftorch_params
        self.electronic_rcut = electronic_rcut
        self.repulsive_rcut = repulsive_rcut
        self.device = device

    def forward(
        self, structure, const, do_scf=True, verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total energy and forces for given density matrix.

        Parameters
        ----------
        density_matrix : (HDIM, HDIM) torch.Tensor
            Density matrix in AO basis.
        verbose : bool
            If True, print timing info from neighbor list routines.

        Returns
        -------
        total_energy : torch.Tensor (scalar)
            Total energy (electronic + repulsive).
        forces : (Nats, 3) torch.Tensor
            Atomic forces in eV/Å.
        """

        # Build the neighborlist

        (
            _,
            _,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            _,
            _,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
        ) = vectorized_nearestneighborlist(
            structure.TYPE,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.cell,
            self.electronic_rcut,
            structure.Nats,
            const,
            upper_tri_only=False,
            remove_self_neigh=False,
            verbose=verbose,
        )

        # Get Hamiltonian, Overlap, etc,
        structure.H0, structure.dH0, structure.S, structure.dS = H0_and_S_vectorized(
            structure.TYPE,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.diagonal,
            structure.H_INDEX_START,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            const,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
            const.R_orb,
            const.coeffs_tensor,
            verbose=verbose,
            store_stress_metadata=const,
        )
        del (
            _,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
        )
        structure.Z = fractional_matrix_power_symm(structure.S, -0.5)

        # nuclear repulsion
        structure.e_repulsion, structure.dVr, structure.stress_repulsion = (
            get_repulsion_energy(
                const.R_rep_tensor,
                const.rep_splines_tensor,
                const.close_exp_tensor,
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                self.repulsive_rcut,
                structure.Nats,
                const,
                verbose=verbose,
            )
        )

        if self.dftorch_params["coul_method"] == "PME":
            structure.C = None
            structure.dCC = None
            # TODO: Full off-diagonal DFTB3 with PME requires building a
            # separate neighbor list for the short-range Γ³ matrix.
            # For now, PME uses diagonal-only DFTB3.
            structure.thirdorder = None
        else:
            Coulomb_acc = self.dftorch_params["Coulomb_acc"]
            SQRTX = math.sqrt(-math.log(Coulomb_acc))
            COULCUT = self.dftorch_params["cutoff"]
            CALPHA = SQRTX / COULCUT
            if COULCUT > 50.0 and structure.cell is not None:
                COULCUT = 50.0
                CALPHA = SQRTX / COULCUT

            # Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
            (
                _,
                nndist,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                nnStruct,
                _,
                neighbor_I,
                neighbor_J,
                IJ_pair_type,
                JI_pair_type,
            ) = vectorized_nearestneighborlist(
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                COULCUT,
                structure.Nats,
                const,
                upper_tri_only=False,
                remove_self_neigh=False,
                verbose=verbose,
            )

            structure.C, structure.dCC = coulomb_matrix_vectorized(
                structure.Hubbard_U,
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                structure.Nats,
                Coulomb_acc,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                neighbor_I,
                neighbor_J,
                CALPHA,
                verbose=verbose,
                h_damp_exp=self.dftorch_params.get("h_damp_exp", None),
                h5_params=self.dftorch_params.get("h5_params", None),
            )

            # ── Full off-diagonal DFTB3 third-order matrices ────────────
            if (
                structure.dU_dq is not None
                and self.dftorch_params.get("dftb3_diagonal_only", False) is False
            ):
                # Compute pairwise distances/unit-vectors for the masked pairs
                Ra = torch.stack(
                    (
                        structure.RX.unsqueeze(-1),
                        structure.RY.unsqueeze(-1),
                        structure.RZ.unsqueeze(-1),
                    ),
                    dim=-1,
                )
                Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
                Rab = Rb - Ra
                dR_full = torch.norm(Rab, dim=-1)
                dR_dxyz_full = Rab / dR_full.unsqueeze(-1).clamp(min=1e-30)
                nn_mask = nnType != -1
                dR_masked = dR_full[nn_mask]
                dR_dxyz_masked = dR_dxyz_full[nn_mask]  # (Npairs, 3)

                structure.thirdorder = create_thirdorder(
                    structure.Hubbard_U,
                    structure.dU_dq,
                    structure.TYPE,
                    h_damp_exp=self.dftorch_params.get("h_damp_exp", 4.0),
                )
                structure.thirdorder.update_coords(
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.cell,
                    neighbor_I,
                    neighbor_J,
                    dR_masked,
                    dR_dxyz_masked,
                )
            else:
                structure.thirdorder = None
            # ────────────────────────────────────────────────────────────

            del (
                _,
                nndist,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                nnStruct,
                neighbor_I,
                neighbor_J,
                IJ_pair_type,
                JI_pair_type,
            )

        if do_scf:
            # --- GBSA / ALPB implicit solvation ---
            if self.dftorch_params.get("solvent", None) is not None:
                structure.gbsa = create_gbsa(
                    structure,
                    self.device,
                    solvent=self.dftorch_params["solvent"],
                    param_file=self.dftorch_params.get("solvent_param_file", None),
                    solvation_model=self.dftorch_params.get("solvation_model", "alpb"),
                )
            else:
                structure.gbsa = None

            # --- D3(BJ) dispersion correction ---
            d3_params = self.dftorch_params.get("d3_params", None)
            if d3_params is not None:
                structure.dftd3 = create_dftd3(
                    atomic_numbers=structure.TYPE.cpu().numpy().astype(int),
                    s6=d3_params.get("s6", 1.0),
                    s8=d3_params.get("s8", 0.5883),
                    a1=d3_params.get("a1", 0.5719),
                    a2=d3_params.get("a2", 3.6017),
                    device=self.device,
                    dtype=torch.float64,
                )
            else:
                structure.dftd3 = None

            if self.dftorch_params.get("UNRESTRICTED", False):  # open-shell
                (
                    structure.H,
                    structure.Hcoul,
                    structure.Hdipole,
                    structure.KK,
                    structure.D,
                    structure.Q,
                    structure.q_spin_atom,
                    structure.q_tot_atom,
                    structure.q_spin_sr,
                    structure.net_spin_sr,
                    structure.f,
                    structure.mu0,
                    structure.e_coul_tmp,
                    structure.f_coul,
                    structure.dq_p1,
                    structure.stress_coulomb,
                ) = scf_x_os(
                    structure.el_per_shell,
                    structure.shell_types,
                    structure.n_shells_per_atom,
                    const.shell_dim,
                    const.w,
                    self.dftorch_params,
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.cell,
                    structure.Nats,
                    structure.Nocc,
                    structure.n_orbitals_per_atom,
                    structure.Znuc,
                    structure.TYPE,
                    structure.Te,
                    structure.Hubbard_U,
                    structure.dU_dq,
                    structure.D0,
                    structure.H0,
                    structure.S,
                    structure.Z,
                    structure.e_field,
                    structure.C,
                    structure.req_grad_xyz,
                    structure.q_spin_sr,
                    gbsa=structure.gbsa,
                    thirdorder=structure.thirdorder,
                )

                structure.q = structure.q_tot_atom

                H_spin = get_h_spin(
                    structure.TYPE,
                    structure.net_spin_sr,
                    const.w,
                    structure.n_shells_per_atom,
                    structure.shell_types,
                )
                structure.H_spin = (
                    0.5
                    * structure.S
                    * H_spin.unsqueeze(0).expand(2, -1, -1)
                    * torch.tensor([[[1]], [[-1]]], device=H_spin.device)
                )
                structure.e_spin = get_spin_energy(
                    structure.TYPE,
                    structure.net_spin_sr,
                    const.w,
                    structure.n_shells_per_atom,
                )

            else:  # closed-shell
                (
                    structure.H,
                    structure.Hcoul,
                    structure.Hdipole,
                    structure.KK,
                    structure.D,
                    structure.Q,
                    structure.q,
                    structure.f,
                    structure.mu0,
                    structure.e_coul_tmp,
                    structure.f_coul,
                    structure.dq_p1,
                    structure.stress_coulomb,
                ) = SCFx(
                    self.dftorch_params,
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.cell,
                    structure.Nats,
                    structure.Nocc,
                    structure.n_orbitals_per_atom,
                    structure.Znuc,
                    structure.TYPE,
                    structure.Te,
                    structure.Hubbard_U,
                    structure.dU_dq,
                    structure.D0,
                    structure.H0,
                    structure.S,
                    structure.Z,
                    structure.e_field,
                    structure.C,
                    structure.req_grad_xyz,
                    structure.q,
                    gbsa=structure.gbsa,
                    thirdorder=structure.thirdorder,
                )
                structure.e_spin = 0.0

            (
                structure.e_elec_tot,
                structure.e_band0,
                structure.e_coul,
                structure.e_dipole,
                structure.e_entropy,
                structure.s_ent,
            ) = energy(
                structure.H0,
                structure.Hubbard_U,
                structure.e_field,
                structure.D0,
                structure.C,
                structure.dq_p1,
                structure.D,
                structure.q,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.f,
                structure.Te,
                structure.dU_dq,
                thirdorder=structure.thirdorder,
            )

            structure.e_tot = (
                structure.e_elec_tot + structure.e_repulsion + structure.e_spin
            )

            # --- GBSA solvation energy (differentiable) ---
            if structure.gbsa is not None:
                coords_ang = torch.stack(
                    [structure.RX, structure.RY, structure.RZ], dim=1
                )
                structure.e_solv = structure.gbsa.get_energy_differentiable(
                    coords_ang,
                    structure.q,
                )
                # Keep non-differentiable e_gb / e_sasa for reporting
                e_gb, e_sasa = structure.gbsa.get_energies(structure.q)
                structure.e_gb = e_gb
                structure.e_sasa = e_sasa
                structure.e_tot = structure.e_tot + structure.e_solv
            else:
                structure.e_gb = 0.0
                structure.e_sasa = 0.0
                structure.e_solv = 0.0

            # --- D3(BJ) dispersion energy ---
            if structure.dftd3 is not None:
                coords_ang = torch.stack(
                    [structure.RX, structure.RY, structure.RZ], dim=1
                )
                structure.e_d3 = structure.dftd3.get_energy(coords_ang)
                structure.e_tot = structure.e_tot + structure.e_d3
            else:
                structure.e_d3 = 0.0

    def calc_forces(self, structure, const):

        with torch.no_grad():
            # Compute solvation shift for force calculation
            solv_shift = None
            if structure.gbsa is not None:
                solv_shift = structure.gbsa.get_shifts(structure.q)

            # Compute third-order shift for force calculation
            to_shift = None
            if structure.thirdorder is not None:
                to_shift = structure.thirdorder.get_shifts(structure.q)

            if (
                self.dftorch_params["coul_method"] == "PME"
            ):  # f_coul was calculated in _scf via calculate_PME_ewald
                # structure.f_coul was calculated in _scf via calculate_PME_ewald
                (
                    f_tot,
                    _,
                    structure.f_band0,
                    structure.f_dipole,
                    structure.f_pulay,
                    structure.f_s_coul,
                    structure.f_s_dipole,
                    structure.f_rep,
                ) = Forces_PME(
                    structure.H,
                    structure.Z,
                    structure.dq_p1,
                    structure.D,
                    structure.D0,
                    structure.dH0,
                    structure.dS,
                    structure.dVr,
                    structure.e_field,
                    structure.Hubbard_U,
                    structure.q,
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.Nats,
                    const,
                    structure.TYPE,
                    structure.dU_dq,
                    solvation_shift=solv_shift,
                )
                structure.f_tot = f_tot + structure.f_coul
            else:
                (
                    structure.f_tot,
                    structure.f_coul,
                    structure.f_band0,
                    structure.f_dipole,
                    structure.f_pulay,
                    structure.f_s_coul,
                    structure.f_s_dipole,
                    structure.f_rep,
                ) = Forces(
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
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.Nats,
                    const,
                    structure.TYPE,
                    structure.dU_dq if structure.thirdorder is None else None,
                    solvation_shift=solv_shift,
                    thirdorder_shift=to_shift,
                )

            if self.dftorch_params.get("UNRESTRICTED", False):  # open-shell
                structure.f_spin = forces_spin(
                    structure.D,
                    structure.dS,
                    structure.q_spin_atom,
                    structure.Nats,
                    const,
                    structure.TYPE,
                )

                structure.f_tot = structure.f_tot + structure.f_spin

            # --- GBSA solvation gradients ---
            if structure.gbsa is not None:
                structure.f_gbsa_sasa = structure.gbsa.get_sasa_gradients(
                    structure.q
                ).T  # (3, N)
                structure.f_gbsa_born = structure.gbsa.get_born_gradients(
                    structure.q
                ).T  # (3, N)
                structure.f_gbsa = structure.f_gbsa_sasa + structure.f_gbsa_born
                structure.f_tot = structure.f_tot + structure.f_gbsa

            # --- Full DFTB3 off-diagonal gradient (dΓ³/dr) ---
            if structure.thirdorder is not None:
                structure.f_thirdorder_dc = structure.thirdorder.get_gradient_dc(
                    structure.q
                )
                structure.f_tot = structure.f_tot + structure.f_thirdorder_dc

            # --- D3(BJ) dispersion forces (analytical) ---
            if structure.dftd3 is not None:
                coords_ang = torch.stack(
                    [structure.RX, structure.RY, structure.RZ], dim=1
                )
                structure.f_d3 = structure.dftd3.get_forces(coords_ang).to(self.device)
                structure.f_tot = structure.f_tot + structure.f_d3

    def calc_stress(self, structure, const):
        stress_dict = get_total_stress_analytical(
            structure,
            const,
            repulsive_rcut=self.repulsive_rcut,
            dftorch_params=self.dftorch_params,
            verbose=False,
        )

        structure.stress_repulsion = stress_dict["repulsion"]
        structure.stress_band = stress_dict["band"]
        structure.stress_overlap = stress_dict["overlap"]
        structure.stress_coulomb = stress_dict["coulomb"]
        structure.stress_tot = stress_dict["total"]


class ESDriverBatch(torch.nn.Module):
    def __init__(
        self,
        dftorch_params,
        electronic_rcut: float,
        repulsive_rcut: float,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dftorch_params = dftorch_params
        self.electronic_rcut = electronic_rcut
        self.repulsive_rcut = repulsive_rcut
        self.device = device

    def forward(
        self, structure, const, do_scf=True, verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total energy and forces for given density matrix.

        Parameters
        ----------
        density_matrix : (HDIM, HDIM) torch.Tensor
            Density matrix in AO basis.
        verbose : bool
            If True, print timing info from neighbor list routines.

        Returns
        -------
        total_energy : torch.Tensor (scalar)
            Total energy (electronic + repulsive).
        forces : (Nats, 3) torch.Tensor
            Atomic forces in eV/Å.
        """

        # Build the neighborlist

        (
            _,
            _,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            _,
            _,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
        ) = vectorized_nearestneighborlist_batch(
            structure.TYPE,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.cell,
            self.electronic_rcut,
            structure.Nats,
            const,
            upper_tri_only=False,
            remove_self_neigh=False,
            min_image_only=False,
            verbose=verbose,
        )

        # Get Hamiltonian, Overlap, etc,
        structure.H0, structure.dH0, structure.S, structure.dS = (
            H0_and_S_vectorized_batch(
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.diagonal,
                structure.H_INDEX_START,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                const,
                neighbor_I,
                neighbor_J,
                IJ_pair_type,
                JI_pair_type,
                const.R_orb,
                const.coeffs_tensor,
                verbose=verbose,
            )
        )

        del (
            _,
            nnRx,
            nnRy,
            nnRz,
            nnType,
            neighbor_I,
            neighbor_J,
            IJ_pair_type,
            JI_pair_type,
        )
        structure.Z = fractional_matrix_power_symm(structure.S, -0.5)

        # nuclear repulsion
        structure.e_repulsion, structure.dVr = get_repulsion_energy_batch(
            const.R_rep_tensor,
            const.rep_splines_tensor,
            const.close_exp_tensor,
            structure.TYPE,
            structure.RX,
            structure.RY,
            structure.RZ,
            structure.cell,
            self.repulsive_rcut,
            structure.Nats,
            const,
            verbose=verbose,
        )

        if self.dftorch_params["coul_method"] == "PME":
            structure.C = None
            structure.dCC = None
            raise ValueError("Batched PME Coulomb not implemented.")
            return
        else:
            Coulomb_acc = self.dftorch_params["Coulomb_acc"]
            SQRTX = math.sqrt(-math.log(Coulomb_acc))
            COULCUT = self.dftorch_params["cutoff"]
            CALPHA = SQRTX / COULCUT
            if COULCUT > 50.0:
                COULCUT = 50.0
                CALPHA = SQRTX / COULCUT

            # Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
            (
                _,
                _,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                _,
                _,
                neighbor_I,
                neighbor_J,
                IJ_pair_type,
                JI_pair_type,
            ) = vectorized_nearestneighborlist_batch(
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                COULCUT,
                structure.Nats,
                const,
                upper_tri_only=False,
                remove_self_neigh=False,
                verbose=verbose,
            )

            structure.C, structure.dCC = coulomb_matrix_vectorized_batch(
                structure.Hubbard_U,
                structure.TYPE,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.cell,
                structure.Nats,
                Coulomb_acc,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                neighbor_I,
                neighbor_J,
                CALPHA,
                verbose=verbose,
                h_damp_exp=self.dftorch_params.get("h_damp_exp", None),
                h5_params=self.dftorch_params.get("h5_params", None),
            )

            del (
                _,
                nnRx,
                nnRy,
                nnRz,
                nnType,
                neighbor_I,
                neighbor_J,
                IJ_pair_type,
                JI_pair_type,
            )

        if do_scf:
            (
                structure.H,
                structure.Hcoul,
                structure.Hdipole,
                structure.KK,
                structure.D,
                structure.q,
                structure.f,
                structure.mu0,
                structure.e_coul_tmp,
                structure.f_coul,
                structure.dq_p1,
            ) = SCFx_batch(
                self.dftorch_params,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.Nats,
                structure.Nocc,
                structure.n_orbitals_per_atom,
                structure.Znuc,
                structure.Te,
                structure.Hubbard_U,
                structure.dU_dq,
                structure.D0,
                structure.H0,
                structure.S,
                structure.Z,
                structure.e_field,
                structure.C,
            )

            (
                structure.e_elec_tot,
                structure.e_band0,
                structure.e_coul,
                structure.e_dipole,
                structure.e_entropy,
                structure.s_ent,
            ) = energy(
                structure.H0,
                structure.Hubbard_U,
                structure.e_field,
                structure.D0,
                structure.C,
                structure.dq_p1,
                structure.D,
                structure.q,
                structure.RX,
                structure.RY,
                structure.RZ,
                structure.f,
                structure.Te,
                structure.dU_dq,
            )

            structure.e_tot = structure.e_elec_tot + structure.e_repulsion

    def calc_forces(self, structure, const):

        # with torch.no_grad():
        if 1:
            if self.dftorch_params["coul_method"] == "PME":
                raise ValueError("Batched PME Coulomb not implemented.")
                return
            else:
                (
                    structure.f_tot,
                    structure.f_coul,
                    structure.f_band0,
                    structure.f_dipole,
                    structure.f_pulay,
                    structure.f_s_coul,
                    structure.f_s_dipole,
                    structure.f_rep,
                ) = forces_batch(
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
                    structure.RX,
                    structure.RY,
                    structure.RZ,
                    structure.Nats,
                    const,
                    structure.TYPE,
                    structure.dU_dq,
                )


__all__ = ["ESDriver", "ESDriverBatch"]
