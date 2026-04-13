import torch
from ._h0ands import H0_and_S_vectorized, H0_and_S_vectorized_batch
from ._repulsive_spline import get_repulsion_energy, get_repulsion_energy_batch
from ._nearestneighborlist import (
    vectorized_nearestneighborlist,
    vectorized_nearestneighborlist_batch,
)
from ._tools import fractional_matrix_power_symm
from ._scf import scf_x_os, SCFx, SCFx_batch, delta_scf_x_os
from ._energy import energy
from ._forces import Forces, forces_spin, Forces_PME
from ._forces_batch import forces_batch
from ._coulomb_matrix import coulomb_matrix_vectorized
from dftorch._coulomb_matrix_batch import coulomb_matrix_vectorized_batch
from dftorch._spin import get_spin_energy, get_h_spin
from ._stress import get_total_stress_analytical
from ._gbsa import create_gbsa, GBSABatch
from ._dftd3 import create_dftd3
from ._thirdorder import create_thirdorder, ThirdOrderBatch


import math
import time


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
        Compute total energy and forces for_ given density matrix.

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
            ml_model_data=getattr(self, "ml_model_data", None),
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
            # separate neighbor list for_ the short-range Γ³ matrix.
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
                # Compute pairwise distances/unit-vectors for_ the masked pairs
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
                    h_damp_exp=self.dftorch_params.get("h_damp_exp", None),
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
            if self.dftorch_params.get("solvent_param_file", None) is not None:
                structure.gbsa = create_gbsa(
                    structure,
                    self.device,
                    param_file=self.dftorch_params.get("solvent_param_file", None),
                    solvation_model=self.dftorch_params.get("solvation_model", "gbsa"),
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
                    structure.e,
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
                # print(f"GS charges: {structure.q}")

                if self.dftorch_params.get("DELTA_SCF", False):
                    # print("detecting delta SCF calculation")
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
                    ) = delta_scf_x_os(
                        structure.el_per_shell,
                        structure.shell_types,
                        structure.n_shells_per_atom,
                        const.shell_dim,
                        const.w,
                        self.dftorch_params,
                        structure.RX,
                        structure.RY,
                        structure.RZ,
                        structure.lattice_vecs,
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
                        structure.H,
                        structure.D,
                        structure.f,
                        structure.mu0,
                        structure.S,
                        structure.Z,
                        structure.e_field,
                        structure.C,
                        structure.req_grad_xyz,
                    )

                    structure.q = structure.q_tot_atom
                    # print(f"ES charges: {structure.q}")

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
                    structure.e,
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

            # --- GBSA solvation energy ---
            if structure.gbsa is not None:
                e_gb, e_sasa = structure.gbsa.get_energies(structure.q)
                structure.e_gb = e_gb
                structure.e_sasa = e_sasa

                if self.dftorch_params.get("gbsa_differentiable", False):
                    # Differentiable path: recomputes Born radii, Born
                    # matrix and SASA with autograd ops.  Enables
                    # backprop through coordinates but allocates O(N²)
                    # and O(N·G·N) intermediates — may OOM on large
                    # systems.
                    coords_ang = torch.stack(
                        [structure.RX, structure.RY, structure.RZ], dim=1
                    )
                    structure.e_solv = structure.gbsa.get_energy_differentiable(
                        coords_ang,
                        structure.q,
                    )
                else:
                    # Non-differentiable path: uses pre-computed Born
                    # matrix and SASA from the GBSA constructor.
                    # Analytical gradients are still available via
                    # calc_forces → get_born_gradients / get_sasa_gradients.
                    structure.e_solv = e_gb + e_sasa

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

        # with torch.no_grad():
        if 1:
            # Compute solvation shift for_ force calculation
            solv_shift = None
            if structure.gbsa is not None:
                solv_shift = structure.gbsa.get_shifts(structure.q)

            # Compute third-order shift for_ force calculation
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
                    net_spin_sr=structure.net_spin_sr,
                    n_shells_per_atom=structure.n_shells_per_atom,
                    shell_types=structure.shell_types,
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

    def calc_hessian(
        self,
        structure,
        const,
        mode: str = "frozen_gbsa",
        batch_size: int = 32,
        delta: float = 1e-4,
        filename: str = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Compute the Hessian matrix d²E/dR² (3N × 3N).

        Two modes are available:

        ``"full"``
            Full finite-difference Hessian.  Each displaced structure gets its
            own GBSA construction (Born radii, SASA, Born matrix recomputed at
            every geometry).  Most accurate; slowest.

        ``"frozen_gbsa"``
            Finite-difference Hessian with frozen GBSA geometry.  A single GBSA
            object is computed once at the reference geometry and reused for all
            displaced structures via ``GBSABatch.from_single``.  For δ ≈ 1e-4 Å
            the frozen-GBSA error is ~0.1–0.6 eV/Å² on individual elements.
            About 3× faster than ``"full"`` when GBSA solvation is enabled.

        .. note::

            An autograd mode (backprop through SCF iterations) was tested and
            found to give **incorrect** Hessians.  Unrolled backprop through
            the self-consistent field iterations does not satisfy the implicit
            function theorem; the resulting second derivatives are off by
            factors of 2–7×.  Correct autograd Hessians would require an
            adjoint / Z-vector formulation, which is not yet implemented.

        Parameters
        ----------
        structure : Structure
            A *non-batch* ``Structure`` that has already been through
            ``forward`` and ``calc_forces``.
        const : Constants
            The ``Constants`` object matching this structure.
        mode : ``{"full", "frozen_gbsa"}``, default ``"frozen_gbsa"``
            Calculation mode (see above).
        batch_size : int, default 32
            Number of displaced structures per batched ES call.  Must be even;
            each DOF consumes two batch slots (+δ, −δ).
        delta : float, default 1e-4
            Cartesian displacement step in Å.
        filename : str
            Path to the XYZ (or PDB) file used to create ``structure``.
            Required (needed to construct ``StructureBatch`` objects internally).
        verbose : bool, default False
            Print progress information.

        Returns
        -------
        hessian : (3N, 3N) torch.Tensor
            Symmetrised Hessian in eV/Å².  Also stored as ``structure.hessian``.
        """
        if mode in ("full", "frozen_gbsa"):
            if filename is None:
                raise ValueError(
                    f"'filename' is required for mode='{mode}'. "
                    "Pass the XYZ/PDB file used to create the Structure."
                )
            return self._hessian_fd(
                structure,
                const,
                mode=mode,
                batch_size=batch_size,
                delta=delta,
                filename=filename,
                verbose=verbose,
            )
        elif mode == "autograd":
            raise ValueError(
                "mode='autograd' is not supported.  Unrolled backprop through "
                "SCF iterations produces incorrect Hessians (errors of 2–7×). "
                "Use 'full' or 'frozen_gbsa' instead."
            )
        else:
            raise ValueError(
                f"Unknown Hessian mode '{mode}'. Choose from 'full' or 'frozen_gbsa'."
            )

    # ─── Vibrational Frequencies ────────────────────────────────────
    @staticmethod
    def calc_frequencies(
        structure,
        hessian: torch.Tensor = None,
        proj_tr: bool = True,
    ):
        """Compute vibrational frequencies from the mass-weighted Hessian.

        Parameters
        ----------
        structure : Structure
            A ``Structure`` with ``Mnuc`` (atomic masses in amu) and ``Nats``.
            If *hessian* is ``None``, ``structure.hessian`` is used.
        hessian : (3N, 3N) torch.Tensor, optional
            Hessian in eV/Å².  Defaults to ``structure.hessian``.
        proj_tr : bool, default True
            If True, project out translational and rotational degrees of
            freedom before diagonalization (Sayvetz conditions).  This
            ensures the 5/6 lowest modes are numerically zero rather than
            noisy small numbers.

        Returns
        -------
        freqs_cm : (3N,) numpy.ndarray
            Vibrational frequencies in cm⁻¹, sorted ascending.  Imaginary
            modes (negative eigenvalues) are returned as negative values.
        eigvecs_mw : (3N, 3N) numpy.ndarray
            Mass-weighted normal-mode eigenvectors (columns).
        results : dict
            ``"eigenvalues"`` – raw eigenvalues of mass-weighted Hessian
            (eV / (Å² · amu)); ``"n_imag"`` – number of imaginary modes
            (freq < −10 cm⁻¹); ``"freqs_cm"`` – same as *freqs_cm*;
            ``"H_mw"`` – mass-weighted Hessian.

        Notes
        -----
        Unit conversion::

            ω = sqrt(λ)  in  sqrt(eV / (Å² · amu))
            ν̃ = ω / (2π c)

        with  1 eV = 1.602 176 634 × 10⁻¹⁹ J,
        1 amu = 1.660 539 066 60 × 10⁻²⁷ kg,  1 Å = 10⁻¹⁰ m,
        c = 2.997 924 58 × 10¹⁰ cm/s.

        This gives a conversion factor of ≈ 521.471 cm⁻¹ per
        √(eV/(Å²·amu)).

        The results are also stored on the *structure* object as
        ``structure.frequencies_cm``, ``structure.normal_modes``, and
        ``structure.freq_results``.
        """
        import numpy as np

        if hessian is None:
            if not hasattr(structure, "hessian") or structure.hessian is None:
                raise RuntimeError(
                    "No Hessian available.  Call calc_hessian() first or "
                    "pass a Hessian tensor explicitly."
                )
            hessian = structure.hessian

        hess = hessian.detach().cpu().numpy()
        masses = structure.Mnuc.detach().cpu().numpy()  # (N,) amu
        N_at = structure.Nats
        n3 = 3 * N_at

        # DOF ordering: [X0..X_{N-1}, Y0..Y_{N-1}, Z0..Z_{N-1}]
        mass_dof = np.tile(masses, 3)  # (3N,)
        sqrt_m = np.sqrt(mass_dof)

        # Mass-weighted Hessian:  H̃_ij = H_ij / sqrt(m_i · m_j)
        H_mw = hess / np.outer(sqrt_m, sqrt_m)

        if proj_tr:
            # ── Project out translations & rotations (Sayvetz) ───────
            # Build the 6 (or 5 for linear) T/R vectors in mass-weighted
            # Cartesian space, orthogonalise, and project them out of H̃.
            coords = np.stack(
                [
                    structure.RX.detach().cpu().numpy(),
                    structure.RY.detach().cpu().numpy(),
                    structure.RZ.detach().cpu().numpy(),
                ],
                axis=1,
            )  # (N, 3)

            # Center of mass
            total_mass = masses.sum()
            com = (masses[:, None] * coords).sum(axis=0) / total_mass
            rc = coords - com  # (N, 3) relative to COM

            tr_vecs = []

            # 3 translations: d_c = sqrt(m_i) * delta_{c, comp}
            for c in range(3):
                v = np.zeros(n3)
                v[c * N_at : (c + 1) * N_at] = sqrt_m[:N_at]
                tr_vecs.append(v)

            # 3 rotations: d_{Rx} = sqrt(m) * (y ẑ - z ŷ), etc.
            # In [X..., Y..., Z...] ordering:
            #   Rx: dY_i = +sqrt(m_i)*z_i,  dZ_i = -sqrt(m_i)*y_i
            #   Ry: dZ_i = +sqrt(m_i)*x_i,  dX_i = -sqrt(m_i)*z_i
            #   Rz: dX_i = +sqrt(m_i)*y_i,  dY_i = -sqrt(m_i)*x_i
            sm = np.sqrt(masses)
            x, y, z = rc[:, 0], rc[:, 1], rc[:, 2]

            for axis in range(3):
                v = np.zeros(n3)
                if axis == 0:  # Rx
                    v[1 * N_at : 2 * N_at] = sm * z  # dY
                    v[2 * N_at : 3 * N_at] = -sm * y  # dZ
                elif axis == 1:  # Ry
                    v[2 * N_at : 3 * N_at] = sm * x  # dZ
                    v[0 * N_at : 1 * N_at] = -sm * z  # dX
                else:  # Rz
                    v[0 * N_at : 1 * N_at] = sm * y  # dX
                    v[1 * N_at : 2 * N_at] = -sm * x  # dY
                tr_vecs.append(v)

            # Orthonormalize via QR (drops linearly dependent vectors
            # automatically — e.g. for linear molecules only 5 remain)
            D = np.column_stack(tr_vecs)  # (3N, 6)
            Q, R_qr = np.linalg.qr(D, mode="reduced")
            # Keep columns whose R diagonal is non-negligible
            keep = np.abs(np.diag(R_qr)) > 1e-10
            Q = Q[:, keep]  # (3N, n_tr)  n_tr = 5 or 6

            # Projector onto vibrational subspace:  P = I - Q Qᵀ
            P = np.eye(n3) - Q @ Q.T
            H_mw = P @ H_mw @ P

        # Diagonalize
        eigvals, eigvecs = np.linalg.eigh(H_mw)

        # Convert eigenvalues → cm⁻¹
        _eV = 1.602176634e-19  # J
        _amu = 1.66053906660e-27  # kg
        _ang = 1e-10  # m
        _c = 2.99792458e10  # cm/s
        factor = np.sqrt(_eV / (_ang**2 * _amu)) / (2.0 * np.pi * _c)
        # ≈ 521.471 cm⁻¹ per sqrt(eV/(Å²·amu))

        freqs_cm = np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * factor

        n_imag = int((freqs_cm < -10.0).sum())

        results = {
            "eigenvalues": eigvals,
            "n_imag": n_imag,
            "freqs_cm": freqs_cm,
            "H_mw": H_mw,
            "conversion_factor": factor,
        }

        # Store on structure
        structure.frequencies_cm = freqs_cm
        structure.normal_modes = eigvecs
        structure.freq_results = results

        return freqs_cm, eigvecs, results

    # ─── Finite-Difference Hessian (full & frozen_gbsa) ──────────────
    def _hessian_fd(
        self,
        structure,
        const,
        mode,
        batch_size,
        delta,
        filename,
        verbose=False,
    ):
        """Central-difference Hessian using batched ES calls."""
        from .Structure import StructureBatch

        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even (each DOF needs +δ and -δ).")

        N = structure.Nats
        n3 = 3 * N
        dtype = structure.RX.dtype
        dev = structure.RX.device

        # Reference coordinates
        RX0 = structure.RX.detach().clone()
        RY0 = structure.RY.detach().clone()
        RZ0 = structure.RZ.detach().clone()

        # Converged charges → initial guess for displaced structures
        q_ref = structure.q.detach().clone()  # (N,)

        # ── Pre-compute frozen GBSA (once) if requested ──────────────
        gbsa_ref = None
        if (
            mode == "frozen_gbsa"
            and self.dftorch_params.get("solvent_param_file", None) is not None
        ):

            class _StructProxy:
                pass

            proxy = _StructProxy()
            proxy.RX = RX0
            proxy.RY = RY0
            proxy.RZ = RZ0
            proxy.TYPE = structure.TYPE
            gbsa_ref = create_gbsa(
                proxy,
                self.device,
                param_file=self.dftorch_params.get("solvent_param_file", None),
                solvation_model=self.dftorch_params.get("solvation_model", "gbsa"),
            )
            if verbose:
                print("Reference GBSA computed (frozen geometry approximation)")

        dofs_per_batch = batch_size // 2  # each DOF → +δ and −δ
        n_batches = math.ceil(n3 / dofs_per_batch)

        if verbose:
            print(
                f"FD Hessian ({mode}): {N} atoms, {n3} DOFs, δ = {delta} Å\n"
                f"  batch_size={batch_size} → {dofs_per_batch} DOFs/batch, "
                f"{n_batches} batched ES calls"
            )

        F_plus = torch.zeros(n3, 3, N, dtype=dtype, device=dev)
        F_minus = torch.zeros(n3, 3, N, dtype=dtype, device=dev)

        t0 = time.perf_counter()

        for batch_idx in range(n_batches):
            dof_start = batch_idx * dofs_per_batch
            dof_end = min(dof_start + dofs_per_batch, n3)
            n_dofs_this = dof_end - dof_start
            bs = 2 * n_dofs_this

            # ── Build displaced coordinates ──────────────────────────
            RX_batch = RX0.unsqueeze(0).expand(bs, -1).clone()
            RY_batch = RY0.unsqueeze(0).expand(bs, -1).clone()
            RZ_batch = RZ0.unsqueeze(0).expand(bs, -1).clone()

            for k, dof in enumerate(range(dof_start, dof_end)):
                comp = dof // N  # 0=X, 1=Y, 2=Z
                atom = dof % N
                idx_plus = 2 * k
                idx_minus = 2 * k + 1

                if comp == 0:
                    RX_batch[idx_plus, atom] += delta
                    RX_batch[idx_minus, atom] -= delta
                elif comp == 1:
                    RY_batch[idx_plus, atom] += delta
                    RY_batch[idx_minus, atom] -= delta
                else:
                    RZ_batch[idx_plus, atom] += delta
                    RZ_batch[idx_minus, atom] -= delta

            # ── Construct StructureBatch from file, then overwrite ───
            # Read from file to populate TYPE, basis info, etc., then
            # replace coordinates with our displaced ones.
            # D0 is zeroed so the SCF uses q_init instead of the atomic
            # density matrix for the initial guess.
            sb = StructureBatch(
                [filename] * bs,
                structure.cell,
                const,
                charge=structure.charge,
                Te=structure.Te,
                device=self.device,
                req_grad_xyz=False,
            )
            sb.RX = RX_batch
            sb.RY = RY_batch
            sb.RZ = RZ_batch
            sb.D0 = sb.D0 * 0.0

            # ── GBSA handling ────────────────────────────────────────
            gbsa_batch_pre = None
            if mode == "frozen_gbsa" and gbsa_ref is not None:
                gbsa_batch_pre = GBSABatch.from_single(gbsa_ref, bs)

            # ── Batched ES + forces ──────────────────────────────────
            es_batch = ESDriverBatch(
                {**self.dftorch_params, "UNRESTRICTED": False},
                electronic_rcut=self.electronic_rcut,
                repulsive_rcut=self.repulsive_rcut,
                device=self.device,
            )
            with torch.no_grad():
                es_batch(
                    sb,
                    const,
                    do_scf=True,
                    q_init=q_ref,
                    gbsa_batch_precomputed=gbsa_batch_pre,
                )
                es_batch.calc_forces(sb, const)

            # ── Harvest forces ───────────────────────────────────────
            for k, dof in enumerate(range(dof_start, dof_end)):
                F_plus[dof] = sb.f_tot[2 * k].detach()
                F_minus[dof] = sb.f_tot[2 * k + 1].detach()

            if verbose:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
                print(
                    f"  batch {batch_idx + 1}/{n_batches}  DOFs {dof_start}-{dof_end - 1}"
                    f"  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

        # ── Assemble Hessian: H[i,j] = -(F_i(+δ_j) - F_i(-δ_j)) / (2δ) ─
        hessian = torch.zeros(n3, n3, dtype=dtype, device=dev)
        for j in range(n3):
            fp = F_plus[j].reshape(-1)  # (3N,)
            fm = F_minus[j].reshape(-1)  # (3N,)
            hessian[:, j] = -(fp - fm) / (2.0 * delta)

        sym_err = (hessian - hessian.T).abs().max().item()
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)  "
                f"max|H-Hᵀ| = {sym_err:.4e}"
            )
        hessian = 0.5 * (hessian + hessian.T)

        structure.hessian = hessian
        return hessian


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
        self,
        structure,
        const,
        do_scf=True,
        verbose: bool = False,
        q_init=None,
        gbsa_batch_precomputed=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total energy and forces for_ given density matrix.

        Parameters
        ----------
        density_matrix : (HDIM, HDIM) torch.Tensor
            Density matrix in AO basis.
        verbose : bool
            If True, print timing info from neighbor list routines.
        q_init : torch.Tensor, optional
            Initial charges for the SCF cycle. Shape (N,) for a single reference
            (broadcast to all batch elements) or (B, N). Skips the expensive
            initial eigendecomposition when provided.
        gbsa_batch_precomputed : GBSABatch, optional
            Pre-computed GBSABatch object to reuse for all batch elements.
            When provided, skips the expensive per-structure GBSA construction
            (Born radii, SASA, Born matrix) and uses the pre-computed geometry-
            dependent quantities directly.  Useful for finite-difference Hessian
            calculations where displacements are tiny and GBSA geometry barely
            changes.

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
            if COULCUT > 50.0 and structure.cell is not None:
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

            # ── Full off-diagonal DFTB3 third-order matrices (batched) ──
            # ThirdOrder is per-structure; we loop over batch elements.
            if (
                structure.dU_dq is not None
                and self.dftorch_params.get("dftb3_diagonal_only", False) is False
            ):
                _to_list = []
                for b in range(structure.batch_size):
                    to = create_thirdorder(
                        structure.Hubbard_U[b],
                        structure.dU_dq[b],
                        structure.TYPE[b],
                        h_damp_exp=self.dftorch_params.get("h_damp_exp", None),
                    )
                    # Build pairwise data for_ this structure from the Coulomb neighbor list.
                    # We rebuild a quick pairwise data from the full Coulomb matrix C[b].
                    # Actually, we need to build from coords. Use a simple all-pairs approach.
                    Nats = structure.Nats
                    rxb, ryb, rzb = structure.RX[b], structure.RY[b], structure.RZ[b]
                    dx = rxb.unsqueeze(0) - rxb.unsqueeze(1)
                    dy = ryb.unsqueeze(0) - ryb.unsqueeze(1)
                    dz = rzb.unsqueeze(0) - rzb.unsqueeze(1)
                    dr = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)
                    # Mask off diagonal and beyond cutoff
                    mask = (dr < 50.0) & (
                        ~torch.eye(Nats, dtype=torch.bool, device=self.device)
                    )
                    ni = (
                        torch.arange(Nats, device=self.device)
                        .unsqueeze(1)
                        .expand(-1, Nats)[mask]
                    )
                    nj = (
                        torch.arange(Nats, device=self.device)
                        .unsqueeze(0)
                        .expand(Nats, -1)[mask]
                    )
                    dR_m = dr[mask]
                    dxyz = torch.stack([dx[mask], dy[mask], dz[mask]], dim=-1)
                    dR_dxyz_m = dxyz / dR_m.unsqueeze(-1)
                    to.update_coords(rxb, ryb, rzb, None, ni, nj, dR_m, dR_dxyz_m)
                    _to_list.append(to)
                structure.thirdorder_batch = ThirdOrderBatch(_to_list)
                structure.thirdorder_list = _to_list  # keep for_ MD rebuild
            else:
                structure.thirdorder_batch = None
                structure.thirdorder_list = None

        if do_scf:
            # ── Create GBSA objects BEFORE SCF so solvation is self-consistent ──
            if gbsa_batch_precomputed is not None:
                # Reuse pre-computed GBSA (frozen geometry approximation)
                structure.gbsa_batch = gbsa_batch_precomputed
                structure.gbsa_list = gbsa_batch_precomputed._list
            elif self.dftorch_params.get("solvent_param_file", None) is not None:
                tic = time.time()
                _gbsa_list = []
                for b in range(structure.batch_size):

                    class _StructProxy:
                        pass

                    proxy = _StructProxy()
                    proxy.RX = structure.RX[b]
                    proxy.RY = structure.RY[b]
                    proxy.RZ = structure.RZ[b]
                    proxy.TYPE = structure.TYPE[b]
                    gbsa_b = create_gbsa(
                        proxy,
                        self.device,
                        param_file=self.dftorch_params.get("solvent_param_file", None),
                        solvation_model=self.dftorch_params.get(
                            "solvation_model", "gbsa"
                        ),
                    )
                    _gbsa_list.append(gbsa_b)
                structure.gbsa_batch = GBSABatch(_gbsa_list)
                structure.gbsa_list = _gbsa_list  # keep for_ rebuild in MD

                toc = time.time()
                print(f"GBSA initialization time: {toc - tic:.2f} seconds")

            else:
                structure.gbsa_batch = None
                structure.gbsa_list = None

            (
                structure.H,
                structure.Hcoul,
                structure.Hdipole,
                structure.KK,
                structure.D,
                structure.Q,
                structure.e,
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
                gbsa_batch=structure.gbsa_batch,
                thirdorder_batch=structure.thirdorder_batch,
                q_init=q_init,
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
                thirdorder=structure.thirdorder_batch,
            )

            structure.e_tot = structure.e_elec_tot + structure.e_repulsion

            # ── GBSA implicit solvation energy (uses already-created gbsa_batch) ──
            if structure.gbsa_batch is not None:
                if gbsa_batch_precomputed is not None:
                    # Fast path: use pre-computed Born matrix + SASA (no recompute)
                    e_gb, e_sasa = structure.gbsa_batch.get_energies(structure.q)
                    structure.e_solv = e_gb + e_sasa  # (B,)
                elif self.dftorch_params.get("gbsa_differentiable", False):
                    # Full differentiable path: recomputes geometry for autograd
                    coords_batch = torch.stack(
                        [structure.RX, structure.RY, structure.RZ], dim=2
                    )  # (B, N, 3)
                    e_solv_list = []
                    for b in range(structure.batch_size):
                        e_solv_list.append(
                            structure.gbsa_batch._list[b].get_energy_differentiable(
                                coords_batch[b], structure.q[b]
                            )
                        )
                    structure.e_solv = torch.stack(e_solv_list)
                else:
                    # Non-differentiable path (default)
                    e_gb, e_sasa = structure.gbsa_batch.get_energies(structure.q)
                    structure.e_solv = e_gb + e_sasa  # (B,)
                structure.e_tot = structure.e_tot + structure.e_solv
            else:
                structure.e_solv = torch.zeros(structure.batch_size, device=self.device)

            # ── D3(BJ) dispersion (single D3 object, batched) ───────────
            d3_params = self.dftorch_params.get("d3_params", None)
            if d3_params is not None:
                # All batch elements share the same atomic numbers — one D3 object
                structure.dftd3 = create_dftd3(
                    atomic_numbers=structure.TYPE[0].cpu().numpy().astype(int),
                    s6=d3_params.get("s6", 1.0),
                    s8=d3_params.get("s8", 0.5883),
                    a1=d3_params.get("a1", 0.5719),
                    a2=d3_params.get("a2", 3.6017),
                    device=self.device,
                    dtype=torch.float64,
                )
                coords_batch = torch.stack(
                    [structure.RX, structure.RY, structure.RZ], dim=2
                )  # (B, N, 3)
                structure.e_d3 = structure.dftd3.get_energy_batch(coords_batch)  # (B,)
                structure.e_tot = structure.e_tot + structure.e_d3
            else:
                structure.dftd3 = None
                structure.e_d3 = torch.zeros(structure.batch_size, device=self.device)

    def calc_forces(self, structure, const):

        # with torch.no_grad():
        if 1:
            # ── Solvation shift for_ force calculation (vectorised) ───────
            solv_shift = None
            if structure.gbsa_batch is not None:
                solv_shift = structure.gbsa_batch.get_shifts(structure.q)  # (B, N)

            # ── ThirdOrder shift for_ force calculation (vectorised) ──────
            to_shift = None
            if structure.thirdorder_batch is not None:
                to_shift = structure.thirdorder_batch.get_shifts(structure.q)  # (B, N)

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
                    structure.dU_dq if structure.thirdorder_batch is None else None,
                    solvation_shift=solv_shift,
                    thirdorder_shift=to_shift,
                )

            # ── GBSA solvation gradients (vectorised) ────────────────────
            if structure.gbsa_batch is not None:
                tic = time.time()
                f_sasa = structure.gbsa_batch.get_sasa_gradients(structure.q).permute(
                    0, 2, 1
                )  # (B, N, 3) → (B, 3, N)
                toc = time.time()
                print(f"GBSA SASA gradient calculation time: {toc - tic:.2f} seconds")

                tic = time.time()
                f_born = structure.gbsa_batch.get_born_gradients(structure.q).permute(
                    0, 2, 1
                )  # (B, N, 3) → (B, 3, N)
                toc = time.time()
                print(f"GBSA Born gradient calculation time: {toc - tic:.2f} seconds")

                structure.f_gbsa = f_sasa + f_born  # (B, 3, N)
                structure.f_tot = structure.f_tot + structure.f_gbsa

            # ── Full DFTB3 off-diagonal gradient (vectorised) ────────────
            if structure.thirdorder_batch is not None:
                structure.f_thirdorder_dc = structure.thirdorder_batch.get_gradient_dc(
                    structure.q
                )  # (B, 3, N)
                structure.f_tot = structure.f_tot + structure.f_thirdorder_dc

            # ── D3(BJ) dispersion forces (vectorised) ────────────────────
            if structure.dftd3 is not None:
                coords_batch = torch.stack(
                    [structure.RX, structure.RY, structure.RZ], dim=2
                )  # (B, N, 3)
                structure.f_d3 = structure.dftd3.get_forces_batch(
                    coords_batch
                )  # (B, 3, N)
                structure.f_tot = structure.f_tot + structure.f_d3


__all__ = ["ESDriver", "ESDriverBatch"]
