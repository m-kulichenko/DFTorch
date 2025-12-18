import torch
from .H0andS import H0_and_S_vectorized, H0_and_S_vectorized_batch
from .RepulsiveSpline import get_repulsion_energy, get_repulsion_energy_batch
from .nearestneighborlist import vectorized_nearestneighborlist, vectorized_nearestneighborlist_batch
from .Tools import fractional_matrix_power_symm
from .SCF import SCF, SCF_adaptive_mixing, SCFx, SCFx_batch
from .Energy import Energy
from .Forces import Forces, Forces_PME
from .ForcesBatch import forces_batch
from .CoulombMatrix import CoulombMatrix_vectorized
from dftorch.CoulombMatrixBatch import CoulombMatrix_vectorized_batch

import math

class ESDriver(torch.nn.Module):
    def __init__(
        self,
        dftorch_params,
        electronic_rcut: float,
        repulsive_rcut: float,
        device: torch.device,
        *args, **kwargs
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
        do_scf = True,
        verbose: bool = False
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

        _, _, nnRx, nnRy, nnRz, nnType, _, _, \
        neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type  = \
        vectorized_nearestneighborlist(
            structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
            self.electronic_rcut, structure.Nats, const,
            upper_tri_only=False, remove_self_neigh=False, min_image_only=False, verbose=verbose)

        # Get Hamiltonian, Overlap, etc, 
        structure.H0, structure.dH0, structure.S, structure.dS = H0_and_S_vectorized(
            structure.TYPE, structure.RX, structure.RY, structure.RZ,
            structure.diagonal, structure.H_INDEX_START,
            nnRx, nnRy, nnRz, nnType,
            const, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type,
            const.R_orb, const.coeffs_tensor,
            verbose=verbose)
        del _, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type   
        structure.Z = fractional_matrix_power_symm(structure.S, -0.5)
        
        # nuclear repulsion
        structure.e_repulsion, structure.dVr = get_repulsion_energy(
            const.R_rep_tensor, const.rep_splines_tensor,
            structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
            self.repulsive_rcut, structure.Nats, 
            const,verbose=verbose)


        if self.dftorch_params['coul_method'] == 'PME':
            structure.C = None
            structure.dCC = None
        else:
            Coulomb_acc = self.dftorch_params['Coulomb_acc']
            SQRTX = math.sqrt(-math.log(Coulomb_acc))
            COULCUT = self.dftorch_params['cutoff']
            CALPHA = SQRTX/COULCUT
            if COULCUT > 50.0:
                COULCUT = 50.0
                CALPHA = SQRTX / COULCUT

            # Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
            _, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, \
            _, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type = \
            vectorized_nearestneighborlist( structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox, COULCUT, structure.Nats, const,
                                            upper_tri_only=False, remove_self_neigh=False, verbose=verbose)
                                            
            structure.C, structure.dCC = CoulombMatrix_vectorized(structure.Hubbard_U, structure.TYPE, structure.RX,structure.RY,structure.RZ,
                                                structure.LBox, structure.lattice_vecs, structure.Nats,
                                                Coulomb_acc, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J,
                                                CALPHA, verbose=verbose)
            del _, nndist, nnRx, nnRy, nnRz, nnType, nnStruct, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type

        if do_scf:
            structure.H, structure.Hcoul, structure.Hdipole, structure.KK, structure.D, \
            structure.Q, structure.q, structure.f, \
            structure.mu0, structure.e_coul_tmp, structure.f_coul, structure.dq_p1 = SCFx(
                self.dftorch_params, 
                structure.RX, structure.RY, structure.RZ, structure.lattice_vecs,
                structure.Nats, structure.Nocc, structure.n_orbitals_per_atom, structure.Znuc, structure.TYPE, structure.Te,
                structure.Hubbard_U,
                structure.D0,
                structure.H0, structure.S, structure.Z, structure.e_field, structure.C,
                structure.req_grad_xyz)            

            structure.e_elec_tot, structure.e_band0, structure.e_coul, structure.e_dipole, \
            structure.e_entropy, structure.s_ent = Energy(
                structure.H0, structure.Hubbard_U, structure.e_field,
                structure.D0, structure.C, structure.dq_p1, structure.D, structure.q,
                structure.RX, structure.RY, structure.RZ, structure.f, structure.Te)

            structure.e_tot = structure.e_elec_tot + structure.e_repulsion

    def calc_forces(self, structure, const):

        with torch.no_grad():
            if self.dftorch_params['coul_method'] == 'PME': # f_coul was calculated in SCF via calculate_PME_ewald
            # structure.f_coul was calculated in SCF via calculate_PME_ewald
                f_tot, _, structure.f_band0, structure.f_dipole, structure.f_pulay, \
                structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
                    Forces_PME( structure.H, structure.Z, structure.dq_p1,
                        structure.D, structure.D0,
                        structure.dH0, structure.dS, structure.dVr,
                        structure.e_field, structure.Hubbard_U, structure.q,
                        structure.RX, structure.RY, structure.RZ,
                        structure.Nats, const, structure.TYPE)
                structure.f_tot = f_tot + structure.f_coul
            else:
                structure.f_tot, structure.f_coul, structure.f_band0, structure.f_dipole, structure.f_pulay, \
                structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
                    Forces( structure.H, structure.Z, structure.C,
                        structure.D, structure.D0,
                        structure.dH0, structure.dS, structure.dCC, structure.dVr,
                        structure.e_field, structure.Hubbard_U, structure.q,
                        structure.RX, structure.RY, structure.RZ,
                        structure.Nats, const, structure.TYPE)
                
class ESDriverBatch(torch.nn.Module):
    def __init__(
        self,
        dftorch_params,
        electronic_rcut: float,
        repulsive_rcut: float,
        device: torch.device,
        *args, **kwargs
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
        do_scf = True,
        verbose: bool = False
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

        _, _, nnRx, nnRy, nnRz, nnType, _, _, \
        neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type = vectorized_nearestneighborlist_batch(
            structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
            self.electronic_rcut, structure.Nats, const,
            upper_tri_only=False, remove_self_neigh=False, min_image_only=False, verbose=verbose)
                
        # Get Hamiltonian, Overlap, etc,         
        structure.H0, structure.dH0, structure.S, structure.dS = H0_and_S_vectorized_batch(
                    structure.TYPE, structure.RX, structure.RY, structure.RZ,
                    structure.diagonal, structure.H_INDEX_START,
                    nnRx, nnRy, nnRz, nnType,
                    const, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type,
                    const.R_orb, const.coeffs_tensor,
                    verbose=verbose)


        del _, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type   
        structure.Z = fractional_matrix_power_symm(structure.S, -0.5)
        
        # nuclear repulsion
        structure.e_repulsion, structure.dVr = get_repulsion_energy_batch(
            const.R_rep_tensor, const.rep_splines_tensor,
            structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
            self.repulsive_rcut, structure.Nats, 
            const,verbose=verbose)


        if self.dftorch_params['coul_method'] == 'PME':
            structure.C = None
            structure.dCC = None
            raise ValueError("Batched PME Coulomb not implemented.")
            return
        else:
            Coulomb_acc = self.dftorch_params['Coulomb_acc']
            SQRTX = math.sqrt(-math.log(Coulomb_acc))
            COULCUT = self.dftorch_params['cutoff']
            CALPHA = SQRTX/COULCUT
            if COULCUT > 50.0:
                COULCUT = 50.0
                CALPHA = SQRTX / COULCUT

            # Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
            _, _, nnRx, nnRy, nnRz, nnType, _, \
            _, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type = \
            vectorized_nearestneighborlist_batch(
                structure.TYPE, structure.RX, structure.RY, structure.RZ, structure.LBox,
                COULCUT, structure.Nats, const,
                upper_tri_only=False, remove_self_neigh=False, verbose=verbose)


            structure.C, structure.dCC = CoulombMatrix_vectorized_batch(structure.Hubbard_U, structure.TYPE, structure.RX, structure.RY, structure.RZ,
                                                    structure.LBox, structure.lattice_vecs, structure.Nats,
                            Coulomb_acc, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J,
                            CALPHA, verbose=verbose)
                                            
            del _, nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, JI_pair_type

        if do_scf:
            structure.H, structure.Hcoul, structure.Hdipole, structure.KK, structure.D, \
            structure.q, structure.f, \
            structure.mu0, structure.e_coul_tmp, structure.f_coul, structure.dq_p1 = SCFx_batch(
                self.dftorch_params, structure.RX, structure.RY, structure.RZ,
                structure.Nats, structure.Nocc, structure.n_orbitals_per_atom, structure.Znuc, structure.Te,
                structure.Hubbard_U,
                structure.D0,
                structure.H0, structure.S, structure.Z, structure.e_field, structure.C)
            

            structure.e_elec_tot, structure.e_band0, structure.e_coul, structure.e_dipole, \
            structure.e_entropy, structure.s_ent = Energy(
                structure.H0, structure.Hubbard_U, structure.e_field,
                structure.D0, structure.C, structure.dq_p1, structure.D, structure.q,
                structure.RX, structure.RY, structure.RZ, structure.f, structure.Te)

            structure.e_tot = structure.e_elec_tot + structure.e_repulsion

    def calc_forces(self, structure, const):

        #with torch.no_grad():
        if 1:
            if self.dftorch_params['coul_method'] == 'PME':
                raise ValueError("Batched PME Coulomb not implemented.")
                return
            else:
                structure.f_tot, structure.f_coul, structure.f_band0, structure.f_dipole, structure.f_pulay, \
                structure.f_s_coul, structure.f_s_dipole, structure.f_rep = \
                    forces_batch( structure.H, structure.Z, structure.C,
                        structure.D, structure.D0,
                        structure.dH0, structure.dS, structure.dCC, structure.dVr,
                        structure.e_field, structure.Hubbard_U, structure.q,
                        structure.RX, structure.RY, structure.RZ,
                        structure.Nats, const, structure.TYPE)
            


     