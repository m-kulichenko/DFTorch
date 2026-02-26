import torch

from collections import deque

from ._tools import fractional_matrix_power_symm
from ._dm_fermi import dm_fermi
from ._dm_fermi_x import (
    dm_fermi_x,
    dm_fermi_x_os_shared,
    dm_fermi_x_batch,
)

# from ._kernel_fermi import _kernel_fermi
from ._tools import calculate_dist_dips
from ._xl_tools import (
    kernel_update_lr,
    kernel_update_lr_os,
    kernel_update_lr_batch,
    calc_q,
    calc_q_os,
    calc_q_batch,
)

from ._spin import get_h_spin


import time
from typing import Optional, Dict, Any, Tuple


def SCFx(
    dftorch_params: Dict[str, Any],
    RX,
    RY,
    RZ,
    lattice_vecs: torch.Tensor,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    TYPE: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    req_grad_xyz: bool,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """
    Self-consistent field (_scf) cycle with finite electronic temperature and
    Fermi–Dirac occupations, using a preconditioned low-rank Krylov charge mixer.
    Supports PME Ewald electrostatics via `sedacs` or a direct Coulomb matrix.

    Parameters
    ----------
    dftorch_params : dict
        _scf/control parameters. Expected keys include:
        - 'coul_method': str, 'PME' or 'direct'
        - 'cutoff': float, real-space cutoff (PME)
        - 'Coulomb_acc': float, accuracy target for PME alpha/grid (PME)
        - 'PME_order': int, B-spline order (PME)
        - other PME/mixing options passed through to helper routines.
    structure : object
        Container providing required system data/attributes:
        - RX, RY, RZ: (Nats,) atomic coordinates (torch.Tensor)
        - lattice_vecs: (3,3) lattice vectors (torch.Tensor), for PME
        - n_orbitals_per_atom: (Nats,) number of AOs per atom (torch.Tensor)
        - Znuc: (Nats,) nuclear charges (torch.Tensor)
        - Hubbard_U: (Nats,) onsite U (torch.Tensor)
        - TYPE: (Nats,) atom types (torch.Tensor)
        - Nocc: int, total electron pairs
        - Te: float, electronic temperature
        - Nats: int, number of atoms
    D0 : torch.Tensor or None
        Reference density matrix for band-energy shifts. Currently unused in the
        _scf loop (kept for compatibility).
    H0 : torch.Tensor
        One-electron Hamiltonian in AO basis, shape (n_orb, n_orb).
    S : torch.Tensor
        Overlap matrix, shape (n_orb, n_orb).
    Z : torch.Tensor
        Symmetric orthogonalizer S^(-1/2) in AO basis, shape (n_orb, n_orb).
        Must satisfy approximately Z.T @ S @ Z = I. Used to transform
        to/from the orthogonal representation where dm_fermi_x is applied.
    Efield : torch.Tensor
        External electric field vector (3,).
    C : torch.Tensor
        Coulomb operator. If dftorch_params['coul_method'] == 'direct',
        used as C @ q to build electrostatic potential; ignored for PME.
    Returns
    -------
    H : torch.Tensor
        Final Hamiltonian including Coulomb and dipole terms, (n_orb, n_orb).
    Hcoul : torch.Tensor
        Final Coulomb contribution to the Hamiltonian, (n_orb, n_orb).
    Hdipole : torch.Tensor
        Symmetrized dipole correction from the external field, (n_orb, n_orb).
    KK : torch.Tensor
        Mixing/preconditioning matrix used by the Krylov _scf accelerator, (Nats, Nats).
    D : torch.Tensor
        Final density matrix in AO basis, (n_orb, n_orb).
    q : torch.Tensor
        Final atomic charges, (Nats,).
    f : torch.Tensor
        Eigenvalues of the orthogonal density (Fermi occupations), (n_orb,).
    mu0 : torch.Tensor
        Fermi level (chemical potential) at convergence.
    Ecoul : torch.Tensor or None
        Coulomb energy from PME (if 'PME') else None.
    forces1 : torch.Tensor or None
        Electrostatic forces from PME (if requested) else None.
    dq_p1 : torch.Tensor or None
        Charge-response-related PME output (if requested) else None.

    Notes
    -----
    - Uses symmetric orthogonalization Z = S^(-1/2) and applies Fermi operator
      expansion in the orthogonal basis.
    - Charge residuals are accelerated by a preconditioned low-rank Krylov method
      after `KRYLOV_START` iterations; before that, linear mixing with `SCF_ALPHA` is used.
    - Electrostatics:
        * 'PME': periodic Ewald via sedacs (real/reciprocal space split).
        * 'direct': direct Coulomb via supplied C.
    """
    print("### Do _scf ###")

    device = H0.device
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=H0.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    Hubbard_U_gathered = Hubbard_U[atom_ids]

    if dftorch_params["coul_method"] == "PME":
        from .ewald_pme import (
            calculate_PME_ewald,
            init_PME_data,
            calculate_alpha_and_num_grids,
        )
        from .ewald_pme.neighbor_list import NeighborState

        # positions = torch.stack((RX, RY, RZ))
        positions = torch.stack(
            (RX, RY, RZ),
        )
        CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
            lattice_vecs.cpu().numpy(),
            dftorch_params["cutoff"],
            dftorch_params["Coulomb_acc"],
        )
        PME_data = init_PME_data(
            grid_dimensions, lattice_vecs, CALPHA, dftorch_params["PME_order"]
        )
        nbr_state = NeighborState(
            positions,
            lattice_vecs,
            None,
            dftorch_params["cutoff"],
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nbr_inds = calculate_dist_dips(
            positions, nbr_state, dftorch_params["cutoff"]
        )
    else:
        PME_data = None
        nbr_inds = None
        disps = None
        dists = None
        CALPHA = None

    with torch.no_grad():
        # if 1:

        # Initial density matrix
        print("  Initial dm_fermi")

        Hdipole = torch.diag(
            -RX[atom_ids] * Efield[0]
            - RY[atom_ids] * Efield[1]
            - RZ[atom_ids] * Efield[2]
        )
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole
        Dorth, Q, e, f, mu0 = dm_fermi_x(
            Z.T @ H0 @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
        )
        D = Z @ Dorth @ Z.T
        DS = 2 * torch.diag(D @ S)
        q = -1.0 * Znuc
        q.scatter_add_(
            0, atom_ids, DS
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
            Nats, device=H0.device
        )  # Initial mixing coefficient for linear mixing
        # KK0 = KK*torch.eye(Nats, device=H0.device)

        ResNorm = torch.tensor([2.0], device=device)
        dEc = torch.tensor([1000.0], device=device)
        it = 0
        Ecoul = torch.tensor([0.0], device=device)

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params["SCF_TOL"])
            or (dEc > dftorch_params["SCF_TOL"] * 100)
        ) and it < dftorch_params["SCF_MAX_ITER"]:
            start_time = time.perf_counter()
            it += 1
            print("Iter {}".format(it))

            if dftorch_params["coul_method"] == "PME":
                # with torch.enable_grad():
                if 1:
                    ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                        positions.detach().clone(),
                        q,
                        lattice_vecs,
                        nbr_inds,
                        disps,
                        dists,
                        CALPHA,
                        dftorch_params["cutoff"],
                        PME_data,
                        hubbard_u=Hubbard_U,
                        atomtypes=TYPE,
                        screening=1,
                        calculate_forces=0,
                        calculate_dq=1,
                    )
            else:
                CoulPot = C @ q
            q_old = q.clone()
            q, H, Hcoul, D, Dorth, Q, e, f, mu0 = calc_q(
                H0,
                Hubbard_U_gathered,
                q[atom_ids],
                CoulPot[atom_ids],
                S,
                Z,
                Te,
                Nocc,
                Znuc,
                atom_ids,
            )
            Res = q - q_old
            ResNorm = torch.norm(Res)
            K0Res = KK @ Res

            if (
                it == dftorch_params["KRYLOV_START"]
            ):  # Calculate full kernel after KRYLOV_START steps
                # KK,D0 = _kernel_fermi(structure, mu0,Te,Nats,H,C,S,Z,Q,e)
                # KK = torch.load("/home/maxim/Projects/DFTB/DFTorch/tests/KK_C840.pt") # For testing purposes
                # KK0 = KK.clone()  # To be kept as preconditioner
                1
            # Preconditioned Low-Rank Krylov _scf acceleration
            if it > dftorch_params["KRYLOV_START"]:
                # Preconditioned residual
                K0Res = kernel_update_lr(
                    RX,
                    RY,
                    RZ,
                    lattice_vecs,
                    TYPE,
                    Nats,
                    Hubbard_U,
                    dftorch_params,
                    dftorch_params["KRYLOV_TOL"],
                    KK.clone(),
                    Res,
                    q,
                    S,
                    Z,
                    PME_data,
                    atom_ids,
                    Q,
                    e,
                    mu0,
                    Te,
                    C,
                    nbr_inds,
                    disps,
                    dists,
                    CALPHA,
                )

            # Mixing update (vector-form)
            q = q_old - K0Res

            Ecoul_old = Ecoul.clone()
            if dftorch_params["coul_method"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q**2 * Hubbard_U)
            else:
                Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * Hubbard_U)

            # dEb = torch.abs(Eband0_old - Eband0)
            dEc = torch.abs(Ecoul_old - Ecoul)

            # print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            print(
                "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                )
            )
            if it == dftorch_params["SCF_MAX_ITER"]:
                print("Did not converge")

        # f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))

    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)
    q = -1.0 * Znuc
    q.scatter_add_(0, atom_ids, DS)

    if dftorch_params["coul_method"] == "PME":
        ewald_e1, forces1, dq_p1 = calculate_PME_ewald(
            positions,  # .detach().clone(),
            q,
            lattice_vecs,
            nbr_inds,
            disps,
            dists,
            CALPHA,
            dftorch_params["cutoff"],
            PME_data,
            hubbard_u=Hubbard_U,
            atomtypes=TYPE,
            screening=1,
            calculate_forces=0 if req_grad_xyz else 1,
            calculate_dq=0 if req_grad_xyz else 1,
        )
        Ecoul = ewald_e1 + 0.5 * torch.sum(q**2 * Hubbard_U)
    else:
        Ecoul, forces1, dq_p1 = None, None, None

    return H, Hcoul, Hdipole, KK, D, Q, q, f, mu0, Ecoul, forces1, dq_p1


def scf_x_os(
    el_per_shell: torch.Tensor,
    shell_types: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
    shell_dim: torch.Tensor,
    w: torch.Tensor,
    dftorch_params: Dict[str, Any],
    RX,
    RY,
    RZ,
    lattice_vecs: torch.Tensor,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    TYPE: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    req_grad_xyz: bool,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """ """
    print("### Do _scf ###")

    device = H0.device
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=H0.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital
    atom_ids_sr = torch.repeat_interleave(
        torch.arange(len(shell_types), device=H0.device), shell_dim[shell_types]
    )  # Generate atom index for each orbital
    shell_to_atom = torch.repeat_interleave(
        torch.arange(len(TYPE), device=S.device), n_shells_per_atom
    )

    Hubbard_U_gathered = Hubbard_U[atom_ids]

    if dftorch_params["coul_method"] == "PME":
        from .ewald_pme import (
            calculate_PME_ewald,
            init_PME_data,
            calculate_alpha_and_num_grids,
        )
        from .ewald_pme.neighbor_list import NeighborState

        # positions = torch.stack((RX, RY, RZ))
        positions = torch.stack(
            (RX, RY, RZ),
        )
        CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
            lattice_vecs.cpu().numpy(),
            dftorch_params["cutoff"],
            dftorch_params["Coulomb_acc"],
        )
        PME_data = init_PME_data(
            grid_dimensions, lattice_vecs, CALPHA, dftorch_params["PME_order"]
        )
        nbr_state = NeighborState(
            positions,
            lattice_vecs,
            None,
            dftorch_params["cutoff"],
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nbr_inds = calculate_dist_dips(
            positions, nbr_state, dftorch_params["cutoff"]
        )
    else:
        PME_data = None
        nbr_inds = None
        disps = None
        dists = None
        CALPHA = None

    with torch.no_grad():
        # if 1:
        # Initial density matrix
        print("  Initial dm_fermi")
        Hdipole = torch.diag(
            -RX[atom_ids] * Efield[0]
            - RY[atom_ids] * Efield[1]
            - RZ[atom_ids] * Efield[2]
        )
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole
        H0 = H0.unsqueeze(0).expand(2, -1, -1)
        # Nocc = torch.tensor([Nocc+1, Nocc-1], device=H0.device)
        # Nocc = torch.tensor([Nocc, Nocc], device=H0.device)
        # Dorth, Q, e, f, mu0 = dm_fermi_x_os(Z.T @ H0 @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, broken_symmetry=True)
        broken_symmetry = dftorch_params.get("BROKEN_SYM", False)
        Dorth, Q, e, f, mu0 = dm_fermi_x_os_shared(
            Z.T @ H0 @ Z,
            Te,
            Nocc,
            mu_0=None,
            eps=1e-9,
            MaxIt=50,
            broken_symmetry=broken_symmetry,
        )
        # print(mu0, mu0_)
        D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
        DS = 1 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)

        q_spin_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
        q_spin_sr.scatter_add_(
            1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        # elec = 0.1
        # q_spin_sr[0,1] += elec
        # q_spin_sr[0,2] -= elec

        # q_spin_sr[1,1] -= elec

        # q_spin_sr[1,2] += elec

        # q_tot_sr = q_spin_sr.sum(dim=0)
        net_spin_sr = q_spin_sr[0] - q_spin_sr[1]

        q_spin_atom = torch.zeros_like(RX.unsqueeze(0).expand(2, -1))
        q_spin_atom.scatter_add_(
            1, shell_to_atom.unsqueeze(0).expand(2, -1), q_spin_sr
        )  # atom-resolved
        q_tot_atom = torch.zeros_like(RX)
        q_tot_atom.scatter_add_(0, shell_to_atom, q_spin_sr.sum(dim=0))  # atom-resolved

        KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
            n_shells_per_atom.sum(), device=H0.device
        ).unsqueeze(0).expand(
            2, -1, -1
        )  # shell-resolved. Initial mixing coefficient for linear mixing
        # KK0 = KK*torch.eye(Nats, device=H0.device)

        ResNorm = torch.tensor([2.0], device=device)
        dEc = torch.tensor([1000.0], device=device)
        it = 0
        Ecoul = torch.tensor([0.0], device=device)

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params["SCF_TOL"])
            or (dEc > dftorch_params["SCF_TOL"] * 100)
        ) and it < dftorch_params["SCF_MAX_ITER"]:
            start_time = time.perf_counter()
            it += 1
            print("Iter {}".format(it))

            if dftorch_params["coul_method"] == "PME":
                # with torch.enable_grad():
                if 1:
                    ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                        positions.detach().clone(),
                        q_tot_atom,
                        lattice_vecs,
                        nbr_inds,
                        disps,
                        dists,
                        CALPHA,
                        dftorch_params["cutoff"],
                        PME_data,
                        hubbard_u=Hubbard_U,
                        atomtypes=TYPE,
                        screening=1,
                        calculate_forces=0,
                        calculate_dq=1,
                    )
            else:
                CoulPot = C @ q_tot_atom
            q_spin_sr_old = q_spin_sr.clone()

            H_spin = get_h_spin(TYPE, net_spin_sr, w, n_shells_per_atom, shell_types)
            q_spin_sr, H, Hcoul, D, Dorth, Q, e, f, mu0 = calc_q_os(
                H0,
                H_spin,
                Hubbard_U_gathered,
                q_tot_atom[atom_ids],
                CoulPot[atom_ids],
                S,
                Z,
                Te,
                Nocc,
                Znuc,
                atom_ids,
                atom_ids_sr,
                el_per_shell,
            )

            q_spin_atom = torch.zeros_like(RX.unsqueeze(0).expand(2, -1))
            q_spin_atom.scatter_add_(
                1, shell_to_atom.unsqueeze(0).expand(2, -1), q_spin_sr
            )  # atom-resolved

            Res = q_spin_sr - q_spin_sr_old
            ResNorm = torch.norm(Res)
            K0Res = torch.bmm(KK, Res.unsqueeze(-1)).squeeze(-1)

            if (
                it == dftorch_params["KRYLOV_START"]
            ):  # Calculate full kernel after KRYLOV_START steps
                # KK,D0 = _kernel_fermi(structure, mu0,Te,Nats,H,C,S,Z,Q,e)
                # KK = torch.load("/home/maxim/Projects/DFTB/DFTorch/tests/KK_C840.pt") # For testing purposes
                # KK0 = KK.clone()  # To be kept as preconditioner
                1
            # Preconditioned Low-Rank Krylov _scf acceleration
            if it > dftorch_params["KRYLOV_START"]:
                # Preconditioned residual
                K0Res = kernel_update_lr_os(
                    RX,
                    RY,
                    RZ,
                    lattice_vecs,
                    TYPE,
                    Nats,
                    Hubbard_U,
                    dftorch_params,
                    dftorch_params["KRYLOV_TOL"],
                    KK.clone(),
                    Res,
                    q_spin_sr,
                    S,
                    Z,
                    PME_data,
                    atom_ids,
                    atom_ids_sr,
                    Q,
                    e,
                    mu0,
                    Te,
                    w,
                    n_shells_per_atom,
                    shell_types,
                    C,
                    nbr_inds,
                    disps,
                    dists,
                    CALPHA,
                )

            # Mixing update (vector-form)
            q_spin_sr = q_spin_sr_old - K0Res
            # q_tot_sr = q_spin_sr.sum(dim=0)
            q_tot_atom = torch.zeros_like(RX)
            q_tot_atom.scatter_add_(
                0, shell_to_atom, q_spin_sr.sum(dim=0)
            )  # atom-resolved
            net_spin_sr = q_spin_sr[0] - q_spin_sr[1]

            Ecoul_old = Ecoul.clone()
            if dftorch_params["coul_method"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q_tot_atom**2 * Hubbard_U)
            else:
                Ecoul = 0.5 * q_tot_atom @ (C @ q_tot_atom) + 0.5 * torch.sum(
                    q_tot_atom**2 * Hubbard_U
                )

            dEc = torch.abs(Ecoul_old - Ecoul)

            # print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            print(
                "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                )
            )
            if it == dftorch_params["SCF_MAX_ITER"]:
                print("Did not converge")

        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.transpose(-1, -2)))

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 1 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)

    q_spin_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
    q_spin_sr.scatter_add_(
        1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    net_spin_sr = q_spin_sr[0] - q_spin_sr[1]

    q_spin_atom = torch.zeros_like(RX.unsqueeze(0).expand(2, -1))
    q_spin_atom.scatter_add_(
        1, shell_to_atom.unsqueeze(0).expand(2, -1), q_spin_sr
    )  # atom-resolved
    q_tot_atom = torch.zeros_like(RX)
    q_tot_atom.scatter_add_(0, shell_to_atom, q_spin_sr.sum(dim=0))  # atom-resolved

    if dftorch_params["coul_method"] == "PME":
        ewald_e1, forces1, dq_p1 = calculate_PME_ewald(
            positions,  # .detach().clone(),
            q_tot_atom,
            lattice_vecs,
            nbr_inds,
            disps,
            dists,
            CALPHA,
            dftorch_params["cutoff"],
            PME_data,
            hubbard_u=Hubbard_U,
            atomtypes=TYPE,
            screening=1,
            calculate_forces=0 if req_grad_xyz else 1,
            calculate_dq=0 if req_grad_xyz else 1,
        )
        Ecoul = ewald_e1 + 0.5 * torch.sum(q_tot_atom**2 * Hubbard_U)
    else:
        Ecoul, forces1, dq_p1 = None, None, None

    return (
        H,
        Hcoul,
        Hdipole,
        KK,
        D,
        Q,
        q_spin_atom,
        q_tot_atom,
        q_spin_sr,
        net_spin_sr,
        f,
        mu0,
        Ecoul,
        forces1,
        dq_p1,
    )


def SCFx_batch(
    dftorch_params: Dict[str, Any],
    RX,
    RY,
    RZ,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """
    Self-consistent field (_scf) cycle with finite electronic temperature and
    Fermi–Dirac occupations, using a preconditioned low-rank Krylov charge mixer.
    Supports PME Ewald electrostatics via `sedacs` or a direct Coulomb matrix.
    """

    batch_size = RX.shape[0]
    device = H0.device
    counts = n_orbitals_per_atom  # shape (B, N)
    cum_counts = torch.cumsum(counts, dim=1)  # cumulative sums per batch
    total_orbs = int(cum_counts[0, -1].item())
    r = torch.arange(total_orbs, device=counts.device).expand(
        counts.size(0), -1
    )  # (B, total_orbs)
    # For each orbital position r[b,k], find first atom index whose cumulative count exceeds r[b,k]
    atom_ids = (
        (r.unsqueeze(2) < cum_counts.unsqueeze(1)).int().argmax(dim=2)
    )  # (B, total_orbs)

    PME_data = None
    nbr_inds = None
    disps = None
    dists = None
    CALPHA = None

    Hubbard_U_gathered = Hubbard_U.gather(1, atom_ids)

    # with torch.no_grad():
    if 1:
        RX_gathered = RX.gather(1, atom_ids)
        RY_gathered = RY.gather(1, atom_ids)
        RZ_gathered = RZ.gather(1, atom_ids)
        Hdipole = torch.diag_embed(
            -RX_gathered * Efield[0] - RY_gathered * Efield[1] - RZ_gathered * Efield[2]
        )
        Hdipole = 0.5 * (torch.matmul(Hdipole, S) + torch.matmul(S, Hdipole))
        H0 = H0 + Hdipole

        H_ortho = torch.matmul(Z.transpose(-1, -2), torch.matmul(H0, Z))
        Dorth, Q, e, f, mu0 = dm_fermi_x_batch(
            H_ortho, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
        )
        D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
        DS = 2 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
        q = -1.0 * Znuc
        q.scatter_add_(
            1, atom_ids, DS
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
            Nats, device=H0.device
        ) + torch.zeros(
            batch_size, Nats, Nats, device=H0.device
        )  # Initial mixing coefficient for linear mixing
        # KK0 = KK*torch.eye(Nats, device=H0.device)

        ResNorm = torch.zeros(batch_size, device=device) + 10.0  # float("inf")
        dEc = torch.zeros(batch_size, device=device) + 1000.0  # float("inf")
        it = 0
        Ecoul = torch.zeros(batch_size, device=device) + 10.0  # float("inf")

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params["SCF_TOL"]).any()
            or (dEc > dftorch_params["SCF_TOL"] * 100).any()
        ) and it < dftorch_params["SCF_MAX_ITER"]:
            it += 1
            print("Iter {}".format(it))

            CoulPot = torch.matmul(C, q.unsqueeze(-1)).squeeze(-1)
            q_old = q.clone()
            q, H, Hcoul, D, Dorth, Q, e, f, mu0 = calc_q_batch(
                H0,
                Hubbard_U_gathered,
                q.gather(1, atom_ids),
                CoulPot.gather(1, atom_ids),
                S,
                Z,
                Te,
                Nocc,
                Znuc,
                atom_ids,
            )
            Res = q - q_old
            ResNorm = torch.norm(Res, dim=1)
            K0Res = torch.matmul(KK, Res.unsqueeze(-1)).squeeze(-1)

            if (
                it == dftorch_params["KRYLOV_START"]
            ):  # Calculate full kernel after KRYLOV_START steps
                # KK,D0 = _kernel_fermi(structure, mu0,Te,Nats,H,C,S,Z,Q,e)
                # KK = torch.load("/home/maxim/Projects/DFTB/DFTorch/tests/KK_C840.pt") # For testing purposes
                # KK0 = KK.clone()  # To be kept as preconditioner
                1
            # Preconditioned Low-Rank Krylov _scf acceleration
            if it > dftorch_params["KRYLOV_START"]:
                # Preconditioned residual
                K0Res = kernel_update_lr_batch(
                    Nats,
                    Hubbard_U_gathered,
                    dftorch_params,
                    dftorch_params["KRYLOV_TOL"],
                    KK.clone(),
                    Res,
                    q,
                    S,
                    Z,
                    PME_data,
                    atom_ids,
                    Q,
                    e,
                    mu0,
                    Te,
                    C,
                    nbr_inds,
                    disps,
                    dists,
                    CALPHA,
                )

            # Mixing update (vector-form)
            q = q_old - K0Res

            Ecoul_old = Ecoul.clone()

            Cq = torch.bmm(C, q.unsqueeze(-1)).squeeze(-1)  # (B,N)
            Ecoul = 0.5 * torch.sum(q * Cq, dim=-1) + 0.5 * torch.sum(
                q**2 * Hubbard_U, dim=1
            )

            dEc = torch.abs(Ecoul_old - Ecoul)

            for b, (rval, dval) in enumerate(zip(ResNorm.tolist(), dEc.tolist())):
                print(f"Batch {b}: Res = {rval:.3e}, dEc = {dval:.3e}")
            # print(f"t = {elapsed:.2f} s")

            if it == dftorch_params["SCF_MAX_ITER"]:
                print("Did not converge")

        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.transpose(-1, -2)))

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 2 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
    q = -1.0 * Znuc
    q.scatter_add_(
        1, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    Ecoul, forces1, dq_p1 = None, None, None

    return H, Hcoul, Hdipole, KK, D, q, f, mu0, Ecoul, forces1, dq_p1


def _scf(
    structure,
    D0,
    H0,
    S,
    Efield,
    C,
    Rx,
    Ry,
    Rz,
    nocc,
    U,
    Znuc,
    Te,
    alpha=0.2,
    acc=1e-7,
    MAX_ITER=200,
    debug=False,
):
    """
    Performs a self-consistent field (_scf) cycle with finite electronic temperature
    and Fermi-Dirac occupations for a DFTB-like semiempirical Hamiltonian.

    Parameters
    ----------
    H0 : torch.Tensor
        Initial one-electron Hamiltonian matrix of shape (n_orb, n_orb).
    S : torch.Tensor
        Overlap matrix of shape (n_orb, n_orb).
    Efield : torch.Tensor
        External electric field vector of shape (3,) in atomic units.
    C : torch.Tensor
        Coulomb matrix used to compute electrostatic potential, shape (n_orb, n_orb).
    TYPE : torch.Tensor
        Atomic type index for each atom, shape (Nats,).
    Rx, Ry, Rz : torch.Tensor
        Atomic x, y, z coordinates respectively, each of shape (Nats,).
    H_Index_Start, H_Index_End : torch.Tensor
        Start and end indices of orbitals for each atom, shape (Nats,).
    nocc : int
        Number of occupied orbitals (electrons / 2).
    U : torch.Tensor
        Hubbard U parameter per atom, shape (Nats,).
    Znuc : torch.Tensor
        Nuclear charges per atom, shape (Nats,).
    Nats : int
        Total number of atoms in the system.
    Te : float
        Electronic temperature in energy units (e.g., eV).
    const : object
        DFTB constants object with attribute `n_orb`, which gives orbitals per atom type.
    alpha : float, optional
        Mixing coefficient for _scf charge update: (1 - alpha) * q_old + alpha * q. Default is 0.2.
    acc : float, optional
        Convergence threshold for _scf loop based on charge difference norm. Default is 1e-7.
    MAX_ITER : int, optional
        Maximum number of _scf iterations. Default is 200.

    Returns
    -------
    H : torch.Tensor
        Final one-electron Hamiltonian matrix after _scf.
    Hcoul : torch.Tensor
        Final Coulomb contribution to the Hamiltonian.
    Hdipole : torch.Tensor
        Dipole correction to the Hamiltonian from external E-field.
    D : torch.Tensor
        Final density matrix.
    q : torch.Tensor
        Final atomic charges.
    f : torch.Tensor
        Final Fermi orbital occupations (eigenvalues of Dorth).

    Notes
    -----
    - _scf convergence is checked via the L2 norm of charge difference between iterations.
    - The Hamiltonian is corrected for the external electric field through a symmetrized dipole term.
    - Charge density is constructed from Fermi-Dirac occupations.
    - Mixing helps stabilize convergence, especially for metallic systems or small gaps.
    """
    print("### Do _scf ###")
    device = H0.device
    atom_ids = torch.repeat_interleave(
        torch.arange(len(structure.n_orbitals_per_atom), device=Rx.device),
        structure.n_orbitals_per_atom,
    )  # Generate atom index for each orbital

    Z = fractional_matrix_power_symm(S, -0.5)
    with torch.no_grad():
        # if 1:
        Hdipole = torch.diag(
            -Rx[atom_ids] * Efield[0]
            - Ry[atom_ids] * Efield[1]
            - Rz[atom_ids] * Efield[2]
        )
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole

        # Initial guess for chemical potential
        # print('Initial guess for chemical potential')
        # h = torch.linalg.eigvalsh(Z.T @ H0 @ Z)
        # mu0 = 0.5 * (h[nocc - 1] + h[nocc])

        # Initial density matrix
        print("  Initial dm_fermi")
        Dorth, mu0 = dm_fermi(
            Z.T @ H0 @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50
        )
        D = Z @ Dorth @ Z.T
        DS = 2 * torch.diag(D @ S)
        q = -1.0 * Znuc
        q.scatter_add_(
            0, atom_ids, DS
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        #####
        # atom_ids_sr = torch.repeat_interleave(torch.arange(len(structure.shell_types), device=Rx.device), const.shell_dim[structure.shell_types]) # Generate atom index for each orbital
        # q_sr = -1.0 * structure.el_per_shell
        # q_sr.scatter_add_(0, atom_ids_sr, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        #####

        Res = torch.tensor([2.0], device=device)
        it = 0
        Eband0 = torch.tensor([0.0], device=device)
        Ecoul = torch.tensor([0.0], device=device)
        dEb = torch.tensor([10.0], device=device)

        #####
        # Eband0_sr = torch.tensor([0.0], device=device)
        # Ecoul_sr = torch.tensor([0.0], device=device)
        #####

        print("\nStarting cycle")
        while ((Res > acc) + (dEb > acc * 20)) and it < MAX_ITER:
            start_time = time.perf_counter()
            it += 1
            print("Iter {}".format(it))
            if it == MAX_ITER:
                print("Did not converge")

            if debug:
                torch.cuda.synchronize()
            start_time1 = time.perf_counter()
            CoulPot = C @ q
            Hcoul_diag = U[atom_ids] * q[atom_ids] + CoulPot[atom_ids]
            # Hcoul_diag = CoulPot[atom_ids]
            Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
            H = H0 + Hcoul

            #####
            # CoulPot_sr = C_sr @ q_sr
            # Hcoul_diag_sr = structure.Hubbard_U_sr[atom_ids_sr] * q_sr[atom_ids_sr] + CoulPot_sr[atom_ids_sr]
            # #Hcoul_diag_sr = CoulPot_sr[atom_ids_sr]
            # Hcoul_sr = 0.5 * (Hcoul_diag_sr.unsqueeze(1) * S + S * Hcoul_diag_sr.unsqueeze(0))
            # H_sr = H0 + Hcoul_sr
            #####

            if debug:
                torch.cuda.synchronize()
            print("  Hcoul {:.1f} s".format(time.perf_counter() - start_time1))

            start_time1 = time.perf_counter()

            # Dorth, mu0 = dm_fermi((Z.T @ H @ Z).to(torch.float64), Te, nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=debug)
            Dorth, mu0 = dm_fermi(
                (Z.T @ H @ Z),
                Te,
                nocc,
                mu_0=None,
                m=18,
                eps=1e-9,
                MaxIt=50,
                debug=debug,
            )
            # Dorth = Dorth.to(torch.get_default_dtype())

            #####
            # Dorth_sr, mu0_sr = dm_fermi((Z.T @ H_sr @ Z).to(torch.float64), Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50, debug=debug)
            # Dorth_sr = Dorth_sr.to(torch.get_default_dtype())
            #####

            if debug:
                torch.cuda.synchronize()
            print("  dm_fermi {:.1f} s".format(time.perf_counter() - start_time1))

            start_time1 = time.perf_counter()

            # D = Z.to(torch.float32) @ Dorth.to(torch.float32) @ Z.T.to(torch.float32)
            D = Z @ Dorth @ Z.T

            #####
            # D_sr = Z @ Dorth_sr @ Z.T
            #####

            if debug:
                torch.cuda.synchronize()
            print("  Z@Dorth@Z.T {:.1f} s".format(time.perf_counter() - start_time1))

            start_time1 = time.perf_counter()

            q_old = q.clone()
            DS = 2 * (D * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
            q = -1.0 * Znuc
            q.scatter_add_(
                0, atom_ids, DS
            )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
            Res = torch.norm(q - q_old)
            print(q.sum())
            q = (1 - alpha) * q_old + alpha * q

            #####
            # q_sr_old = q_sr.clone()
            # DS_sr = 2 * (D_sr * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
            # q_sr = -1.0 * structure.el_per_shell
            # q_sr.scatter_add_(0, atom_ids_sr, DS_sr) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
            # Res_sr = torch.norm(q_sr - q_sr_old)
            # print(q_sr.sum())
            # q_sr = (1-alpha)*q_sr_old + alpha * q_sr
            #####

            if debug:
                torch.cuda.synchronize()
            print("  update q {:.1f} s".format(time.perf_counter() - start_time1))

            Eband0_old = Eband0.clone()
            Ecoul_old = Ecoul.clone()
            Eband0 = 2 * torch.trace(H0 @ (D - D0))
            Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)

            dEb = (Eband0_old - Eband0).abs()

            #####
            # Eband0_sr_old = Eband0_sr.clone()
            # Ecoul_sr_old = Ecoul_sr.clone()
            # Eband0_sr = 2 * torch.trace(H0 @ (D_sr-D0))
            # Ecoul_sr = 0.5 * q_sr @ (C_sr @ q_sr) + 0.5 * torch.sum(q_sr**2 * structure.Hubbard_U_sr)
            #####
            # print(Eband0 - Eband0_sr, Ecoul - Ecoul_sr)
            # print(Eband0 + Ecoul - Eband0_sr - Ecoul_sr)

            print(
                "Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    Res.item(),
                    torch.abs(Eband0_old - Eband0).item(),
                    torch.abs(Ecoul_old - Ecoul).item(),
                    time.perf_counter() - start_time,
                )
            )
            # print("Res_sr = {:.9f}, t = {:.1f} s\n".format(Res_sr.item(), time.perf_counter()-start_time ))

        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))

    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)
    q = -1.0 * Znuc
    q.scatter_add_(0, atom_ids, DS)

    return H, Hcoul, Hdipole, D, q, f


# --- simple Anderson (Type-I) mixer on charges --------------------------------
class AndersonMixer:
    """
    Anderson acceleration for fixed-point q = F(q).
    We are given (q_in -> q_out), residual r := q_out - q_in.
    Update: q_next = q_out - ΔX * beta,  beta solves  min || r_k - ΔR * beta ||_2
    with small Tikhonov regularization for stability.
    """

    def __init__(
        self, dim, m=5, lam=1e-10, damping=1.0, device=None, dtype=torch.float64
    ):
        self.m = int(m)
        self.lam = lam
        self.damping = damping
        self.device = device
        self.dtype = dtype
        self.q_hist = deque([], maxlen=self.m + 1)  # store q_k
        self.r_hist = deque([], maxlen=self.m + 1)  # store r_k
        # prealloc scratch
        self._last_beta = None

    def reset(self):
        self.q_hist.clear()
        self.r_hist.clear()
        self._last_beta = None

    @torch.no_grad()
    def step(self, q_in, q_out):
        """
        q_in, q_out: (Natoms,) charge vectors on same device/dtype
        returns q_next
        """
        r = q_out - q_in  # residual
        self.q_hist.append(q_in.clone())
        self.r_hist.append(r.clone())

        # not enough history -> fall back to damped linear mix towards q_out
        if len(self.q_hist) < 2:
            return (1.0 - self.damping) * q_in + self.damping * q_out

        # build ΔR and ΔX from history (columns are differences)
        p = min(len(self.q_hist) - 1, self.m)
        # take last (p+1) entries
        Q = list(self.q_hist)[-(p + 1) :]
        R = list(self.r_hist)[-(p + 1) :]
        # differences (N,p)
        dR_cols = []
        dX_cols = []
        for i in range(1, p + 1):
            dR_cols.append(R[i] - R[i - 1])
            dX_cols.append(Q[i] - Q[i - 1])
        dR = torch.stack(dR_cols, dim=1)  # (N,p)
        dX = torch.stack(dX_cols, dim=1)  # (N,p)

        # solve least squares: min_beta || r_k - dR * beta ||_2^2 + lam ||beta||^2
        # normal equations: (dR^T dR + lam I) beta = dR^T r
        # shapes: (p,p) (p,)
        G = dR.T @ dR
        if self.lam > 0:
            G = G + self.lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        rhs = dR.T @ r
        beta = torch.linalg.solve(G, rhs)  # (p,)
        self._last_beta = beta

        # Anderson update
        q_star = q_out - dX @ beta
        # optional damping towards q_in (helps early iterations)
        q_next = (1.0 - self.damping) * q_in + self.damping * q_star
        return q_next


def SCF_adaptive_mixing(
    H0,
    S,
    Efield,
    C,
    TYPE,
    Rx,
    Ry,
    Rz,
    H_Index_Start,
    H_Index_End,
    nocc,
    U,
    Znuc,
    Nats,
    Te,
    const,
    mixing="adaptive",  # "adaptive" or "anderson"
    alpha0=0.2,  # initial linear-mix alpha (for "adaptive")
    alpha_min=0.02,
    alpha_max=0.7,  # clamp for adaptive alpha
    grow=1.15,
    shrink=0.5,  # alpha *= grow if improving, *= shrink if worsening
    anderson_m=6,
    anderson_lam=1e-10,
    anderson_damp=1.0,  # for "anderson"
    acc=1e-7,
    MAX_ITER=200,
):
    """
    _scf with adaptive mixing:
    - mixing='adaptive': auto-tunes linear α from residual history
    - mixing='anderson': Anderson (Pulay-like) charge mixing (recommended)

    Returns: H, Hcoul, Hdipole, D, q, f
    """
    print("### Do _scf (adaptive mixing) ###")
    dtype = H0.dtype
    device = H0.device

    n_orbitals_per_atom = const.n_orb[TYPE]  # (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom
    )

    # Symmetric orthogonalizer
    Z = fractional_matrix_power_symm(S, -0.5)

    # External field dipole term (symmetrized)
    Hdipole = torch.diag(
        -Rx[atom_ids] * Efield[0] - Ry[atom_ids] * Efield[1] - Rz[atom_ids] * Efield[2]
    )
    Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
    H0 = H0 + Hdipole

    # Initial density via Fermi operator on H0
    print("  Initial dm_fermi")
    Dorth, mu0 = dm_fermi(Z.T @ H0 @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50)
    D = Z @ Dorth @ Z.T
    DS = 2 * torch.diag(D @ S)  # AO populations per orbital
    # Build initial atomic charges q (Mulliken-like): q_A = sum_occ_on_A - Z_A
    q = -Znuc.to(DS.dtype, copy=False).to(DS.device).clone()

    q.scatter_add_(0, atom_ids, DS)

    # Mixers
    alpha = float(alpha0)
    last_Res = None
    if mixing.lower() == "anderson":
        mixer = AndersonMixer(
            dim=Nats,
            m=anderson_m,
            lam=anderson_lam,
            damping=anderson_damp,
            device=device,
            dtype=dtype,
        )
    else:
        mixer = None

    it = 0
    Res = torch.tensor([float("inf")], device=device, dtype=dtype)

    print("\nStarting cycle")
    while Res > acc and it < MAX_ITER:
        t0 = time.perf_counter()
        it += 1
        print(f"Iter {it}")

        # --- Build Coulomb contribution from current charges q ------------
        t1 = time.perf_counter()
        CoulPot = C @ q
        Hcoul_diag = U[atom_ids] * q[atom_ids] + CoulPot[atom_ids]
        Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
        H = H0 + Hcoul
        print("  Hcoul {:.3f} s".format(time.perf_counter() - t1))

        # --- New density (out) for current H ------------------------------
        t1 = time.perf_counter()
        Dorth, mu0 = dm_fermi(
            Z.T @ H @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50
        )
        print("  dm_fermi {:.3f} s".format(time.perf_counter() - t1))

        # --- Back-transform density --------------------------------------
        t1 = time.perf_counter()
        D = Z @ Dorth @ Z.T
        print("  Z@Dorth@Z.T {:.3f} s".format(time.perf_counter() - t1))

        # --- Build output charges q_out from this density -----------------
        t1 = time.perf_counter()
        DS = 2 * (D * S.T).sum(dim=1)  # same as 2*diag(D@S) but faster
        q_out = -Znuc.to(DS.dtype, copy=False).to(DS.device).clone()
        q_out.scatter_add_(0, atom_ids, DS)

        # --- Residual & mixing -------------------------------------------
        # residual based on atomic charges (L2)
        Res = torch.norm(q_out - q)
        # choose mixing strategy
        if mixing.lower() == "anderson":
            q_next = mixer.step(q, q_out)
        else:
            # adaptive linear mixing: if improving fast, increase α; else decrease
            if last_Res is not None:
                if Res < 0.7 * last_Res:
                    alpha = min(alpha * grow, alpha_max)
                elif Res > 1.05 * last_Res:
                    alpha = max(alpha * shrink, alpha_min)
            q_next = (1.0 - alpha) * q + alpha * q_out

        # update and report
        last_Res = Res.clone()
        q = q_next
        print(
            f"  mix: method={mixing}, Res={Res.item():.3e}, "
            + (
                f"alpha={alpha:.3f}"
                if mixing.lower() != "anderson"
                else f"Anderson m={anderson_m}, damp={anderson_damp:.2f}"
            )
        )
        print("  update q {:.3f} s".format(time.perf_counter() - t1))
        print("  iter wall {:.3f} s\n".format(time.perf_counter() - t0))

    if it >= MAX_ITER and Res > acc:
        print("WARNING: _scf did not converge (Res={:.3e})".format(Res.item()))

    f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))
    return H, Hcoul, Hdipole, D, q, f
