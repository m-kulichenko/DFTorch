from __future__ import annotations

from typing import Any

import torch

from ._fermi_prt import Canon_DM_PRT


def _kernel_fermi(
    structure: Any,
    mu0: torch.Tensor | float,
    T: torch.Tensor | float,
    Nr_atoms: int,
    H: torch.Tensor,
    C: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Q: torch.Tensor,
    e: torch.Tensor,
    gbsa: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the atomic charge-response kernel for SCF mixing.

    This routine applies a unit perturbation to each atomic charge degree of
    freedom, computes the induced AO Hamiltonian perturbation, and evaluates
    the first-order density response via canonical finite-temperature
    perturbation theory. The resulting charge-response matrix is converted
    into the mixing kernel ``(dq_dn.T - I)^(-1)``.

    Parameters
    ----------
    structure : Any
        Structure-like object exposing ``n_orbitals_per_atom`` and ``Hubbard_U``.
    mu0 : torch.Tensor or float
        Chemical potential at the current SCF point.
    T : torch.Tensor or float
        Electronic temperature in Kelvin.
    Nr_atoms : int
        Number of atoms.
    H : torch.Tensor
        AO Hamiltonian of shape ``(n_orb, n_orb)``.
    C : torch.Tensor
        Atomic Coulomb interaction matrix of shape ``(Nats, Nats)``.
    S : torch.Tensor
        AO overlap matrix of shape ``(n_orb, n_orb)``.
    Z : torch.Tensor
        Symmetric orthogonalizer ``S^{-1/2}``.
    Q : torch.Tensor
        Eigenvectors of the orthogonalized Hamiltonian.
    e : torch.Tensor
        Eigenvalues corresponding to ``Q``.
    gbsa : Any, optional
        Optional solvation object exposing ``get_shifts``.

    Returns
    -------
    KK : torch.Tensor
        Charge-mixing kernel of shape ``(Nats, Nats)``.
    D0 : torch.Tensor
        Unperturbed density matrix in the orthogonal basis returned by
        :func:`Canon_DM_PRT`.
    """
    dq_dn = torch.zeros(Nr_atoms, Nr_atoms, device=H.device)
    dq_J = torch.zeros(Nr_atoms, device=H.device)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(structure.n_orbitals_per_atom), device=H.device),
        structure.n_orbitals_per_atom,
    )  # Generate atom index for each orbital

    for J in range(0, Nr_atoms):
        print("Building kernel row ", J + 1, " of ", Nr_atoms)

        dq_J[J] = 1

        d_CoulPot = C @ dq_J
        if gbsa is not None:
            d_CoulPot = d_CoulPot + gbsa.get_shifts(dq_J)
        d_Hcoul_diag = (
            structure.Hubbard_U[atom_ids] * dq_J[atom_ids] + d_CoulPot[atom_ids]
        )
        d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0))

        H1 = Z.T @ d_Hcoul @ Z
        # [D0,D_dq_J] = fermi_prt(H0,H1,T,Q,e,mu0);
        D0, D_dq_J = Canon_DM_PRT(H1, T, Q, e, mu0, 10)

        D_dq_J = 2 * Z @ D_dq_J @ Z.T
        D_diag = torch.diag(D_dq_J @ S)
        dqI_dqJ = torch.zeros(Nr_atoms, device=H.device)

        dqI_dqJ.scatter_add_(
            0, atom_ids, D_diag
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        dq_dn[J, :] = dqI_dqJ
        dq_J[J] = 0

    II = torch.eye(Nr_atoms, device=H.device)
    KK = torch.linalg.matrix_power(dq_dn.T - II, -1)
    return KK, D0
