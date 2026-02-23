import torch
from .Fermi_PRT import Canon_DM_PRT


def Kernel_Fermi(structure, mu0, T, Nr_atoms, H, C, S, Z, Q, e):
    """
    Build the charge-response kernel for Krylov/Anderson SCF acceleration.

    This computes the atomic charge response matrix dq_dn with elements
      dq_dn[J, I] = ∂q_I / ∂n_J
    by applying a unit perturbation on atomic population n_J, forming the
    induced Coulomb/Hubbard AO Hamiltonian perturbation dH, and obtaining
    the density response D1 via canonical Fermi perturbation theory in the
    orthogonalized AO basis. The SCF kernel returned is
      KK = (dq_dn^T − I)^{-1},
    which maps residuals on atomic charges to a mixed update.

    Args:
        structure: Structure with fields
            - n_orbitals_per_atom (list/1D tensor int)
            - Hubbard_U (Nats,)
        mu0 (tensor or float): Chemical potential at the current SCF point.
        T (tensor or float): Electronic temperature (Kelvin).
        Nr_atoms (int): Number of atoms (Nats).
        H (tensor): Current AO Hamiltonian, shape (NAO, NAO).
        C (tensor): Coulomb interaction matrix between atoms, shape (Nats, Nats).
        S (tensor): AO overlap matrix, shape (NAO, NAO).
        Z (tensor): Symmetric inverse square root of S, Z = S^{-1/2}, shape (NAO, NAO).
        Q (tensor): Eigenvectors of the orthogonalized Hamiltonian (columns), shape (NAO, NAO).
        e (tensor): Eigenvalues corresponding to Q, shape (NAO,).

    Returns:
        KK (tensor): SCF kernel, shape (Nats, Nats), KK = (dq_dn^T − I)^{-1}.
        D0 (tensor): Unperturbed density matrix in the orthogonal basis as returned by Canon_DM_PRT.

    Notes:
        - For each atom J, a unit perturbation on n_J produces an AO-diagonal
          shift U_i δ_{i∈J} and a Coulomb shift (C @ δ_J) at each AO via atom mapping.
          The AO perturbation is symmetrized with S: dH = 0.5(d* S + S* d).
        - Response D1 is computed in the orthogonal basis and back-transformed to AO.
          Atomic charge response is accumulated from 2·diag(D1 @ S) over AOs per atom.
        - A factor 2 is used for spin degeneracy.
        - Complexity is roughly O(Nats · NAO^3) due to repeated linear responses.

    Shape/dtype/device:
        All tensors are expected on H.device with consistent dtype; outputs follow inputs.
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
        d_Hcoul_diag = (
            structure.Hubbard_U[atom_ids] * dq_J[atom_ids] + d_CoulPot[atom_ids]
        )
        d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0))

        H1 = Z.T @ d_Hcoul @ Z
        # [D0,D_dq_J] = Fermi_PRT(H0,H1,T,Q,e,mu0);
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
