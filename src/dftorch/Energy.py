import torch
def Energy(H0, U, Efield, D0, C, D, q, Rx, Ry, Rz, f, Te):
    """
    Computes the total DFTB energy, including band, Coulomb, dipole, and entropy contributions.

    Args:
        H0 (torch.Tensor): Hamiltonian matrix in the orthogonal basis (N x N).
        U (torch.Tensor): Hubbard U parameters per atom (N,).
        Efield (torch.Tensor): External electric field vector (3,).
        D0 (torch.Tensor): Reference density matrix, diagonal or full (N x N or N,).
        C (torch.Tensor): Coulomb interaction matrix (N x N).
        D (torch.Tensor): Current density matrix (N x N).
        q (torch.Tensor): SCC atomic charges (N,).
        Rx (torch.Tensor): X coordinates of atoms (N,).
        Ry (torch.Tensor): Y coordinates of atoms (N,).
        Rz (torch.Tensor): Z coordinates of atoms (N,).
        f (torch.Tensor): Orbital occupation numbers (N,).
        Te (float): Electronic temperature in Kelvin.

    Returns:
        Etot (float): Total energy.
        Eband0 (float): Band structure energy.
        Ecoul (float): Coulomb (electrostatic) energy.
        Edipole (float): Interaction energy with external electric field.
        S_ent (float): Electronic entropy contribution.
    """
    kB = 8.61739e-5  # eV/K
    eps = 1e-10

    # Ensure D0 is diagonal for consistent subtraction
    if D0.ndim == 2:
        D0_diag = torch.diag(D0)
    else:
        D0_diag = D0

    # Band energy
    #Eband0 = 2 * torch.trace(H0 @ (D - torch.diag(D0_diag)))
    Eband0 = 2 * torch.trace(H0 @ (D))

    # Coulomb energy
    if C.dim() == 1:
        Ecoul = 0.5 * q @ C + 0.5 * torch.sum(q**2 * U)
    else:
        Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)

    # Dipole energy
    Efield_term = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    Edipole = -torch.sum(q * Efield_term)

    # Entropy
    mask = (f > eps) & (f < 1 - eps)
    S_ent = -kB * torch.sum(f[mask] * torch.log(f[mask]) + (1 - f[mask]) * torch.log(1 - f[mask]))

    # Total energy
    Etot = Eband0 + Ecoul + Edipole - 2 * Te * S_ent

    return Etot, Eband0, Ecoul, Edipole, S_ent

def EnergyShadow(H0, U, Efield, D0, C, D, q, n, Rx, Ry, Rz, f, Te):
    """
    Computes the total DFTB energy, including band, Coulomb, dipole, and entropy contributions.

    Args:
        H0 (torch.Tensor): Hamiltonian matrix in the orthogonal basis (N x N).
        U (torch.Tensor): Hubbard U parameters per atom (N,).
        Efield (torch.Tensor): External electric field vector (3,).
        D0 (torch.Tensor): Reference density matrix, diagonal or full (N x N or N,).
        C (torch.Tensor): Coulomb interaction matrix (N x N).
        D (torch.Tensor): Current density matrix (N x N).
        q (torch.Tensor): SCC atomic charges (N,).
        Rx (torch.Tensor): X coordinates of atoms (N,).
        Ry (torch.Tensor): Y coordinates of atoms (N,).
        Rz (torch.Tensor): Z coordinates of atoms (N,).
        f (torch.Tensor): Orbital occupation numbers (N,).
        Te (float): Electronic temperature in Kelvin.

    Returns:
        Etot (float): Total energy.
        Eband0 (float): Band structure energy.
        Ecoul (float): Coulomb (electrostatic) energy.
        Edipole (float): Interaction energy with external electric field.
        S_ent (float): Electronic entropy contribution.
    """
    kB = 8.61739e-5  # eV/K
    eps = 1e-10

    # Ensure D0 is diagonal for consistent subtraction
    if D0.ndim == 2:
        D0_diag = torch.diag(D0)
    else:
        D0_diag = D0

    # Band energy
    #Eband0 = 2 * torch.trace(H0 @ (D - torch.diag(D0_diag)))
    Eband0 = 2 * torch.trace(H0 @ (D))

    # Coulomb energy
    if C.dim() == 1:
        Ecoul = 0.5 * (2*q-n) @ C + 0.5 * torch.sum((2.0*q - n) * U * n)
    else:
        Ecoul = 0.5 * (2*q-n) @ (C @ n) + 0.5 * torch.sum((2.0*q - n) * U * n)

    # Dipole energy
    Efield_term = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    Edipole = -torch.sum(q * Efield_term)

    # Entropy
    mask = (f > eps) & (f < 1 - eps)
    S_ent = -kB * torch.sum(f[mask] * torch.log(f[mask]) + (1 - f[mask]) * torch.log(1 - f[mask]))

    # Total energy
    Etot = Eband0 + Ecoul + Edipole - 2 * Te * S_ent

    return Etot, Eband0, Ecoul, Edipole, S_ent