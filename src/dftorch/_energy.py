import torch


def Energy(
    H0: torch.Tensor,
    U: torch.Tensor,
    Efield: torch.Tensor,
    D0: torch.Tensor,
    C: torch.Tensor,
    dq_p1: torch.Tensor,
    D: torch.Tensor,
    q: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    f: torch.Tensor,
    Te: float,
):
    """
    Compute the total DFTB energy in the standard SCC formulation.

    The energy includes band-structure, Coulomb (SCC), interaction with an
    external electric field (dipole term), and electronic entropy.

    Parameters
    ----------
    H0 : torch.Tensor
        One-electron Hamiltonian in the orthogonal basis, shape (n_orb, n_orb).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    Efield : torch.Tensor
        External electric field vector, shape (3,).
    D0 : torch.Tensor
        Reference (atomic) density. If a matrix, shape (n_orb, n_orb); if a
        vector, shape (n_orb,). Used only to build a diagonal reference.
    C : torch.Tensor
        Coulomb interaction. Either:
        - full matrix of shape (Nats, Nats), or
        - effective Coulomb potential per atom of shape (Nats,).
    D : torch.Tensor
        Current (self-consistent) density matrix, shape (n_orb, n_orb).
    q : torch.Tensor
        SCC atomic charges, shape (Nats,).
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    f : torch.Tensor
        Orbital occupation numbers, shape (n_orb,).
    Te : float
        Electronic temperature in Kelvin.

    Returns
    -------
    Etot : torch.Tensor
        Total energy (scalar tensor).
    Eband0 : torch.Tensor
        Band-structure energy (scalar tensor).
    Ecoul : torch.Tensor
        Coulomb (electrostatic) energy (scalar tensor).
    Edipole : torch.Tensor
        Interaction energy with the external electric field (scalar tensor).
    S_ent : torch.Tensor
        Electronic entropy contribution (dimensionless, in units of k_B).

    Notes
    -----
    The total energy is assembled as::

        Etot = Eband0 + Ecoul + Edipole - 2 * Te * S_ent

    with the factor 2 accounting for spin degeneracy in the entropy term.
    """
    kB = 8.61739e-5  # eV/K

    if torch.get_default_dtype() == torch.float32:
        eps = 1e-7
    elif torch.get_default_dtype() == torch.float64:
        eps = 1e-10

    # Band energy
    if Rx.dim() == 1:  # non-batched. both cs and os.
        factor = 2 if D.dim() == 2 else 1  # closed-shell or open-shell
        Eband0 = factor * (H0 @ D).diagonal(offset=0, dim1=-2, dim2=-1).sum()
    else:  # batched
        Eband0 = 2 * (H0 @ D).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

    # Coulomb energy
    if Rx.dim() == 1:  # non-batched
        if C is None and dq_p1 is None:
            Ecoul = 0  # structure.e_coul_tmp
        elif C is None and dq_p1 is not None:  # PME
            Ecoul = 0.5 * q @ dq_p1 + 0.5 * torch.sum(q**2 * U)
        elif C is not None and dq_p1 is None:  # full Coulomb matrix
            Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        elif C is not None and dq_p1 is not None:
            raise ValueError("Both C and dq_p1 are provided; only one expected.")
    else:  # batched, only full Coulomb matrix for batches
        if dq_p1 is not None:
            raise ValueError("Batched PME Coulomb not implemented.")
        Cq = torch.bmm(C, q.unsqueeze(-1)).squeeze(-1)  # (B,N)
        Ecoul = 0.5 * torch.sum(q * Cq, dim=-1) + 0.5 * torch.sum(q**2 * U, dim=1)

    # Dipole energy
    Efield_term = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    Edipole = -torch.sum(q * Efield_term, dim=-1)

    # Entropy
    # mask = (f > eps) & (f < 1 - eps)
    # S_ent = -kB * torch.sum(f[mask] * torch.log(f[mask]) + (1 - f[mask]) * torch.log(1 - f[mask]))
    mask = (f > eps) & (f < 1 - eps)  # (B, n_orb)
    f_safe = f.clamp(eps, 1 - eps)  # avoid log(0)
    term = f_safe * torch.log(f_safe) + (1 - f_safe) * torch.log(1 - f_safe)
    term = term * mask  # zero out invalid entries

    if Rx.dim() == 1:  # non-batched. both cs and os.
        S_ent = -kB * term.sum()
    else:
        S_ent = -kB * term.sum(dim=-1)  # (B,)

    E_entropy = -2 * Te * S_ent

    # Total energy
    Etot = Eband0 + Ecoul + Edipole + E_entropy

    return Etot, Eband0, Ecoul, Edipole, E_entropy, S_ent


def EnergyShadow(
    H0: torch.Tensor,
    U: torch.Tensor,
    Efield: torch.Tensor,
    D0: torch.Tensor,
    C: torch.Tensor,
    dq_p1: torch.Tensor,
    D: torch.Tensor,
    q: torch.Tensor,
    n: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    f: torch.Tensor,
    Te: float,
):
    """
    Compute the total “shadow” DFTB energy.

    This variant modifies the SCC Coulomb term by introducing an auxiliary
    charge-like vector ``n``. The standard SCC energy is recovered if
    ``n == q``. Band, dipole, and entropy terms are unchanged relative to
    :func:`Energy`.

    Parameters
    ----------
    H0 : torch.Tensor
        One-electron Hamiltonian in the orthogonal basis, shape (n_orb, n_orb).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    Efield : torch.Tensor
        External electric field vector, shape (3,).
    D0 : torch.Tensor
        Reference (atomic) density. If a matrix, shape (n_orb, n_orb); if a
        vector, shape (n_orb,).
    C : torch.Tensor
        Coulomb interaction. Either:
        - full matrix of shape (Nats, Nats), or
        - effective Coulomb potential per atom of shape (Nats,).
    D : torch.Tensor
        Current (self-consistent) density matrix, shape (n_orb, n_orb).
    q : torch.Tensor
        SCC atomic charges, shape (Nats,). Appears together with ``n`` in the
        modified Coulomb term and in the dipole term.
    n : torch.Tensor
        Shadow charge (or occupancy) vector per atom, shape (Nats,). Used in
        place of ``q`` inside the Coulomb energy and in the ``(2*q - n)`` prefactor.
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    f : torch.Tensor
        Orbital occupation numbers, shape (n_orb,).
    Te : float
        Electronic temperature in Kelvin.

    Returns
    -------
    Etot : torch.Tensor
        Total energy (scalar tensor).
    Eband0 : torch.Tensor
        Band-structure energy (scalar tensor).
    Ecoul : torch.Tensor
        Shadow Coulomb (electrostatic) energy (scalar tensor).
    Edipole : torch.Tensor
        Interaction energy with the external electric field (scalar tensor).
    S_ent : torch.Tensor
        Electronic entropy contribution (dimensionless, in units of k_B).

    Notes
    -----
    The shadow Coulomb energy is::

        Ecoul = 0.5 * (2*q - n)^T C n + 0.5 * sum_i (2*q_i - n_i) U_i n_i

    in the matrix ``C`` case, with the obvious simplification when ``C`` is
    provided as an effective potential per atom. The total energy is::

        Etot = Eband0 + Ecoul + Edipole - 2 * Te * S_ent

    Setting ``n = q`` reduces this to the standard SCC expression in
    :func:`Energy`.
    """
    kB = 8.61739e-5  # eV/K

    if torch.get_default_dtype() == torch.float32:
        eps = 1e-7
    elif torch.get_default_dtype() == torch.float64:
        eps = 1e-10

    # Band energy
    if Rx.dim() == 1:  # non-batched. both cs and os.
        factor = 2 if D.dim() == 2 else 1  # closed-shell or open-shell
        Eband0 = factor * (H0 @ D).diagonal(offset=0, dim1=-2, dim2=-1).sum()
    else:  # batched
        Eband0 = 2 * (H0 @ D).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)

    # Coulomb energy
    if Rx.dim() == 1:  # non-batched
        if C is None and dq_p1 is None:
            Ecoul = 0  # structure.e_coul_tmp
        elif C is None and dq_p1 is not None:  # PME
            Ecoul = 0.5 * (2 * q - n) @ dq_p1 + 0.5 * torch.sum((2.0 * q - n) * U * n)
        elif C is not None and dq_p1 is None:  # full Coulomb matrix
            Ecoul = 0.5 * (2 * q - n) @ (C @ n) + 0.5 * torch.sum((2.0 * q - n) * U * n)
        elif C is not None and dq_p1 is not None:
            raise ValueError("Both C and dq_p1 are provided; only one expected.")
    else:  # batched, only full Coulomb matrix for batches
        if dq_p1 is not None:
            raise ValueError("Batched PME Coulomb not implemented.")
        Cn = torch.bmm(C, n.unsqueeze(-1)).squeeze(-1)  # (B,N)
        Ecoul = 0.5 * torch.sum((2 * q - n) * Cn, dim=-1) + 0.5 * torch.sum(
            (2.0 * q - n) * U * n, dim=1
        )

    # Dipole energy
    Efield_term = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    Edipole = -torch.sum(q * Efield_term, dim=-1)

    # Entropy
    # mask = (f > eps) & (f < 1 - eps)
    # S_ent = -kB * torch.sum(f[mask] * torch.log(f[mask]) + (1 - f[mask]) * torch.log(1 - f[mask]))
    mask = (f > eps) & (f < 1 - eps)  # (B, n_orb)
    f_safe = f.clamp(eps, 1 - eps)  # avoid log(0)
    term = f_safe * torch.log(f_safe) + (1 - f_safe) * torch.log(1 - f_safe)
    term = term * mask  # zero out invalid entries

    if Rx.dim() == 1:  # non-batched. both cs and os.
        S_ent = -kB * term.sum()
    else:
        S_ent = -kB * term.sum(dim=-1)  # (B,)

    E_entropy = -2 * Te * S_ent

    # Total energy
    Etot = Eband0 + Ecoul + Edipole + E_entropy

    return Etot, Eband0, Ecoul, Edipole, E_entropy, S_ent
