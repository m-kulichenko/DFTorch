import torch


@torch.compile
def Forces(
    H: torch.Tensor,
    Z: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    D0: torch.Tensor,
    dH: torch.Tensor,
    dS: torch.Tensor,
    dC: torch.Tensor,
    dVr: torch.Tensor,
    Efield: torch.Tensor,
    U: torch.Tensor,
    q: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
    verbose: bool = False,
):
    """
    Compute atomic forces for a DFTB-like total energy expression in gas phase
    (non-PME, standard SCC).

    Parameters
    ----------
    H : torch.Tensor
        Two-electron (effective) Hamiltonian matrix of shape (n_orb, n_orb).
    Z : torch.Tensor
        Transformation / renormalization matrix (e.g. orthogonalization) used
        in the Pulay term, shape (n_orb, n_orb).
    C : torch.Tensor
        Coulomb interaction matrix of shape (Nats, Nats). It is multiplied by
        the atomic charges `q` to obtain the Coulomb potential.
    D : torch.Tensor
        Self-consistent density matrix of shape (n_orb, n_orb).
    D0 : torch.Tensor
        Reference (atomic) density vector of shape (n_orb,). Internally turned
        into a diagonal matrix.
    dH : torch.Tensor
        Derivatives of the Hamiltonian with respect to Cartesian coordinates,
        shape (3, n_orb, n_orb). The first dimension corresponds to x, y, z.
    dS : torch.Tensor
        Derivatives of the overlap matrix with respect to Cartesian
        coordinates, shape (3, n_orb, n_orb).
    dC : torch.Tensor
        Derivatives of the Coulomb matrix with respect to atomic coordinates,
        shape (Nats, Nats, 3) or broadcastable equivalent; it is contracted
        with the charges in `q`.
    dVr : torch.Tensor
        Derivatives of the short-range repulsive potential with respect to
        atomic coordinates, shape (3, Nats, Nats).
    Efield : torch.Tensor
        External electric field vector of shape (3,).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    q : torch.Tensor
        Self-consistent charges per atom, shape (Nats,).
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    Nats : int
        Number of atoms in the system.
    const : object
        Container with model constants. Must provide `n_orb`, the number of
        orbitals per element type.
    TYPE : torch.Tensor
        Element type indices for each atom, shape (Nats,). Used to map into
        `const.n_orb`.
    verbose : bool, optional
        If True, allows callers to hook in additional logging (currently not
        used inside this routine).

    Returns
    -------
    Ftot : torch.Tensor
        Total forces on atoms, shape (3, Nats). Convention: forces are
        negative gradients of the total energy.
    Fcoul : torch.Tensor
        Coulomb interaction contribution to the forces, shape (3, Nats).
    Fband0 : torch.Tensor
        Band-structure (Hamiltonian) contribution to the forces, shape
        (3, Nats).
    Fdipole : torch.Tensor
        Direct electric-field (dipole) contribution to the forces, shape
        (3, Nats).
    FPulay : torch.Tensor
        Pulay correction forces due to non-orthogonal orbitals, shape
        (3, Nats).
    FScoul : torch.Tensor
        Overlap-derivative contribution associated with the SCC Coulomb
        energy, shape (3, Nats).
    FSdipole : torch.Tensor
        Overlap-derivative contribution associated with the dipole / external
        field term, shape (3, Nats).
    Frep : torch.Tensor
        Short-range repulsive potential contribution to the forces, shape
        (3, Nats).

    Notes
    -----
    The total force is assembled as::

        Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    where all components are computed as negative derivatives of the
    corresponding energy contributions with respect to atomic positions.
    """
    # Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T
    dtype = H.dtype
    device = H.device

    if D.dim() == 3:  # os
        D_tot = D.sum(0) / 2
    else:  # cs
        D_tot = D

    n_orbitals_per_atom = const.n_orb[
        TYPE
    ]  # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    Fcoul = q * (q @ dC)

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = C @ q
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    factor = (U * q + CoulPot) * 2
    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D_tot).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Pulay forces (Here H includes H_spin)
    SIHD = 4 * Z @ Z.T @ H @ D  # removes S factor from H
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    # Collapse alpha and beta channels if present: dS @ (SIHD[0] + SIHD[1]) == dS @ SIHD.sum(0)
    if SIHD.dim() == 2:  # cs
        TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    else:  # os
        TMP = -0.5 * torch.matmul(dS.unsqueeze(0), SIHD.unsqueeze(1)).sum(0).diagonal(
            offset=0, dim1=1, dim2=2
        )  # (2, 3, 4, 4)
    FPulay.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    tmp1 = (D_tot - D0) @ dS
    tmp2 = -2 * (tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    D_diff = D_tot - D0
    n_orb = dS.shape[1]
    a = dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
    outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
    new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
    FSdipole -= 2 * new_fs

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


@torch.compile
def forces_spin(
    D: torch.Tensor,
    dS: torch.Tensor,
    q_spin_atom: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
):
    """ """
    # Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T
    dtype = q_spin_atom.dtype
    device = q_spin_atom.device

    n_orbitals_per_atom = const.n_orb[
        TYPE
    ]  # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    net_spin = q_spin_atom[0] - q_spin_atom[1]
    FSspinA = torch.zeros((3, Nats), dtype=dtype, device=device)
    factor = (net_spin * const.w[TYPE]) * 2
    tmp = (
        (torch.tensor([[[1]], [[-1]]], device=device) * D).unsqueeze(1)
        * dS.unsqueeze(0)
    ).sum(0)
    dS_times_D = tmp * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FSspinA.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FSspinA.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    return FSspinA / 2


@torch.compile
def Forces_PME(
    H: torch.Tensor,
    Z: torch.Tensor,
    dq_p1: torch.Tensor,
    D: torch.Tensor,
    D0: torch.Tensor,
    dH: torch.Tensor,
    dS: torch.Tensor,
    dVr: torch.Tensor,
    Efield: torch.Tensor,
    U: torch.Tensor,
    q: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
    verbose: bool = False,
):
    """
    Compute atomic forces for a DFTB-like total energy expression using
    PME-based electrostatics (periodic boundary conditions).

    In contrast to :func:`Forces`, the PME variant expects the Coulomb
    potential per atom (already including long-range Ewald / PME effects)
    in ``dq_p1`` and does not directly use the Coulomb force derivative tensor
    ``dC`` (the PME forces are assumed to be provided separately).

    Parameters
    ----------
    H : torch.Tensor
        Two-electron (effective) Hamiltonian matrix of shape (n_orb, n_orb).
    Z : torch.Tensor
        Transformation / renormalization matrix (e.g. orthogonalization) used
        in the Pulay term, shape (n_orb, n_orb).
    dq_p1 : torch.Tensor
        Coulomb potential per atom, typically including PME / Ewald
        contributions, shape (Nats,) or (Nats, 1). Used only to build the
        SCC-related overlap term.
    D : torch.Tensor
        Self-consistent density matrix of shape (n_orb, n_orb).
    D0 : torch.Tensor
        Reference (atomic) density vector of shape (n_orb,). Internally turned
        into a diagonal matrix when needed.
    dH : torch.Tensor
        Derivatives of the Hamiltonian with respect to Cartesian coordinates,
        shape (3, n_orb, n_orb). The first dimension corresponds to x, y, z.
    dS : torch.Tensor
        Derivatives of the overlap matrix with respect to Cartesian
        coordinates, shape (3, n_orb, n_orb).
    dVr : torch.Tensor
        Derivatives of the short-range repulsive potential with respect to
        atomic coordinates, shape (3, Nats, Nats).
    Efield : torch.Tensor
        External electric field vector of shape (3,).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    q : torch.Tensor
        Self-consistent charges per atom, shape (Nats,).
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    Nats : int
        Number of atoms in the system.
    const : object
        Container with model constants. Must provide `n_orb`, the number of
        orbitals per element type.
    TYPE : torch.Tensor
        Element type indices for each atom, shape (Nats,). Used to map into
        `const.n_orb`.
    verbose : bool, optional
        If True, allows callers to hook in additional logging (currently not
        used inside this routine).

    Returns
    -------
    Ftot : torch.Tensor
        Total forces on atoms, shape (3, Nats). Convention: forces are
        negative gradients of the total energy.
    Fcoul : torch.Tensor
        PME Coulomb interaction contribution to the forces, shape (3, Nats).
        In this implementation, it is set to zero and should be added
        externally if available.
    Fband0 : torch.Tensor
        Band-structure (Hamiltonian) contribution to the forces, shape
        (3, Nats).
    Fdipole : torch.Tensor
        Direct electric-field (dipole) contribution to the forces, shape
        (3, Nats).
    FPulay : torch.Tensor
        Pulay correction forces due to non-orthogonal orbitals, shape
        (3, Nats).
    FScoul : torch.Tensor
        Overlap-derivative contribution associated with the SCC Coulomb
        energy, using the PME Coulomb potential, shape (3, Nats).
    FSdipole : torch.Tensor
        Overlap-derivative contribution associated with the dipole / external
        field term, shape (3, Nats).
    Frep : torch.Tensor
        Short-range repulsive potential contribution to the forces, shape
        (3, Nats).

    Notes
    -----
    The total force is assembled as::

        Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    where all components are computed as negative derivatives of the
    corresponding energy contributions with respect to atomic positions.
    PME / periodicity affects only the Coulomb-related pieces through the
    supplied Coulomb potential ``dq_p1``.
    """

    # Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T

    dtype = H.dtype
    device = H.device

    if D.dim() == 3:  # os
        D_tot = D.sum(0) / 2
    else:  # cs
        D_tot = D

    n_orbitals_per_atom = const.n_orb[
        TYPE
    ]  # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    Fcoul = torch.zeros((3, Nats), dtype=dtype, device=device)

    # FScoul
    CoulPot = dq_p1
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    factor = (U * q + CoulPot) * 2
    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D_tot).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Pulay forces
    SIHD = 4 * Z @ Z.T @ H @ D
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    if SIHD.dim() == 2:  # cs
        TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    else:  # os
        TMP = -0.5 * torch.matmul(dS.unsqueeze(0), SIHD.unsqueeze(1)).sum(0).diagonal(
            offset=0, dim1=1, dim2=2
        )  # (2, 3, 4, 4)
    FPulay.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    tmp1 = (D_tot - D0) @ dS
    tmp2 = -2 * (tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    D_diff = D_tot - D0
    n_orb = dS.shape[1]
    a = dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
    outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
    new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
    FSdipole -= 2 * new_fs

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


@torch.compile
def forces_shadow(
    H: torch.Tensor,
    Z: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    D0: torch.Tensor,
    dH: torch.Tensor,
    dS: torch.Tensor,
    dC: torch.Tensor,
    dVr: torch.Tensor,
    Efield: torch.Tensor,
    U: torch.Tensor,
    q: torch.Tensor,
    n: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
    verbose: bool = False,
):
    """
    Computes atomic forces from a DFTB-like total energy expression.

    This variant uses an auxiliary charge-like vector ``n`` instead of the
    standard SCC charges ``q`` in the Coulomb- and overlap-related terms.
    The usual SCC force is recovered when ``n == q``. The other contributions
    (band-structure, Pulay, dipole, repulsive) are treated analogously to
    :func:`Forces`.

    Parameters
    ----------
    H : torch.Tensor
        Two-electron (effective) Hamiltonian matrix of shape (n_orb, n_orb).
    Z : torch.Tensor
        Transformation / renormalization matrix (e.g. orthogonalization) used
        in the Pulay term, shape (n_orb, n_orb).
    C : torch.Tensor
        Coulomb interaction matrix of shape (Nats, Nats). It is multiplied by
        the “shadow” charges ``n`` to obtain the Coulomb potential.
    D : torch.Tensor
        Self-consistent density matrix of shape (n_orb, n_orb).
    D0 : torch.Tensor
        Reference (atomic) density vector of shape (n_orb,). Internally turned
        into a diagonal matrix.
    dH : torch.Tensor
        Derivatives of the Hamiltonian with respect to Cartesian coordinates,
        shape (3, n_orb, n_orb). The first dimension corresponds to x, y, z.
    dS : torch.Tensor
        Derivatives of the overlap matrix with respect to Cartesian
        coordinates, shape (3, n_orb, n_orb).
    dC : torch.Tensor
        Derivatives of the Coulomb matrix with respect to atomic coordinates,
        shape (Nats, Nats, 3) or broadcastable equivalent; it is contracted
        with the “shadow” charges ``n`` and the combination ``(2*q - n)`` in
        the Coulomb force.
    dVr : torch.Tensor
        Derivatives of the short-range repulsive potential with respect to
        atomic coordinates, shape (3, Nats, Nats).
    Efield : torch.Tensor
        External electric field vector of shape (3,).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    q : torch.Tensor
        Self-consistent SCC charges per atom, shape (Nats,). Used together
        with ``n`` in the shadow Coulomb force term.
    n : torch.Tensor
        Shadow charge (or occupancy) vector per atom, shape (Nats,). Used in
        place of ``q`` in the Coulomb energy and overlap-related terms.
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    Nats : int
        Number of atoms in the system.
    const : object
        Container with model constants. Must provide `n_orb`, the number of
        orbitals per element type.
    TYPE : torch.Tensor
        Element type indices for each atom, shape (Nats,). Used to map into
        `const.n_orb`.
    verbose : bool, optional
        If True, allows callers to hook in additional logging (currently not
        used inside this routine).

    Returns
    -------
    Ftot : torch.Tensor
        Total forces on atoms, shape (3, Nats). Convention: forces are
        negative gradients of the total energy.
    Fcoul : torch.Tensor
        Shadow Coulomb interaction contribution to the forces, shape (3, Nats).
        Uses the combination ``(2*q - n) * (n @ dC)``.
    Fband0 : torch.Tensor
        Band-structure (Hamiltonian) contribution to the forces, shape
        (3, Nats).
    Fdipole : torch.Tensor
        Direct electric-field (dipole) contribution to the forces, shape
        (3, Nats), still built from the SCC charges ``q``.
    FPulay : torch.Tensor
        Pulay correction forces due to non-orthogonal orbitals, shape
        (3, Nats).
    FScoul : torch.Tensor
        Overlap-derivative contribution associated with the shadow Coulomb
        energy, using ``n`` in place of ``q``, shape (3, Nats).
    FSdipole : torch.Tensor
        Overlap-derivative contribution associated with the dipole / external
        field term, shape (3, Nats).
    Frep : torch.Tensor
        Short-range repulsive potential contribution to the forces, shape
        (3, Nats).

    Notes
    -----
    The total force is assembled as::

        Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    Setting ``n = q`` recovers the standard SCC force decomposition used in
    :func:`Forces`.
    """

    # Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T

    dtype = H.dtype
    device = H.device

    if D.dim() == 3:  # os
        D_tot = D.sum(0) / 2
    else:  # cs
        D_tot = D

    n_orbitals_per_atom = const.n_orb[
        TYPE
    ]  # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    Fcoul = (2 * q - n) * (n @ dC)

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = C @ n
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    factor = (U * n + CoulPot) * 2
    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D_tot).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Pulay forces (Here H includes H_spin)
    SIHD = 4 * Z @ Z.T @ H @ D
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    # Collapse alpha and beta channels if present: dS @ (SIHD[0] + SIHD[1]) == dS @ SIHD.sum(0)
    if SIHD.dim() == 2:  # cs
        TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    else:  # os
        TMP = -0.5 * torch.matmul(dS.unsqueeze(0), SIHD.unsqueeze(1)).sum(0).diagonal(
            offset=0, dim1=1, dim2=2
        )  # (2, 3, 4, 4)
    FPulay.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    D_diff = D_tot - D0
    tmp1 = D_diff @ dS
    tmp2 = -2 * (tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    n_orb = dS.shape[1]
    a = dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
    outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
    new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
    FSdipole -= 2 * new_fs

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


@torch.compile
def forces_shadow_pme(
    H: torch.Tensor,
    Z: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    D0: torch.Tensor,
    dH: torch.Tensor,
    dS: torch.Tensor,
    dVr: torch.Tensor,
    Efield: torch.Tensor,
    U: torch.Tensor,
    q: torch.Tensor,
    n: torch.Tensor,
    Rx: torch.Tensor,
    Ry: torch.Tensor,
    Rz: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
):
    """
    Compute atomic forces for a “shadow” DFTB-like total energy expression
    using PME-based electrostatics (periodic boundary conditions).

    This PME variant mirrors :func:`forces_shadow` but assumes that the
    long-range Coulomb forces are handled externally and provides only the
    band-structure, Pulay, dipole, SCC-overlap, and repulsive contributions.
    As in :func:`forces_shadow`, an auxiliary charge-like vector ``n`` is used
    instead of the standard SCC charges ``q`` in the Coulomb-overlap term.

    Parameters
    ----------
    H : torch.Tensor
        Two-electron (effective) Hamiltonian matrix of shape (n_orb, n_orb).
    Z : torch.Tensor
        Transformation / renormalization matrix (e.g. orthogonalization) used
        in the Pulay term, shape (n_orb, n_orb).
    C : torch.Tensor
        Coulomb potential per atom, typically including PME / Ewald
        contributions, shape (Nats,) or (Nats, 1). Used only to build the
        shadow SCC-related overlap term.
    D : torch.Tensor
        Self-consistent density matrix of shape (n_orb, n_orb).
    D0 : torch.Tensor
        Reference (atomic) density vector of shape (n_orb,). Internally turned
        into a diagonal matrix when needed.
    dH : torch.Tensor
        Derivatives of the Hamiltonian with respect to Cartesian coordinates,
        shape (3, n_orb, n_orb). The first dimension corresponds to x, y, z.
    dS : torch.Tensor
        Derivatives of the overlap matrix with respect to Cartesian
        coordinates, shape (3, n_orb, n_orb).
    dVr : torch.Tensor
        Derivatives of the short-range repulsive potential with respect to
        atomic coordinates, shape (3, Nats, Nats).
    Efield : torch.Tensor
        External electric field vector of shape (3,).
    U : torch.Tensor
        On-site Hubbard U parameters per atom, shape (Nats,).
    q : torch.Tensor
        Self-consistent SCC charges per atom, shape (Nats,). Used to build the
        direct dipole term and appears implicitly in the shadow formalism.
    n : torch.Tensor
        Shadow charge (or occupancy) vector per atom, shape (Nats,). Used in
        place of ``q`` in the shadow Coulomb-overlap term.
    Rx, Ry, Rz : torch.Tensor
        Cartesian coordinates of atoms along x, y, z respectively, each of
        shape (Nats,).
    Nats : int
        Number of atoms in the system.
    const : object
        Container with model constants. Must provide `n_orb`, the number of
        orbitals per element type.
    TYPE : torch.Tensor
        Element type indices for each atom, shape (Nats,). Used to map into
        `const.n_orb`.

    Returns
    -------
    Ftot : torch.Tensor
        Total forces on atoms, shape (3, Nats). Convention: forces are
        negative gradients of the total energy.
    Fcoul : torch.Tensor
        PME shadow Coulomb interaction contribution to the forces,
        shape (3, Nats). In this implementation it is set to zero and should
        be added externally if available.
    Fband0 : torch.Tensor
        Band-structure (Hamiltonian) contribution to the forces, shape
        (3, Nats).
    Fdipole : torch.Tensor
        Direct electric-field (dipole) contribution to the forces, shape
        (3, Nats).
    FPulay : torch.Tensor
        Pulay correction forces due to non-orthogonal orbitals, shape
        (3, Nats).
    FScoul : torch.Tensor
        Overlap-derivative contribution associated with the shadow Coulomb
        energy, using ``n`` and the PME Coulomb potential ``C``, shape
        (3, Nats).
    FSdipole : torch.Tensor
        Overlap-derivative contribution associated with the dipole / external
        field term, shape (3, Nats).
    Frep : torch.Tensor
        Short-range repulsive potential contribution to the forces, shape
        (3, Nats).

    Notes
    -----
    The total force is assembled as::

        Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    As in :func:`forces_shadow`, setting ``n = q`` recovers the standard SCC
    decomposition, now in a PME / periodic setting.
    """
    # Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T

    dtype = H.dtype
    device = H.device

    n_orbitals_per_atom = const.n_orb[TYPE]
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device),
        n_orbitals_per_atom,
    )

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    Fcoul = torch.zeros((3, Nats), dtype=dtype, device=device)

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = C
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    factor = (U * n + CoulPot) * 2
    dS_times_D = D * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Pulay forces
    SIHD = 4 * Z @ Z.T @ H @ D
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    FPulay.scatter_add_(
        1, atom_ids.expand(3, -1), TMP
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    D_diff = D - D0
    tmp1 = D_diff @ dS
    tmp2 = -2 * (tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    n_orb = dS.shape[1]
    a = dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
    outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
    new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
    FSdipole -= 2 * new_fs

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep
