import torch

@torch.compile
def forces_batch(
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
    #Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T
    
    n_orbitals_per_atom = const.n_orb[TYPE] # Compute number of orbitals per atom. shape: (Nats,)
    
    batch_size = H.shape[0]
    dtype = H.dtype
    device = H.device
    counts = n_orbitals_per_atom          # shape (B, N)
    cum_counts = torch.cumsum(counts, dim=1)        # cumulative sums per batch
    total_orbs = int(cum_counts[0, -1].item())
    r = torch.arange(total_orbs, device=counts.device).expand(counts.size(0), -1)  # (B, total_orbs)
    # For each orbital position r[b,k], find first atom index whose cumulative count exceeds r[b,k]
    atom_ids = (r.unsqueeze(2) < cum_counts.unsqueeze(1)).int().argmax(dim=2)      # (B, total_orbs)

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    t = torch.einsum('bn,bknm->bkm', q, dC)   # = (q @ dC) per batch and component
    Fcoul = q.unsqueeze(1) * t                # elementwise multiply by q -> (B,3,N)


    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = torch.bmm(C, q.unsqueeze(-1)).squeeze(-1)          # (B, N)
    FScoul = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    factor = (U * q + CoulPot) * 2                               # (B, N)
    # Map atom-based factor to orbitals
    orbital_factor = factor.gather(1, atom_ids)                  # (B, n_orb_total)
    # Build (B,3,n_orb,n_orb) tensor: (D * dS) scaled per row (orbital) by orbital_factor
    dS_times_D = (D.unsqueeze(1) * dS) * orbital_factor.unsqueeze(1).unsqueeze(-1)
    # Row sum (sum over columns j)
    dDS_row_sum = dS_times_D.sum(dim=3)                          # (B,3,n_orb_total)
    # Column sum (sum over rows i)
    dDS_col_sum = dS_times_D.sum(dim=2)                          # (B,3,n_orb_total)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                # (B,3,n_orb_total)
    FScoul.scatter_add_(2, idx, dDS_row_sum)
    FScoul.scatter_add_(2, idx, -dDS_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D.unsqueeze(1)).diagonal(offset=0, dim1=-2, dim2=-1)  # (B,3,n_orb)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    Fband0.scatter_add_(2, idx, TMP)

    # Pulay forces
    SIHD = 4 * Z @ Z.transpose(-2, -1) @ H @ D
    FPulay = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    TMP = -(dS @ SIHD.unsqueeze(1)).diagonal(offset=0, dim1=-2, dim2=-1)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    FPulay.scatter_add_(2, idx, TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    
    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(1) * Efield.view(3, 1)
    
    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag_embed(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    tmp1 = (D-D0).unsqueeze(1)@dS
    tmp2 = -2*(tmp1).diagonal(offset=0, dim1=-2, dim2=-1)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    FSdipole.scatter_add_(2, idx, tmp2)
    FSdipole *= dotRE.unsqueeze(-2)

    D_diff = D - D0
    n_orb = dS.shape[-1]
    a = dS*D_diff.permute(0,2,1).unsqueeze(1) # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((batch_size,3,n_orb,Nats),dtype=dtype,device=device)
    idx4 = atom_ids.unsqueeze(1).unsqueeze(1).expand(-1, 3, n_orb, n_orb)  # (B,3,n_orb,Nats)
    outs_by_atom.scatter_add_(3, idx4, a)   
    new_fs = torch.einsum('bjan, bn -> bja', outs_by_atom.permute(0, 1, 3, 2), dotRE.gather(1, atom_ids))  # (B,3,Nats)
    FSdipole += -2.0 * new_fs

    Frep = dVr.sum(dim=-1)
    
    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep

@torch.compile
def forces_shadow_batch(
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
    
    n_orbitals_per_atom = const.n_orb[TYPE] # Compute number of orbitals per atom. shape: (Nats,)
    
    batch_size = H.shape[0]
    dtype = H.dtype
    device = H.device
    counts = n_orbitals_per_atom          # shape (B, N)
    cum_counts = torch.cumsum(counts, dim=1)        # cumulative sums per batch
    total_orbs = int(cum_counts[0, -1].item())
    r = torch.arange(total_orbs, device=counts.device).expand(counts.size(0), -1)  # (B, total_orbs)
    # For each orbital position r[b,k], find first atom index whose cumulative count exceeds r[b,k]
    atom_ids = (r.unsqueeze(2) < cum_counts.unsqueeze(1)).int().argmax(dim=2)      # (B, total_orbs)

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # Fcoul = -q_i * sum_j q_j * dCj/dRi
    t = torch.einsum('bn,bknm->bkm', n, dC)   # = (q @ dC) per batch and component
    Fcoul = (2*q - n).unsqueeze(1) * t                # elementwise multiply by q -> (B,3,N)
    
    
    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = torch.bmm(C, n.unsqueeze(-1)).squeeze(-1)          # (B, N)
    FScoul = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    factor = (U * n + CoulPot) * 2                               # (B, N)
    # Map atom-based factor to orbitals
    orbital_factor = factor.gather(1, atom_ids)                  # (B, n_orb_total)
    # Build (B,3,n_orb,n_orb) tensor: (D * dS) scaled per row (orbital) by orbital_factor
    dS_times_D = (D.unsqueeze(1) * dS) * orbital_factor.unsqueeze(1).unsqueeze(-1)
    # Row sum (sum over columns j)
    dDS_row_sum = dS_times_D.sum(dim=3)                          # (B,3,n_orb_total)
    # Column sum (sum over rows i)
    dDS_col_sum = dS_times_D.sum(dim=2)                          # (B,3,n_orb_total)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                # (B,3,n_orb_total)
    FScoul.scatter_add_(2, idx, dDS_row_sum)
    FScoul.scatter_add_(2, idx, -dDS_col_sum)

    
    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH @ D.unsqueeze(1)).diagonal(offset=0, dim1=-2, dim2=-1)  # (B,3,n_orb)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    Fband0.scatter_add_(2, idx, TMP)

            
    # Pulay forces
    SIHD = 4 * Z @ Z.transpose(-2, -1) @ H @ D
    FPulay = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    TMP = -(dS @ SIHD.unsqueeze(1)).diagonal(offset=0, dim1=-2, dim2=-1)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    FPulay.scatter_add_(2, idx, TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        
    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(1) * Efield.view(3, 1)


    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag_embed(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((batch_size, 3, Nats), dtype=dtype, device=device)
    tmp1 = (D-D0).unsqueeze(1)@dS
    tmp2 = -2*(tmp1).diagonal(offset=0, dim1=-2, dim2=-1)
    idx = atom_ids.unsqueeze(1).expand(-1, 3, -1)                         # (B,3,n_orb)
    FSdipole.scatter_add_(2, idx, tmp2)
    FSdipole *= dotRE.unsqueeze(-2)

    D_diff = D - D0
    n_orb = dS.shape[-1]
    a = dS*D_diff.permute(0,2,1).unsqueeze(1) # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((batch_size,3,n_orb,Nats),dtype=dtype,device=device)
    idx4 = atom_ids.unsqueeze(1).unsqueeze(1).expand(-1, 3, n_orb, n_orb)  # (B,3,n_orb,Nats)
    outs_by_atom.scatter_add_(3, idx4, a)   
    new_fs = torch.einsum('bjan, bn -> bja', outs_by_atom.permute(0, 1, 3, 2), dotRE.gather(1, atom_ids))  # (B,3,Nats)
    FSdipole += -2.0 * new_fs


    Frep = dVr.sum(dim=-1)

    
    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


