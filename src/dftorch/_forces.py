import torch

from ._tools import _maybe_compile


# @torch.compile  # Disabled: stale inductor cache produces incorrect Ftot
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
    dU_dq: torch.Tensor = None,
    verbose: bool = False,
    solvation_shift: torch.Tensor = None,
    thirdorder_shift: torch.Tensor = None,
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
    dU_dq : torch.Tensor, optional
        Derivative of Hubbard U with respect to charge, shape (Nats,). If provided, it is used to compute the DFTB3 correction terms in both the forces and the energy. If None, the DFTB3 correction is not applied.
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
    # Fcoul = q * (q @ dC)

    # ── DFTB3: add Fcoul3 = dU/dq * q^2 * (q @ dC) term ─────────────────
    # if dU_dq is not None:
    #     Fcoul = (q + dU_dq * q**2) * (q @ dC)  # equivalent, more efficient
    # else:
    Fcoul = q * (q @ dC)
    # ─────────────────────────────────────────────────────────────────────

    # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
    # FScoul
    CoulPot = C @ q
    if solvation_shift is not None:
        CoulPot = CoulPot + solvation_shift
    if thirdorder_shift is not None:
        CoulPot = CoulPot + thirdorder_shift
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)

    # ── DFTB3: modify factor with extra dU/dq * q^2 term ─────────────────
    # When thirdorder_shift is provided, it already includes the full
    # off-diagonal third-order contribution and dU_dq should be None.
    if dU_dq is not None:
        factor = (U * q + CoulPot + 0.5 * dU_dq * q**2) * 2
    else:
        factor = (U * q + CoulPot) * 2
    # ─────────────────────────────────────────────────────────────────────

    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum)
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    # O(n²): diag(dH @ D) = (dH * D.T).sum(-1)
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH * D_tot.T).sum(dim=-1)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Pulay forces (Here H includes H_spin)
    if D.dim() == 3:  # os
        HD_sum = H[0] @ D[0] + H[1] @ D[1]
        SIHD_sum = 2 * Z @ (Z.T @ HD_sum)
    else:  # cs
        SIHD_sum = 4 * Z @ (Z.T @ (H @ D))
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS * SIHD_sum.T).sum(dim=-1)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole — skip entirely when no external field
    D0 = torch.diag(D0)
    Efield_norm = Efield[0] * Efield[0] + Efield[1] * Efield[1] + Efield[2] * Efield[2]
    if Efield_norm > 1e-30:
        dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
        D_diff = D_tot - D0
        tmp2 = -2 * (D_diff.unsqueeze(0) * dS.transpose(1, 2)).sum(dim=-1)
        FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
        FSdipole *= dotRE

        n_orb = dS.shape[1]
        a = dS * D_diff.permute(1, 0).unsqueeze(0)
        outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
        outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
        new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
        FSdipole -= 2 * new_fs
    else:
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


# @torch.compile  # Disabled: stale inductor cache produces incorrect results
def forces_spin(
    D: torch.Tensor,
    dS: torch.Tensor,
    q_spin_atom: torch.Tensor,
    Nats: int,
    const,
    TYPE: torch.Tensor,
    net_spin_sr: torch.Tensor = None,
    n_shells_per_atom: torch.Tensor = None,
    shell_types: torch.Tensor = None,
):
    """Compute spin contribution to atomic forces.

    When ``const.w`` is a 1-D tensor (scalar spin constant per element),
    the original atom-resolved formula is used.  When ``const.w`` is a
    3-D tensor (shell-resolved W matrix, i.e. ``magnetic_hubbard_ldep=True``),
    the per-orbital spin potential mu_orb is built from *net_spin_sr* and W,
    and the force is evaluated as

        F_A = -1/2 * sum_{mu,nu} DeltaD_{mu,nu} * dS_{mu,nu}/dR_A * (mu_mu + mu_nu)
    """
    dtype = q_spin_atom.dtype
    device = q_spin_atom.device

    n_orbitals_per_atom = const.n_orb[
        TYPE
    ]  # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    # tmp = (D_up - D_down) * dS  — shape (3, Norb, Norb)
    tmp = (
        (torch.tensor([[[1]], [[-1]]], device=device) * D).unsqueeze(1)
        * dS.unsqueeze(0)
    ).sum(0)

    w = const.w[TYPE]

    if w.dim() == 1:
        # --- scalar W per atom (original code path) ---
        net_spin = q_spin_atom[0] - q_spin_atom[1]
        factor = (net_spin * w) * 2  # (Nats,)
        dS_times_D = tmp * factor[atom_ids].unsqueeze(-1)

        FSspinA = torch.zeros((3, Nats), dtype=dtype, device=device)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)
        FSspinA.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum)
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FSspinA.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)
        return FSspinA / 2

    # --- shell-resolved W matrix (magnetic_hubbard_ldep = True) ---
    # Build per-orbital spin potential mu_orb, mirroring get_h_spin logic.
    n_orb_per_shell = torch.tensor([0, 1, 3, 5], device=device)
    n_orb_per_shell_global = n_orb_per_shell[shell_types]
    mu_sr = torch.zeros(
        n_shells_per_atom.sum(), dtype=dtype, device=device
    )  # per-shell mu

    # s atoms (1 shell)
    mask1 = n_shells_per_atom == 1
    if mask1.any():
        w_tmp = w[mask1][:, 0:1, 0:1]
        mask_sh = mask1.repeat_interleave(n_shells_per_atom)
        q_tmp = net_spin_sr[mask_sh].view(-1, 1)
        mu_sr[mask_sh] = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()

    # sp atoms (2 shells)
    mask2 = n_shells_per_atom == 2
    if mask2.any():
        w_tmp = w[mask2][:, 0:2, 0:2]
        mask_sh = mask2.repeat_interleave(n_shells_per_atom)
        q_tmp = net_spin_sr[mask_sh].view(-1, 2)
        mu_sr[mask_sh] = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()

    # spd atoms (3 shells)
    mask3 = n_shells_per_atom == 3
    if mask3.any():
        w_tmp = w[mask3][:, 0:3, 0:3]
        mask_sh = mask3.repeat_interleave(n_shells_per_atom)
        q_tmp = net_spin_sr[mask_sh].view(-1, 3)
        mu_sr[mask_sh] = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()

    # Expand shell mu to orbital mu
    mu_orb = mu_sr.repeat_interleave(n_orb_per_shell_global)  # (Norb,)

    # F_A = -1/2 * sum_{mu in A, nu} DeltaD * dS * (mu_mu + mu_nu)
    #      + 1/2 * sum_{nu in A, mu} DeltaD * dS * (mu_mu + mu_nu)
    # Combined factor matrix: (mu_mu + mu_nu)
    mu_row = mu_orb.unsqueeze(-1)  # (Norb, 1)
    mu_col = mu_orb.unsqueeze(-2)  # (1, Norb)
    combined = tmp * (mu_row + mu_col)  # (3, Norb, Norb)

    FSspinA = torch.zeros((3, Nats), dtype=dtype, device=device)
    dDS_XYZ_row_sum = torch.sum(combined, dim=2)
    FSspinA.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum)
    dDS_XYZ_col_sum = torch.sum(combined, dim=1)
    FSspinA.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)
    return FSspinA / 2


# @torch.compile  # Disabled: stale inductor cache produces incorrect results
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
    dU_dq: torch.Tensor = None,
    verbose: bool = False,
    solvation_shift: torch.Tensor = None,
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
    dU_dq : torch.Tensor, optional
        Derivative of Hubbard U with respect to charge, shape (Nats,). If provided, it is used to compute the DFTB3 correction terms in both the forces and the energy. If None, the DFTB3 correction is not applied.
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
    if solvation_shift is not None:
        CoulPot = CoulPot + solvation_shift
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    # ── DFTB3: modify factor with extra dU/dq * q^2 term ─────────────────
    if dU_dq is not None:
        factor = (U * q + CoulPot + 0.5 * dU_dq * q**2) * 2
    else:
        factor = (U * q + CoulPot) * 2
    # ─────────────────────────────────────────────────────────────────────

    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum)
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    # O(n²): diag(dH @ D) = (dH * D.T).sum(-1)
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH * D_tot.T).sum(dim=-1)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Pulay forces
    if D.dim() == 3:  # os
        HD_sum = H[0] @ D[0] + H[1] @ D[1]
        SIHD_sum = 2 * Z @ (Z.T @ HD_sum)
    else:  # cs
        SIHD_sum = 4 * Z @ (Z.T @ (H @ D))
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS * SIHD_sum.T).sum(dim=-1)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole — skip entirely when no external field
    D0 = torch.diag(D0)
    Efield_norm = Efield[0] * Efield[0] + Efield[1] * Efield[1] + Efield[2] * Efield[2]
    if Efield_norm > 1e-30:
        dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
        D_diff = D_tot - D0
        tmp2 = -2 * (D_diff.unsqueeze(0) * dS.transpose(1, 2)).sum(dim=-1)
        FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
        FSdipole *= dotRE

        n_orb = dS.shape[1]
        a = dS * D_diff.permute(1, 0).unsqueeze(0)
        outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
        outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
        new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
        FSdipole -= 2 * new_fs
    else:
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


# @torch.compile  # Disabled: stale inductor cache produces incorrect results
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
    dU_dq: torch.Tensor = None,
    verbose: bool = False,
    solvation_shift: torch.Tensor = None,
    thirdorder_shift: torch.Tensor = None,
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
    dU_dq : torch.Tensor, optional
        Derivative of Hubbard U with respect to charge, shape (Nats,). If provided, it is used to compute the DFTB3 correction terms in both the forces and the energy.
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
    if solvation_shift is not None:
        CoulPot = CoulPot + solvation_shift
    if thirdorder_shift is not None:
        CoulPot = CoulPot + thirdorder_shift
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    # ── DFTB3: modify factor with extra dU/dq * q^2 term ─────────────────
    # When full off-diagonal thirdorder is active, its shift is already in
    # CoulPot via thirdorder_shift, so the diagonal dU_dq term is suppressed.
    if dU_dq is not None and thirdorder_shift is None:
        factor = (U * n + CoulPot + 0.5 * dU_dq * (2.0 * q - n) * n) * 2
    else:
        factor = (U * n + CoulPot) * 2
    # ─────────────────────────────────────────────────────────────────────

    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(
        1, atom_ids.expand(3, -1), dDS_XYZ_row_sum
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    # O(n²): diag(dH @ D) = (dH * D.T).sum(-1), avoids 3 full O(n³) matmuls
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH * D_tot.T).sum(dim=-1)  # (3, n_orb)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Pulay forces (Here H includes H_spin)
    # For OS: sum spin channels first, then one chain of matmuls
    if D.dim() == 3:  # os
        HD_sum = H[0] @ D[0] + H[1] @ D[1]  # (n_orb, n_orb), 2 matmuls
        SIHD_sum = 2 * Z @ (Z.T @ HD_sum)  # 2 matmuls (instead of 6)
    else:  # cs
        SIHD_sum = 4 * Z @ (Z.T @ (H @ D))  # 3 matmuls
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    # O(n²): diag(dS @ M) = (dS * M.T).sum(-1)
    TMP = -(dS * SIHD_sum.T).sum(dim=-1)  # (3, n_orb)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole — skip entirely when no external field
    D0 = torch.diag(D0)
    Efield_norm = Efield[0] * Efield[0] + Efield[1] * Efield[1] + Efield[2] * Efield[2]
    if Efield_norm > 1e-30:
        dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
        D_diff = D_tot - D0
        # O(n²): diag(D_diff @ dS) = (D_diff * dS.transpose(1,2)).sum(-1)
        tmp2 = -2 * (D_diff.unsqueeze(0) * dS.transpose(1, 2)).sum(dim=-1)  # (3, n_orb)
        FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
        FSdipole *= dotRE

        n_orb = dS.shape[1]
        a = dS * D_diff.permute(1, 0).unsqueeze(0)  # 3, n_ham, n_ham
        outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
        outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
        new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
        FSdipole -= 2 * new_fs
    else:
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


# @torch.compile  # Disabled: stale inductor cache produces incorrect results
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
    dU_dq: torch.Tensor = None,
    solvation_shift: torch.Tensor = None,
    thirdorder_shift: torch.Tensor = None,
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
    dU_dq : torch.Tensor, optional
        Derivative of Hubbard U with respect to charge, shape (Nats,). If provided, it is used to compute the DFTB3 correction terms in both the forces and the energy. If None, the DFTB3 correction is not applied.

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

    if D.dim() == 3:  # os
        D_tot = D.sum(0) / 2
    else:  # cs
        D_tot = D

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
    if solvation_shift is not None:
        CoulPot = CoulPot + solvation_shift
    if thirdorder_shift is not None:
        CoulPot = CoulPot + thirdorder_shift
    FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
    # ── DFTB3 shadow PME ──────────────────────────────────────────────────
    # When full off-diagonal thirdorder is active, its shift is already in
    # CoulPot via thirdorder_shift, so the diagonal dU_dq term is suppressed.
    if dU_dq is not None and thirdorder_shift is None:
        factor = (U * n + CoulPot + 0.5 * dU_dq * (2.0 * q - n) * n) * 2
    else:
        factor = (U * n + CoulPot) * 2
    # ─────────────────────────────────────────────────────────────────────

    dS_times_D = D_tot * dS * factor[atom_ids].unsqueeze(-1)
    dDS_XYZ_row_sum = torch.sum(dS_times_D, dim=2)  # sum of elements in each row
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum)
    dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
    FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    # O(n²): diag(dH @ D) = (dH * D.T).sum(-1)
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4 * (dH * D_tot.T).sum(dim=-1)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Pulay forces
    if D.dim() == 3:  # os
        HD_sum = H[0] @ D[0] + H[1] @ D[1]
        SIHD_sum = 2 * Z @ (Z.T @ HD_sum)
    else:  # cs
        SIHD_sum = 4 * Z @ (Z.T @ (H @ D))
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS * SIHD_sum.T).sum(dim=-1)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP)

    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)

    # FSdipole — skip entirely when no external field
    D0 = torch.diag(D0)
    Efield_norm = Efield[0] * Efield[0] + Efield[1] * Efield[1] + Efield[2] * Efield[2]
    if Efield_norm > 1e-30:
        dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
        D_diff = D_tot - D0
        tmp2 = -2 * (D_diff.unsqueeze(0) * dS.transpose(1, 2)).sum(dim=-1)
        FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
        FSdipole *= dotRE

        n_orb = dS.shape[1]
        a = dS * D_diff.permute(1, 0).unsqueeze(0)
        outs_by_atom = torch.zeros((3, n_orb, Nats), dtype=a.dtype, device=a.device)
        outs_by_atom = outs_by_atom.index_add(2, atom_ids, a)
        new_fs = outs_by_atom.permute(0, 2, 1) @ dotRE[atom_ids]
        FSdipole -= 2 * new_fs
    else:
        FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)

    Frep = dVr.sum(dim=2)

    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep


Forces_eager = Forces
forces_spin_eager = forces_spin
Forces_PME_eager = Forces_PME
forces_shadow_eager = forces_shadow
forces_shadow_pme_eager = forces_shadow_pme

Forces = _maybe_compile(
    Forces,
    fullgraph=False,
    dynamic=False,
)
forces_spin = _maybe_compile(
    forces_spin,
    fullgraph=False,
    dynamic=False,
)
Forces_PME = _maybe_compile(
    Forces_PME,
    fullgraph=False,
    dynamic=False,
)
forces_shadow = _maybe_compile(
    forces_shadow,
    fullgraph=False,
    dynamic=False,
)
forces_shadow_pme = _maybe_compile(
    forces_shadow_pme,
    fullgraph=False,
    dynamic=False,
)
