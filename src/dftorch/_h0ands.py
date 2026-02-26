import torch
import time
from ._slater_koster_pair import (
    Slater_Koster_Pair_SKF_vectorized,
    Slater_Koster_Pair_SKF_vectorized_batch,
)


# @torch.compile
def H0_and_S_vectorized(
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    diagonal: torch.Tensor,
    H_INDEX_START: torch.Tensor,
    nnRx: torch.Tensor,
    nnRy: torch.Tensor,
    nnRz: torch.Tensor,
    nnType: torch.Tensor,
    const,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    IJ_pair_type: torch.Tensor,
    JI_pair_type: torch.Tensor,
    R_orb: torch.Tensor,
    coeffs_tensor: torch.Tensor,
    verbose: bool = False,
):
    """
    Build one-electron Hamiltonian H0, overlap S, their Cartesian derivatives,
    and an initial atomic density matrix, using vectorized Slater–Koster
    interpolation from tabulated data.

    Parameters
    ----------
    TYPE : torch.Tensor
        Integer element/type indices for each atom, shape (Nats,). Used to
        look up the number of orbitals per atom via ``const.n_orb`` and for
        Slater–Koster parameter selection.
    RX, RY, RZ : torch.Tensor
        Cartesian coordinates of atoms along x, y, z, each of shape (Nats,).
    diagonal : torch.Tensor
        On‑site orbital energies laid out in AO order, shape (HDIM,). These
        are added to the diagonal of the Hamiltonian after assembling the
        off‑diagonal couplings.
    H_INDEX_START : torch.Tensor
        For each atom a, index of its first AO in the global Hamiltonian,
        shape (Nats,).
    nnRx, nnRy, nnRz : torch.Tensor
        Neighbor coordinates for each atom in the neighbor list, shape
        (Nats, Nmax_neigh). These are typically pre‑wrapped or given in the
        same coordinate frame as RX/RY/RZ.
    nnType : torch.Tensor
        Neighbor element/type indices, shape (Nats, Nmax_neigh). Entries
        equal to ``-1`` denote padded / non‑existent neighbors and are
        excluded via a mask.
    const : object
        Container for model constants and lookup tables. Must at least
        provide ``n_orb``, the number of orbitals for each element type.
    neighbor_I : torch.Tensor
        Flattened list of “central” atom indices for each neighbor pair,
        shape (Npairs,). Points into the atom index range [0, Nats).
    neighbor_J : torch.Tensor
        Flattened list of neighbor atom indices for each pair, shape
        (Npairs,). Used together with ``neighbor_I`` to index TYPE and
        coordinates.
    IJ_pair_type, JI_pair_type : torch.Tensor
        Encoded pair/type indices used to select the proper block in
        ``coeffs_tensor`` for a given atom pair and direction, shape
        (Npairs,).
    R_orb : torch.Tensor
        1D grid of radii at which Slater–Koster coefficients are tabulated,
        shape (Nr_grid,). Used for searchsorted interpolation.
    coeffs_tensor : torch.Tensor
        Tabulated Slater–Koster coefficients on the ``R_orb`` grid, with a
        layout consistent with :func:`Slater_Koster_Pair_SKF_vectorized`.
        This tensor is shared between H and S constructions; the last
        argument to the SK routine selects which block to use.
    verbose : bool, optional
        If True, print timing/debug information to stdout.

    Returns
    -------
    D0 : torch.Tensor
        Initial atomic density matrix in the AO basis, shape (HDIM, HDIM).
        Constructed from ``Znuc`` and the atom‑orbital mapping, and scaled
        by 1/2.
    H0 : torch.Tensor
        One‑electron Hamiltonian in the AO basis, including on‑site
        ``diagonal`` contribution, shape (HDIM, HDIM).
    dH0 : torch.Tensor
        Cartesian derivatives of H0 w.r.t. nuclear coordinates, shape
        (3, HDIM, HDIM). The first axis corresponds to x, y, z.
    S : torch.Tensor
        Overlap matrix in the AO basis, shape (HDIM, HDIM). Built from
        Slater–Koster integrals, scaled by 1/27.21138625 and with the AO
        identity added on the diagonal.
    dS : torch.Tensor
        Cartesian derivatives of S, shape (3, HDIM, HDIM), scaled by the
        same factor as S.

    Notes
    -----
    The global AO dimension is

        HDIM = len(diagonal),

    and must be consistent with the orbital counts implied by ``TYPE`` and
    ``const.n_orb``. Neighbor lists are assumed to be pre‑built; only entries
    with ``nnType != -1`` participate in the Slater–Koster sums. The same
    vectorized SK routine is used for both H and S; a selector flag in the
    call controls which coefficient block is used.
    """
    # Map atom type to properties
    # Support both str and int input
    if verbose:
        print("H0_and_S")
    start_time1 = time.perf_counter()
    start_time3 = time.perf_counter()

    if verbose:
        print("  Do H off-diag")
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)

    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)

    L = Rab_X / dR
    L_dx = (Rab_Y**2 + Rab_Z**2) / (dR**3)
    L_dy = -Rab_X * Rab_Y / (dR**3)
    L_dz = -Rab_X * Rab_Z / (dR**3)

    M = Rab_Y / dR
    M_dx = -Rab_Y * Rab_X / (dR**3)
    M_dy = (Rab_X**2 + Rab_Z**2) / (dR**3)
    M_dz = -Rab_Y * Rab_Z / (dR**3)

    N = Rab_Z / dR
    N_dx = -Rab_Z * Rab_X / (dR**3)
    N_dy = -Rab_Z * Rab_Y / (dR**3)
    N_dz = (Rab_X**2 + Rab_Y**2) / (dR**3)

    # HDIM = sum(non_hydro_mask)*4 + sum(hydro_mask)
    HDIM = len(diagonal)
    pair_mask_HH = (const.n_orb[TYPE[neighbor_I]] == 1) & (
        const.n_orb[TYPE[neighbor_J]] == 1
    )
    pair_mask_HX = (const.n_orb[TYPE[neighbor_I]] == 1) & (
        const.n_orb[TYPE[neighbor_J]] == 4
    )
    pair_mask_XH = (const.n_orb[TYPE[neighbor_I]] == 4) & (
        const.n_orb[TYPE[neighbor_J]] == 1
    )
    pair_mask_XX = (const.n_orb[TYPE[neighbor_I]] == 4) & (
        const.n_orb[TYPE[neighbor_J]] == 4
    )

    pair_mask_HY = (const.n_orb[TYPE[neighbor_I]] == 1) & (
        const.n_orb[TYPE[neighbor_J]] == 9
    )
    pair_mask_XY = (const.n_orb[TYPE[neighbor_I]] == 4) & (
        const.n_orb[TYPE[neighbor_J]] == 9
    )
    pair_mask_YH = (const.n_orb[TYPE[neighbor_I]] == 9) & (
        const.n_orb[TYPE[neighbor_J]] == 1
    )
    pair_mask_YX = (const.n_orb[TYPE[neighbor_I]] == 9) & (
        const.n_orb[TYPE[neighbor_J]] == 4
    )
    pair_mask_YY = (const.n_orb[TYPE[neighbor_I]] == 9) & (
        const.n_orb[TYPE[neighbor_J]] == 9
    )

    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    L_mskd = L[nn_mask]
    M_mskd = M[nn_mask]
    N_mskd = N[nn_mask]

    L_dxyz = torch.stack((L_dx, L_dy, L_dz), dim=0)[:, nn_mask]
    M_dxyz = torch.stack((M_dx, M_dy, M_dz), dim=0)[:, nn_mask]
    N_dxyz = torch.stack((N_dx, N_dy, N_dz), dim=0)[:, nn_mask]

    dR_dxyz = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=0)[:, nn_mask] / dR_mskd

    if verbose:
        print(
            "  t <dR and pair mask> {:.1f} s\n".format(
                time.perf_counter() - start_time3
            )
        )
    start_time4 = time.perf_counter()

    idx = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
    idx = torch.clamp(idx, 0, len(R_orb))
    dx = dR_mskd - R_orb[idx]

    if verbose:
        print("  t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))

    if verbose:
        print("  Do H and S")
    H0, dH0 = Slater_Koster_Pair_SKF_vectorized(
        HDIM,
        dR_dxyz,
        L_mskd,
        M_mskd,
        N_mskd,
        L_dxyz,
        M_dxyz,
        N_dxyz,
        pair_mask_HH,
        pair_mask_HX,
        pair_mask_XH,
        pair_mask_XX,
        pair_mask_HY,
        pair_mask_XY,
        pair_mask_YH,
        pair_mask_YX,
        pair_mask_YY,
        dx,
        idx,
        IJ_pair_type,
        JI_pair_type,
        coeffs_tensor,
        neighbor_I,
        neighbor_J,
        H_INDEX_START,
        0,
    )

    H0 = H0.reshape(HDIM, HDIM)
    H0 = H0 + torch.diag(diagonal)
    H0 = (
        H0 + H0.transpose(0, 1)
    ) * 0.5  # Enforce exact symmetry. Some DFTB files give slightly asymmetric data.
    dH0 = dH0.reshape(3, HDIM, HDIM)
    dH0 = (dH0 - dH0.transpose(1, 2)) * 0.5  # Enforce exact symmetry.

    #### S PART ###
    S, dS = Slater_Koster_Pair_SKF_vectorized(
        HDIM,
        dR_dxyz,
        L_mskd,
        M_mskd,
        N_mskd,
        L_dxyz,
        M_dxyz,
        N_dxyz,
        pair_mask_HH,
        pair_mask_HX,
        pair_mask_XH,
        pair_mask_XX,
        pair_mask_HY,
        pair_mask_XY,
        pair_mask_YH,
        pair_mask_YX,
        pair_mask_YY,
        dx,
        idx,
        IJ_pair_type,
        JI_pair_type,
        coeffs_tensor,
        neighbor_I,
        neighbor_J,
        H_INDEX_START,
        1,
    )

    S = S.reshape(HDIM, HDIM) / 27.21138625
    S = S + torch.eye(HDIM, device=S.device)
    S = (
        S + S.transpose(0, 1)
    ) * 0.5  # Enforce exact symmetry. Some DFTB files give slightly asymmetric data.
    dS = dS.reshape(3, HDIM, HDIM) / 27.21138625
    dS = (dS - dS.transpose(1, 2)) * 0.5  # Enforce exact symmetry.

    if verbose:
        print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))
    return H0, dH0, S, dS


def H0_and_S_vectorized_batch(
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    diagonal: torch.Tensor,
    H_INDEX_START: torch.Tensor,
    nnRx: torch.Tensor,
    nnRy: torch.Tensor,
    nnRz: torch.Tensor,
    nnType: torch.Tensor,
    const,
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    IJ_pair_type: torch.Tensor,
    JI_pair_type: torch.Tensor,
    R_orb: torch.Tensor,
    coeffs_tensor: torch.Tensor,
    verbose: bool = False,
):
    """
    Build one-electron Hamiltonian H0, overlap S, their Cartesian derivatives,
    and an initial atomic density matrix, using vectorized Slater–Koster
    interpolation from tabulated data.

    Parameters
    ----------
    H_INDEX_START : torch.Tensor
        For each atom a, index of its first AO in the global Hamiltonian,
        shape (Nats,).
    nnRx, nnRy, nnRz : torch.Tensor
        Neighbor coordinates for each atom in the neighbor list, shape
        (Nats, Nmax_neigh). These are typically pre‑wrapped or given in the
        same coordinate frame as RX/RY/RZ.
    nnType : torch.Tensor
        Neighbor element/type indices, shape (Nats, Nmax_neigh). Entries
        equal to ``-1`` denote padded / non‑existent neighbors and are
        excluded via a mask.
    const : object
        Container for model constants and lookup tables. Must at least
        provide ``n_orb``, the number of orbitals for each element type.
    neighbor_I : torch.Tensor
        Flattened list of “central” atom indices for each neighbor pair,
        shape (Npairs,). Points into the atom index range [0, Nats).
    neighbor_J : torch.Tensor
        Flattened list of neighbor atom indices for each pair, shape
        (Npairs,). Used together with ``neighbor_I`` to index TYPE and
        coordinates.
    IJ_pair_type, JI_pair_type : torch.Tensor
        Encoded pair/type indices used to select the proper block in
        ``coeffs_tensor`` for a given atom pair and direction, shape
        (Npairs,).
    R_orb : torch.Tensor
        1D grid of radii at which Slater–Koster coefficients are tabulated,
        shape (Nr_grid,). Used for searchsorted interpolation.
    coeffs_tensor : torch.Tensor
        Tabulated Slater–Koster coefficients on the ``R_orb`` grid, with a
        layout consistent with :func:`Slater_Koster_Pair_SKF_vectorized`.
        This tensor is shared between H and S constructions; the last
        argument to the SK routine selects which block to use.
    verbose : bool, optional
        If True, print timing/debug information to stdout.

    Returns
    -------
    D0 : torch.Tensor
        Initial atomic density matrix in the AO basis, shape (HDIM, HDIM).
        Constructed from ``Znuc`` and the atom‑orbital mapping, and scaled
        by 1/2.
    H0 : torch.Tensor
        One‑electron Hamiltonian in the AO basis, including on‑site
        ``diagonal`` contribution, shape (HDIM, HDIM).
    dH0 : torch.Tensor
        Cartesian derivatives of H0 w.r.t. nuclear coordinates, shape
        (3, HDIM, HDIM). The first axis corresponds to x, y, z.
    S : torch.Tensor
        Overlap matrix in the AO basis, shape (HDIM, HDIM). Built from
        Slater–Koster integrals, scaled by 1/27.21138625 and with the AO
        identity added on the diagonal.
    dS : torch.Tensor
        Cartesian derivatives of S, shape (3, HDIM, HDIM), scaled by the
        same factor as S.

    Notes
    -----
    The global AO dimension is

        HDIM = len(diagonal),

    and must be consistent with the orbital counts implied by ``TYPE`` and
    ``const.n_orb``. Neighbor lists are assumed to be pre‑built; only entries
    with ``nnType != -1`` participate in the Slater–Koster sums. The same
    vectorized SK routine is used for both H and S; a selector flag in the
    call controls which coefficient block is used.
    """
    # Map atom type to properties
    # Support both str and int input
    if verbose:
        print("H0_and_S")
    start_time1 = time.perf_counter()
    start_time3 = time.perf_counter()

    batch_size = RX.shape[0]

    if verbose:
        print("  Do H off-diag")
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)

    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)

    L = Rab_X / dR
    L_dx = (Rab_Y**2 + Rab_Z**2) / (dR**3)
    L_dy = -Rab_X * Rab_Y / (dR**3)
    L_dz = -Rab_X * Rab_Z / (dR**3)

    M = Rab_Y / dR
    M_dx = -Rab_Y * Rab_X / (dR**3)
    M_dy = (Rab_X**2 + Rab_Z**2) / (dR**3)
    M_dz = -Rab_Y * Rab_Z / (dR**3)

    N = Rab_Z / dR
    N_dx = -Rab_Z * Rab_X / (dR**3)
    N_dy = -Rab_Z * Rab_Y / (dR**3)
    N_dz = (Rab_X**2 + Rab_Y**2) / (dR**3)

    # HDIM = sum(non_hydro_mask)*4 + sum(hydro_mask)
    HDIM = diagonal.shape[-1]
    # neighbor_I, neighbor_J: (B, Npairs) with -1 padding
    valid_pairs = (neighbor_I >= 0) & (
        neighbor_J >= 0
    )  # $$$ maybe '& (neighbor_J >= 0)' is not necessary???
    safe_I = neighbor_I.clamp(min=0)
    safe_J = neighbor_J.clamp(min=0)

    # Element types per atom: (B, Nats). Gather types for each pair safely.
    type_I = TYPE.gather(1, safe_I)  # (B, Npairs)
    type_J = TYPE.gather(1, safe_J)  # (B, Npairs)
    # Map to number of orbitals per atom; shapes match (B, Npairs)
    norb_I = const.n_orb[type_I]
    norb_J = const.n_orb[type_J]

    # Pair masks (invalid pairs stay False)
    pair_mask_HH = valid_pairs & (norb_I == 1) & (norb_J == 1)
    pair_mask_HX = valid_pairs & (norb_I == 1) & (norb_J == 4)
    pair_mask_XH = valid_pairs & (norb_I == 4) & (norb_J == 1)
    pair_mask_XX = valid_pairs & (norb_I == 4) & (norb_J == 4)
    pair_mask_HY = valid_pairs & (norb_I == 1) & (norb_J == 9)
    pair_mask_XY = valid_pairs & (norb_I == 4) & (norb_J == 9)
    pair_mask_YH = valid_pairs & (norb_I == 9) & (norb_J == 1)
    pair_mask_YX = valid_pairs & (norb_I == 9) & (norb_J == 4)
    pair_mask_YY = valid_pairs & (norb_I == 9) & (norb_J == 9)

    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]
    L_mskd = L[nn_mask]
    M_mskd = M[nn_mask]
    N_mskd = N[nn_mask]

    L_dxyz = torch.stack((L_dx, L_dy, L_dz), dim=0)[:, nn_mask]
    M_dxyz = torch.stack((M_dx, M_dy, M_dz), dim=0)[:, nn_mask]
    N_dxyz = torch.stack((N_dx, N_dy, N_dz), dim=0)[:, nn_mask]

    dR_dxyz = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=0)[:, nn_mask] / dR_mskd

    if verbose:
        print(
            "  t <dR and pair mask> {:.1f} s\n".format(
                time.perf_counter() - start_time3
            )
        )
    start_time4 = time.perf_counter()

    idx = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
    idx = torch.clamp(idx, 0, len(R_orb))
    dx = dR_mskd - R_orb[idx]

    if verbose:
        print("  t <SKF> {:.1f} s\n".format(time.perf_counter() - start_time4))

    if verbose:
        print("  Do H and S")
    H0, dH0 = Slater_Koster_Pair_SKF_vectorized_batch(
        batch_size,
        HDIM,
        dR_dxyz,
        L_mskd,
        M_mskd,
        N_mskd,
        L_dxyz,
        M_dxyz,
        N_dxyz,
        pair_mask_HH,
        pair_mask_HX,
        pair_mask_XH,
        pair_mask_XX,
        pair_mask_HY,
        pair_mask_XY,
        pair_mask_YH,
        pair_mask_YX,
        pair_mask_YY,
        dx,
        idx,
        IJ_pair_type,
        JI_pair_type,
        coeffs_tensor,
        neighbor_I,
        neighbor_J,
        safe_I,
        safe_J,
        valid_pairs,
        H_INDEX_START,
        0,
    )

    H0 = H0.reshape(batch_size, HDIM, HDIM)
    H0 = H0 + torch.diag_embed(diagonal)
    H0 = (
        H0 + H0.transpose(1, 2)
    ) * 0.5  # Enforce exact symmetry. Some DFTB files give slightly asymmetric data.
    dH0 = dH0.view(3, batch_size, HDIM, HDIM)
    dH0 = dH0.permute(1, 0, 2, 3).contiguous()
    dH0 = (dH0 - dH0.transpose(2, 3)) * 0.5  # Enforce exact symmetry.

    #### S PART ###
    S, dS = Slater_Koster_Pair_SKF_vectorized_batch(
        batch_size,
        HDIM,
        dR_dxyz,
        L_mskd,
        M_mskd,
        N_mskd,
        L_dxyz,
        M_dxyz,
        N_dxyz,
        pair_mask_HH,
        pair_mask_HX,
        pair_mask_XH,
        pair_mask_XX,
        pair_mask_HY,
        pair_mask_XY,
        pair_mask_YH,
        pair_mask_YX,
        pair_mask_YY,
        dx,
        idx,
        IJ_pair_type,
        JI_pair_type,
        coeffs_tensor,
        neighbor_I,
        neighbor_J,
        safe_I,
        safe_J,
        valid_pairs,
        H_INDEX_START,
        1,
    )

    S = S.reshape(batch_size, HDIM, HDIM) / 27.21138625
    S = S + torch.eye(HDIM, device=S.device)
    S = (
        S + S.transpose(1, 2)
    ) * 0.5  # Enforce exact symmetry. Some DFTB files give slightly asymmetric data.
    dS = dS.view(3, batch_size, HDIM, HDIM) / 27.21138625
    dS = dS.permute(1, 0, 2, 3).contiguous()
    dS = (dS - dS.transpose(2, 3)) * 0.5  # Enforce exact symmetry.

    if verbose:
        print("H0_and_S t {:.1f} s\n".format(time.perf_counter() - start_time1))
    return H0, dH0, S, dS
