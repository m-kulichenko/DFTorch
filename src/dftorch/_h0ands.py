import time

import torch

from ._slater_koster_pair import (
    Slater_Koster_Pair_SKF_vectorized,
    Slater_Koster_Pair_SKF_vectorized_batch,
)


def _pair_knot_lookup( #Different SKFs can have different R's so we need to keep it consistent in lookup
    R_tensor: torch.Tensor,
    n_grid: torch.Tensor,
    pair_type: torch.Tensor,
    dR: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each distance's spline knot against ITS OWN pair's radial grid.

    ``get_skf_tensors`` stores every element pair's radial grid in its own row
    of ``R_tensor``, but historically exported a single global ``R_orb`` chosen
    as the longest grid seen, and every pair type was interpolated against that
    one array. A pair whose real grid differed was therefore evaluated against
    another pair's ruler. This helper closes that (decision D-01, requirement
    REG-06): entry ``k`` of ``dR`` is looked up in row ``pair_type[k]``.

    The arithmetic per row is IDENTICAL to the global expression it replaces:
    ``searchsorted(..., right=True) - 1``, clamped below at 0 and above at the
    row's own grid length, then ``dx = dR - R_row[idx]``. Reproducing it exactly
    is the point; for a directory whose grids all share one step every returned
    ``idx`` and ``dx`` is bit-identical to the old global result.

    A closed-form ``floor(dR / step) - 1`` would be cheaper and is arithmetically
    equal for a uniform grid, but it disagrees with ``searchsorted`` by one when
    a distance lands exactly on a knot, so it is deliberately NOT used.

    LOOP, DO NOT BATCH. ``torch.searchsorted`` accepts a 2-D boundary argument,
    which would let this be written as a single call over
    ``R_tensor[pair_type]``. That form materialises a ``(len(dR), 1301)`` tensor,
    which for a real neighbour list is gigabytes of allocation for no benefit.
    The number of DISTINCT pair types in a system is small (nine for a
    three-element system), so looping over ``torch.unique(pair_type)`` and
    running a 1-D search on each masked subset is both cheap and bounded.

    Parameters
    ----------
    R_tensor : torch.Tensor
        Per-pair radial grids, shape ``(n_pairs, n_columns)``. Every row must be
        strictly increasing across its full width; ``get_skf_tensors`` guarantees
        that by continuing each pair's arithmetic progression past its tabulated
        end. ``torch.searchsorted`` against an unsorted row is undefined.
    n_grid : torch.Tensor
        Per-pair tabulated grid length, integer tensor of shape ``(n_pairs,)``.
        Used as the upper clamp so that a distance beyond a pair's own cutoff
        lands on that pair's trailing all-zero spline interval rather than on a
        coefficient borrowed from a longer grid.
    pair_type : torch.Tensor
        Row selector for each entry of ``dR``, shape ``(N,)``.
    dR : torch.Tensor
        Pair separations in Angstrom, shape ``(N,)``. Same units as the grid:
        the SKF file declares its step in Bohr and ``read_skf_table`` multiplies
        it by ``BOHR_TO_ANGSTROM`` when building the grid, so nothing here is
        mixed-unit. See the "Units" note in
        :func:`dftorch._ml_sk.build_pair_type_rcut`, which records the
        measurement that settled this and the superseded claim it replaced.

    Returns
    -------
    idx : torch.Tensor
        Interval index into ``coeffs_tensor``, shape ``(N,)``, dtype int64.
    dx : torch.Tensor
        Offset of ``dR`` from the selected knot, shape ``(N,)``.
    """
    idx = torch.zeros(dR.shape, dtype=torch.int64, device=dR.device)
    dx = torch.zeros_like(dR)

    for pt in torch.unique(pair_type):
        mask = pair_type == pt
        row = R_tensor[pt]
        sub = dR[mask]
        sub_idx = torch.searchsorted(row.contiguous(), sub.contiguous(), right=True) - 1
        sub_idx = torch.clamp(sub_idx, 0, int(n_grid[pt]))
        idx[mask] = sub_idx
        dx[mask] = sub - row[sub_idx]

    return idx, dx


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
    store_stress_metadata: object = None,
    ml_model_data: object = None,
    R_tensor: torch.Tensor = None,
    n_grid: torch.Tensor = None,
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
        shape (Nr_grid,). This is the LONGEST grid seen across all element
        pairs, retained for backward compatibility: ``_stress``, ``_ml_sk``,
        the SEDACS interface and the batched H0/S path all still read it.

        The knot lookup no longer uses it when ``R_tensor`` and ``n_grid`` are
        supplied. It then uses ``R_tensor`` per pair, so a pair whose real grid
        differs is no longer evaluated against another pair's ruler (decision
        D-01, requirement REG-06). ``R_orb`` remains the fallback boundary array
        when either new argument is omitted.
    coeffs_tensor : torch.Tensor
        Tabulated Slater–Koster coefficients on the ``R_orb`` grid, with a
        layout consistent with :func:`Slater_Koster_Pair_SKF_vectorized`.
        This tensor is shared between H and S constructions; the last
        argument to the SK routine selects which block to use.
    verbose : bool, optional
        If True, print timing/debug information to stdout.
    R_tensor : torch.Tensor, optional
        Per-pair radial grids, shape (n_pairs, n_columns), from
        ``const.R_tensor``. Supplying it together with ``n_grid`` selects the
        per-pair knot lookup :func:`_pair_knot_lookup`.
    n_grid : torch.Tensor, optional
        Per-pair tabulated grid length, shape (n_pairs,), from ``const.n_grid``.
        Both default to None so that callers outside this phase's scope, such as
        ``sedacs_interface``, keep working on the global path rather than
        failing on a missing argument.

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

    # 16-orbital (spdf) pair classes. "Z" denotes an atom with n_orb == 16.
    # Before these masks existed, every f-containing neighbor pair fell through
    # all nine 1/4/9 masks and was silently omitted from H0/S off-diagonal
    # assembly. Routing them explicitly means an unimplemented f angular block
    # now fails loudly inside Slater_Koster_Pair_SKF_vectorized instead of
    # producing a finite, symmetric, and wrong matrix.
    pair_mask_HZ = (const.n_orb[TYPE[neighbor_I]] == 1) & (
        const.n_orb[TYPE[neighbor_J]] == 16
    )
    pair_mask_ZH = (const.n_orb[TYPE[neighbor_I]] == 16) & (
        const.n_orb[TYPE[neighbor_J]] == 1
    )
    pair_mask_XZ = (const.n_orb[TYPE[neighbor_I]] == 4) & (
        const.n_orb[TYPE[neighbor_J]] == 16
    )
    pair_mask_ZX = (const.n_orb[TYPE[neighbor_I]] == 16) & (
        const.n_orb[TYPE[neighbor_J]] == 4
    )
    pair_mask_YZ = (const.n_orb[TYPE[neighbor_I]] == 9) & (
        const.n_orb[TYPE[neighbor_J]] == 16
    )
    pair_mask_ZY = (const.n_orb[TYPE[neighbor_I]] == 16) & (
        const.n_orb[TYPE[neighbor_J]] == 9
    )
    pair_mask_ZZ = (const.n_orb[TYPE[neighbor_I]] == 16) & (
        const.n_orb[TYPE[neighbor_J]] == 16
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

    # ── Choose between spline-based (default) and ML-based SK integrals ──
    _ml_ctx = None
    if ml_model_data is not None:
        _ml_ctx = {
            "model": ml_model_data["model"],
            "TYPE": TYPE,
            "neighbor_I": neighbor_I,
            "neighbor_J": neighbor_J,
            "dR_mskd": dR_mskd,
        }
        if verbose:
            print("  Using ML model for SK integrals (lazy per-call)")

    # Spline grid lookup (used in spline mode; in ML mode _get_val_dR ignores these)
    #
    # Per-pair path (decision D-01, requirement REG-06): each distance finds its
    # knot on ITS OWN pair's grid row rather than on the single global R_orb,
    # which was the longest grid seen and therefore the wrong ruler for any pair
    # whose grid differed. Bit-identical to the fallback for a directory whose
    # grids all share one step, which is every parameter set shipped today.
    #
    # The None fallback is deliberate, not a leftover: sedacs_interface calls
    # this function without the new arguments and is out of scope for this
    # phase, so it keeps working on the global path rather than crashing on a
    # missing argument.
    if R_tensor is not None and n_grid is not None:
        idx_use, dx_use = _pair_knot_lookup(R_tensor, n_grid, IJ_pair_type, dR_mskd)
    else:
        idx_use = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
        idx_use = torch.clamp(idx_use, 0, len(R_orb))
        dx_use = dR_mskd - R_orb[idx_use]
    coeffs_tensor_use = coeffs_tensor
    IJ_use = IJ_pair_type
    JI_use = JI_pair_type

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
        dx_use,
        idx_use,
        IJ_use,
        JI_use,
        coeffs_tensor_use,
        neighbor_I,
        neighbor_J,
        H_INDEX_START,
        "H",
        ml_ctx=_ml_ctx,
        pair_mask_HZ=pair_mask_HZ,
        pair_mask_ZH=pair_mask_ZH,
        pair_mask_XZ=pair_mask_XZ,
        pair_mask_ZX=pair_mask_ZX,
        pair_mask_YZ=pair_mask_YZ,
        pair_mask_ZY=pair_mask_ZY,
        pair_mask_ZZ=pair_mask_ZZ,
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
        dx_use,
        idx_use,
        IJ_use,
        JI_use,
        coeffs_tensor_use,
        neighbor_I,
        neighbor_J,
        H_INDEX_START,
        "S",
        ml_ctx=_ml_ctx,
        pair_mask_HZ=pair_mask_HZ,
        pair_mask_ZH=pair_mask_ZH,
        pair_mask_XZ=pair_mask_XZ,
        pair_mask_ZX=pair_mask_ZX,
        pair_mask_YZ=pair_mask_YZ,
        pair_mask_ZY=pair_mask_ZY,
        pair_mask_ZZ=pair_mask_ZZ,
    )

    S = S.reshape(HDIM, HDIM) / 27.21138625
    S = S + torch.eye(HDIM, device=S.device)
    S = (
        S + S.transpose(0, 1)
    ) * 0.5  # Enforce exact symmetry. Some DFTB files give slightly asymmetric data.
    dS = dS.reshape(3, HDIM, HDIM) / 27.21138625
    dS = (dS - dS.transpose(1, 2)) * 0.5  # Enforce exact symmetry.

    # Optionally store pair-level metadata for analytical stress computation.
    # We store the minimum needed: bond vectors, spline indices, pair types,
    # pre-computed AO offsets (i0/j0), and per-pair orbital counts (2 uint8
    # tensors) from which all 9 boolean masks can be reconstructed on the fly.
    if store_stress_metadata is not None:
        Rab_mskd = torch.stack((Rab_X[nn_mask], Rab_Y[nn_mask], Rab_Z[nn_mask]), dim=-1)
        store_stress_metadata._stress_metadata = {
            "Rab_mskd": Rab_mskd,  # (P, 3) bond vectors
            "IJ_pair_type": IJ_use,  # (P,) long
            "JI_pair_type": JI_use,  # (P,) long
            "idx": idx_use,  # (P,) long — spline interval
            "i0": H_INDEX_START[neighbor_I],  # (P,) long — AO offset of atom I
            "j0": H_INDEX_START[neighbor_J],  # (P,) long — AO offset of atom J
            "n_orb_I": const.n_orb[TYPE[neighbor_I]].to(torch.uint8),  # (P,) uint8
            "n_orb_J": const.n_orb[TYPE[neighbor_J]].to(torch.uint8),  # (P,) uint8
        }

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

    # Batch f-orbital routing is deferred to a later phase (Phase 3 decision
    # D-01 scopes f support to the single-system path). The nine masks above
    # cover only n_orb in {1, 4, 9}, so a 16-orbital atom would be dropped from
    # every off-diagonal block without warning. Fail explicitly instead.
    if bool((valid_pairs & ((norb_I == 16) | (norb_J == 16))).any()):
        raise NotImplementedError(
            "H0_and_S_vectorized_batch: batched f-orbital (n_orb == 16) H0/S "
            "assembly is not supported. The f angular formulas are wired into "
            "the single-system path H0_and_S_vectorized only; the batched "
            "Slater-Koster routine still reconstructs the 1/4/9 orbital masks "
            "alone. Refusing to return H0/S with silently omitted f blocks."
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
        "H",
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
        "S",
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
