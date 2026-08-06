import contextlib
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import torch

from ._dm_fermi_x import (
    dm_fermi_x,
    dm_fermi_x_batch,
    dm_fermi_x_batch_degen,
    dm_fermi_x_os,
    nonaufbau_constraints,
)
from ._spin import get_h_spin

# from ._kernel_fermi import _kernel_fermi
from ._tools import (
    calculate_dist_dips,
    library_output_enabled,
    normalize_coulomb_settings,
)
from ._xl_tools import (
    calc_q,
    calc_q_batch,
    calc_q_os,
    kernel_update_lr,
    kernel_update_lr_batch,
    kernel_update_lr_os,
)


# ---------------------------------------------------------------------------
# Anderson / Pulay DIIS charge mixer
# ---------------------------------------------------------------------------
class _AndersonMixer:
    """History-based Anderson (Pulay/DIIS) charge mixer.

    Keeps the last *depth* (input, residual) pairs and solves the
    least-squares problem

        min_{c} ||Σ_i c_i R_i||²   s.t.  Σ c_i = 1

    to return the extrapolated update:

        q_new = Σ_i c_i (q_i + α R_i)

    where α = ``alpha`` (damping / mixing parameter).

    For the first step (no history) it falls back to simple linear mixing.
    """

    def __init__(self, alpha: float = 0.2, depth: int = 8):
        self.alpha = alpha
        self.depth = depth
        self._q_hist: deque = deque(maxlen=depth)  # input charges
        self._r_hist: deque = deque(maxlen=depth)  # residuals

    def reset(self):
        self._q_hist.clear()
        self._r_hist.clear()

    def mix(self, q_in: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Return next mixed charge vector.

        Handles both single-system (N,), open-shell (2, N), and batch (B, N)
        inputs correctly by computing per-element DIIS coefficients when
        the input has a leading batch dimension.
        """
        self._q_hist.append(q_in)
        self._r_hist.append(residual)

        m = len(self._r_hist)
        if m == 1:
            # Simple linear mixing on first step
            return q_in + self.alpha * residual

        # Build overlap matrix of residuals  A_{ij} = R_i · R_j
        R = torch.stack(list(self._r_hist))  # (m, N)  or (m, 2, N) or (m, B, N)

        # Detect batch dimension: if input is (B, N) with B > 1 and
        # ndim == 2, we need per-batch-element Anderson mixing.
        # Non-batch: q_in is (N,) → R is (m, N)
        # Open-shell: q_in is (2, N) → R is (m, 2, N)  (dim0=spin)
        # Batch: q_in is (B, N) → R is (m, B, N)
        # We distinguish batch from open-shell by checking if the class
        # was marked as batch-aware.
        is_batch = getattr(self, "_batch_mode", False)

        if is_batch and R.ndim == 3:
            Q = torch.stack(list(self._q_hist))  # (m, B, N)
            mixed = Q + self.alpha * R  # (m, B, N)
            q_new = torch.empty_like(q_in)

            # Solve each batch element independently. This matches the scalar
            # DIIS path and avoids numerical pathologies seen with batched
            # linear solves on heterogeneous SCC histories.
            for batch_idx in range(q_in.shape[0]):
                R_b = R[:, batch_idx, :].reshape(m, -1)
                A_b = R_b @ R_b.T

                reg = 1e-12 * torch.eye(m, device=q_in.device, dtype=q_in.dtype)
                A_b = A_b + reg * A_b.diag().max().clamp(min=1e-30)

                Bmat = torch.zeros(m + 1, m + 1, device=q_in.device, dtype=q_in.dtype)
                Bmat[:m, :m] = A_b
                Bmat[:m, m] = 1.0
                Bmat[m, :m] = 1.0
                rhs = torch.zeros(m + 1, device=q_in.device, dtype=q_in.dtype)
                rhs[m] = 1.0

                try:
                    sol = torch.linalg.solve(Bmat, rhs)
                except torch.linalg.LinAlgError:
                    q_new[batch_idx] = (
                        q_in[batch_idx] + self.alpha * residual[batch_idx]
                    )
                    continue

                c = sol[:m]
                q_new[batch_idx] = torch.einsum(
                    "i,i...->...", c, mixed[:, batch_idx, :]
                )
            return q_new
        else:
            # Non-batch or open-shell path (original)
            R_flat = R.reshape(m, -1)
            A = R_flat @ R_flat.T  # (m, m)

            # Tikhonov regularization to prevent ill-conditioned DIIS
            reg = 1e-12 * torch.eye(m, device=q_in.device, dtype=q_in.dtype)
            A = A + reg * A.diag().max().clamp(min=1e-30)

            # Constrained least squares via bordered matrix:
            #   [ A  1 ] [ c  ]   [ 0 ]
            #   [ 1  0 ] [ λ  ] = [ 1 ]
            Bmat = torch.zeros(m + 1, m + 1, device=q_in.device, dtype=q_in.dtype)
            Bmat[:m, :m] = A
            Bmat[:m, m] = 1.0
            Bmat[m, :m] = 1.0
            rhs = torch.zeros(m + 1, device=q_in.device, dtype=q_in.dtype)
            rhs[m] = 1.0

            try:
                sol = torch.linalg.solve(Bmat, rhs)
            except torch.linalg.LinAlgError:
                # Singular — fall back to simple mixing
                return q_in + self.alpha * residual

            c = sol[:m]  # coefficients that sum to 1

            # Extrapolated update:  Σ c_i (q_i + α R_i)
            Q = torch.stack(list(self._q_hist))  # (m, N) or (m, 2, N)
            q_new = torch.einsum("i,i...->...", c, Q + self.alpha * R)
            return q_new


#: The five pieces of per-orbital-group data ``SCFx`` needs to run at the finer
#: resolution, named once so the all-or-nothing check and its error message
#: cannot drift apart.
_SHELL_RESOLVED_ARGUMENT_NAMES = (
    "shell_types",
    "n_shells_per_atom",
    "el_per_shell",
    "Hubbard_U_sr",
    "C_sr",
)


def _shell_resolved_requested(supplied: Dict[str, Any]) -> bool:
    """Return whether the caller asked for per-orbital-group charge, or raise.

    All five arguments together mean yes; none of them means no. Any partial
    combination is a mistake and raises ``ValueError`` naming exactly what is
    missing. The alternative — falling through to the per-atom arm — is the
    silent-fallback failure this whole design is arranged against: the caller
    would hold a per-atom answer while believing it was the finer one.
    """
    present = [name for name in _SHELL_RESOLVED_ARGUMENT_NAMES if supplied[name] is not None]
    if not present:
        return False
    if len(present) == len(_SHELL_RESOLVED_ARGUMENT_NAMES):
        return True
    missing = [name for name in _SHELL_RESOLVED_ARGUMENT_NAMES if supplied[name] is None]
    raise ValueError(
        "Per-orbital-group (shell-resolved) charge needs all of "
        f"{', '.join(_SHELL_RESOLVED_ARGUMENT_NAMES)}. "
        f"Supplied: {', '.join(present)}. Missing: {', '.join(missing)}. "
        "Supply all five or none; a partial request is not served by the "
        "per-atom path."
    )


def _sum_shells_over_atoms(
    values: torch.Tensor, shell_to_atom: torch.Tensor, Nats: int
) -> torch.Tensor:
    """Add up a per-orbital-group quantity into one number per atom.

    This is how ``q`` keeps being populated while the loop runs at the finer
    resolution: the per-atom charges are a *view* of the per-group ones, summed,
    never a separately computed second answer that could drift from them.
    """
    per_atom = torch.zeros(Nats, dtype=values.dtype, device=values.device)
    per_atom.scatter_add_(0, shell_to_atom, values)
    return per_atom


def SCFx(
    dftorch_params: Dict[str, Any],
    RX,
    RY,
    RZ,
    cell: torch.Tensor,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    TYPE: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    dU_dq: Optional[torch.Tensor],
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    req_grad_xyz: bool,
    q_init: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
    *,
    shell_types: Optional[torch.Tensor] = None,
    n_shells_per_atom: Optional[torch.Tensor] = None,
    el_per_shell: Optional[torch.Tensor] = None,
    Hubbard_U_sr: Optional[torch.Tensor] = None,
    C_sr: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # Q (eigenvectors of the orthogonalized Hamiltonian)
    torch.Tensor,  # e (eigenvalues of the orthogonalized Hamiltonian)
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
    Optional[torch.Tensor],  # stress_coul (PME only)
    int,  # scf_iter_count: pass count on success, -1 if the loop gave up
    Optional[torch.Tensor],  # q_sr: per-orbital-group charges, None when off
]:
    """
    Self-consistent field (_scf) cycle with finite electronic temperature and
    Fermi–Dirac occupations, using a preconditioned low-rank Krylov charge mixer.
    Supports PME Ewald electrostatics via `sedacs` or a direct Coulomb matrix.

    Parameters
    ----------
    dftorch_params : dict
        _scf/control parameters. Expected keys include:
        - 'COUL_METHOD': str, 'PME' or 'direct'
        - 'cutoff': float, real-space cutoff (PME)
        - 'Coulomb_acc': float, accuracy target for PME alpha/grid (PME)
        - 'PME_ORDER': int, B-spline order (PME)
        - other PME/mixing options passed through to helper routines.
    structure : object
        Container providing required system data/attributes:
        - RX, RY, RZ: (Nats,) atomic coordinates (torch.Tensor)
        - cell: (3,3) lattice vectors (torch.Tensor), for PME
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
        Coulomb operator. If dftorch_params['COUL_METHOD'] == 'direct',
        used as C @ q to build electrostatic potential; ignored for PME.
    shell_types, n_shells_per_atom, el_per_shell, Hubbard_U_sr, C_sr :
        torch.Tensor or None, keyword-only
        The five pieces of per-orbital-group ("shell-resolved") data. Supply
        **all five** to run the loop at the finer resolution; leave all five
        absent — the default — for exactly the per-atom behaviour this function
        had before they existed. Supplying some but not all is a mistake and
        raises ``ValueError`` naming the missing ones, rather than quietly
        taking the per-atom arm.

        What the finer resolution does, in plain words: instead of tracking one
        charge number and one electron-repulsion strength per *atom*, the loop
        tracks them separately for each atom's s, p, d and f groups. That
        matters wherever an element's groups do not share one strength — for
        europium the s group costs about 5.7 eV per unit of charge and the f
        group about 13.6 eV, and seven of its nine outer electrons live in the
        f group. Requirement SCC-03 in Phase 6.

        - ``shell_types``: (n_shells,) the group's angular label, 1/2/3/4 for
          s/p/d/f, in the layout ``Structure`` builds.
        - ``n_shells_per_atom``: (Nats,) how many groups each atom contributes,
          which is what maps groups back to atoms.
        - ``el_per_shell``: (n_shells,) the reference occupation the per-group
          tally subtracts, playing the role ``Znuc`` plays per atom.
        - ``Hubbard_U_sr``: (n_shells,) each group's own repulsion strength.
        - ``C_sr``: (n_shells, n_shells) the shell-resolved Coulomb matrix.

        Three limits of the finer resolution, all of which raise rather than
        fall back: PME electrostatics, the third-order DFTB3 correction and
        GBSA solvation are per-atom constructions with no shell-resolved
        counterpart. ``q_init`` is *ignored* at the finer resolution — a
        per-atom starting guess cannot be split across an atom's groups without
        inventing information — so the loop always starts from a reference
        diagonalization there. That changes only where the loop starts, never
        where it lands.
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
    scf_iter_count : int
        How the loop ended, as a number rather than as printed text. On success
        it is the number of passes taken to reach tolerance, always ``>= 1``.
        On failure it is the literal ``-1``, meaning the loop exhausted
        ``SCF_MAX_ITER`` with at least one of its two tolerance conditions
        still unmet. The shape follows scipy's iterative solvers, which return
        a positive count on success and a sentinel on failure. It is never a
        boolean: a caller that only wants "did it work" can test
        ``scf_iter_count != -1``, but the count itself is not thrown away.
        A ``-1`` still comes back alongside the last iterate; the loop warns
        and returns, it never raises (Phase 4 decision D-13).
    q_sr : torch.Tensor or None
        The converged charges one per orbital group, (n_shells,), when the loop
        ran at the finer resolution; ``None`` when it did not. ``q`` above is
        populated in **both** cases — at the finer resolution it is derived by
        summing ``q_sr`` over each atom — so every existing caller keeps
        reading the same attribute it always read.

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
    # D-02: one read of VERBOSE_LIBRARY_OUTPUT for the whole call, rather
    # than a dict lookup per print. Defaults to True, so a caller who never
    # sets the key sees exactly the output they saw before it existed.
    _lib_out = library_output_enabled(dftorch_params)
    if _lib_out:
        print("### Do _scf ###")

    device = H0.device
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=H0.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    # --- Per-orbital-group ("shell-resolved") mode --------------------------
    # All five arguments or none; a partial request raises rather than being
    # served by the per-atom path.
    shell_resolved = _shell_resolved_requested(
        {
            "shell_types": shell_types,
            "n_shells_per_atom": n_shells_per_atom,
            "el_per_shell": el_per_shell,
            "Hubbard_U_sr": Hubbard_U_sr,
            "C_sr": C_sr,
        }
    )
    q_sr = None
    shell_ids = None
    shell_to_atom = None
    if shell_resolved:
        # Three constructions that have no per-orbital-group counterpart. Each
        # is per-atom by construction, so serving the request would mean
        # quietly mixing resolutions inside one energy. Refuse instead.
        if dftorch_params["COUL_METHOD"] == "PME":
            raise NotImplementedError(
                "COUL_METHOD='PME' cannot serve a MAGNETIC_HUBBARD_LDEP "
                "(per-orbital-group) charge loop: no shell-resolved "
                "reciprocal-space Coulomb matrix exists. Use a real-space "
                "COUL_METHOD, or unset MAGNETIC_HUBBARD_LDEP."
            )
        if dU_dq is not None or thirdorder is not None:
            raise NotImplementedError(
                "The third-order DFTB3 correction is per-atom only and cannot "
                "be combined with a MAGNETIC_HUBBARD_LDEP "
                "(per-orbital-group) charge loop."
            )
        if gbsa is not None:
            raise NotImplementedError(
                "GBSA solvation shifts are per-atom only and cannot be "
                "combined with a MAGNETIC_HUBBARD_LDEP (per-orbital-group) "
                "charge loop."
            )
        # Orbitals per group: 1, 3, 5, 7 for s, p, d, f. This is exactly
        # ``const.shell_dim[shell_types]`` for the table [0, 1, 3, 5, 7] that
        # Constants.py:106 defines, written as arithmetic so the loop does not
        # need a sixth argument to carry a table it can derive.
        orbitals_per_shell = 2 * shell_types - 1
        # Both maps are built exactly as the open-shell routine builds them
        # (it calls the first one ``atom_ids_sr``, despite the name).
        shell_ids = torch.repeat_interleave(
            torch.arange(len(shell_types), device=H0.device), orbitals_per_shell
        )  # Generate orbital-group index for each orbital
        shell_to_atom = torch.repeat_interleave(
            torch.arange(len(n_shells_per_atom), device=H0.device), n_shells_per_atom
        )  # Generate atom index for each orbital group
        # The groups must tile each atom's orbitals exactly. If they do not,
        # every gather below lands on the wrong orbital and the answer would
        # still look plausible, so this is checked rather than assumed.
        orbitals_from_shells = _sum_shells_over_atoms(
            orbitals_per_shell, shell_to_atom, Nats
        )
        if not bool(
            torch.equal(
                orbitals_from_shells.to(n_orbitals_per_atom.dtype),
                n_orbitals_per_atom,
            )
        ):
            raise ValueError(
                "The orbital groups do not tile the atoms' orbitals: groups "
                f"give {orbitals_from_shells.tolist()} orbitals per atom but "
                f"n_orbitals_per_atom is {n_orbitals_per_atom.tolist()}."
            )

    if shell_resolved:
        Hubbard_U_gathered = Hubbard_U_sr[shell_ids]
    else:
        Hubbard_U_gathered = Hubbard_U[atom_ids]
    if dU_dq is not None:
        dU_dq_gathered = dU_dq[atom_ids]
    else:
        dU_dq_gathered = None

    normalize_coulomb_settings(dftorch_params, cell, context="SCFx")
    coulomb_cutoff = dftorch_params.get("COULOMB_CUTOFF", 10.0)
    if dftorch_params["COUL_METHOD"] == "PME":
        from .ewald_pme import (
            calculate_alpha_and_num_grids,
            calculate_PME_ewald,
            init_PME_data,
        )
        from .ewald_pme.neighbor_list import NeighborState

        # positions = torch.stack((RX, RY, RZ))
        positions = torch.stack(
            (RX, RY, RZ),
        )
        CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
            cell.cpu().numpy(),
            coulomb_cutoff,
            dftorch_params.get("COULOMB_ACC", 1e-5),
        )
        PME_data = init_PME_data(
            grid_dimensions, cell, CALPHA, dftorch_params.get("PME_ORDER", 4)
        )
        nbr_state = NeighborState(
            positions,
            cell,
            None,
            coulomb_cutoff,
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nbr_inds = calculate_dist_dips(
            positions, nbr_state, coulomb_cutoff
        )
    else:
        PME_data = None
        nbr_inds = None
        disps = None
        dists = None
        CALPHA = None

    _no_grad_ctx = (
        contextlib.nullcontext()
        if dftorch_params.get("SCF_GRAD", False)
        else torch.no_grad()
    )
    with _no_grad_ctx:
        # if 1:

        # Initial density matrix
        if _lib_out:
            print("  Initial dm_fermi")

        Hdipole = torch.diag(
            -RX[atom_ids] * Efield[0]
            - RY[atom_ids] * Efield[1]
            - RZ[atom_ids] * Efield[2]
        )
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole

        if q_init is None or shell_resolved:
            if shell_resolved and q_init is not None and _lib_out:
                # Not silent: a per-atom starting guess carries no information
                # about how the charge is split across an atom's groups, so it
                # cannot be honoured here. Only the starting point is affected.
                print(
                    "  Ignoring the per-atom initial charges: this loop tracks "
                    "charge per orbital group and starts from a reference "
                    "diagonalization."
                )
            Dorth, Q, e, f, mu0 = dm_fermi_x(
                Z.T @ H0 @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
            )

            if _lib_out:
                print("  Initial mu = {:.4f}".format(mu0.item()))

            D = Z @ Dorth @ Z.T
            DS = 2 * torch.diag(D @ S)
            if shell_resolved:
                # The per-group tally subtracts each group's reference
                # occupation where the per-atom one subtracts nuclear charge;
                # the two totals agree, so the two resolutions describe one
                # answer.  Same construction as the open-shell routine.
                q_sr = -1.0 * el_per_shell.to(DS.dtype)
                q_sr.scatter_add_(0, shell_ids, DS)
                q = _sum_shells_over_atoms(q_sr, shell_to_atom, Nats)
            else:
                q = -1.0 * Znuc
                q.scatter_add_(
                    0, atom_ids, DS
                )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        else:
            q = q_init.clone()

        KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
            Nats, device=H0.device
        )  # Initial mixing coefficient for linear mixing
        # KK0 = KK*torch.eye(Nats, device=H0.device)

        # Anderson / Pulay DIIS mixer for pre-Krylov phase
        anderson_depth = dftorch_params.get("ANDERSON_DEPTH", 8)
        if anderson_depth > 0:
            anderson_alpha = dftorch_params.get(
                "ANDERSON_ALPHA", max(dftorch_params["SCF_ALPHA"], 0.2)
            )
            _mixer = _AndersonMixer(
                alpha=anderson_alpha,
                depth=anderson_depth,
            )
        else:
            _mixer = None

        ResNorm = torch.tensor([2.0], device=device)
        dEc = torch.tensor([1000.0], device=device)
        it = 0
        Ecoul = torch.tensor([0.0], device=device)

        if _lib_out:
            print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
            if _lib_out:
                print("Iter {}".format(it))

            if dftorch_params["COUL_METHOD"] == "PME":
                # with torch.enable_grad():
                if 1:
                    ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                        positions,
                        q,
                        cell,
                        nbr_inds,
                        disps,
                        dists,
                        CALPHA,
                        coulomb_cutoff,
                        PME_data,
                        hubbard_u=Hubbard_U,
                        atomtypes=TYPE,
                        screening=1,
                        calculate_forces=0,
                        calculate_dq=1,
                        h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
                        h5_params=dftorch_params.get("H5_PARAMS", None),
                    )

            elif shell_resolved:
                CoulPot = C_sr @ q_sr
            else:
                CoulPot = C @ q

            # Add GBSA Born shift to Coulomb potential
            if gbsa is not None:
                CoulPot = CoulPot + gbsa.get_shifts(q)

            # Add full off-diagonal DFTB3 shift to Coulomb potential
            # (replaces the diagonal-only dU_dq term inside calc_q)
            if thirdorder is not None:
                CoulPot = CoulPot + thirdorder.get_shifts(q)

            q_old = q.clone()
            q_sr_old = q_sr.clone() if shell_resolved else None

            # calc_q is orbital-level throughout except for its closing
            # per-atom tally, so the finer resolution enters by handing it the
            # same three orbital-length arrays gathered from *groups* instead
            # of from atoms. calc_q itself is untouched.
            if shell_resolved:
                charge_gathered = q_sr[shell_ids]
                potential_gathered = CoulPot[shell_ids]
            else:
                charge_gathered = q[atom_ids]
                potential_gathered = CoulPot[atom_ids]

            q_new, H, Hcoul, D, Dorth, Q, e, f, mu0 = calc_q(
                H0,
                Hubbard_U_gathered,
                charge_gathered,
                potential_gathered,
                S,
                Z,
                Te,
                Nocc,
                Znuc,
                atom_ids,
                dU_dq_gathered if thirdorder is None else None,
            )
            if shell_resolved:
                # Ignore calc_q's per-atom tally and re-tally per group from
                # the density matrix it returned, then derive the per-atom
                # charges by summing the groups over each atom so ``q`` stays
                # correct for every downstream consumer.
                DS = 2 * (D * S.T).sum(dim=1)
                q_sr = -1.0 * el_per_shell.to(DS.dtype)
                q_sr.scatter_add_(0, shell_ids, DS)
                q = _sum_shells_over_atoms(q_sr, shell_to_atom, Nats)
                Res = q_sr - q_sr_old
            else:
                q = q_new
                Res = q - q_old
            ResNorm = torch.norm(Res)

            # --- Charge mixing ---
            # The Krylov accelerator preconditions with the *per-atom* Coulomb
            # matrix and per-atom Hubbard U, so it has no counterpart at the
            # finer resolution: running it there would be precisely the silent
            # per-atom substitution this design refuses. The Anderson/DIIS
            # mixer below handles a one-dimensional vector of any length and
            # is what the finer resolution uses. This changes how the loop
            # travels, never where it lands -- the fixed point is set by C_sr
            # and Hubbard_U_sr either way.
            use_krylov = (not shell_resolved) and it > dftorch_params.get(
                "KRYLOV_START", 10
            )

            if use_krylov:
                K0Res = KK @ Res
                # Preconditioned Low-Rank Krylov _scf acceleration
                K0Res = kernel_update_lr(
                    RX,
                    RY,
                    RZ,
                    cell,
                    TYPE,
                    Nats,
                    Hubbard_U,
                    dftorch_params,
                    dftorch_params.get("KRYLOV_TOL", 1e-6),
                    KK,
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
                    dU_dq if thirdorder is None else None,
                    gbsa,
                    thirdorder=thirdorder,
                )
                q = q_old - K0Res
            elif _mixer is not None:
                # Anderson / DIIS mixing (pre-Krylov)
                if shell_resolved:
                    q_sr = _mixer.mix(q_sr_old, Res)
                    q = _sum_shells_over_atoms(q_sr, shell_to_atom, Nats)
                else:
                    q = _mixer.mix(q_old, Res)
            else:
                # Simple linear mixing fallback
                if shell_resolved:
                    # KK is -SCF_ALPHA * I at the per-atom size and is handed
                    # back to callers that expect that size, so the finer
                    # resolution applies the same linear step directly rather
                    # than resizing a matrix other code reads.
                    q_sr = q_sr_old + dftorch_params["SCF_ALPHA"] * Res
                    q = _sum_shells_over_atoms(q_sr, shell_to_atom, Nats)
                else:
                    K0Res = KK @ Res
                    q = q_old - K0Res

            Ecoul_old = Ecoul
            if dftorch_params["COUL_METHOD"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q**2 * Hubbard_U)
            elif shell_resolved:
                # The same expression as the per-atom line below, read at the
                # finer resolution throughout. It must match what calc_q was
                # handed above, or the reported energy would describe a
                # different charge state than the one the loop converged to.
                Ecoul = 0.5 * q_sr @ (C_sr @ q_sr) + 0.5 * torch.sum(
                    q_sr**2 * Hubbard_U_sr
                )
            else:
                Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * Hubbard_U)

            # Third-order energy contribution
            if thirdorder is not None:
                Ecoul = Ecoul + thirdorder.get_energy(q)
            elif dU_dq is not None:
                Ecoul = Ecoul + (1.0 / 3.0) * torch.sum(0.5 * dU_dq * q**3)

            # dEb = torch.abs(Eband0_old - Eband0)
            dEc = torch.abs(Ecoul_old - Ecoul)

            # print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            if _lib_out:
                print(
                    "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                        ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                    )
                )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        # How the loop ended, as a number a caller can read rather than as
        # printed text.  The two clauses below are the negation of the
        # while-condition's own tolerance tests above -- mirrored from it, not
        # retyped with fresh thresholds -- so a loop that merely ran out of
        # passes can never present itself as converged (threat T-06-03).  The
        # printed warning above stays: D-13 requires non-convergence to warn as
        # well as to report.
        _scf_tol = dftorch_params.get("SCF_TOL", 1e-6)
        _converged = bool(ResNorm <= _scf_tol) and bool(dEc <= _scf_tol * 100)
        scf_iter_count = int(it) if _converged else -1

        # f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))

    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)
    if shell_resolved:
        q_sr = -1.0 * el_per_shell.to(DS.dtype)
        q_sr.scatter_add_(0, shell_ids, DS)
        q = _sum_shells_over_atoms(q_sr, shell_to_atom, Nats)
    else:
        q = -1.0 * Znuc
        q.scatter_add_(0, atom_ids, DS)

    if dftorch_params["COUL_METHOD"] == "PME":
        ewald_e1, forces1, dq_p1, stress_coul = calculate_PME_ewald(
            positions,  # .detach().clone(),
            q,
            cell,
            nbr_inds,
            disps,
            dists,
            CALPHA,
            coulomb_cutoff,
            PME_data,
            hubbard_u=Hubbard_U,
            atomtypes=TYPE,
            screening=1,
            calculate_forces=0 if req_grad_xyz else 1,
            calculate_dq=0 if req_grad_xyz else 1,
            calculate_stress=0 if req_grad_xyz else 1,
            h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
            h5_params=dftorch_params.get("H5_PARAMS", None),
        )
        Ecoul = ewald_e1 + 0.5 * torch.sum(q**2 * Hubbard_U)
    else:
        Ecoul, forces1, dq_p1, stress_coul = None, None, None, None

    return (
        H,
        Hcoul,
        Hdipole,
        KK,
        D,
        Q,
        e,
        q,
        f,
        mu0,
        Ecoul,
        forces1,
        dq_p1,
        stress_coul,
        scf_iter_count,
        q_sr,
    )


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
    cell: torch.Tensor,
    Nats: int,
    Nocc: int,
    n_orbitals_per_atom: torch.Tensor,
    Znuc: torch.Tensor,
    TYPE: torch.Tensor,
    Te: float,
    Hubbard_U: torch.Tensor,
    dU_dq: Optional[torch.Tensor],
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    req_grad_xyz: bool,
    q_spin_sr_init: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
) -> Tuple[
    torch.Tensor,  # H
    torch.Tensor,  # Hcoul
    torch.Tensor,  # Hdipole
    torch.Tensor,  # KK (preconditioner / mixing kernel)
    torch.Tensor,  # D
    torch.Tensor,  # Q (eigenvectors of the orthogonalized Hamiltonian)
    torch.Tensor,  # e (eigenvalues of the orthogonalized Hamiltonian)
    torch.Tensor,  # q_spin_atom
    torch.Tensor,  # q_tot_atom
    torch.Tensor,  # q_spin_sr (shell-resolved)
    torch.Tensor,  # net_spin_sr (shell-resolved)
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
    Optional[torch.Tensor],  # stress_coul (PME only)
    int,  # scf_iter_count: pass count on success, -1 if the loop gave up
]:
    """Open-shell (spin-polarized) self-consistent charge loop.

    The last returned value, ``scf_iter_count``, is how the loop ended stated as
    a number rather than as printed text: the number of passes taken on success,
    always ``>= 1``, or the literal ``-1`` when the loop exhausted
    ``SCF_MAX_ITER`` with a tolerance still unmet. A ``-1`` still comes back
    alongside the last iterate; the loop warns and returns, it never raises.
    """
    # D-02: one read of VERBOSE_LIBRARY_OUTPUT for the whole call, rather
    # than a dict lookup per print. Defaults to True, so a caller who never
    # sets the key sees exactly the output they saw before it existed.
    _lib_out = library_output_enabled(dftorch_params)
    if _lib_out:
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
    if dU_dq is not None:
        dU_dq_gathered = dU_dq[atom_ids]
    else:
        dU_dq_gathered = None

    normalize_coulomb_settings(dftorch_params, cell, context="scf_x_os")
    coulomb_cutoff = dftorch_params.get("COULOMB_CUTOFF", 10.0)
    if dftorch_params["COUL_METHOD"] == "PME":
        from .ewald_pme import (
            calculate_alpha_and_num_grids,
            calculate_PME_ewald,
            init_PME_data,
        )
        from .ewald_pme.neighbor_list import NeighborState

        # positions = torch.stack((RX, RY, RZ))
        positions = torch.stack(
            (RX, RY, RZ),
        )
        CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
            cell.cpu().numpy(),
            coulomb_cutoff,
            dftorch_params.get("COULOMB_ACC", 1e-5),
        )
        PME_data = init_PME_data(
            grid_dimensions, cell, CALPHA, dftorch_params.get("PME_ORDER", 4)
        )
        nbr_state = NeighborState(
            positions,
            cell,
            None,
            coulomb_cutoff,
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nbr_inds = calculate_dist_dips(
            positions, nbr_state, coulomb_cutoff
        )
    else:
        PME_data = None
        nbr_inds = None
        disps = None
        dists = None
        CALPHA = None

    _no_grad_ctx = (
        contextlib.nullcontext()
        if dftorch_params.get("SCF_GRAD", False)
        else torch.no_grad()
    )
    with _no_grad_ctx:
        # if 1:
        # Initial density matrix
        if _lib_out:
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

        # if shared_mu0:
        #     Dorth, Q, e, f, mu0 = dm_fermi_x_os_shared(
        #         Z.T @ H0 @ Z,
        #         Te,
        #         Nocc,
        #         mu_0=None,
        #         eps=1e-9,
        #         MaxIt=50,
        #         broken_symmetry=False,
        #     )
        # else:
        if q_spin_sr_init is not None:
            q_spin_sr = q_spin_sr_init.clone()
        else:
            Dorth, Q, e, f, mu0 = dm_fermi_x_os(
                Z.T @ H0 @ Z,
                Te,
                Nocc,
                mu_0=None,
                eps=1e-9,
                MaxIt=50,
                broken_symmetry=broken_symmetry,
            )

            D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
            DS = 1 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)

            q_spin_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
            q_spin_sr.scatter_add_(
                1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS
            )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        # break spin symmetry on q_spin_sr, not on density matrix
        # if broken_symmetry:
        #     # shift spin density: add electrons to alpha, remove from beta
        #     # on shells belonging to atom with highest Znuc (most polarizable)
        #     most_polarizable_atom = Znuc.argmax()
        #     atom_shells = (shell_to_atom == most_polarizable_atom).nonzero().squeeze()
        #     delta = 0.01
        #     q_spin_sr[0, atom_shells] += delta   # alpha gets more
        #     q_spin_sr[1, atom_shells] -= delta   # beta gets less
        #     print(f"  Broken symmetry: perturbed shells {atom_shells.tolist()} "
        #           f"on atom {most_polarizable_atom.item()} (Znuc={Znuc[most_polarizable_atom].item()})")

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

        # Anderson / Pulay DIIS mixer for pre-Krylov phase (open-shell)
        anderson_depth = dftorch_params.get("ANDERSON_DEPTH", 8)
        if anderson_depth > 0:
            anderson_alpha = dftorch_params.get(
                "ANDERSON_ALPHA", max(dftorch_params["SCF_ALPHA"], 0.2)
            )
            _mixer = _AndersonMixer(
                alpha=anderson_alpha,
                depth=anderson_depth,
            )
        else:
            _mixer = None

        ResNorm = torch.tensor(2.0, device=device)
        dEc = torch.tensor(1000.0, device=device)
        it = 0
        Ecoul = torch.tensor(0.0, device=device)

        if _lib_out:
            print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
            if _lib_out:
                print("Iter {}".format(it))

            if dftorch_params["COUL_METHOD"] == "PME":
                # with torch.enable_grad():
                if 1:
                    ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                        positions,
                        q_tot_atom,
                        cell,
                        nbr_inds,
                        disps,
                        dists,
                        CALPHA,
                        coulomb_cutoff,
                        PME_data,
                        hubbard_u=Hubbard_U,
                        atomtypes=TYPE,
                        screening=1,
                        calculate_forces=0,
                        calculate_dq=1,
                        h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
                        h5_params=dftorch_params.get("H5_PARAMS", None),
                    )
            else:
                CoulPot = C @ q_tot_atom

            # Add GBSA Born shift to Coulomb potential
            if gbsa is not None:
                CoulPot = CoulPot + gbsa.get_shifts(q_tot_atom)

            # Add full off-diagonal DFTB3 shift
            if thirdorder is not None:
                CoulPot = CoulPot + thirdorder.get_shifts(q_tot_atom)

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
                dU_dq_gathered if thirdorder is None else None,
                dftorch_params.get("SHARED_MU", False),
            )

            q_spin_atom = torch.zeros_like(RX.unsqueeze(0).expand(2, -1))
            q_spin_atom.scatter_add_(
                1, shell_to_atom.unsqueeze(0).expand(2, -1), q_spin_sr
            )  # atom-resolved

            Res = q_spin_sr - q_spin_sr_old
            ResNorm = torch.norm(Res)

            # --- Charge mixing ---
            use_krylov = it > dftorch_params.get("KRYLOV_START", 10)

            if use_krylov:
                K0Res = torch.bmm(KK, Res.unsqueeze(-1)).squeeze(-1)
                # Preconditioned Low-Rank Krylov _scf acceleration
                K0Res = kernel_update_lr_os(
                    RX,
                    RY,
                    RZ,
                    cell,
                    TYPE,
                    Nats,
                    Hubbard_U,
                    dftorch_params,
                    dftorch_params.get("KRYLOV_TOL", 1e-6),
                    KK,
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
                    dU_dq if thirdorder is None else None,
                    gbsa,
                    thirdorder=thirdorder,
                )
                q_spin_sr = q_spin_sr_old - K0Res
            elif _mixer is not None:
                # Anderson / DIIS mixing (pre-Krylov)
                q_spin_sr = _mixer.mix(q_spin_sr_old, Res)
            else:
                # Simple linear mixing fallback
                K0Res = torch.bmm(KK, Res.unsqueeze(-1)).squeeze(-1)
                q_spin_sr = q_spin_sr_old - K0Res

            # q_tot_sr = q_spin_sr.sum(dim=0)
            q_tot_atom = torch.zeros_like(RX)
            q_tot_atom.scatter_add_(
                0, shell_to_atom, q_spin_sr.sum(dim=0)
            )  # atom-resolved
            net_spin_sr = q_spin_sr[0] - q_spin_sr[1]

            Ecoul_old = Ecoul
            if dftorch_params["COUL_METHOD"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q_tot_atom**2 * Hubbard_U)
            else:
                Ecoul = 0.5 * q_tot_atom @ (C @ q_tot_atom) + 0.5 * torch.sum(
                    q_tot_atom**2 * Hubbard_U
                )

            dEc = torch.abs(Ecoul_old - Ecoul)

            # print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            if _lib_out:
                print(
                    "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                        ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                    )
                )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        # How the loop ended, as a number a caller can read rather than as
        # printed text.  The two clauses are the negation of this loop's own
        # while-condition tolerance tests, read from it rather than copied from
        # a sibling loop.  See ``SCFx`` for the full reasoning.
        _scf_tol = dftorch_params.get("SCF_TOL", 1e-6)
        _converged = bool(ResNorm <= _scf_tol) and bool(dEc <= _scf_tol * 100)
        scf_iter_count = int(it) if _converged else -1

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

    if dftorch_params["COUL_METHOD"] == "PME":
        ewald_e1, forces1, dq_p1, stress_coul = calculate_PME_ewald(
            positions,  # .detach().clone(),
            q_tot_atom,
            cell,
            nbr_inds,
            disps,
            dists,
            CALPHA,
            coulomb_cutoff,
            PME_data,
            hubbard_u=Hubbard_U,
            atomtypes=TYPE,
            screening=1,
            calculate_forces=0 if req_grad_xyz else 1,
            calculate_dq=0 if req_grad_xyz else 1,
            calculate_stress=0 if req_grad_xyz else 1,
            h_damp_exp=dftorch_params.get("H_DAMP_EXP", None),
            h5_params=dftorch_params.get("H5_PARAMS", None),
        )
        Ecoul = ewald_e1 + 0.5 * torch.sum(q_tot_atom**2 * Hubbard_U)
    else:
        Ecoul, forces1, dq_p1, stress_coul = None, None, None, None

    return (
        H,
        Hcoul,
        Hdipole,
        KK,
        D,
        Q,
        e,
        q_spin_atom,
        q_tot_atom,
        q_spin_sr,
        net_spin_sr,
        f,
        mu0,
        Ecoul,
        forces1,
        dq_p1,
        stress_coul,
        scf_iter_count,
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
    dU_dq: Optional[torch.Tensor],
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Efield: torch.Tensor,
    C: torch.Tensor,
    gbsa_batch=None,
    thirdorder_batch=None,
    q_init: Optional[torch.Tensor] = None,
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
    int,  # scf_iter_count: pass count on success, -1 if the loop gave up
]:
    """
    Self-consistent field (_scf) cycle with finite electronic temperature and
    Fermi–Dirac occupations, using a preconditioned low-rank Krylov charge mixer.
    Supports PME Ewald electrostatics via `sedacs` or a direct Coulomb matrix.

    Batched over many structures at once.  The last returned value,
    ``scf_iter_count``, describes **the batch as a whole and never one structure
    within it**: the loop's stopping test is taken across every member, so the
    count is the number of passes after which the last remaining member met
    tolerance, and ``-1`` means the cap ran out with at least one member still
    unconverged -- it does not say which.
    """

    # D-02: one read of VERBOSE_LIBRARY_OUTPUT for the whole call, rather
    # than a dict lookup per print. Defaults to True, so a caller who never
    # sets the key sees exactly the output they saw before it existed.
    _lib_out = library_output_enabled(dftorch_params)

    batch_size = RX.shape[0]
    device = H0.device
    counts = n_orbitals_per_atom  # shape (B, N)
    cum_counts = torch.cumsum(counts, dim=1)  # cumulative sums per batch
    total_orbs = H0.shape[-1]
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
    if dU_dq is not None:
        dU_dq_gathered = dU_dq.gather(1, atom_ids)
    else:
        dU_dq_gathered = None

    _no_grad_ctx = (
        contextlib.nullcontext()
        if dftorch_params.get("SCF_GRAD", False)
        else torch.no_grad()
    )
    with _no_grad_ctx:
        RX_gathered = RX.gather(1, atom_ids)
        RY_gathered = RY.gather(1, atom_ids)
        RZ_gathered = RZ.gather(1, atom_ids)
        Hdipole = torch.diag_embed(
            -RX_gathered * Efield[0] - RY_gathered * Efield[1] - RZ_gathered * Efield[2]
        )
        Hdipole = 0.5 * (torch.matmul(Hdipole, S) + torch.matmul(S, Hdipole))
        H0 = H0 + Hdipole

        _scf_degen = dftorch_params.get("SCF_DEGEN", False)
        _dm_solver = dm_fermi_x_batch_degen if _scf_degen else dm_fermi_x_batch
        if q_init is None:
            H_ortho = torch.matmul(Z.transpose(-1, -2), torch.matmul(H0, Z))
            Dorth, Q, e, f, mu0 = _dm_solver(
                H_ortho, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
            )
            if _lib_out:
                print("  Initial mu", mu0)
            D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
            DS = 2 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
            q = -1.0 * Znuc
            q.scatter_add_(
                1, atom_ids, DS
            )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        else:
            # Use provided initial charges — skip expensive eigendecomposition
            if q_init.dim() == 1:
                # Single reference charges → broadcast to all batch elements
                q = q_init.unsqueeze(0).expand(batch_size, -1).clone()
            else:
                q = q_init.clone()
            if _lib_out:
                print("  Using q_init (skipping initial dm_fermi)")

        KK = -dftorch_params["SCF_ALPHA"] * torch.eye(
            Nats, device=H0.device
        ) + torch.zeros(
            batch_size, Nats, Nats, device=H0.device
        )  # Initial mixing coefficient for linear mixing
        # KK0 = KK*torch.eye(Nats, device=H0.device)

        # Anderson / Pulay DIIS mixer for pre-Krylov phase (batch)
        anderson_depth = dftorch_params.get("ANDERSON_DEPTH", 8)
        if anderson_depth > 0:
            anderson_alpha = dftorch_params.get(
                "ANDERSON_ALPHA", max(dftorch_params["SCF_ALPHA"], 0.2)
            )
            _mixers = [
                _AndersonMixer(alpha=anderson_alpha, depth=anderson_depth)
                for _ in range(batch_size)
            ]
        else:
            _mixers = None

        scf_tol = dftorch_params.get("SCF_TOL", 1e-6)
        ResNorm = torch.zeros(batch_size, device=device) + 2.0  # float("inf")
        dEc = torch.zeros(batch_size, device=device) + 1000.0  # float("inf")
        it = 0
        Ecoul = torch.zeros(batch_size, device=device) + 0.0  # float("inf")

        if _lib_out:
            print("\nStarting cycle")
        while (
            (ResNorm > scf_tol).any() or (dEc > scf_tol * 100).any()
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            it += 1
            if _lib_out:
                print("Iter {}".format(it))

            CoulPot = torch.matmul(C, q.unsqueeze(-1)).squeeze(-1)

            # Add GBSA Born shift to Coulomb potential (vectorised over batch)
            if gbsa_batch is not None:
                CoulPot = CoulPot + gbsa_batch.get_shifts(q)

            # Add full off-diagonal DFTB3 shift to Coulomb potential
            if thirdorder_batch is not None:
                CoulPot = CoulPot + thirdorder_batch.get_shifts(q)

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
                dU_dq_gathered if thirdorder_batch is None else None,
                degen=_scf_degen,
            )
            Res = q - q_old
            ResNorm = torch.norm(Res, dim=1)
            active_mask = (ResNorm > scf_tol) | (dEc > scf_tol * 100)

            # --- Charge mixing ---
            use_krylov = it > dftorch_params.get("KRYLOV_START", 10)

            if use_krylov:
                K0Res = torch.matmul(KK, Res.unsqueeze(-1)).squeeze(-1)
                # Preconditioned Low-Rank Krylov _scf acceleration
                K0Res = kernel_update_lr_batch(
                    Nats,
                    Hubbard_U_gathered,
                    dftorch_params,
                    dftorch_params.get("KRYLOV_TOL", 1e-6),
                    KK,
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
                    dU_dq_gathered if thirdorder_batch is None else None,
                    gbsa=gbsa_batch,
                    thirdorder=thirdorder_batch,
                )
                q_candidate = q_old - K0Res
                q = torch.where(active_mask.unsqueeze(-1), q_candidate, q_old)
            elif _mixers is not None:
                # Anderson / DIIS mixing (pre-Krylov)
                q = q_old.clone()
                for batch_idx, mixer in enumerate(_mixers):
                    if active_mask[batch_idx]:
                        q[batch_idx] = mixer.mix(q_old[batch_idx], Res[batch_idx])
            else:
                # Simple linear mixing fallback
                K0Res = torch.matmul(KK, Res.unsqueeze(-1)).squeeze(-1)
                q_candidate = q_old - K0Res
                q = torch.where(active_mask.unsqueeze(-1), q_candidate, q_old)

            Ecoul_old = Ecoul

            Cq = torch.bmm(C, q.unsqueeze(-1)).squeeze(-1)  # (B,N)
            Ecoul = 0.5 * torch.sum(q * Cq, dim=-1) + 0.5 * torch.sum(
                q**2 * Hubbard_U, dim=1
            )

            # Third-order energy contribution
            if thirdorder_batch is not None:
                Ecoul = Ecoul + thirdorder_batch.get_energy(q)
            elif dU_dq is not None:
                Ecoul = Ecoul + (1.0 / 3.0) * torch.sum(0.5 * dU_dq * q**3, dim=1)

            dEc = torch.abs(Ecoul_old - Ecoul)

            for b, (rval, dval) in enumerate(zip(ResNorm.tolist(), dEc.tolist())):
                if _lib_out:
                    print(f"Batch {b}: Res = {rval:.3e}, dEc = {dval:.3e}")
            # print(f"t = {elapsed:.2f} s")

            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        # How the loop ended, for the batch as a whole.  This loop's
        # while-condition tests ``.any()`` across the batch, so it stops on
        # tolerance only when *every* member has met it -- the count that comes
        # back therefore describes the batch, never one structure within it, and
        # ``-1`` means the cap ran out with at least one member still moving.
        _converged = bool((ResNorm <= scf_tol).all()) and bool(
            (dEc <= scf_tol * 100).all()
        )
        scf_iter_count = int(it) if _converged else -1

        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.transpose(-1, -2)))

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 2 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
    q = -1.0 * Znuc
    q.scatter_add_(
        1, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    Ecoul, forces1, dq_p1 = None, None, None

    return (
        H,
        Hcoul,
        Hdipole,
        KK,
        D,
        Q,
        e,
        q,
        f,
        mu0,
        Ecoul,
        forces1,
        dq_p1,
        scf_iter_count,
    )


def delta_scf_x_os(
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
    dU_dq: Optional[torch.Tensor],
    D0: Optional[torch.Tensor],
    H0: torch.Tensor,
    H: torch.Tensor,
    D: torch.Tensor,
    f: torch.Tensor,
    mu0: torch.Tensor,
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
    torch.Tensor,  # Q (eigenvectors of the orthogonalized Hamiltonian)
    torch.Tensor,  # q_spin_atom
    torch.Tensor,  # q_tot_atom
    torch.Tensor,  # q_spin_sr (shell-resolved)
    torch.Tensor,  # net_spin_sr (shell-resolved)
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
    int,  # scf_iter_count: pass count on success, -1 if the loop gave up
]:
    """Delta-SCF loop for an excited state, on top of a converged ground state.

    The last returned value, ``scf_iter_count``, is how the loop ended stated as
    a number rather than as printed text: the number of passes taken on success,
    always ``>= 1``, or the literal ``-1`` when the loop exhausted
    ``SCF_MAX_ITER`` with a tolerance still unmet. A ``-1`` still comes back
    alongside the last iterate; the loop warns and returns, it never raises.
    """
    # D-02: one read of VERBOSE_LIBRARY_OUTPUT for the whole call, rather
    # than a dict lookup per print. Defaults to True, so a caller who never
    # sets the key sees exactly the output they saw before it existed.
    _lib_out = library_output_enabled(dftorch_params)
    if _lib_out:
        print("### Do Delta_scf ###")

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
    if dU_dq is not None:
        dU_dq_gathered = dU_dq[atom_ids]
    else:
        dU_dq_gathered = None

    normalize_coulomb_settings(dftorch_params, lattice_vecs, context="delta_scf_x_os")
    coulomb_cutoff = dftorch_params.get("COULOMB_CUTOFF", 10.0)
    if dftorch_params["COUL_METHOD"] == "PME":
        from .ewald_pme import (
            calculate_alpha_and_num_grids,
            calculate_PME_ewald,
            init_PME_data,
        )
        from .ewald_pme.neighbor_list import NeighborState

        # positions = torch.stack((RX, RY, RZ))
        positions = torch.stack(
            (RX, RY, RZ),
        )
        CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
            lattice_vecs.cpu().numpy(),
            coulomb_cutoff,
            dftorch_params.get("COULOMB_ACC", 1e-5),
        )
        PME_data = init_PME_data(
            grid_dimensions, lattice_vecs, CALPHA, dftorch_params.get("PME_ORDER", 4)
        )
        nbr_state = NeighborState(
            positions,
            lattice_vecs,
            None,
            coulomb_cutoff,
            is_dense=True,
            buffer=0.0,
            use_triton=False,
        )
        disps, dists, nbr_inds = calculate_dist_dips(
            positions, nbr_state, coulomb_cutoff
        )
    else:
        PME_data = None
        nbr_inds = None
        disps = None
        dists = None
        CALPHA = None

    _no_grad_ctx = (
        contextlib.nullcontext()
        if dftorch_params.get("SCF_GRAD", False)
        else torch.no_grad()
    )
    with _no_grad_ctx:
        # if 1:
        # Initial density matrix
        if _lib_out:
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
        broken_symmetry = dftorch_params.get("BROKEN_SYM", False)  # noqa: F841
        mu_0 = mu0
        ES_config = dftorch_params.get("DELTA_SCF_TARGET", "")
        ES_smearing = dftorch_params.get("DELTA_SCF_SMEARING", False)

        Dorth, Q, e, f, mu0 = nonaufbau_constraints(
            Z.T @ H @ Z,
            Te,
            Nocc,
            mu_0,
            ES_config,
            ES_smearing,
        )

        D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
        DS = 1 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)

        q_spin_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
        q_spin_sr.scatter_add_(
            1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS
        )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        # break spin symmetry on q_spin_sr, not on density matrix
        # if broken_symmetry:
        #     # shift spin density: add electrons to alpha, remove from beta
        #     # on shells belonging to atom with highest Znuc (most polarizable)
        #     most_polarizable_atom = Znuc.argmax()
        #     atom_shells = (shell_to_atom == most_polarizable_atom).nonzero().squeeze()
        #     delta = 0.01
        #     q_spin_sr[0, atom_shells] += delta   # alpha gets more
        #     q_spin_sr[1, atom_shells] -= delta   # beta gets less
        #     print(f"  Broken symmetry: perturbed shells {atom_shells.tolist()} "
        #           f"on atom {most_polarizable_atom.item()} (Znuc={Znuc[most_polarizable_atom].item()})")

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

        ResNorm = torch.tensor(2.0, device=device)
        dEc = torch.tensor(1000.0, device=device)
        it = 0
        Ecoul = torch.tensor(0.0, device=device)

        if _lib_out:
            print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
            if _lib_out:
                print("Iter {}".format(it))

            if dftorch_params["COUL_METHOD"] == "PME":
                # with torch.enable_grad():
                if 1:
                    ewald_e1, forces1, CoulPot = calculate_PME_ewald(
                        positions,
                        q_tot_atom,
                        lattice_vecs,
                        nbr_inds,
                        disps,
                        dists,
                        CALPHA,
                        coulomb_cutoff,
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
                dU_dq_gathered,
                dftorch_params.get("SHARED_MU", False),
                dftorch_params["DELTA_SCF"],
                dftorch_params,
            )

            q_spin_atom = torch.zeros_like(RX.unsqueeze(0).expand(2, -1))
            q_spin_atom.scatter_add_(
                1, shell_to_atom.unsqueeze(0).expand(2, -1), q_spin_sr
            )  # atom-resolved

            Res = q_spin_sr - q_spin_sr_old
            ResNorm = torch.norm(Res)
            K0Res = torch.bmm(KK, Res.unsqueeze(-1)).squeeze(-1)

            if it == dftorch_params.get(
                "KRYLOV_START", 10
            ):  # Calculate full kernel after KRYLOV_START steps
                # KK,D0 = _kernel_fermi(structure, mu0,Te,Nats,H,C,S,Z,Q,e)
                # KK = torch.load("/home/maxim/Projects/DFTB/DFTorch/tests/KK_C840.pt") # For testing purposes
                # KK0 = KK.clone()  # To be kept as preconditioner
                1
            # Preconditioned Low-Rank Krylov _scf acceleration
            if it > dftorch_params.get("KRYLOV_START", 10):
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
                    dftorch_params.get("KRYLOV_TOL", 1e-6),
                    KK,
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
                    dU_dq,
                )

            # Mixing update (vector-form)
            q_spin_sr = q_spin_sr_old - K0Res
            # q_tot_sr = q_spin_sr.sum(dim=0)
            q_tot_atom = torch.zeros_like(RX)
            q_tot_atom.scatter_add_(
                0, shell_to_atom, q_spin_sr.sum(dim=0)
            )  # atom-resolved
            net_spin_sr = q_spin_sr[0] - q_spin_sr[1]

            Ecoul_old = Ecoul
            if dftorch_params["COUL_METHOD"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q_tot_atom**2 * Hubbard_U)
            else:
                Ecoul = 0.5 * q_tot_atom @ (C @ q_tot_atom) + 0.5 * torch.sum(
                    q_tot_atom**2 * Hubbard_U
                )

            dEc = torch.abs(Ecoul_old - Ecoul)

            # print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(ResNorm.item(), dEb.item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            if _lib_out:
                print(
                    "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                        ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                    )
                )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        # How the loop ended, as a number a caller can read rather than as
        # printed text.  The two clauses are the negation of this loop's own
        # while-condition tolerance tests, read from it rather than copied from
        # a sibling loop.  See ``SCFx`` for the full reasoning.
        _scf_tol = dftorch_params.get("SCF_TOL", 1e-6)
        _converged = bool(ResNorm <= _scf_tol) and bool(dEc <= _scf_tol * 100)
        scf_iter_count = int(it) if _converged else -1

        # f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.transpose(-1, -2))) # supersedes non-aufbau contraint if calculated, at least in appearance

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

    if dftorch_params["COUL_METHOD"] == "PME":
        ewald_e1, forces1, dq_p1 = calculate_PME_ewald(
            positions,  # .detach().clone(),
            q_tot_atom,
            lattice_vecs,
            nbr_inds,
            disps,
            dists,
            CALPHA,
            coulomb_cutoff,
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
        scf_iter_count,
    )
