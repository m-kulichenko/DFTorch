import torch

from collections import deque


from ._dm_fermi_x import (
    dm_fermi_x,
    dm_fermi_x_os,
    dm_fermi_x_batch,
    nonaufbau_constraints,
)

# from ._kernel_fermi import _kernel_fermi
from ._tools import calculate_dist_dips, normalize_coulomb_settings
from ._xl_tools import (
    kernel_update_lr,
    kernel_update_lr_os,
    kernel_update_lr_batch,
    calc_q,
    calc_q_os,
    calc_q_batch,
)

from ._spin import get_h_spin


import time
from typing import Optional, Dict, Any, Tuple


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
            # Batched: R is (m, B, N) → per-batch overlap (B, m, m)
            # A[b, i, j] = R[i, b, :] · R[j, b, :]
            A = torch.einsum("ibn,jbn->bij", R, R)  # (B, m, m)

            # Tikhonov regularization per batch element
            reg_eye = 1e-12 * torch.eye(m, device=q_in.device, dtype=q_in.dtype)
            A_diag_max = A.diagonal(dim1=-2, dim2=-1).max(dim=-1).values  # (B,)
            A = A + reg_eye.unsqueeze(0) * A_diag_max.clamp(min=1e-30).unsqueeze(
                -1
            ).unsqueeze(-1)

            # Bordered system per batch element: (B, m+1, m+1)
            Bmat = torch.zeros(
                q_in.shape[0], m + 1, m + 1, device=q_in.device, dtype=q_in.dtype
            )
            Bmat[:, :m, :m] = A
            Bmat[:, :m, m] = 1.0
            Bmat[:, m, :m] = 1.0
            rhs = torch.zeros(
                q_in.shape[0], m + 1, device=q_in.device, dtype=q_in.dtype
            )
            rhs[:, m] = 1.0

            try:
                sol = torch.linalg.solve(Bmat, rhs)  # (B, m+1)
            except torch.linalg.LinAlgError:
                return q_in + self.alpha * residual

            c = sol[:, :m]  # (B, m)

            # Extrapolated update per batch: Σ_i c[b,i] * (Q[i,b,:] + α R[i,b,:])
            Q = torch.stack(list(self._q_hist))  # (m, B, N)
            mixed = Q + self.alpha * R  # (m, B, N)
            # q_new[b, n] = Σ_i c[b, i] * mixed[i, b, n]
            q_new = torch.einsum("bi,ibn->bn", c, mixed)
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
    print("### Do _scf ###")

    device = H0.device
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=H0.device), n_orbitals_per_atom
    )  # Generate atom index for each orbital

    Hubbard_U_gathered = Hubbard_U[atom_ids]
    if dU_dq is not None:
        dU_dq_gathered = dU_dq[atom_ids]
    else:
        dU_dq_gathered = None

    normalize_coulomb_settings(dftorch_params, cell, context="SCFx")
    coulomb_cutoff = dftorch_params.get("COULOMB_CUTOFF", 10.0)
    if dftorch_params["COUL_METHOD"] == "PME":
        from .ewald_pme import (
            calculate_PME_ewald,
            init_PME_data,
            calculate_alpha_and_num_grids,
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

    with torch.no_grad():
        # if 1:

        # Initial density matrix
        print("  Initial dm_fermi")

        Hdipole = torch.diag(
            -RX[atom_ids] * Efield[0]
            - RY[atom_ids] * Efield[1]
            - RZ[atom_ids] * Efield[2]
        )
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole

        if q_init is None:
            Dorth, Q, e, f, mu0 = dm_fermi_x(
                Z.T @ H0 @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
            )

            print("  Initial mu = {:.4f}".format(mu0.item()))

            D = Z @ Dorth @ Z.T
            DS = 2 * torch.diag(D @ S)
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

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
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

            q, H, Hcoul, D, Dorth, Q, e, f, mu0 = calc_q(
                H0,
                Hubbard_U_gathered,
                q[atom_ids],
                CoulPot[atom_ids],
                S,
                Z,
                Te,
                Nocc,
                Znuc,
                atom_ids,
                dU_dq_gathered if thirdorder is None else None,
            )
            Res = q - q_old
            ResNorm = torch.norm(Res)

            # --- Charge mixing ---
            use_krylov = it > dftorch_params.get("KRYLOV_START", 10)

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
                q = _mixer.mix(q_old, Res)
            else:
                # Simple linear mixing fallback
                K0Res = KK @ Res
                q = q_old - K0Res

            Ecoul_old = Ecoul
            if dftorch_params["COUL_METHOD"] == "PME":
                Ecoul = ewald_e1 + 0.5 * torch.sum(q**2 * Hubbard_U)
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
            print(
                "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                )
            )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        # f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))

    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)
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

    return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1, stress_coul


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
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """ """
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
            calculate_PME_ewald,
            init_PME_data,
            calculate_alpha_and_num_grids,
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

    with torch.no_grad():
        # if 1:
        # Initial density matrix
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

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
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
            print(
                "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                )
            )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

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
]:
    """
    Self-consistent field (_scf) cycle with finite electronic temperature and
    Fermi–Dirac occupations, using a preconditioned low-rank Krylov charge mixer.
    Supports PME Ewald electrostatics via `sedacs` or a direct Coulomb matrix.
    """

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

    # with torch.no_grad():
    if 1:
        RX_gathered = RX.gather(1, atom_ids)
        RY_gathered = RY.gather(1, atom_ids)
        RZ_gathered = RZ.gather(1, atom_ids)
        Hdipole = torch.diag_embed(
            -RX_gathered * Efield[0] - RY_gathered * Efield[1] - RZ_gathered * Efield[2]
        )
        Hdipole = 0.5 * (torch.matmul(Hdipole, S) + torch.matmul(S, Hdipole))
        H0 = H0 + Hdipole

        if q_init is None:
            H_ortho = torch.matmul(Z.transpose(-1, -2), torch.matmul(H0, Z))
            Dorth, Q, e, f, mu0 = dm_fermi_x_batch(
                H_ortho, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50
            )
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
            _mixer = _AndersonMixer(
                alpha=anderson_alpha,
                depth=anderson_depth,
            )
            _mixer._batch_mode = True  # per-batch-element DIIS
        else:
            _mixer = None

        ResNorm = torch.zeros(batch_size, device=device) + 2.0  # float("inf")
        dEc = torch.zeros(batch_size, device=device) + 1000.0  # float("inf")
        it = 0
        Ecoul = torch.zeros(batch_size, device=device) + 0.0  # float("inf")

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6)).any()
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100).any()
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            it += 1
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
            )
            Res = q - q_old
            ResNorm = torch.norm(Res, dim=1)

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
                q = q_old - K0Res
            elif _mixer is not None:
                # Anderson / DIIS mixing (pre-Krylov)
                q = _mixer.mix(q_old, Res)
            else:
                # Simple linear mixing fallback
                K0Res = torch.matmul(KK, Res.unsqueeze(-1)).squeeze(-1)
                q = q_old - K0Res

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
                print(f"Batch {b}: Res = {rval:.3e}, dEc = {dval:.3e}")
            # print(f"t = {elapsed:.2f} s")

            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.transpose(-1, -2)))

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 2 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)
    q = -1.0 * Znuc
    q.scatter_add_(
        1, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    Ecoul, forces1, dq_p1 = None, None, None

    return H, Hcoul, Hdipole, KK, D, Q, e, q, f, mu0, Ecoul, forces1, dq_p1


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
    torch.Tensor,  # q
    torch.Tensor,  # f
    torch.Tensor,  # mu0
    Optional[torch.Tensor],  # Ecoul (PME only)
    Optional[torch.Tensor],  # forces1 (PME only)
    Optional[torch.Tensor],  # dq_p1 (PME only)
]:
    """ """
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
            calculate_PME_ewald,
            init_PME_data,
            calculate_alpha_and_num_grids,
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

    with torch.no_grad():
        # if 1:
        # Initial density matrix
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

        print("\nStarting cycle")
        while (
            (ResNorm > dftorch_params.get("SCF_TOL", 1e-6))
            or (dEc > dftorch_params.get("SCF_TOL", 1e-6) * 100)
        ) and it < dftorch_params.get("SCF_MAX_ITER", 100):
            start_time = time.perf_counter()
            it += 1
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
            print(
                "Res = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(
                    ResNorm.item(), dEc.item(), time.perf_counter() - start_time
                )
            )
            if it == dftorch_params.get("SCF_MAX_ITER", 100):
                print("Did not converge")

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
    )
