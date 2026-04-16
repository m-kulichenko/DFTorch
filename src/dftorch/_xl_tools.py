import torch


from ._fermi_prt import (
    fermi_prt_D1_only,
    fermi_prt_batch_D1_only,
)  # noqa: F401
from ._dm_fermi_x import (
    dm_fermi_x,
    dm_fermi_x_os,  # noqa: F401
    dm_fermi_x_batch,
    dm_fermi_x_os_shared,
    nonaufbau_constraints,
)
from ._spin import get_h_spin_diag
from typing import Any, Optional, Tuple, Dict


@torch.compile(fullgraph=False, dynamic=False)
def calc_q(
    H0: torch.Tensor,
    U: torch.Tensor,
    n: torch.Tensor,
    CoulPot: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Te: float,
    Nocc: int,
    Znuc: torch.Tensor,
    atom_ids: torch.Tensor,
    dU_dq: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build _scf quantities from AO-level inputs and return updated atomic charges.
    Parameters
    ----------
    H0 : (n_orb, n_orb) torch.Tensor
        One-electron Hamiltonian in AO basis.
    U : (n_orb,) torch.Tensor
        AO-mapped on-site Hubbard U (U[atom_ids]).
    n : (n_orb,) torch.Tensor
        AO-mapped atomic charges (q[atom_ids]) used to form the Coulomb diagonal.
    CoulPot : (n_orb,) torch.Tensor
        AO-level Coulomb potential per orbital (from PME or direct).
    S : (n_orb, n_orb) torch.Tensor
        AO overlap matrix.
    Z : (n_orb, n_orb) torch.Tensor
        Symmetric orthogonalizer S^(-1/2), approximately satisfying Z.T @ S @ Z ≈ I.
        Used to enter/leave the orthogonal representation for dm_fermi_x.
    Te : float
        Electronic temperature.
    Nocc : int
        Number of occupied pairs.
    Znuc : (Nats,) torch.Tensor
        Nuclear charges per atom.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
    dU_dq : Optional[(n_orb,) torch.Tensor]
        AO-mapped derivative of Hubbard U with respect to atomic charge, used to form the Coulomb diagonal. If None, dU_dq is ignored and not included in the Coulomb diagonal.
    Returns
    -------
    q : (Nats,) torch.Tensor
        Updated atomic charges.
    H : (n_orb, n_orb) torch.Tensor
        Total Hamiltonian H0 + Hcoul.
    Hcoul : (n_orb, n_orb) torch.Tensor
        Coulomb contribution constructed from U, n, and CoulPot.
    D : (n_orb, n_orb) torch.Tensor
        Density matrix in AO basis.
    Dorth : (n_orb, n_orb) torch.Tensor
        Density matrix in orthogonal basis.
    Q : (n_orb, n_orb) torch.Tensor
        Eigenvectors of the orthogonal Hamiltonian (from dm_fermi_x).
    e : (n_orb,) torch.Tensor
        Eigenvalues of the orthogonal Hamiltonian (from dm_fermi_x).
    f : (n_orb,) torch.Tensor
        Fermi–Dirac occupations (from dm_fermi_x).
    mu0 : () torch.Tensor
        Chemical potential.
    """
    Hcoul_diag = U * n + CoulPot
    # ── DFTB3: add (1/2) * dU/dq * q^2 to diagonal ───────────────────────
    # This comes from d/dq [(1/3) dU/dq * q^3] = dU/dq * q^2,
    # and the factor 1/2 is from symmetrization of the Hamiltonian
    if dU_dq is not None:
        Hcoul_diag = Hcoul_diag + 0.5 * dU_dq * n**2
    # ─────────────────────────────────────────────────────────────────────

    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
    H = H0 + Hcoul

    Dorth, Q, e, f, mu0 = dm_fermi_x(
        (Z.T @ H @ Z), Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False
    )
    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
    q = -1.0 * Znuc
    q.scatter_add_(
        0, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    return q, H, Hcoul, D, Dorth, Q, e, f, mu0


@torch.compile(fullgraph=False, dynamic=False)
def calc_q_os(
    H0: torch.Tensor,
    H_spin: torch.Tensor,
    U: torch.Tensor,
    n: torch.Tensor,
    CoulPot: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Te: float,
    Nocc: int,
    Znuc: torch.Tensor,
    atom_ids: torch.Tensor,
    atom_ids_sr: torch.Tensor,
    el_per_shell: torch.Tensor,
    dU_dq: Optional[torch.Tensor] = None,
    shared_mu: bool = False,
    deltaSCF: bool = False,
    dftorch_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Open-shell version of calc_q.
    ----------
    """
    Hcoul_diag = U * n + CoulPot
    if dU_dq is not None:
        Hcoul_diag = Hcoul_diag + 0.5 * dU_dq * n**2
    # ─────────────────────────────────────────────────────────────────────

    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
    H = (
        H0
        + Hcoul
        + 0.5
        * S
        * H_spin.unsqueeze(0).expand(2, -1, -1)
        * torch.tensor([[[1]], [[-1]]], device=H_spin.device)
    )

    if shared_mu:
        Dorth, Q, e, f, mu0 = dm_fermi_x_os_shared(
            Z.T @ H @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False
        )
    else:
        Dorth, Q, e, f, mu0 = dm_fermi_x_os(
            Z.T @ H @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False
        )
    # print(mu0)

    if deltaSCF:
        mu_0 = mu0
        Dorth, Q, e, f, mu0 = nonaufbau_constraints(
            Z.T @ H @ Z,
            Te,
            Nocc,
            mu_0,
            dftorch_params["DELTA_SCF_TARGET"],
            dftorch_params["DELTA_SCF_SMEARING"],
        )

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = (D * S.transpose(-1, -2)).sum(dim=-1)  # O(n²) diagonal extraction

    q_sr = -0.5 * el_per_shell.unsqueeze(0).expand(2, -1)
    q_sr.scatter_add_(
        1, atom_ids_sr.unsqueeze(0).expand(2, -1), DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    return q_sr, H, Hcoul, D, Dorth, Q, e, f, mu0


@torch.compile(fullgraph=False, dynamic=False)
def calc_q_batch(
    H0: torch.Tensor,
    U: torch.Tensor,
    n: torch.Tensor,
    CoulPot: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Te: float,
    Nocc: int,
    Znuc: torch.Tensor,
    atom_ids: torch.Tensor,
    dU_dq: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build _scf quantities from AO-level inputs and return updated atomic charges.
    Parameters
    ----------
    """
    Hcoul_diag = U * n + CoulPot

    # ── DFTB3: add (1/2) * dU/dq * q^2 to diagonal ───────────────────────
    # This comes from d/dq [(1/3) dU/dq * q^3] = dU/dq * q^2,
    # and the factor 1/2 is from symmetrization of the Hamiltonian
    if dU_dq is not None:
        Hcoul_diag = Hcoul_diag + 0.5 * dU_dq * n**2
    # ─────────────────────────────────────────────────────────────────────

    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(-1) * S + S * Hcoul_diag.unsqueeze(-2))
    H = H0 + Hcoul
    H_ortho = torch.matmul(Z.transpose(-1, -2), torch.matmul(H, Z))
    Dorth, Q, e, f, mu0 = dm_fermi_x_batch(
        H_ortho, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False
    )
    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 2 * (D * S.transpose(-1, -2)).sum(dim=2)  # same as DS = 2 * torch.diag(D @ S)
    q = -1.0 * Znuc
    q.scatter_add_(
        1, atom_ids, DS
    )  # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    return q, H, Hcoul, D, Dorth, Q, e, f, mu0


@torch.compile(fullgraph=False, dynamic=False)
def calc_dq(
    U: torch.Tensor,
    v: torch.Tensor,
    d_CoulPot: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Te: float,
    Q: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    Nats: int,
    atom_ids: torch.Tensor,
    dU_dq: Optional[torch.Tensor] = None,
    n: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Linear charge response dq for a perturbation direction v in atomic charge space.

    Parameters
    ----------
    U : (n_orb,) torch.Tensor
        AO-mapped on-site Hubbard U (U[atom_ids]).
    v : (Nats,) torch.Tensor
        Atomic direction vector (e.g., Krylov search direction).
    d_CoulPot : (n_orb,) torch.Tensor
        AO-level change in Coulomb potential induced by v (e.g., from PME).
    S : (n_orb, n_orb) torch.Tensor
        AO overlap matrix.
    Z : (n_orb, n_orb) torch.Tensor
        Symmetric orthogonalizer S^(-1/2), with Z.T @ S @ Z ≈ I.
    Te : float
        Electronic temperature.
    Q : (n_orb, n_orb) torch.Tensor
        Orthogonal eigenvectors used by fermi_prt.
    e : (n_orb,) torch.Tensor
        Orthogonal eigenvalues used by fermi_prt.
    mu0 : () torch.Tensor
        Chemical potential corresponding to (Q, e).
    Nats : int
        Number of atoms.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
    dU_dq : Optional[(n_orb,) torch.Tensor]
        Change in Hubbard U with respect to charge. If not None, included in the Coulomb diagonal response as dU_dq * n_gathered * v, where n_gathered is the AO-mapped atomic charge (n[atom_ids]) gathered from the orthogonal density matrix response. This term captures how changes in atomic charge affect the Hubbard U contribution to the Hamiltonian, and thus the overall charge response.
    n : Optional[(n_orb,) torch.Tensor]
        AO-mapped atomic charge (n[atom_ids]) gathered from the orthogonal density matrix response.
    Returns
    -------
    dq : (Nats,) torch.Tensor
        Atomic charge response to the perturbation v.
    """
    d_Hcoul_diag = U * v + d_CoulPot

    # ── DFTB3: linearized response of third-order term ────────────────────
    # d/dlambda [(1/2) dU/dq * (q + lambda*v)^2]|_{lambda=0} = dU/dq * q * v
    if dU_dq is not None:
        d_Hcoul_diag = d_Hcoul_diag + dU_dq * n * v
    # ─────────────────────────────────────────────────────────────────────

    d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0))

    H1_orth = Z.T @ d_Hcoul @ Z
    # First-order density response (D1 only — skip D0 to save 2 matmuls)
    D1 = fermi_prt_D1_only(H1_orth, Te, Q, e, mu0)
    D1 = Z @ D1 @ Z.T
    # O(n²) diagonal extraction instead of O(n³) full matmul
    D1S = 2 * (D1 * S.T).sum(dim=1)
    # dq (atomic) from AO response
    dq = torch.zeros(Nats, dtype=S.dtype, device=S.device)
    dq.scatter_add_(0, atom_ids, D1S)
    return dq


@torch.compile(fullgraph=False, dynamic=False)
def calc_dq_batch(
    U: torch.Tensor,
    v: torch.Tensor,
    d_CoulPot: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    Te: float,
    Q: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    Nats: int,
    atom_ids: torch.Tensor,
    dU_dq: Optional[torch.Tensor] = None,
    n: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Linear charge response dq for a perturbation direction v in atomic charge space.
    Returns
    -------
    dq : (Nats,) torch.Tensor
        Atomic charge response to the perturbation v.
    """
    batch_size = U.shape[0]
    d_Hcoul_diag = U * v + d_CoulPot
    # ── DFTB3: linearized response of third-order term ────────────────────
    # d/dlambda [(1/2) dU/dq * (q + lambda*v)^2]|_{lambda=0} = dU/dq * q * v
    if dU_dq is not None:
        d_Hcoul_diag = d_Hcoul_diag + dU_dq * n * v
    # ─────────────────────────────────────────────────────────────────────

    d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(-1) * S + S * d_Hcoul_diag.unsqueeze(-2))

    H1_orth = torch.matmul(Z.transpose(-1, -2), torch.matmul(d_Hcoul, Z))
    # First-order density response (D1 only — skip D0 to save 2 matmuls)
    D1 = fermi_prt_batch_D1_only(H1_orth, Te, Q, e, mu0)
    D1 = torch.matmul(Z, torch.matmul(D1, Z.transpose(-1, -2)))
    # O(n²) diagonal extraction instead of O(n³) full matmul
    D1S = 2 * (D1 * S.transpose(-1, -2)).sum(dim=-1)
    # dq (atomic) from AO response
    dq = torch.zeros(batch_size, Nats, dtype=S.dtype, device=S.device)
    dq.scatter_add_(1, atom_ids, D1S)
    return dq


def kernel_update_lr(
    RX,
    RY,
    RZ,
    lattice_vecs: torch.Tensor,
    TYPE: torch.Tensor,
    n_atoms: int,
    Hubbard_U: torch.Tensor,
    dftorch_params: Any,
    FelTol: float,
    KK0: torch.Tensor,
    Res: torch.Tensor,
    q: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    PME_data: Any,
    atom_ids: torch.Tensor,
    Q: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    Te: float,
    C: torch.Tensor = None,
    nbr_inds: torch.Tensor = None,
    disps: torch.Tensor = None,
    dists: torch.Tensor = None,
    CALPHA: torch.Tensor = None,
    dU_dq: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for _scf charge mixing.

    Builds an orthonormal basis in charge space using the preconditioned residual,
    evaluates a small projected system to determine the optimal correction, and
    returns a step K0Res to update charges.

    Parameters
    ----------
    structure : object
        Must provide RX, RY, RZ, lattice_vecs, Hubbard_U, TYPE, Nats.
    KK0 : (Nats, Nats) torch.Tensor
        Preconditioner (e.g., initial mixing kernel).
    Res : (Nats,) torch.Tensor
        Current charge residual (q - q_old).
    q : (Nats,) torch.Tensor
        Current atomic charges.
    FelTol : float
        Residual norm tolerance in the Krylov subspace.
    S, Z : (n_orb, n_orb) torch.Tensor
        AO overlap and symmetric orthogonalizer S^(-1/2) (Z.T @ S @ Z ≈ I).
    nbr_inds, disps, dists, CALPHA, dftorch_params, PME_data : Any/torch.Tensor
        PME neighbor information and parameters for sedacs calls.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
    Q, e, mu0 : torch.Tensor
        Electronic structure data in orthogonal basis for first-order response.

    Returns
    -------
    K0Res : (Nats,) torch.Tensor
        Preconditioned low-rank correction step in charge space.
    """

    Hubbard_U_gathered = Hubbard_U[atom_ids]
    if dU_dq is not None:
        dU_dq_gathered = dU_dq[atom_ids]
    else:
        dU_dq_gathered = None

    vi = torch.zeros(n_atoms, dftorch_params["KRYLOV_MAXRANK"], device=S.device)
    fi = torch.zeros(n_atoms, dftorch_params["KRYLOV_MAXRANK"], device=S.device)

    # Preconditioned residual
    K0Res = KK0 @ Res
    dr = K0Res.clone()
    krylov_rank = 0
    Fel = torch.tensor(float("inf"), device=S.device)

    while (krylov_rank < dftorch_params["KRYLOV_MAXRANK"]) and (Fel > FelTol):
        # Normalize current direction
        norm_dr = torch.norm(dr)
        if norm_dr < 1e-9:
            print("zero norm_dr")
            break
        vi[:, krylov_rank] = dr / norm_dr

        # Modified Gram-Schmidt against previous vi
        if krylov_rank > 0:
            Vprev = vi[:, :krylov_rank]  # (Nats, krylov_rank)
            vi[:, krylov_rank] = vi[:, krylov_rank] - Vprev @ (
                Vprev.T @ vi[:, krylov_rank]
            )

        norm_vi = torch.norm(vi[:, krylov_rank])
        if norm_vi < 1e-9:
            print("zero norm_vi")
            break
        vi[:, krylov_rank] = vi[:, krylov_rank] / norm_vi
        v = vi[:, krylov_rank].clone()  # current search direction

        # dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
        if dftorch_params["coul_method"] == "PME":
            from .ewald_pme import calculate_PME_ewald

            # PME Coulomb case
            _, _, d_CoulPot = calculate_PME_ewald(
                torch.stack((RX, RY, RZ)),
                v,
                lattice_vecs,
                nbr_inds,
                disps,
                dists,
                CALPHA,
                dftorch_params["cutoff"],
                PME_data,
                hubbard_u=Hubbard_U,
                atomtypes=TYPE,
                screening=1,
                calculate_forces=0,
                calculate_dq=1,
                h_damp_exp=dftorch_params.get("h_damp_exp", None),
                h5_params=dftorch_params.get("h5_params", None),
            )
        else:  # Direct Coulomb case
            d_CoulPot = C @ v

        # Add GBSA Born shift response (linear in v)
        if gbsa is not None:
            d_CoulPot = d_CoulPot + gbsa.get_shifts(v)

        # Add full DFTB3 off-diagonal linearized shift response
        if thirdorder is not None:
            d_CoulPot = d_CoulPot + thirdorder.get_dshifts_dq(q, v)

        dq = calc_dq(
            Hubbard_U_gathered,
            v[atom_ids],
            d_CoulPot[atom_ids],
            S,
            Z,
            Te,
            Q,
            e,
            mu0,
            n_atoms,
            atom_ids,
            dU_dq_gathered,
            q[atom_ids],
        )

        # New residual (df/dlambda), preconditioned
        dr = dq - v
        dr = KK0 @ dr

        # Store fi column
        fi[:, krylov_rank] = dr

        # Small overlap O and RHS (vectorized)
        rank_m = krylov_rank + 1
        F_small = fi[:, :rank_m]  # (Nats, r)
        O = F_small.T @ F_small  # (r, r) # noqa: E741
        rhs = F_small.T @ K0Res  # (r,)

        # Solve O Y = rhs (stable) instead of explicit inverse
        Y = torch.linalg.solve(O, rhs)  # (r,)

        # Residual norm in the subspace
        Fel = torch.norm(F_small @ Y - K0Res)
        print("  rank: {:}, Fel = {:.6f}".format(krylov_rank, Fel.item()))
        krylov_rank += 1

    # If no Krylov steps were taken, return preconditioned residual
    if krylov_rank == 0:
        return K0Res

    # Combine correction: K0Res := V Y
    step = vi[:, :rank_m] @ Y

    # Optional trust region (kept commented)
    # base = torch.norm(KK0 @ Res)
    # sn   = torch.norm(step)
    # if sn > 1.25 * base and sn > 0:
    #     step = step * ((1.25 * base) / sn)

    K0Res = step  # (Nats,)
    return K0Res


def kernel_update_lr_os(
    RX,
    RY,
    RZ,
    lattice_vecs: torch.Tensor,
    TYPE: torch.Tensor,
    n_atoms: int,
    Hubbard_U: torch.Tensor,
    dftorch_params: Any,
    FelTol: float,
    KK0: torch.Tensor,
    Res: torch.Tensor,
    q: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    PME_data: Any,
    atom_ids: torch.Tensor,
    atom_ids_sr: torch.Tensor,
    Q: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    Te: float,
    w,
    n_shells_per_atom,
    shell_types,
    C: torch.Tensor = None,
    nbr_inds: torch.Tensor = None,
    disps: torch.Tensor = None,
    dists: torch.Tensor = None,
    CALPHA: torch.Tensor = None,
    dU_dq: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for _scf charge mixing.

    Builds an orthonormal basis in charge space using the preconditioned residual,
    evaluates a small projected system to determine the optimal correction, and
    returns a step K0Res to update charges.

    Parameters
    ----------
    structure : object
        Must provide RX, RY, RZ, lattice_vecs, Hubbard_U, TYPE, Nats.
    KK0 : (Nats, Nats) torch.Tensor
        Preconditioner (e.g., initial mixing kernel).
    Res : (Nats,) torch.Tensor
        Current charge residual (q - q_old).
    q : (Nats,) torch.Tensor
        Current atomic charges.
    FelTol : float
        Residual norm tolerance in the Krylov subspace.
    S, Z : (n_orb, n_orb) torch.Tensor
        AO overlap and symmetric orthogonalizer S^(-1/2) (Z.T @ S @ Z ≈ I).
    nbr_inds, disps, dists, CALPHA, dftorch_params, PME_data : Any/torch.Tensor
        PME neighbor information and parameters for sedacs calls.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
    Q, e, mu0 : torch.Tensor
        Electronic structure data in orthogonal basis for first-order response.
    gbsa : optional GBSA object
        If not None, adds solvation Born shift response to d_CoulPot.

    Returns
    -------
    K0Res : (Nats,) torch.Tensor
        Preconditioned low-rank correction step in charge space.
    """
    Nshells = n_shells_per_atom.sum().item()

    vi = torch.zeros(
        2, n_shells_per_atom.sum(), dftorch_params["KRYLOV_MAXRANK"], device=S.device
    )
    fi = torch.zeros(
        2, n_shells_per_atom.sum(), dftorch_params["KRYLOV_MAXRANK"], device=S.device
    )

    # Preconditioned residual
    K0Res = torch.bmm(KK0, Res.unsqueeze(-1)).squeeze(-1)
    dr = K0Res.clone()
    krylov_rank = 0
    Fel = torch.tensor(float("inf"), device=S.device)

    shell_to_atom = torch.repeat_interleave(
        torch.arange(len(TYPE), device=S.device), n_shells_per_atom
    )
    q_atomic = torch.zeros_like(RX)
    q_atomic.scatter_add_(0, shell_to_atom, q.sum(dim=0))

    spin_sign = torch.tensor([1.0, -1.0], device=S.device).view(2, 1)

    # ── Precompute fused eigenvector transform W = Z @ Q ─────────────────
    # Fuses the four-matmul chain  Q.T @ Z.T @ dH @ Z @ Q  into  W.T @ dH @ W
    # and the back-transform  Z @ Q @ X @ Q.T @ Z.T  into  W @ X @ W.T
    W = torch.matmul(Z.unsqueeze(0).expand(2, -1, -1), Q)  # (2, n_orb, n_orb)
    SW = torch.matmul(S.unsqueeze(0).expand(2, -1, -1), W)  # (2, n_orb, n_orb)

    # Susceptibility kernel χ_ij (depends only on eigenvalues, constant in Krylov loop)
    kB = 8.61739e-5
    beta = 1.0 / (kB * Te)
    fe = 1.0 / (torch.exp(beta * (e - mu0.unsqueeze(-1))) + 1.0)  # (2, n_orb)
    de = e[:, :, None] - e[:, None, :]  # (2, n_orb, n_orb)
    chi = torch.empty_like(de)
    off = de.abs() > 1e-12
    chi[off] = (fe[:, :, None] - fe[:, None, :])[off] / de[off]
    f_diag = -beta * fe * (1.0 - fe)  # (2, n_orb)
    chi[~off] = f_diag.unsqueeze(1).expand_as(de)[~off]
    f_diag_sum = f_diag.sum(dim=1)  # (2,) — for μ1 conservation
    # ─────────────────────────────────────────────────────────────────────

    while (krylov_rank < dftorch_params["KRYLOV_MAXRANK"]) and (Fel > FelTol):
        # Normalize current direction
        norm_dr = torch.norm(dr)
        if norm_dr < 1e-9:
            print("zero norm_dr")
            break

        vi[:, :, krylov_rank] = dr / norm_dr.unsqueeze(-1)

        # FIX: Gram-Schmidt in full (2*Nshells) space
        if krylov_rank > 0:
            Vprev = vi[:, :, :krylov_rank]  # (2, Nshells, rank)
            # dot product over BOTH dimensions 0 and 1 (full vector)
            # vi_flat: (2*Nshells,), Vprev_flat: (2*Nshells, rank)
            vi_flat = vi[:, :, krylov_rank].reshape(-1)  # (2*Nshells,)
            Vprev_flat = Vprev.reshape(-1, krylov_rank)  # (2*Nshells, rank)
            coeffs = Vprev_flat.T @ vi_flat  # (rank,)
            correction = (Vprev_flat @ coeffs).reshape(2, Nshells)  # (2, Nshells)
            vi[:, :, krylov_rank] = vi[:, :, krylov_rank] - correction

        norm_vi = torch.norm(vi[:, :, krylov_rank])  # scalar over full space
        if norm_vi < 1e-9:
            break
        vi[:, :, krylov_rank] = vi[:, :, krylov_rank] / norm_vi
        v = vi[:, :, krylov_rank].clone()

        v_atomic = torch.zeros_like(RX)
        v_atomic.scatter_add_(0, shell_to_atom, v.sum(dim=0))  # atom-resolved
        v_net_spin = v[0] - v[1]  # shell-resolved

        # dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
        if dftorch_params["coul_method"] == "PME":
            from .ewald_pme import calculate_PME_ewald

            # PME Coulomb case
            _, _, d_CoulPot = calculate_PME_ewald(
                torch.stack((RX, RY, RZ)),
                v_atomic,
                lattice_vecs,
                nbr_inds,
                disps,
                dists,
                CALPHA,
                dftorch_params["cutoff"],
                PME_data,
                hubbard_u=Hubbard_U,
                atomtypes=TYPE,
                screening=1,
                calculate_forces=0,
                calculate_dq=1,
                h_damp_exp=dftorch_params.get("h_damp_exp", None),
                h5_params=dftorch_params.get("h5_params", None),
            )
        else:  # Direct Coulomb case
            d_CoulPot = C @ v_atomic

        # Add GBSA Born shift response (linear in v)
        if gbsa is not None:
            d_CoulPot = d_CoulPot + gbsa.get_shifts(v_atomic)

        # ── Full DFTB3: linearized response of off-diagonal third-order term ──
        if thirdorder is not None:
            d_CoulPot = d_CoulPot + thirdorder.get_dshifts_dq(q_atomic, v_atomic)
        # ─────────────────────────────────────────────────────────────────────

        # Compute spin potential as a vector (avoid building full n_orb×n_orb H_spin matrix)
        mu_spin = get_h_spin_diag(TYPE, v_net_spin, w, n_shells_per_atom, shell_types)

        d_Hcoul_diag = Hubbard_U[atom_ids] * v_atomic[atom_ids] + d_CoulPot[atom_ids]
        # ── DFTB3: linearized response of diagonal third-order term ───────────
        # d/dlambda [(1/2) dU/dq * (q + lambda*v)^2]|_{lambda=0} = dU/dq * q * v
        # Skipped when full off-diagonal thirdorder is active (already included).
        if dU_dq is not None and thirdorder is None:
            d_Hcoul_diag = (
                d_Hcoul_diag + dU_dq[atom_ids] * q_atomic[atom_ids] * v_atomic[atom_ids]
            )
        # ─────────────────────────────────────────────────────────────────────

        # Combine Coulomb + spin into effective diagonal per spin channel
        # spin-up: d_Hcoul_diag + mu_spin,  spin-down: d_Hcoul_diag - mu_spin
        eff_diag = d_Hcoul_diag.unsqueeze(0) + spin_sign * mu_spin.unsqueeze(
            0
        )  # (2, n_orb)

        # ── Fused forward transform: W.T @ dH @ W via row-scaling ────────
        # dH = 0.5*(diag(d)@S + S@diag(d)), so W.T@dH@W = 0.5*(A + A.T)
        # where A = (d*W).T @ (S@W),  with S@W = SW precomputed
        Wd = eff_diag.unsqueeze(-1) * W  # (2, n_orb, n_orb) row-scaling O(n²)
        A = torch.matmul(Wd.transpose(-1, -2), SW)  # 1 batched matmul
        QH1Q = 0.5 * (A + A.transpose(-1, -2))

        # ── Response in eigenbasis (chi precomputed) ──────────────────────
        X = chi * QH1Q
        mu1_mask = (torch.abs(f_diag_sum) > 1e-12).to(S.dtype)
        mu1 = (
            X.diagonal(dim1=-2, dim2=-1).sum(dim=1) / (f_diag_sum + (1.0 - mu1_mask))
        ) * mu1_mask
        X = X - torch.diag_embed(f_diag) * mu1.unsqueeze(-1).unsqueeze(-1)

        # ── Fused backward: D1S = diag(W @ X @ W.T @ S) = row-dot(W@X, SW) ─
        WX = torch.matmul(W, X)  # 1 batched matmul
        D1S = (WX * SW).sum(dim=-1)  # (2, n_orb), O(n²)
        # dq (atomic) from AO response
        dq = torch.zeros(2, n_shells_per_atom.sum(), dtype=S.dtype, device=S.device)
        dq.scatter_add_(
            1, atom_ids_sr.unsqueeze(0).expand(2, -1), D1S
        )  # sums elements from D1S into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        # New residual (df/dlambda), preconditioned
        dr = dq - v
        dr = torch.bmm(KK0, dr.unsqueeze(-1)).squeeze(-1)
        fi[:, :, krylov_rank] = dr

        # FIX: O and rhs in full (2*Nshells) space
        rank_m = krylov_rank + 1
        F_small = fi[:, :, :rank_m]  # (2, Nshells, rank)
        F_flat = F_small.reshape(-1, rank_m)  # (2*Nshells, rank)
        K0Res_flat = K0Res.reshape(-1)  # (2*Nshells,)

        O = F_flat.T @ F_flat  # (rank, rank) # noqa: E741
        rhs = F_flat.T @ K0Res_flat  # (rank,)
        Y = torch.linalg.solve(O, rhs)  # (rank,)

        Fel = torch.norm(F_flat @ Y - K0Res_flat)
        print("  rank: {:}, Fel = {:.6f}".format(krylov_rank, Fel.item()))
        krylov_rank += 1

    if krylov_rank == 0:
        return K0Res

    rank_m = krylov_rank
    Vi_flat = vi[:, :, :rank_m].reshape(-1, rank_m)  # (2*Nshells, rank)
    step = (Vi_flat @ Y).reshape(2, Nshells)  # (2, Nshells)
    return step


def kernel_update_lr_batch(
    n_atoms: int,
    Hubbard_U_gathered: torch.Tensor,
    dftorch_params,
    FelTol,
    KK0: torch.Tensor,
    Res: torch.Tensor,
    q: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    PME_data: Any,
    atom_ids: torch.Tensor,
    Q: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    Te: float,
    C: torch.Tensor = None,
    nbr_inds: torch.Tensor = None,
    disps: torch.Tensor = None,
    dists: torch.Tensor = None,
    CALPHA: torch.Tensor = None,
    dU_dq_gathered: Optional[torch.Tensor] = None,
    gbsa=None,
    thirdorder=None,
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for _scf charge mixing.

    Builds an orthonormal basis in charge space using the preconditioned residual,
    evaluates a small projected system to determine the optimal correction, and
    returns a step K0Res to update charges.

    Parameters
    ----------
    structure : object
        Must provide RX, RY, RZ, lattice_vecs, Hubbard_U, TYPE, Nats.
    KK0 : (Nats, Nats) torch.Tensor
        Preconditioner (e.g., initial mixing kernel).
    Res : (Nats,) torch.Tensor
        Current charge residual (q - q_old).
    q : (Nats,) torch.Tensor
        Current atomic charges.
    S, Z : (n_orb, n_orb) torch.Tensor
        AO overlap and symmetric orthogonalizer S^(-1/2) (Z.T @ S @ Z ≈ I).
    nbr_inds, disps, dists, CALPHA, dftorch_params, PME_data : Any/torch.Tensor
        PME neighbor information and parameters for sedacs calls.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
    Q, e, mu0 : torch.Tensor
        Electronic structure data in orthogonal basis for first-order response.

    Returns
    -------
    K0Res : (Nats,) torch.Tensor
        Preconditioned low-rank correction step in charge space.
    """
    batch_size = q.shape[0]
    vi = torch.zeros(
        batch_size, n_atoms, dftorch_params["KRYLOV_MAXRANK"], device=S.device
    )
    fi = torch.zeros(
        batch_size, n_atoms, dftorch_params["KRYLOV_MAXRANK"], device=S.device
    )
    step = torch.zeros_like(q)

    # Preconditioned residual
    K0Res = torch.matmul(KK0, Res.unsqueeze(-1)).squeeze(-1)
    dr = K0Res.clone()
    krylov_rank = 0
    Fel_all = torch.tensor([float("inf")] * batch_size, device=q.device)

    active_mask = torch.tensor([True] * batch_size, dtype=torch.bool, device=q.device)

    while (krylov_rank < dftorch_params["KRYLOV_MAXRANK"]) and (Fel_all > FelTol).any():
        # Normalize current direction
        if krylov_rank > 0:
            norm_dr = torch.norm(fi[active_mask, :, krylov_rank - 1], dim=1)
            if (norm_dr < 1e-9).any():
                print("zero norm_dr")
                break
            vi[active_mask, :, krylov_rank] = fi[
                active_mask, :, krylov_rank - 1
            ] / norm_dr.unsqueeze(-1)
        else:
            norm_dr = torch.norm(dr, dim=1)
            if (norm_dr < 1e-9).any():
                print("zero norm_dr")
                break
            vi[active_mask, :, krylov_rank] = dr / norm_dr.unsqueeze(-1)

        # Modified Gram-Schmidt against previous vi
        if krylov_rank > 0:
            Vprev = vi[active_mask, :, :krylov_rank]  # (Nats, krylov_rank)
            coeffs = torch.bmm(
                Vprev.transpose(-1, -2), vi[active_mask, :, krylov_rank].unsqueeze(-1)
            )  # (B_active, krylov_rank, 1)
            vi[active_mask, :, krylov_rank] = vi[
                active_mask, :, krylov_rank
            ] - torch.bmm(Vprev, coeffs).squeeze(-1)

        norm_vi = torch.norm(vi[active_mask, :, krylov_rank], dim=1)
        if (norm_vi < 1e-9).any():
            print("zero norm_vi")
            break
        vi[active_mask, :, krylov_rank] = vi[
            active_mask, :, krylov_rank
        ] / norm_vi.unsqueeze(-1)
        v = vi[active_mask, :, krylov_rank].clone()  # current search direction

        # dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
        if dftorch_params["coul_method"] == "PME":
            # PME Coulomb case
            NotImplementedError(
                "PME Coulomb method not implemented in batch Krylov updater."
            )
        else:  # Direct Coulomb case
            d_CoulPot = torch.bmm(C[active_mask], v.unsqueeze(-1)).squeeze(-1)

        # Add GBSA Born shift response (linear in v, vectorised)
        if gbsa is not None:
            # gbsa is a GBSABatch object — extract active born_mat rows
            active_born = gbsa.born_mat[active_mask]  # (B_active, N, N)
            active_sasa = gbsa.sasa[active_mask]  # (B_active, N)
            hb = gbsa._hbond_strength  # (N,)
            shift_v = 2.0 * active_sasa * hb.unsqueeze(0) * v  # (B_active, N)
            shift_v = shift_v + torch.bmm(active_born, v.unsqueeze(-1)).squeeze(-1)
            d_CoulPot = d_CoulPot + shift_v

        # Add full DFTB3 off-diagonal linearized shift response
        if thirdorder is not None:
            # Subset gamma matrices for active batch elements
            active_g3ab = thirdorder.gamma3ab[active_mask]  # (B_active, N, N)
            active_g3ba = thirdorder.gamma3ba[active_mask]  # (B_active, N, N)
            q_active = q[active_mask]  # (B_active, N)
            g3ab_v = torch.bmm(active_g3ab, v.unsqueeze(-1)).squeeze(-1)
            g3ab_q = torch.bmm(active_g3ab, q_active.unsqueeze(-1)).squeeze(-1)
            dt1 = 2.0 * (g3ab_v * q_active + g3ab_q * v)
            dt2 = 2.0 * torch.bmm(active_g3ba, (q_active * v).unsqueeze(-1)).squeeze(-1)
            d_CoulPot = d_CoulPot + (1.0 / 3.0) * (dt1 + dt2)

        dq = calc_dq_batch(
            Hubbard_U_gathered[active_mask],
            v.gather(1, atom_ids[active_mask]),
            d_CoulPot.gather(1, atom_ids[active_mask]),
            S[active_mask],
            Z[active_mask],
            Te,
            Q[active_mask],
            e[active_mask],
            mu0[active_mask],
            n_atoms,
            atom_ids[active_mask],
            dU_dq_gathered[active_mask] if dU_dq_gathered is not None else None,
            q.gather(1, atom_ids[active_mask]),
        )

        # New residual (df/dlambda), preconditioned
        dr = dq - v
        dr = torch.bmm(KK0[active_mask], dr.unsqueeze(-1)).squeeze(-1)

        # Store fi column
        fi[active_mask, :, krylov_rank] = dr

        # Small overlap O and RHS (vectorized)
        rank_m = krylov_rank + 1
        F_small = fi[active_mask, :, :rank_m]
        O = torch.bmm(F_small.transpose(-1, -2), F_small)  # noqa: E741
        rhs = torch.bmm(
            F_small.transpose(-1, -2), K0Res[active_mask].unsqueeze(-1)
        ).squeeze(-1)

        Y = torch.linalg.solve(O, rhs)

        # Residual norm in the subspace
        proj = torch.bmm(F_small, Y.unsqueeze(-1)).squeeze(-1)
        Fel = torch.norm(proj - K0Res[active_mask], dim=1)

        # Keep latest Krylov step for all active systems
        step_active = torch.bmm(vi[active_mask, :, :rank_m], Y.unsqueeze(-1)).squeeze(
            -1
        )
        step[active_mask] = step_active

        Fel_all[active_mask] = Fel
        active_mask = Fel_all > FelTol

        fel_list = Fel_all.detach().cpu().tolist()
        for b, val in enumerate(fel_list):
            print(f"  rank: {krylov_rank}, batch {b}, Fel = {val:.6f}")
        print(f"  Not converged: {active_mask.sum()}")
        krylov_rank += 1

    # If no Krylov steps were taken, return preconditioned residual
    if krylov_rank == 0:
        return K0Res

    # Combine correction: K0Res := V Y
    # step = vi[:, :rank_m] @ Y

    # Optional trust region (kept commented)
    # base = torch.norm(KK0 @ Res)
    # sn   = torch.norm(step)
    # if sn > 1.25 * base and sn > 0:
    #     step = step * ((1.25 * base) / sn)

    K0Res = step  # (Nats,)
    return K0Res
