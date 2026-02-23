import torch

# from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from .ewald_pme import (
    calculate_PME_ewald,
)


from .Fermi_PRT import Canon_DM_PRT, Fermi_PRT, Fermi_PRT_batch  # noqa: F401
from .DM_Fermi_x import (
    DM_Fermi_x,
    dm_fermi_x_os,  # noqa: F401
    DM_Fermi_x_batch,
    dm_fermi_x_os_shared,
)
from .Spin import get_h_spin
from typing import Any, Tuple


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
    Build SCF quantities from AO-level inputs and return updated atomic charges.
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
        Used to enter/leave the orthogonal representation for DM_Fermi_x.
    Te : float
        Electronic temperature.
    Nocc : int
        Number of occupied pairs.
    Znuc : (Nats,) torch.Tensor
        Nuclear charges per atom.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.
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
        Eigenvectors of the orthogonal Hamiltonian (from DM_Fermi_x).
    e : (n_orb,) torch.Tensor
        Eigenvalues of the orthogonal Hamiltonian (from DM_Fermi_x).
    f : (n_orb,) torch.Tensor
        Fermi–Dirac occupations (from DM_Fermi_x).
    mu0 : () torch.Tensor
        Chemical potential.
    """
    Hcoul_diag = U * n + CoulPot
    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
    H = H0 + Hcoul

    # if H0.dtype == torch.float32:
    # 	Dorth, Q, e, f, mu0 = DM_Fermi_x((Z.T @ H @ Z).to(torch.float64), Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False)
    # 	D = Z.to(torch.float64) @ Dorth @ Z.T.to(torch.float64)
    # 	DS = 2 * (D * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
    # 	q = -1.0 * Znuc.to(torch.float64)
    # 	q.scatter_add_(0, atom_ids, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

    # 	Dorth = Dorth.to(torch.get_default_dtype())
    # 	Q = Q.to(torch.get_default_dtype())
    # 	e = e.to(torch.get_default_dtype())
    # 	D = D.to(torch.get_default_dtype())
    # 	q = q.to(torch.get_default_dtype())

    # else:
    Dorth, Q, e, f, mu0 = DM_Fermi_x(
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
    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
    H = (
        H0
        + Hcoul
        + 0.5
        * S
        * H_spin.unsqueeze(0).expand(2, -1, -1)
        * torch.tensor([[[1]], [[-1]]], device=H_spin.device)
    )

    # Dorth, Q, e, f, mu0 = dm_fermi_x_os(Z.T @ H @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False)
    Dorth, Q, e, f, mu0 = dm_fermi_x_os_shared(
        Z.T @ H @ Z, Te, Nocc, mu_0=None, eps=1e-9, MaxIt=50, debug=False
    )
    # print(mu0, mu0_)

    D = torch.matmul(Z, torch.matmul(Dorth, Z.transpose(-1, -2)))
    DS = 1 * torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1)

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
    Build SCF quantities from AO-level inputs and return updated atomic charges.
    Parameters
    ----------
    """
    Hcoul_diag = U * n + CoulPot
    Hcoul = 0.5 * (Hcoul_diag.unsqueeze(-1) * S + S * Hcoul_diag.unsqueeze(-2))
    H = H0 + Hcoul
    H_ortho = torch.matmul(Z.transpose(-1, -2), torch.matmul(H, Z))
    Dorth, Q, e, f, mu0 = DM_Fermi_x_batch(
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
        Orthogonal eigenvectors used by Fermi_PRT.
    e : (n_orb,) torch.Tensor
        Orthogonal eigenvalues used by Fermi_PRT.
    mu0 : () torch.Tensor
        Chemical potential corresponding to (Q, e).
    Nats : int
        Number of atoms.
    atom_ids : (n_orb,) torch.Tensor
        Map from AO index to atom index.

    Returns
    -------
    dq : (Nats,) torch.Tensor
        Atomic charge response to the perturbation v.
    """
    d_Hcoul_diag = U * v + d_CoulPot
    d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0))

    H1_orth = Z.T @ d_Hcoul @ Z
    # First-order density response (canonical Fermi PRT)
    # _, D1 = Canon_DM_PRT(H1_orth, structure.Te, Q, e, mu0, 10)
    _, D1 = Fermi_PRT(H1_orth, Te, Q, e, mu0)
    D1 = Z @ D1 @ Z.T
    D1S = 2 * torch.diag(D1 @ S)
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
    d_Hcoul = 0.5 * (d_Hcoul_diag.unsqueeze(-1) * S + S * d_Hcoul_diag.unsqueeze(-2))

    H1_orth = torch.matmul(Z.transpose(-1, -2), torch.matmul(d_Hcoul, Z))
    # First-order density response (canonical Fermi PRT)
    _, D1 = Fermi_PRT_batch(H1_orth, Te, Q, e, mu0)
    D1 = torch.matmul(Z, torch.matmul(D1, Z.transpose(-1, -2)))
    tmp = torch.matmul(D1, S)
    D1S = 2 * tmp.diagonal(dim1=-2, dim2=-1)
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
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for SCF charge mixing.

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
            )
        else:  # Direct Coulomb case
            d_CoulPot = C @ v

        dq = calc_dq(
            Hubbard_U[atom_ids],
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
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for SCF charge mixing.

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
    Fel = torch.tensor(float("inf"), dtype=q.dtype, device=q.device)

    while (krylov_rank < dftorch_params["KRYLOV_MAXRANK"]) and (Fel > FelTol):
        # Normalize current direction
        norm_dr = torch.norm(dr)
        if (norm_dr < 1e-9).any():
            print("zero norm_dr")
            break

        vi[:, :, krylov_rank] = dr / norm_dr.unsqueeze(-1)

        # Modified Gram-Schmidt against previous vi
        if krylov_rank > 0:
            Vprev = vi[:, :, :krylov_rank]  # (Nats, krylov_rank)
            coeffs = torch.bmm(
                Vprev.transpose(-1, -2), vi[:, :, krylov_rank].unsqueeze(-1)
            )  # (B_active, krylov_rank, 1)
            vi[:, :, krylov_rank] = vi[:, :, krylov_rank] - torch.bmm(
                Vprev, coeffs
            ).squeeze(-1)

        norm_vi = torch.norm(vi[:, :, krylov_rank])
        if (norm_vi < 1e-9).any():
            print("zero norm_vi")
            break
        vi[:, :, krylov_rank] = vi[:, :, krylov_rank] / norm_vi.unsqueeze(-1)
        v = vi[:, :, krylov_rank].clone()  # current search direction

        v_atomic = torch.zeros_like(RX)
        shell_to_atom = torch.repeat_interleave(
            torch.arange(len(TYPE), device=S.device), n_shells_per_atom
        )
        v_atomic.scatter_add_(0, shell_to_atom, v.sum(dim=0))  # atom-resolved
        v_net_spin = v[0] - v[1]  # shell-resolved

        # dHcoul from a unit step along v (atomic) mapped to AO via atom_ids
        if dftorch_params["coul_method"] == "PME":
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
            )
        else:  # Direct Coulomb case
            d_CoulPot = C @ v_atomic

        H_spin = get_h_spin(TYPE, v_net_spin, w, n_shells_per_atom, shell_types)
        d_Hcoul_diag = Hubbard_U[atom_ids] * v_atomic[atom_ids] + d_CoulPot[atom_ids]
        d_Hcoul = 0.5 * (
            d_Hcoul_diag.unsqueeze(1) * S + S * d_Hcoul_diag.unsqueeze(0)
        ) + 0.5 * S * H_spin.unsqueeze(0).expand(2, -1, -1) * torch.tensor(
            [[[1]], [[-1]]], device=H_spin.device
        )

        H1_orth = Z.T @ d_Hcoul @ Z
        # First-order density response (canonical Fermi PRT)
        # _, D1 = Canon_DM_PRT(H1_orth, structure.Te, Q, e, mu0, 10)
        _, D1 = Fermi_PRT_batch(H1_orth, Te, Q, e, mu0)
        D1 = Z @ D1 @ Z.T
        D1S = 1 * torch.diagonal(torch.matmul(D1, S), dim1=-2, dim2=-1)
        # dq (atomic) from AO response
        dq = torch.zeros(2, n_shells_per_atom.sum(), dtype=S.dtype, device=S.device)
        dq.scatter_add_(
            1, atom_ids_sr.unsqueeze(0).expand(2, -1), D1S
        )  # sums elements from D1S into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        # New residual (df/dlambda), preconditioned
        dr = dq - v
        dr = torch.bmm(KK0, dr.unsqueeze(-1)).squeeze(-1)

        # Store fi column
        fi[:, :, krylov_rank] = dr

        # Small overlap O and RHS (vectorized)
        rank_m = krylov_rank + 1
        F_small = fi[:, :, :rank_m]  # (Nats, r)
        O = torch.bmm(F_small.transpose(-1, -2), F_small)  # noqa: E741
        rhs = torch.bmm(F_small.transpose(-1, -2), K0Res.unsqueeze(-1)).squeeze(
            -1
        )  # (B, r)

        # Solve O Y = rhs (stable) instead of explicit inverse
        Y = torch.linalg.solve(O, rhs)  # (r,)

        # Residual norm in the subspace
        proj = torch.bmm(F_small, Y.unsqueeze(-1)).squeeze(-1)
        Fel = torch.norm(proj - K0Res)
        print("  rank: {:}, Fel = {:.6f}".format(krylov_rank, Fel.item()))
        krylov_rank += 1

    # If no Krylov steps were taken, return preconditioned residual
    if krylov_rank == 0:
        return K0Res

    # Combine correction: K0Res := V Y
    step = torch.bmm(vi[:, :, :rank_m], Y.unsqueeze(-1)).squeeze(-1)
    K0Res = step  # (Nats,)
    return K0Res


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
) -> torch.Tensor:
    """
    Preconditioned low-rank Krylov update for SCF charge mixing.

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
    Fel_all = torch.tensor([float("inf")] * batch_size, dtype=q.dtype, device=q.device)

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
        )

        # New residual (df/dlambda), preconditioned
        dr = dq - v
        dr = torch.bmm(KK0[active_mask], dr.unsqueeze(-1)).squeeze(-1)

        # Store fi column
        fi[active_mask, :, krylov_rank] = dr

        # Small overlap O and RHS (vectorized)
        rank_m = krylov_rank + 1
        F_small = fi[active_mask, :, :rank_m]  # (Nats, r)
        # O = F_small.transpose(-1,-2) @ F_small                  # (r, r)
        O = torch.bmm(F_small.transpose(-1, -2), F_small)  # noqa: E741
        rhs = torch.bmm(
            F_small.transpose(-1, -2), K0Res[active_mask].unsqueeze(-1)
        ).squeeze(-1)  # (B, r)

        # Solve O Y = rhs (stable) instead of explicit inverse
        Y = torch.linalg.solve(O, rhs)  # (r,)
        Y_all = torch.zeros(batch_size, rank_m, device=S.device, dtype=S.dtype)
        Y_all[active_mask] = Y

        # Residual norm in the subspace
        proj = torch.bmm(F_small, Y.unsqueeze(-1)).squeeze(-1)
        Fel = torch.norm(proj - K0Res[active_mask], dim=1)
        Fel_all[active_mask] = Fel
        active_mask_old = active_mask.clone()
        active_mask = Fel_all > FelTol

        cur_change_mask = active_mask_old != active_mask

        step[cur_change_mask] = torch.bmm(
            vi[cur_change_mask, :, :rank_m], Y_all[cur_change_mask].unsqueeze(-1)
        ).squeeze(-1)

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
