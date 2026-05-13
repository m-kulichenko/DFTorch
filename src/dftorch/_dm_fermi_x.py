from __future__ import annotations

from typing import Optional

import torch

from ._tools import _maybe_compile

# Pairs whose eigenvalue difference is smaller than this are treated as
# degenerate: the 1/(λᵢ−λⱼ) term in the eigenvector gradient is zeroed out
# to avoid NaN/inf when backpropagating through symmetric/padded systems.
DEGEN_THRESHOLD: float = float(torch.finfo(torch.float64).eps ** 0.6)  # ≈1.3e-10


class _degen_symeig(torch.autograd.Function):
    """Batched symmetric eigensolver with degenerate-safe backward.

    Forward: identical to ``torch.linalg.eigh``.

    Backward: the ill-posed ``1/(λᵢ−λⱼ)`` terms are **zeroed out** when
    ``|λᵢ−λⱼ| ≤ DEGEN_THRESHOLD``, preventing NaN/inf gradients through
    degenerate eigenspaces (e.g. symmetric molecules or padded orbitals).

    Handles both single ``(N, N)`` and batched ``(B, N, N)`` inputs.
    Based on M. F. Kasim, arXiv:2011.04366 (2020).
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor):
        eival, eivec = torch.linalg.eigh(A, UPLO="U")
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        eival, eivec = ctx.saved_tensors
        eivecT = eivec.transpose(-2, -1).conj()

        if grad_eivec is not None:
            # (..., N, N) difference matrix — works for both batched and single
            delta = eival.unsqueeze(-2) - eival.unsqueeze(-1)
            idx = torch.abs(delta) > DEGEN_THRESHOLD
            delta_inv = torch.zeros_like(delta)
            delta_inv[idx] = delta[idx].reciprocal()
            dC_proj = delta_inv * torch.matmul(eivecT, grad_eivec)
            CdCCT = torch.matmul(eivec, torch.matmul(dC_proj, eivecT))
        else:
            CdCCT = torch.zeros_like(eivec)

        dA = CdCCT
        if grad_eival is not None:
            CdLCT = torch.matmul(eivec, grad_eival.unsqueeze(-1) * eivecT)
            dA = dA + CdLCT

        dA = (dA + dA.transpose(-2, -1).conj()) * 0.5
        return dA


def dm_fermi_x(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finite-temperature density matrix via Fermi–Dirac occupations + Newton μ.

    Diagonalizes the (orthonormal-basis) Hamiltonian ``H0``:

        H0 = V diag(h) Vᵀ

    Finds the chemical potential μ such that the total occupation matches ``nocc``:

        N(μ) = ∑_i f_i(μ)  with  f_i(μ) = 1 / (exp(β (h_i − μ)) + 1)

    and constructs the density matrix

        P0 = V diag(f) Vᵀ

    Parameters
    ----------
    H0:
        Hamiltonian matrix, shape `(n_orb, n_orb)`.
    T:
        Electronic temperature (K).
    nocc:
        Target total occupation (spin-summed), integer.
    mu_0:
        Initial guess for μ. If ``None``, set to midpoint between HOMO/LUMO
        eigenvalues (indices `nocc-1` and `nocc`).
    eps:
        Convergence threshold for occupation error ``|nocc - N(μ)|``.
    MaxIt:
        Maximum number of Newton iterations.
    debug:
        If True, synchronizes CUDA at key points (kept for compatibility; minimal use).

    Returns
    -------
    P0:
        Density matrix, shape `(n_orb, n_orb)`.
    v:
        Eigenvectors, shape `(n_orb, n_orb)`.
    h:
        Eigenvalues, shape `(n_orb,)`.
    f:
        Occupations at converged μ, shape `(n_orb,)`.
    mu0:
        Chemical potential as a scalar tensor.

    Notes
    -----
    - Newton update uses a small lower bound on `dN/dμ` to avoid division by ~0.
    """

    h, v = torch.linalg.eigh(H0)
    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)

        mu0 = 0.5 * (h[nocc - 1] + h[nocc])
    else:
        mu0 = torch.as_tensor(mu_0, dtype=h.dtype, device=h.device)

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    target_occ = torch.as_tensor(nocc, dtype=h.dtype, device=h.device)
    occ_err_val = torch.full((), float("inf"), dtype=h.dtype, device=h.device)
    cnt = 0
    while (occ_err_val > eps) and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)  # occupations (N,)

        dOcc = beta * torch.sum(f * (1.0 - f))
        Occ = torch.sum(f)

        occ_err_val = torch.abs(target_occ - Occ)
        active = occ_err_val > 1e-9

        if active:
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-16))
            mu0 = mu0 + ((target_occ - Occ) / denom) * active.to(h.dtype)
        cnt += 1

    if cnt == MaxIt:
        print(
            "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format(
                MaxIt, occ_err_val
            )
        )

    # Final adjustment of occupation
    # P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)

    return P0, v, h, f, mu0


def dm_fermi_x_os(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    broken_symmetry: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Open-shell variant of :func:`DM_Fermi_x` (batched eigenproblems).

    Important
    ---------
    This function expects **batched** inputs:
    - `H0`: `(B, n_orb, n_orb)`
    - `nocc`: `(B,)` integer tensor

    Returns `mu0` as a tensor of shape `(B,)`.
    """

    h, v = torch.linalg.eigh(H0)
    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)

        lumo = nocc.unsqueeze(0).T
        if broken_symmetry:
            v[0, :, nocc[0] - 1] = 0.9 * v[0, :, nocc[0] - 1] + 0.1 * v[0, :, nocc[0]]
        mu0 = 0.5 * (h.gather(1, lumo) + h.gather(1, lumo - 1)).reshape(-1)
    else:
        mu0 = torch.as_tensor(mu_0, dtype=h.dtype, device=h.device)

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(
        nocc, dtype=torch.float64, device=nocc.device
    ) + float("inf")
    cnt = 0
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)  # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f), dim=1)
        Occ = torch.sum(f, dim=1)

        occ_err_val = abs(nocc - Occ)
        active = occ_err_val > 1e-9

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc - Occ) / denom) * active  # guard tiny derivative
        cnt += 1

    if cnt == MaxIt:
        print(
            "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format(
                MaxIt, occ_err_val
            )
        )

    # Final adjustment of occupation
    # P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)

    return P0, v, h, f, mu0


def dm_fermi_x_os_shared(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    broken_symmetry: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Open-shell "shared μ" variant.

    This version computes a single chemical potential `mu0` shared across the batch
    by sorting/flattening all eigenvalues (i.e., distributing total `nocc.sum()`).
    """

    h, v = torch.linalg.eigh(H0)
    h_all = h.flatten().sort()[0]
    target_occ = nocc.sum()
    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)

        lumo = target_occ
        if broken_symmetry:
            mix_coeff = 0.02
            v[0, :, nocc[0] - 1] = (1 - mix_coeff) * v[
                0, :, nocc[0] - 1
            ] + mix_coeff * v[0, :, nocc[0]]
        mu0 = 0.5 * (h_all[lumo] + h_all[lumo - 1])
    else:
        mu0 = torch.as_tensor(mu_0, dtype=h.dtype, device=h.device)

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(
        target_occ, dtype=torch.float64, device=nocc.device
    ) + float("inf")
    cnt = 0

    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h_all - mu0)) + 1.0)  # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f))
        Occ = torch.sum(f)

        occ_err_val = abs(target_occ - Occ)
        active = occ_err_val > 1e-9

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((target_occ - Occ) / denom) * active  # guard tiny derivative
        cnt += 1

    if cnt == MaxIt:
        print(
            "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format(
                MaxIt, occ_err_val
            )
        )
    f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)  # occupations (N,)

    # Final adjustment of occupation
    # P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)

    return P0, v, h, f, mu0


def dm_fermi_x_batch(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched density matrices with per-batch chemical potentials.

    Parameters are kept identical to the original signature. In practice this
    routine expects:
    - `H0`: `(B, n_orb, n_orb)`
    - `nocc`: `(B,)` integer tensor

    Returns `mu0` as `(B,)`.
    """

    h, v = torch.linalg.eigh(H0)
    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)
        mu0 = 0.5 * (
            h.gather(1, (nocc.unsqueeze(0).T - 1)) + h.gather(1, nocc.unsqueeze(0).T)
        ).squeeze(-1)
    else:
        mu0 = torch.as_tensor(mu_0, dtype=h.dtype, device=h.device)

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(
        nocc, dtype=torch.float64, device=nocc.device
    ) + float("inf")
    cnt = 0
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)  # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f), dim=1)
        Occ = torch.sum(f, dim=1)

        occ_err_val = abs(nocc - Occ)
        active = occ_err_val > 1e-9

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc - Occ) / denom) * active  # guard tiny derivative
        cnt += 1
    if cnt == MaxIt:
        print(
            "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format(
                MaxIt, occ_err_val
            )
        )

    # Final adjustment of occupation
    # P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)
    return P0, v, h, f, mu0


def dm_fermi_x_batch_degen(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like :func:`dm_fermi_x_batch` but with a degenerate-safe eigensolver.

    Replaces ``torch.linalg.eigh`` with :class:`_degen_symeig`, which zeros out
    the ``1/(λᵢ−λⱼ)`` gradient terms for eigenvalue pairs closer than
    ``DEGEN_THRESHOLD``.  Use this variant when training on symmetric molecules
    or batches that contain degenerate orbitals.

    Parameters / returns are identical to :func:`dm_fermi_x_batch`.
    """
    h, v = _degen_symeig.apply(H0)
    if mu_0 is None:
        mu0 = 0.5 * (
            h.gather(1, (nocc.unsqueeze(0).T - 1)) + h.gather(1, nocc.unsqueeze(0).T)
        ).squeeze(-1)
    else:
        mu0 = torch.as_tensor(mu_0, dtype=h.dtype, device=h.device)

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(
        nocc, dtype=torch.float64, device=nocc.device
    ) + float("inf")
    cnt = 0
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)
        dOcc = beta * torch.sum(f * (1.0 - f), dim=1)
        Occ = torch.sum(f, dim=1)
        occ_err_val = abs(nocc - Occ)
        active = occ_err_val > 1e-9
        if active.any():
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc - Occ) / denom) * active
        cnt += 1
    if cnt == MaxIt:
        print(
            "Warning: dm_fermi_batch_degen did not converge in {} iterations, occ error = {}".format(
                MaxIt, occ_err_val
            )
        )

    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)
    return P0, v, h, f, mu0


dm_fermi_x_eager = dm_fermi_x
dm_fermi_x_os_eager = dm_fermi_x_os
dm_fermi_x_os_shared_eager = dm_fermi_x_os_shared
dm_fermi_x_batch_eager = dm_fermi_x_batch
dm_fermi_x_batch_degen_eager = dm_fermi_x_batch_degen

dm_fermi_x = _maybe_compile(dm_fermi_x)
dm_fermi_x_os = _maybe_compile(dm_fermi_x_os)
dm_fermi_x_os_shared = _maybe_compile(dm_fermi_x_os_shared)
dm_fermi_x_batch = _maybe_compile(dm_fermi_x_batch)
dm_fermi_x_batch_degen = _maybe_compile(dm_fermi_x_batch_degen)


def nonaufbau_constraints(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: float,
    state: str,
    smearing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforces nonaufbau occupation constraints after aufbau calculation (dm_fermi_x).
    Therefore, overwrites previous aufbau f and D. Required for deltaSCF.

    Parameters:
    ----------
    state:
         Target lowest energy excited state. Defined in dftparams{}.
    smearing:
         Utilize half excitation by setting relevant orbital occupcations to 0.5. Defined in dftparams{}.


    TODO:
    ----
         Provide option for user defined occupations to target higher energy excited states.
    """

    mu0 = mu_0
    h, v = torch.linalg.eigh(H0)
    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)  # occupations (N,)

    if state == "SINGLET":
        f[1, nocc - 1] = 0
        f[1, nocc] = 1
        if smearing:
            f[1, nocc - 1] = 0.5
            f[1, nocc] = 0.5
    elif state == "TRIPLET":
        f[1, nocc - 1] = 0
        f[0, nocc] = 1
        if smearing:
            f[1, nocc - 1] = 0.5
            f[0, nocc] = 0.5
    else:
        raise ValueError(
            "target excited state required for deltaSCF (SINGLET OR TRIPLET currently supported)"
        )

    # Final adjustment of occupation
    # P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)

    return P0, v, h, f, mu0


nonaufbau_constraints_eager = nonaufbau_constraints
nonaufbau_constraints = _maybe_compile(nonaufbau_constraints)
