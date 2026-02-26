from __future__ import annotations

from typing import Optional

import torch
import torch._dynamo as dynamo

dynamo.config.capture_scalar_outputs = True


@torch.compile(dynamic=False)
def dm_fermi_x(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
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
        Chemical potential as Python float.

    Notes
    -----
    - Newton update uses a small lower bound on `dN/dμ` to avoid division by ~0.
    - This function is compiled with `torch.compile`; Python-side `while` control
      is enabled via `dynamo.config.capture_scalar_outputs = True`.
    """

    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        h, v = torch.linalg.eigh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)

        mu0 = 0.5 * (h[nocc - 1] + h[nocc])
    else:
        mu0 = mu_0

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = float("inf")
    cnt = 0
    while (occ_err_val > eps) and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)  # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = (beta * torch.sum(f * (1.0 - f))).item()
        Occ = torch.sum(f).item()

        occ_err_val = abs(nocc - Occ)

        if occ_err_val > 1e-9:
            # Newton step on mu
            mu0 = mu0 + (nocc - Occ) / max(dOcc, 1e-16)  # guard tiny derivative
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

    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        h, v = torch.linalg.eigh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)

        lumo = nocc.unsqueeze(0).T
        if broken_symmetry:
            v[0, :, nocc[0] - 1] = 0.8 * v[0, :, nocc[0] - 1] + 0.2 * v[0, :, nocc[0]]
        mu0 = 0.5 * (h.gather(1, lumo) + h.gather(1, lumo - 1)).reshape(-1)
    else:
        mu0 = mu_0

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

    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        h, v = torch.linalg.eigh(H0)
        h_all = h.flatten().sort()[0]

        lumo = nocc.sum()
        if broken_symmetry:
            v[0, :, nocc[0] - 1] = 0.98 * v[0, :, nocc[0] - 1] + 0.02 * v[0, :, nocc[0]]
        mu0 = 0.5 * (h_all[lumo] + h_all[lumo - 1])
    else:
        mu0 = mu_0

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(
        lumo, dtype=torch.float64, device=nocc.device
    ) + float("inf")
    cnt = 0

    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h_all - mu0)) + 1.0)  # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f))
        Occ = torch.sum(f)

        occ_err_val = abs(nocc.sum() - Occ)
        active = occ_err_val > 1e-9

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc.sum() - Occ) / denom) * active  # guard tiny derivative
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


@torch.compile(dynamic=False)
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

    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        h, v = torch.linalg.eigh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)
        mu0 = 0.5 * (
            h.gather(1, (nocc.unsqueeze(0).T - 1)) + h.gather(1, nocc.unsqueeze(0).T)
        ).squeeze(-1)
    else:
        mu0 = mu_0

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
