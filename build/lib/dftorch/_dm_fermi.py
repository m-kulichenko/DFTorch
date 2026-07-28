from __future__ import annotations

import time
from typing import Optional

import torch


def dm_fermi(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    m: int,
    eps: float,
    MaxIt: int,
    debug: bool = False,
) -> tuple[torch.Tensor, float]:
    """Finite-temperature density matrix via Recursive Fermi Operator Expansion (RFOE).

    This function approximates the Fermi–Dirac occupation function using a recursive
    polynomial/rational expansion, avoiding explicit matrix exponentials. The Hamiltonian
    is diagonalized once, the occupation function is evaluated in the eigenvalue basis,
    and the density matrix is reconstructed as:

        H0 = V diag(h) Vᵀ
        P0 = V diag(p0) Vᵀ

    The chemical potential μ is adjusted by Newton iterations so that:

        trace(P0) = ∑_i p0_i ≈ nocc

    Parameters
    ----------
    H0:
        Symmetric/Hermitian Hamiltonian in an orthonormal basis, shape `(N, N)`.
    T:
        Electronic temperature in Kelvin.
    nocc:
        Target number of occupied electrons (spin-summed), integer in `[1, N-1]`.
    mu_0:
        Initial guess for chemical potential. If `None`, initialized as midpoint
        between HOMO/LUMO eigenvalues: `0.5 * (h[nocc-1] + h[nocc])`.
    m:
        Depth of the recursion. Larger values increase sharpness/accuracy but
        increase cost. Must be `>= 0`.
    eps:
        Convergence threshold for occupation error `|trace(P0) - nocc|`.
    MaxIt:
        Maximum number of Newton iterations for μ.
    debug:
        If True, synchronizes CUDA and prints basic timings.

    Returns
    -------
    P0:
        Density matrix, shape `(N, N)`.
    mu0:
        Converged chemical potential as Python float.

    Notes
    -----
    - The recursion operates on eigenvalues only (vector operations), then reconstructs `P0`.
    - The update uses `dPdmu = Σ β p0 (1-p0)`; when near 0, the update is skipped.
    - Units: `kB` is in eV/K, so `H0` eigenvalues should be in eV for consistent β.
    """
    if debug:
        torch.cuda.synchronize()
    start_time1 = time.perf_counter()

    if mu_0 is None:
        # h = torch.linalg.eigvalsh(H0)
        h, v = torch.linalg.eigh(H0)
        mu0 = 0.5 * (h[nocc - 1] + h[nocc])
    else:
        mu0 = mu_0

    if debug:
        torch.cuda.synchronize()
        print("    eigh     {:.1f} s".format(time.perf_counter() - start_time1))

    start_time1 = time.perf_counter()

    kB = 8.61739e-5  # eV/K
    beta = 1.0 / (kB * T)
    cnst = 2 ** (-2 - m) * beta
    OccErr = 1.0
    Cnt = 0
    while OccErr > eps and Cnt < MaxIt:
        p0 = 0.5 - cnst * (h - mu0)  # $$$ should be exp?
        for _ in range(m):
            p02 = p0 * p0
            iD0 = 1 / (2 * (p02 - p0) + 1)
            p0 = iD0 * p02
        dPdmu = torch.sum(beta * p0 * (1 - p0))
        occ = torch.sum(p0)

        if abs(dPdmu) > 1e-8:
            mu0 = mu0 + (nocc - occ) / dPdmu
            OccErr = abs(occ - nocc)
        else:
            OccErr = 0.0

        Cnt += 1
    if Cnt == MaxIt:
        print(
            "Warning: dm_fermi did not converge in {} iterations, occ error = {}".format(
                MaxIt, OccErr
            )
        )
    if debug:
        torch.cuda.synchronize()
        print("    dm ptr   {:.1f} s".format(time.perf_counter() - start_time1))
    start_time1 = time.perf_counter()

    # Final adjustment of occupation
    P0 = (v * p0.unsqueeze(0)) @ v.T  # same as v@(torch.diag_embed(p0)@v.T)
    if debug:
        torch.cuda.synchronize()
        print("    v*p0*v.T {:.1f} s".format(time.perf_counter() - start_time1))

    return P0, mu0
