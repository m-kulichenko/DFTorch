import torch
import time


def _dm_fermi(H0, T, nocc, mu_0, m, eps, MaxIt, debug=False):
    """
    Computes the finite-temperature density matrix using Recursive Fermi Operator Expansion.

    This function implements the recursive expansion of the Fermi-Dirac distribution to
    evaluate the electronic density matrix `P0` at finite temperature without explicitly computing
    matrix exponentials. It iteratively adjusts the chemical potential `mu0` to ensure the trace of the
    density matrix matches the desired number of occupied electrons.

    Args:
        H0 (torch.Tensor): Hamiltonian matrix in the orthonormal basis (N x N).
        T (float): Electronic temperature in Kelvin.
        nocc (int): Number of occupied electrons (expected trace of the density matrix).
        mu_0 (float or None): Initial guess for chemical potential. If None, estimated from eigenvalues.
        m (int): Depth of the recursive expansion. Higher values increase accuracy.
        eps (float): Convergence threshold for occupation error.
        MaxIt (int): Maximum number of _scf iterations to adjust the chemical potential.

    Returns:
        P0 (torch.Tensor): Finite-temperature density matrix (N x N).
        mu0 (float): Converged chemical potential.

    Notes:
        - The recursion approximates the Fermi function at finite temperature, avoiding costly exponentials.
        - The eigenbasis is used to construct the density matrix after recursion.
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
            "Warning: _dm_fermi did not converge in {} iterations, occ error = {}".format(
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
