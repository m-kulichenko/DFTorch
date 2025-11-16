from typing import Optional
import torch
import time
import torch, torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

@torch.compile(dynamic=False, mode="reduce-overhead")  # or mode="default"
def DM_Fermi_x(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    m: int,
    eps: float,
    MaxIt: int,
    debug: bool = False,
    ):
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
        MaxIt (int): Maximum number of SCF iterations to adjust the chemical potential.

    Returns:
        P0 (torch.Tensor): Finite-temperature density matrix (N x N).
        mu0 (float): Converged chemical potential.

    Notes:
        - The recursion approximates the Fermi function at finite temperature, avoiding costly exponentials.
        - The eigenbasis is used to construct the density matrix after recursion.
    """    
    if debug: torch.cuda.synchronize()
    #start_time1 = time.perf_counter()
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
        mu0 = 0.5 * (h[nocc - 1] + h[nocc])
    else:
        mu0 = mu_0
        
    if debug: 
        torch.cuda.synchronize()
        #print("    eigh     {:.1f} s".format( time.perf_counter()-start_time1 ))

    #start_time1 = time.perf_counter()

    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = float("inf")
    cnt = 0
    while (occ_err_val > eps) and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)   # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = (beta * torch.sum(f * (1.0 - f))).item()
        Occ  = torch.sum(f).item()

        occ_err_val = abs(nocc - Occ)

        if occ_err_val > 1e-10:
            # Newton step on mu
            mu0 = mu0 + (nocc - Occ) / max(dOcc, 1e-30)   # guard tiny derivative
        cnt += 1
        
    if cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, OccErr))
    if debug:
        torch.cuda.synchronize()
        #print("    dm ptr   {:.1f} s".format( time.perf_counter()-start_time1 ))
    #start_time1 = time.perf_counter()
    
    # Final adjustment of occupation    
    #P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)

    if debug:
        torch.cuda.synchronize()
        #print("    v*p0*v.T {:.1f} s".format( time.perf_counter()-start_time1 ))


    
    return P0, v,h,f, mu0