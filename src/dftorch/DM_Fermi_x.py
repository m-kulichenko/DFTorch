import torch
import time
@torch.compile
def DM_Fermi_x(H0, T, nocc, mu_0, m, eps, MaxIt, debug=False):
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
    start_time1 = time.perf_counter()
    dtype = H0.dtype
    device = H0.device
    N = H0.shape[0]
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
        mu0 = 0.5 * (h[nocc - 1] + h[nocc])
    else:
        mu0 = mu_0
        
    if debug: 
        torch.cuda.synchronize()
        print("    eigh     {:.1f} s".format( time.perf_counter()-start_time1 ))

    start_time1 = time.perf_counter()

    kB = 8.61739e-5  # eV/K
    beta = 1.0 / (kB * T)
    cnst = 2 ** (-2 - m) * beta
    OccErr = 1.0
    Cnt = 0
    while OccErr > eps and Cnt < MaxIt:
        f = 1./(torch.exp(beta*(h-mu0))+1)
        dOcc = beta*torch.sum(f*(1.0-f))
        Occ = torch.sum(f)
        OccErr = nocc-Occ
        if abs(OccErr) > 1e-10:
            mu0 = mu0 + OccErr/dOcc
        Cnt += 1
        
    if Cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, OccErr))
    if debug:
        torch.cuda.synchronize()
        print("    dm ptr   {:.1f} s".format( time.perf_counter()-start_time1 ))
    start_time1 = time.perf_counter()
    
    # Final adjustment of occupation    
    P0 = v@(torch.diag_embed(f)@v.T)
    if debug:
        torch.cuda.synchronize()
        print("    v*p0*v.T {:.1f} s".format( time.perf_counter()-start_time1 ))


    
    return P0, v,h,f, mu0