from typing import Optional
import torch
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

@torch.compile(dynamic=False)
def DM_Fermi_x(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
):
    """
    Compute the finite‑temperature density matrix using a Fermi–Dirac
    diagonalization and Newton iteration on the chemical potential.

    Diagonalizes the orthonormal‑basis Hamiltonian ``H0`, determines the
    chemical potential ``mu0`` such that the total occupation matches
    ``nocc`` at temperature ``T``, and constructs the density matrix

        P0 = V diag(f) Vᵀ

    where ``V`` and ``h`` are eigenvectors/eigenvalues of ``H0`` and
    ``f_i = 1 / (exp(β (h_i − μ)) + 1)`` are Fermi–Dirac occupations.

    Parameters
    ----------
    H0 : torch.Tensor
        Hamiltonian matrix in an orthonormal basis, shape (n_orb, n_orb).
    T : float
        Electronic temperature in Kelvin.
    nocc : int
        Target number of (spin‑summed) occupied electrons; the trace of the
        density matrix is driven to this value.
    mu_0 : float or None
        Initial guess for the chemical potential μ. If ``None``, it is
        estimated as the midpoint between the nocc‑th and (nocc+1)‑th
        eigenvalues of ``H0``.
    eps : float
        Convergence threshold for the occupation error
        ``|nocc − ∑_i f_i|``.
    MaxIt : int
        Maximum number of Newton iterations on μ.
    debug : bool, optional
        If True, performs CUDA synchronizations around the main steps for
        more reliable timing.

    Returns
    -------
    P0 : torch.Tensor
        Finite‑temperature density matrix, shape (n_orb, n_orb).
    v : torch.Tensor
        Eigenvector matrix of ``H0`` (columns are eigenvectors), shape
        (n_orb, n_orb).
    h : torch.Tensor
        Eigenvalues of ``H0`` corresponding to columns of ``v``, shape
        (n_orb,).
    f : torch.Tensor
        Fermi–Dirac occupation numbers at the converged μ, shape (n_orb,).
    mu0 : float
        Converged chemical potential.

    Notes
    -----
    The chemical potential is updated via a scalar Newton step

        μ_{k+1} = μ_k + (nocc − N(μ_k)) / (dN/dμ),

    where ``N(μ) = ∑_i f_i(μ)`` and

        dN/dμ = β ∑_i f_i (1 − f_i).

    A small lower bound is imposed on ``dN/dμ`` to avoid division by very
    small derivatives in pathological cases.
    """    
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
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
        f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)   # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = (beta * torch.sum(f * (1.0 - f))).item()
        Occ  = torch.sum(f).item()

        occ_err_val = abs(nocc - Occ)

        if occ_err_val > 1e-9:
            # Newton step on mu
            mu0 = mu0 + (nocc - Occ) / max(dOcc, 1e-16)   # guard tiny derivative
        cnt += 1
        
    if cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, occ_err_val))
    
    # Final adjustment of occupation    
    #P0 = v@(torch.diag_embed(f)@v.T)
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
):
    """
    Open-shell version of DM_Fermi_x.
    """    
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)

        lumo = nocc.unsqueeze(0).T
        if broken_symmetry:
            v[0,:,nocc[0]-1] = 0.8*v[0,:,nocc[0]-1] + 0.2*v[0,:,nocc[0]]
        mu0 = 0.5*(h.gather(1, lumo) + h.gather(1, lumo-1)).reshape(-1)
    else:
        mu0 = mu_0
    
    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(nocc, dtype=torch.float64, device=nocc.device) + float("inf")
    cnt = 0
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)   # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f), dim=1)
        Occ  = torch.sum(f, dim=1)

        occ_err_val = abs(nocc - Occ)
        active = (occ_err_val > 1e-9)

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc - Occ) / denom)*active   # guard tiny derivative
        cnt += 1
        
    if cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, occ_err_val))
    
    # Final adjustment of occupation    
    #P0 = v@(torch.diag_embed(f)@v.T)
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
):
    """
    Open-shell version of DM_Fermi_x.
    """    
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
        h_all = h.flatten().sort()[0]

        lumo = nocc.sum()
        if broken_symmetry:
            v[0,:,nocc[0]-1] = 0.8*v[0,:,nocc[0]-1] + 0.2*v[0,:,nocc[0]]
        mu0 = 0.5*(h_all[lumo] + h_all[lumo-1])
    else:
        mu0 = mu_0
    
    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(lumo, dtype=torch.float64, device=nocc.device) + float("inf")
    cnt = 0
    
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h_all - mu0)) + 1.0)   # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f))
        Occ  = torch.sum(f)

        occ_err_val = abs(nocc.sum() - Occ)
        active = (occ_err_val > 1e-9)

        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc.sum() - Occ) / denom)*active   # guard tiny derivative
        cnt += 1
        
    if cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, occ_err_val))
    f = 1.0 / (torch.exp(beta * (h - mu0)) + 1.0)   # occupations (N,)
    
    # Final adjustment of occupation    
    #P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)
    
    return P0, v, h, f, mu0

@torch.compile(dynamic=False)
def DM_Fermi_x_batch(
    H0: torch.Tensor,
    T: float,
    nocc: int,
    mu_0: Optional[float],
    eps: float,
    MaxIt: int,
    debug: bool = False,
):
    """
    Notes
    -----
    The chemical potential is updated via a scalar Newton step

        μ_{k+1} = μ_k + (nocc − N(μ_k)) / (dN/dμ),

    where ``N(μ) = ∑_i f_i(μ)`` and

        dN/dμ = β ∑_i f_i (1 − f_i).

    A small lower bound is imposed on ``dN/dμ`` to avoid division by very
    small derivatives in pathological cases.
    """    
    
    if mu_0 == None:
        #h = torch.linalg.eigvalsh(H0)
        h,v = torch.linalg.eigh(H0)
        # h,v = torch.linalg.eigh(H0.to(torch.float64))
        # h = h.to(H0.dtype)
        # v = v.to(H0.dtype)
        mu0 = 0.5 * (h.gather(1, (nocc.unsqueeze(0).T - 1)) + h.gather(1, nocc.unsqueeze(0).T)).squeeze(-1)
    else:
        mu0 = mu_0
        
    kB = torch.tensor(8.61739e-5, dtype=h.dtype, device=h.device)  # eV/K
    beta = 1.0 / (kB * T)
    occ_err_val = torch.zeros_like(nocc, dtype=torch.float64, device=nocc.device) + float("inf")
    cnt = 0
    while (occ_err_val > eps).any() and (cnt < MaxIt):
        # Clamp small eigvals if needed by your physics; leave as-is here.
        f = 1.0 / (torch.exp(beta * (h - mu0.unsqueeze(-1))) + 1.0)   # occupations (N,)

        # dOcc and Occ are scalar tensors; convert to Python floats for loop control.
        dOcc = beta * torch.sum(f * (1.0 - f), dim=1)
        Occ  = torch.sum(f, dim=1)

        occ_err_val = abs(nocc - Occ)
        active = (occ_err_val > 1e-9)
        
        if active.any():
            # Newton step on mu
            denom = torch.maximum(dOcc, torch.full_like(dOcc, 1e-14))
            mu0 = mu0 + ((nocc - Occ) / denom)*active   # guard tiny derivative
        cnt += 1
    if cnt == MaxIt:
        print("Warning: DM_Fermi did not converge in {} iterations, occ error = {}".format(MaxIt, occ_err_val))

    # Final adjustment of occupation    
    #P0 = v@(torch.diag_embed(f)@v.T)
    # Build density matrix P0 = V diag(f) V^T without forming diag explicitly
    # Column-scale trick: V @ diag(f) == V * f[None, :]
    P0 = (v * f.unsqueeze(-2)) @ v.transpose(-2, -1)    
    return P0, v, h, f, mu0