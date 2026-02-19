import torch

@torch.compile(fullgraph=False, dynamic=False)
def Fermi_PRT(
    H1: torch.Tensor,
    Te: float,
    Q: torch.Tensor,
    ev: torch.Tensor,
    mu0: float,
):
    """
    First‑order Fermi–Dirac density matrix perturbation in the eigenbasis.

    Given a first‑order perturbation of the Fock/Hamiltonian matrix ``H1``
    (in the AO basis), this routine computes the unperturbed density matrix
    ``D0`` and its first-order response ``D1`` at temperature ``Te`` and
    chemical potential ``mu0`` using finite‑temperature linear response.

    Parameters
    ----------
    H1 : torch.Tensor
        First‑order perturbation of the Hamiltonian in the AO basis,
        shape (n_orb, n_orb).
    Te : float
        Electronic temperature in Kelvin.
    Q : torch.Tensor
        Eigenvector matrix of the unperturbed Hamiltonian, columns are
        eigenvectors, shape (n_orb, n_orb). It is used to transform between
        AO and eigenbases.
    ev : torch.Tensor
        Eigenvalues (orbital energies) corresponding to columns of ``Q``,
        shape (n_orb,).
    mu0 : float
        Zeroth‑order chemical potential.

    Returns
    -------
    D0 : torch.Tensor
        Zeroth‑order (unperturbed) density matrix in the AO basis,
        shape (n_orb, n_orb).
    D1 : torch.Tensor
        First‑order correction to the density matrix in the AO basis,
        shape (n_orb, n_orb).

    Notes
    -----
    The routine builds the susceptibility kernel

        χ_ij = (f_i - f_j) / (e_i - e_j),

    in the eigenbasis, with the diagonal limit

        χ_ii = -β f_i (1 - f_i),

    where ``f_i`` are Fermi–Dirac occupations at temperature ``Te`` and
    chemical potential ``mu0``. The first‑order response in the eigenbasis
    is then

        X = χ ⊙ (Qᵀ H1 Q),

    which is transformed back to the AO basis as ``D1 = Q X Qᵀ``.

    Particle-number conservation is enforced via a first‑order correction
    to the chemical potential, ``mu1``, obtained from

        dN/dμ = -β ∑_i f_i (1 - f_i),

    and correcting ``X`` along the diagonal. A small numerical safeguard is
    applied so that if ``|dN/dμ|`` is below a threshold, the correction is
    skipped to avoid division by a near‑zero value.
    """
    kB = 8.61739e-5  # eV/K
    beta = 1 / (kB * Te)

    QH1Q = Q.T @ H1 @ Q
    fe = 1./(torch.exp(beta*(ev-mu0))+1.0)
    ei = ev[:, None]
    ej = ev[None, :]
    de = ei - ej                                     # (N,N)

    # Susceptibility kernel χ_ij = (f_i - f_j)/(e_i - e_j), with diagonal limit -β f(1-f)
    chi = torch.empty_like(de)
    off = (de.abs() > 1e-12)
    chi[off] = (fe[:, None] - fe[None, :])[off] / de[off]
    diag = -beta * fe * (1.0 - fe)                   # (N,)
    chi[~off] = diag.expand_as(de)[~off]

    # Response in eigenbasis
    X = chi * QH1Q

    # Enforce particle conservation via μ1
    dN_dmu = diag.sum()                              # d tr(D)/dμ = -β ∑ f(1-f)
    # if torch.abs(dN_dmu) > 1e-15:
    #     mu1 = X.diagonal().sum() / dN_dmu
    #     X = X - torch.diag_embed(diag) * mu1

    # Numerically stable μ1: if |dN_dmu| is small, skip the correction.
    mask = (torch.abs(dN_dmu) > 1e-12).to(H1.dtype)
    mu1 = (X.diagonal().sum() / (dN_dmu + (1.0 - mask))) * mask
    X = X - torch.diag_embed(diag) * mu1
    dpdmu = -diag

    D0 = Q @ torch.diag_embed(fe) @ Q.T
    D1 = Q @ X @ Q.T
    return D0, D1

@torch.compile(fullgraph=False, dynamic=False)
def Fermi_PRT_batch(
    H1: torch.Tensor,
    Te: float,
    Q: torch.Tensor,
    ev: torch.Tensor,
    mu0: float,
):
    """
    First‑order Fermi–Dirac density matrix perturbation in the eigenbasis.
    """
    kB = 8.61739e-5  # eV/K
    beta = 1 / (kB * Te)

    QH1Q = torch.matmul(Q.transpose(-1, -2), torch.matmul(H1, Q))
    fe = 1./(torch.exp(beta*(ev-mu0.unsqueeze(-1)))+1.0)
    ei = ev[:, :, None]
    ej = ev[:, None, :]
    de = ei - ej                                     # (N,N)

    # Susceptibility kernel χ_ij = (f_i - f_j)/(e_i - e_j), with diagonal limit -β f(1-f)
    chi = torch.empty_like(de)
    off = (de.abs() > 1e-12)
    chi[off] = (fe[:, :, None] - fe[:, None, :])[off] / de[off]
    diag = -beta * fe * (1.0 - fe)                   # (N,)
    chi[~off] = diag.unsqueeze(1).expand_as(de)[~off]

    # Response in eigenbasis
    X = chi * QH1Q

    # Enforce particle conservation via μ1
    dN_dmu = diag.sum(dim=1)                              # d tr(D)/dμ = -β ∑ f(1-f)

    # if torch.abs(dN_dmu) > 1e-15:
    #     mu1 = X.diagonal().sum() / dN_dmu
    #     X = X - torch.diag_embed(diag) * mu1

    # Numerically stable μ1: if |dN_dmu| is small, skip the correction.
    mask = (torch.abs(dN_dmu) > 1e-12).to(H1.dtype)
    mu1 = (X.diagonal(dim1=-2, dim2=-1).sum(dim=1) / (dN_dmu + (1.0 - mask))) * mask
    X = X - torch.diag_embed(diag) * mu1.unsqueeze(-1).unsqueeze(-1)

    D0 = torch.matmul(Q, torch.matmul(torch.diag_embed(fe), Q.transpose(-1,-2)))
    D1 = torch.matmul(Q, torch.matmul(X, Q.transpose(-1,-2)))
    return D0, D1

def Canon_DM_PRT(F1,T,Q,ev,mu_0,m):
    '''
    canonical density matrix perturbation theory
    Alg.2 from https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00264
    '''

    mu0 = mu_0 #Intial guess
    FO1 = Q.T@(F1@Q)    # Main cost I = O(N^3) -> GPU
    kB = 8.61739e-5; # eV/K;

    beta = 1/(kB*T)    # Temp in Kelvin
    cnst = (2**(-2-m))*beta

    # $$$ maybe Occ_mask is redundant ??
    p0 = (0.5 - cnst*(ev-mu0)) #################################
    #p0 = p0.unsqueeze(1)                                                #
    P1 = -cnst*FO1  #{P1 = -cnst*(FO1-mu1*I);}                          #
    for i in range(0,m):                                                #
        p02 = p0*p0                                                     #
        dX1 = p0*P1+P1*p0.T        # Cost O(N^2)           #
        iD0 = 1./(2*(p02-p0)+1)                                         #
        p0 = iD0*p02                                                    #
        P1 = iD0*(dX1+2*(P1-dX1)*p0.T) #Cost O(N^2) <#mask##


    # Original approximate μ1 using dp/dμ ≈ beta*p0*(1-p0)
    dpdmu = beta * p0 * (1.0 - p0)
    dmu1 = -(P1.diagonal().sum()) / dpdmu.sum()
    P1 = P1 + torch.diag_embed(dpdmu) * dmu1
    P1 = Q@P1@Q.T  # Main cost II = O(N^3) -> GPU

    #####
    # I = torch.eye(F1.shape[-1], device=F1.device)
    # F_0 = torch.diag_embed(ev)
    # F_1 = Q.T@(FO1@Q)
    # X_0 = (0.5*I - cnst*(F_0 - mu0*I))
    # X_1 = -cnst*F_1
    # for i in range(m):
    #     X_1_tmp = X_0@X_1 + X_1@X_0
    #     Y_0_tmp = torch.linalg.matrix_power(
    #             (2*X_0@(X_0 - I) + I ),
    #             -1)
    #     X_0 = Y_0_tmp@X_0@X_0
    #     X_1 = Y_0_tmp@(X_1_tmp + 2*(X_1 - X_1_tmp)@X_0)
    # D_0_mu = beta * X_0 @ (I - X_0)
    # mu1 = -torch.trace(X_1)/torch.trace(D_0_mu)
    # D_1 = X_1 + D_0_mu*mu1
    # D_1 = Q@D_1@Q.T
    



    return p0, P1