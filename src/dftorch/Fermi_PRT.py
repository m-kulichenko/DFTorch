import torch

def Fermi_PRT(H1, Te, Q, ev, mu0):
    kB = 8.61739e-5; # eV/K;
    beta = 1/(kB*Te)

    N  = max(H1.shape)

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
    if torch.abs(dN_dmu) > 1e-15:
        mu1 = X.diagonal().sum() / dN_dmu
        X = X - torch.diag_embed(diag) * mu1

    D0 = Q @ torch.diag_embed(fe) @ Q.T
    D1 = Q @ X @ Q.T
    return D0, D1

    # dDtmp = torch.zeros(N,N, device = H1.device)
    # X = torch.zeros(N,N, device = H1.device)
    # for i in range(N):
    #     for j in range(N):
    #         if abs(ev[i]-ev[j]) < 1e-4:
    #             xx = (ev[i]+ev[j])/2
    #             tmp = beta*(xx-mu0)
    #             if abs(tmp) > 25:
    #                 dDtmp[i,j] = 0.0
    #             else:
    #                 dDtmp[i,j] = -beta*torch.exp(beta*(xx-mu0))/(torch.exp(beta*(xx-mu0))+1)**2
    #         else:
    #             dDtmp[i,j] = (fe[i]-fe[j])/(ev[i]-ev[j])
    #         X[i,j] = dDtmp[i,j]*QH1Q[i,j]
    # TrdD = torch.trace(dDtmp)
    # if abs(TrdD) > 10e-9:
    #     mu1 = torch.trace(X)/torch.trace(dDtmp)
    #     X = X-torch.diag_embed(torch.diag(dDtmp))*mu1
    # D0 = Q @ torch.diag_embed(fe) @ Q.T
    # D1 = Q @ X @ Q.T
    # return D0, D1

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