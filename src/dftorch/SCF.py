import torch
from .Tools import fractional_matrix_power_symm
from .DM_Fermi import DM_Fermi
import time

def SCF(structure, H0, S, Efield, C, C_sr, TYPE, Rx, Ry, Rz, H_Index_Start, H_Index_End, nocc,
        U, Znuc, Nats, Te, const, alpha=0.2, acc=1e-7, MAX_ITER=200, debug=False):
    """
    Performs a self-consistent field (SCF) cycle with finite electronic temperature
    and Fermi-Dirac occupations for a DFTB-like semiempirical Hamiltonian.

    Parameters
    ----------
    H0 : torch.Tensor
        Initial one-electron Hamiltonian matrix of shape (n_orb, n_orb).
    S : torch.Tensor
        Overlap matrix of shape (n_orb, n_orb).
    Efield : torch.Tensor
        External electric field vector of shape (3,) in atomic units.
    C : torch.Tensor
        Coulomb matrix used to compute electrostatic potential, shape (n_orb, n_orb).
    TYPE : torch.Tensor
        Atomic type index for each atom, shape (Nats,).
    Rx, Ry, Rz : torch.Tensor
        Atomic x, y, z coordinates respectively, each of shape (Nats,).
    H_Index_Start, H_Index_End : torch.Tensor
        Start and end indices of orbitals for each atom, shape (Nats,).
    nocc : int
        Number of occupied orbitals (electrons / 2).
    U : torch.Tensor
        Hubbard U parameter per atom, shape (Nats,).
    Znuc : torch.Tensor
        Nuclear charges per atom, shape (Nats,).
    Nats : int
        Total number of atoms in the system.
    Te : float
        Electronic temperature in energy units (e.g., eV).
    const : object
        DFTB constants object with attribute `n_orb`, which gives orbitals per atom type.
    alpha : float, optional
        Mixing coefficient for SCF charge update: (1 - alpha) * q_old + alpha * q. Default is 0.2.
    acc : float, optional
        Convergence threshold for SCF loop based on charge difference norm. Default is 1e-7.
    MAX_ITER : int, optional
        Maximum number of SCF iterations. Default is 200.

    Returns
    -------
    H : torch.Tensor
        Final one-electron Hamiltonian matrix after SCF.
    Hcoul : torch.Tensor
        Final Coulomb contribution to the Hamiltonian.
    Hdipole : torch.Tensor
        Dipole correction to the Hamiltonian from external E-field.
    D : torch.Tensor
        Final density matrix.
    q : torch.Tensor
        Final atomic charges.
    f : torch.Tensor
        Final Fermi orbital occupations (eigenvalues of Dorth).

    Notes
    -----
    - SCF convergence is checked via the L2 norm of charge difference between iterations.
    - The Hamiltonian is corrected for the external electric field through a symmetrized dipole term.
    - Charge density is constructed from Fermi-Dirac occupations.
    - Mixing helps stabilize convergence, especially for metallic systems or small gaps.
    """
    print('### Do SCF ###')
    dtype = H0.dtype
    device = H0.device
    N = H0.shape[0]
    atom_ids = torch.repeat_interleave(torch.arange(len(structure.n_orbitals_per_atom), device=Rx.device), structure.n_orbitals_per_atom) # Generate atom index for each orbital

    Z = fractional_matrix_power_symm(S, -0.5)
    with torch.no_grad():
    #if 1:
        Hdipole = torch.diag(-Rx[atom_ids] * Efield[0] - Ry[atom_ids] * Efield[1] - Rz[atom_ids] * Efield[2])
        Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
        H0 = H0 + Hdipole

        # Initial guess for chemical potential
        #print('Initial guess for chemical potential')
        # h = torch.linalg.eigvalsh(Z.T @ H0 @ Z)
        # mu0 = 0.5 * (h[nocc - 1] + h[nocc])

        # Initial density matrix
        print('  Initial DM_Fermi')
        Dorth, mu0 = DM_Fermi(Z.T @ H0 @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50)
        D = Z @ Dorth @ Z.T
        DS = 2 * torch.diag(D @ S)
        q = -1.0 * Znuc
        q.scatter_add_(0, atom_ids, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        #####
        atom_ids_sr = torch.repeat_interleave(torch.arange(len(structure.shell_types), device=Rx.device), const.shell_dim[structure.shell_types]) # Generate atom index for each orbital
        q_sr = -1.0 * structure.el_per_shell
        q_sr.scatter_add_(0, atom_ids_sr, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        #####
    
        Res = torch.tensor([2.0], device=device)
        it = 0
        Eband0 = torch.tensor([0.0], device=device) 
        Ecoul = torch.tensor([0.0], device=device)
        dEb = torch.tensor([10.0], device=device)

        #####
        Eband0_sr = torch.tensor([0.0], device=device) 
        Ecoul_sr = torch.tensor([0.0], device=device)
        #####

        print('\nStarting cycle')
        while ((Res > acc) + (dEb > acc*20)) and it < MAX_ITER:
            start_time = time.perf_counter()
            it += 1
            print("Iter {}".format(it))
            if it == MAX_ITER:
                print('Did not converge')
            
            if debug: torch.cuda.synchronize()
            start_time1 = time.perf_counter()
            CoulPot = C @ q
            Hcoul_diag = U[atom_ids] * q[atom_ids] + CoulPot[atom_ids]        
            #Hcoul_diag = CoulPot[atom_ids]   
            Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))  
            H = H0 + Hcoul

            #####
            CoulPot_sr = C_sr @ q_sr
            Hcoul_diag_sr = structure.Hubbard_U_sr[atom_ids_sr] * q_sr[atom_ids_sr] + CoulPot_sr[atom_ids_sr] 
            #Hcoul_diag_sr = CoulPot_sr[atom_ids_sr] 
            Hcoul_sr = 0.5 * (Hcoul_diag_sr.unsqueeze(1) * S + S * Hcoul_diag_sr.unsqueeze(0))
            H_sr = H0 + Hcoul_sr
            #####

            if debug: torch.cuda.synchronize()
            print("  Hcoul {:.1f} s".format( time.perf_counter()-start_time1 ))

            start_time1 = time.perf_counter()

            Dorth, mu0 = DM_Fermi((Z.T @ H @ Z).to(torch.float64), Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50, debug=debug)
            #Dorth, mu0 = DM_Fermi((Z.T @ H @ Z), Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50, debug=debug)
            Dorth = Dorth.to(torch.get_default_dtype())

            #####
            Dorth_sr, mu0_sr = DM_Fermi((Z.T @ H_sr @ Z).to(torch.float64), Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50, debug=debug)
            #Dorth, mu0 = DM_Fermi((Z.T @ H @ Z), Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50, debug=debug)
            Dorth_sr = Dorth_sr.to(torch.get_default_dtype())
            #####

            if debug: torch.cuda.synchronize()
            print("  DM_Fermi {:.1f} s".format( time.perf_counter()-start_time1 ))

            start_time1 = time.perf_counter()
            
            #D = Z.to(torch.float32) @ Dorth.to(torch.float32) @ Z.T.to(torch.float32)
            D = Z @ Dorth @ Z.T

            #####
            D_sr = Z @ Dorth_sr @ Z.T
            #####


            if debug: torch.cuda.synchronize()
            print("  Z@Dorth@Z.T {:.1f} s".format( time.perf_counter()-start_time1 ))

            start_time1 = time.perf_counter()
            
            q_old = q.clone()
            DS = 2 * (D * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
            q = -1.0 * Znuc
            q.scatter_add_(0, atom_ids, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
            Res = torch.norm(q - q_old)
            print(q.sum())
            q = (1-alpha)*q_old + alpha * q 

            #####
            q_sr_old = q_sr.clone()
            DS_sr = 2 * (D_sr * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
            q_sr = -1.0 * structure.el_per_shell
            q_sr.scatter_add_(0, atom_ids_sr, DS_sr) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
            Res_sr = torch.norm(q_sr - q_sr_old)
            print(q_sr.sum())
            q_sr = (1-alpha)*q_sr_old + alpha * q_sr 
            #####


            if debug: torch.cuda.synchronize()
            print("  update q {:.1f} s".format( time.perf_counter()-start_time1 ))
            
            Eband0_old = Eband0.clone()
            Ecoul_old = Ecoul.clone()
            Eband0 = 2 * torch.trace(H0 @ (D))
            Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)

            dEb = (Eband0_old-Eband0).abs()

            #####
            Eband0_sr_old = Eband0_sr.clone()
            Ecoul_sr_old = Ecoul_sr.clone()
            Eband0_sr = 2 * torch.trace(H0 @ (D_sr))
            Ecoul_sr = 0.5 * q_sr @ (C_sr @ q_sr) + 0.5 * torch.sum(q_sr**2 * structure.Hubbard_U_sr)
            #####
            print(Eband0 - Eband0_sr, Ecoul - Ecoul_sr)
            
            print("Res = {:.9f}, dEb = {:.9f}, dEc = {:.9f}, t = {:.1f} s\n".format(Res.item(), torch.abs(Eband0_old-Eband0).item(), torch.abs(Ecoul_old-Ecoul).item(), time.perf_counter()-start_time ))
            print("Res_sr = {:.9f}, t = {:.1f} s\n".format(Res_sr.item(), time.perf_counter()-start_time ))
            
        f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))

    D = Z @ Dorth @ Z.T
    DS = 2 * (D * S.T).sum(dim=1)
    q = -1.0 * Znuc
    q.scatter_add_(0, atom_ids, DS)

    return H, H_sr, Hcoul, Hdipole, D, q, q_sr, f

import time
import torch
from collections import deque

# --- simple Anderson (Type-I) mixer on charges --------------------------------
class AndersonMixer:
    """
    Anderson acceleration for fixed-point q = F(q).
    We are given (q_in -> q_out), residual r := q_out - q_in.
    Update: q_next = q_out - ΔX * beta,  beta solves  min || r_k - ΔR * beta ||_2
    with small Tikhonov regularization for stability.
    """
    def __init__(self, dim, m=5, lam=1e-10, damping=1.0, device=None, dtype=torch.float64):
        self.m = int(m)
        self.lam = lam
        self.damping = damping
        self.device = device
        self.dtype = dtype
        self.q_hist = deque([], maxlen=self.m+1)  # store q_k
        self.r_hist = deque([], maxlen=self.m+1)  # store r_k
        # prealloc scratch
        self._last_beta = None

    def reset(self):
        self.q_hist.clear()
        self.r_hist.clear()
        self._last_beta = None

    @torch.no_grad()
    def step(self, q_in, q_out):
        """
        q_in, q_out: (Natoms,) charge vectors on same device/dtype
        returns q_next
        """
        r = q_out - q_in  # residual
        self.q_hist.append(q_in.clone())
        self.r_hist.append(r.clone())

        # not enough history -> fall back to damped linear mix towards q_out
        if len(self.q_hist) < 2:
            return (1.0 - self.damping) * q_in + self.damping * q_out

        # build ΔR and ΔX from history (columns are differences)
        p = min(len(self.q_hist) - 1, self.m)
        # take last (p+1) entries
        Q = list(self.q_hist)[- (p+1) :]
        R = list(self.r_hist)[- (p+1) :]
        # differences (N,p)
        dR_cols = []
        dX_cols = []
        for i in range(1, p+1):
            dR_cols.append(R[i] - R[i-1])
            dX_cols.append(Q[i] - Q[i-1])
        dR = torch.stack(dR_cols, dim=1)  # (N,p)
        dX = torch.stack(dX_cols, dim=1)  # (N,p)

        # solve least squares: min_beta || r_k - dR * beta ||_2^2 + lam ||beta||^2
        # normal equations: (dR^T dR + lam I) beta = dR^T r
        # shapes: (p,p) (p,)
        G = dR.T @ dR
        if self.lam > 0:
            G = G + self.lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        rhs = dR.T @ r
        beta = torch.linalg.solve(G, rhs)  # (p,)
        self._last_beta = beta

        # Anderson update
        q_star = q_out - dX @ beta
        # optional damping towards q_in (helps early iterations)
        q_next = (1.0 - self.damping) * q_in + self.damping * q_star
        return q_next


def SCF_adaptive_mixing(
    H0, S, Efield, C, TYPE, Rx, Ry, Rz, H_Index_Start, H_Index_End, nocc,
    U, Znuc, Nats, Te, const,
    mixing="adaptive",              # "adaptive" or "anderson"
    alpha0=0.2,                     # initial linear-mix alpha (for "adaptive")
    alpha_min=0.02, alpha_max=0.7,  # clamp for adaptive alpha
    grow=1.15, shrink=0.5,          # alpha *= grow if improving, *= shrink if worsening
    anderson_m=6, anderson_lam=1e-10, anderson_damp=1.0,  # for "anderson"
    acc=1e-7, MAX_ITER=200
):
    """
    SCF with adaptive mixing:
    - mixing='adaptive': auto-tunes linear α from residual history
    - mixing='anderson': Anderson (Pulay-like) charge mixing (recommended)

    Returns: H, Hcoul, Hdipole, D, q, f
    """
    print('### Do SCF (adaptive mixing) ###')
    dtype = H0.dtype
    device = H0.device
    N = H0.shape[0]

    n_orbitals_per_atom = const.n_orb[TYPE]  # (Nats,)
    atom_ids = torch.repeat_interleave(
        torch.arange(len(n_orbitals_per_atom), device=Rx.device),
        n_orbitals_per_atom
    )

    # Symmetric orthogonalizer
    Z = fractional_matrix_power_symm(S, -0.5)

    # External field dipole term (symmetrized)
    Hdipole = torch.diag(-Rx[atom_ids] * Efield[0] - Ry[atom_ids] * Efield[1] - Rz[atom_ids] * Efield[2])
    Hdipole = 0.5 * Hdipole @ S + 0.5 * S @ Hdipole
    H0 = H0 + Hdipole

    # Initial density via Fermi operator on H0
    print('  Initial DM_Fermi')
    Dorth, mu0 = DM_Fermi(Z.T @ H0 @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50)
    D = Z @ Dorth @ Z.T
    DS = 2 * torch.diag(D @ S)  # AO populations per orbital
    # Build initial atomic charges q (Mulliken-like): q_A = sum_occ_on_A - Z_A
    q = -Znuc.to(DS.dtype, copy=False).to(DS.device).clone()

    q.scatter_add_(0, atom_ids, DS)

    # Mixers
    alpha = float(alpha0)
    last_Res = None
    if mixing.lower() == "anderson":
        mixer = AndersonMixer(dim=Nats, m=anderson_m, lam=anderson_lam, damping=anderson_damp,
                              device=device, dtype=dtype)
    else:
        mixer = None

    it = 0
    Res = torch.tensor([float('inf')], device=device, dtype=dtype)

    print('\nStarting cycle')
    while Res > acc and it < MAX_ITER:
        t0 = time.perf_counter()
        it += 1
        print(f"Iter {it}")

        # --- Build Coulomb contribution from current charges q ------------
        t1 = time.perf_counter()
        CoulPot = C @ q
        Hcoul_diag = U[atom_ids] * q[atom_ids] + CoulPot[atom_ids]
        Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))
        H = H0 + Hcoul
        print("  Hcoul {:.3f} s".format(time.perf_counter() - t1))

        # --- New density (out) for current H ------------------------------
        t1 = time.perf_counter()
        Dorth, mu0 = DM_Fermi(Z.T @ H @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50)
        print("  DM_Fermi {:.3f} s".format(time.perf_counter() - t1))

        # --- Back-transform density --------------------------------------
        t1 = time.perf_counter()
        D = Z @ Dorth @ Z.T
        print("  Z@Dorth@Z.T {:.3f} s".format(time.perf_counter() - t1))

        # --- Build output charges q_out from this density -----------------
        t1 = time.perf_counter()
        DS = 2 * (D * S.T).sum(dim=1)  # same as 2*diag(D@S) but faster
        q_out = -Znuc.to(DS.dtype, copy=False).to(DS.device).clone()
        q_out.scatter_add_(0, atom_ids, DS)

        # --- Residual & mixing -------------------------------------------
        # residual based on atomic charges (L2)
        Res = torch.norm(q_out - q)
        # choose mixing strategy
        if mixing.lower() == "anderson":
            q_next = mixer.step(q, q_out)
        else:
            # adaptive linear mixing: if improving fast, increase α; else decrease
            if last_Res is not None:
                if Res < 0.7 * last_Res:
                    alpha = min(alpha * grow, alpha_max)
                elif Res > 1.05 * last_Res:
                    alpha = max(alpha * shrink, alpha_min)
            q_next = (1.0 - alpha) * q + alpha * q_out

        # update and report
        last_Res = Res.clone()
        q = q_next
        print(f"  mix: method={mixing}, Res={Res.item():.3e}, "
              + (f"alpha={alpha:.3f}" if mixing.lower()!='anderson' else
                 f"Anderson m={anderson_m}, damp={anderson_damp:.2f}") )
        print("  update q {:.3f} s".format(time.perf_counter() - t1))
        print("  iter wall {:.3f} s\n".format(time.perf_counter() - t0))

    if it >= MAX_ITER and Res > acc:
        print("WARNING: SCF did not converge (Res={:.3e})".format(Res.item()))

    f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))
    return H, Hcoul, Hdipole, D, q, f
