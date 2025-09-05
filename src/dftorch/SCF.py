import torch
from .Tools import fractional_matrix_power_symm
from .DM_Fermi import DM_Fermi
import time

def SCF(H0, S, Efield, C, TYPE, Rx, Ry, Rz, H_Index_Start, H_Index_End, nocc,
        U, Znuc, Nats, Te, const, alpha=0.2, acc=1e-7, MAX_ITER=200):
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
    n_orbitals_per_atom = const.n_orb[TYPE] # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom) # Generate atom index for each orbital

    Z = fractional_matrix_power_symm(S, -0.5)
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

    Res = torch.tensor([1.0], device=device)
    it = 0

    print('\nStarting cycle')
    while Res > acc and it < MAX_ITER:
        start_time = time.perf_counter()
        it += 1
        print("Iter {}".format(it))
        if it == MAX_ITER:
            print('Did not converge')
        
        #torch.cuda.synchronize()
        start_time1 = time.perf_counter()
        CoulPot = C @ q
        Hcoul_diag = U[atom_ids] * q[atom_ids] + CoulPot[atom_ids]        
        Hcoul = 0.5 * (Hcoul_diag.unsqueeze(1) * S + S * Hcoul_diag.unsqueeze(0))  
        H = H0 + Hcoul
        #torch.cuda.synchronize()
        print("  Hcoul {:.1f} s".format( time.perf_counter()-start_time1 ))

        start_time1 = time.perf_counter()

        Dorth, mu0 = DM_Fermi(Z.T @ H @ Z, Te, nocc, mu_0=None, m=18, eps=1e-9, MaxIt=50)
        
        #torch.cuda.synchronize()
        print("  DM_Fermi {:.1f} s".format( time.perf_counter()-start_time1 ))

        start_time1 = time.perf_counter()
        
        #D = Z @ Dorth @ Z.T
        tmp = torch.matmul(Z, Dorth)
        D = torch.matmul(tmp, Z.T)
        del tmp
        
        #torch.cuda.synchronize()
        print("  Z@Dorth@Z.T {:.1f} s".format( time.perf_counter()-start_time1 ))

        start_time1 = time.perf_counter()
        
        q_old = q.clone()
        DS = 2 * (D * S.T).sum(dim=1)  # same as DS = 2 * torch.diag(D @ S)
        q = -1.0 * Znuc
        q.scatter_add_(0, atom_ids, DS) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen

        Res = torch.norm(q - q_old)
        q = (1-alpha)*q_old + alpha * q 
        
        #torch.cuda.synchronize()
        print("  update q {:.1f} s".format( time.perf_counter()-start_time1 ))
        
        

        
        print("Res = {:.9f}, t = {:.1f} s\n".format(Res.item(), time.perf_counter()-start_time ))
        
    f = torch.linalg.eigvalsh(0.5 * (Dorth + Dorth.T))
    return H, Hcoul, Hdipole, D, q, f