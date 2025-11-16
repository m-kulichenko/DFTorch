import torch
from .Tools import fractional_matrix_power_symm
import time

def Forces(H, H0, S, C, D, D0, dH, dS,
                   dC, dVr, Efield, U, q, Rx, Ry, Rz,
                   Nats, H_INDEX_START, H_INDEX_END, const, TYPE,
                   verbose=False):
    """
    Computes atomic forces from a DFTB-like total energy expression.

    Args:
        H (Tensor): Two-electron Hamiltonian matrix (n_orb, n_orb).
        H0 (Tensor): One-electron Hamiltonian matrix (n_orb, n_orb).
        S (Tensor): Overlap matrix (n_orb, n_orb).
        C (Tensor): Coulomb interaction matrix (n_orb, n_orb) or coulomb potential vector (n_orb,).
        D (Tensor): Optimized density matrix (n_orb, n_orb).
        D0 (Tensor): Atomic reference density matrix (n_orb,).
        dH (Tensor): Derivatives of the Hamiltonian (3, n_orb, n_orb).
        dS (Tensor): Derivatives of the overlap matrix (3, n_orb, n_orb).
        dC (Tensor): Derivatives of the Coulomb matrix (3, n_orb, n_orb).
        Efield (Tensor): External electric field vector (3,).
        U (Tensor): Hubbard U parameters per atom (Nats,).
        q (Tensor): Self-consistent charge (SCC) vector (Nats,).
        Rx (Tensor): X-coordinates of atoms (Nats,).
        Ry (Tensor): Y-coordinates of atoms (Nats,).
        Rz (Tensor): Z-coordinates of atoms (Nats,).
        Nats (int): Number of atoms.
        H_INDEX_START (Tensor): Index of first orbital for each atom (Nats,).
        H_INDEX_END (Tensor): Index of last orbital for each atom (Nats,).
        const (object): Container with model constants (e.g. orbital numbers).
        TYPE (Tensor): Element type vector (Nats,).

    Returns:
        Ftot (Tensor): Total forces on atoms (3, Nats).
        Fcoul (Tensor): Coulomb interaction forces (3, Nats).
        Fband0 (Tensor): Band structure energy forces (3, Nats).
        Fdipole (Tensor): Electric dipole interaction forces (3, Nats).
        FPulay (Tensor): Pulay correction forces (3, Nats).
        FScoul (Tensor): Coulomb-related overlap derivatives contribution (3, Nats).
        FSdipole (Tensor): Dipole-related overlap derivatives contribution (3, Nats).

    Notes:
        - All forces are computed as negative gradients of the total energy.
        - Electric field forces include both direct dipole terms and overlap-derivative corrections.
        - SCC (self-consistent charge) and Pulay forces are included.
    """
    
    #Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T

    start_time1 = time.perf_counter()

    dtype = H.dtype
    device = H.device
    HDIM = H0.size(0)
    
    n_orbitals_per_atom = const.n_orb[TYPE] # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom) # Generate atom index for each orbital
    
    if dC is not None:
        if verbose: print('Doing Fcoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # Fcoul = -q_i * sum_j q_j * dCj/dRi
        Fcoul = q * (q @ dC)

        if verbose: print('Doing FScoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # FScoul
        CoulPot = C @ q
        FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
        factor = (U * q + CoulPot)*2
        dS_times_D = D*dS*factor[atom_ids].unsqueeze(-1)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim = 2) # sum of elements in each row
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)
    else:
        if verbose: print('Skipping Fcoul, done in PME.')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # Fcoul = -q_i * sum_j q_j * dCj/dRi
        Fcoul = torch.zeros((3, Nats), dtype=dtype, device=device)

        if verbose: print('Doing FScoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # FScoul
        CoulPot = C
        FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
        factor = (U * q + CoulPot)*2
        dS_times_D = D*dS*factor[atom_ids].unsqueeze(-1)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim = 2) # sum of elements in each row
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    if verbose: print('Doing Fband0')
    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4*(dH @ D).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        
    if verbose: print('Doing Pulay')
    # Pulay forces
    Z = fractional_matrix_power_symm(S, -0.5)
    SIHD = 4 * Z @ Z.T @ H @ D
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    
    if verbose: print('Doing Fdipole') # $$$ ??? a bug in Efield calculations.
    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)
    
    if verbose: print('Doing FSdipole')
    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    tmp1 = (D-D0)@dS
    tmp2 = -2*(tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    D_diff = D - D0
    n_orb = dS.shape[1]
    a = dS*D_diff.permute(1,0).unsqueeze(0) # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3,n_orb,Nats),dtype=a.dtype,device=a.device)
    outs_by_atom=outs_by_atom.index_add(2, atom_ids,a)
    new_fs = outs_by_atom.permute(0,2,1) @ dotRE[atom_ids]
    FSdipole -= 2*new_fs

    if verbose: print('Doing Repulsion')
    Frep = dVr.sum(dim=2)
    
    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    print("Forces t {:.1f} s\n".format( time.perf_counter()-start_time1 ))

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep

def ForcesShadow(H, H0, S, C, D, D0, dH, dS,
                   dC, dVr, Efield, U, q, n, Rx, Ry, Rz,
                   Nats, H_INDEX_START, H_INDEX_END, const, TYPE,
                   verbose=False):
    """
    Computes atomic forces from a DFTB-like total energy expression.

    Args:
        H (Tensor): Two-electron Hamiltonian matrix (n_orb, n_orb).
        H0 (Tensor): One-electron Hamiltonian matrix (n_orb, n_orb).
        S (Tensor): Overlap matrix (n_orb, n_orb).
        C (Tensor): Coulomb interaction matrix (n_orb, n_orb).
        D (Tensor): Optimized density matrix (n_orb, n_orb).
        D0 (Tensor): Atomic reference density matrix (n_orb,).
        dH (Tensor): Derivatives of the Hamiltonian (3, n_orb, n_orb).
        dS (Tensor): Derivatives of the overlap matrix (3, n_orb, n_orb).
        dC (Tensor): Derivatives of the Coulomb matrix (3, n_orb, n_orb).
        Efield (Tensor): External electric field vector (3,).
        U (Tensor): Hubbard U parameters per atom (Nats,).
        q (Tensor): Self-consistent charge (SCC) vector (Nats,).
        Rx (Tensor): X-coordinates of atoms (Nats,).
        Ry (Tensor): Y-coordinates of atoms (Nats,).
        Rz (Tensor): Z-coordinates of atoms (Nats,).
        Nats (int): Number of atoms.
        H_INDEX_START (Tensor): Index of first orbital for each atom (Nats,).
        H_INDEX_END (Tensor): Index of last orbital for each atom (Nats,).
        const (object): Container with model constants (e.g. orbital numbers).
        TYPE (Tensor): Element type vector (Nats,).

    Returns:
        Ftot (Tensor): Total forces on atoms (3, Nats).
        Fcoul (Tensor): Coulomb interaction forces (3, Nats).
        Fband0 (Tensor): Band structure energy forces (3, Nats).
        Fdipole (Tensor): Electric dipole interaction forces (3, Nats).
        FPulay (Tensor): Pulay correction forces (3, Nats).
        FScoul (Tensor): Coulomb-related overlap derivatives contribution (3, Nats).
        FSdipole (Tensor): Dipole-related overlap derivatives contribution (3, Nats).

    Notes:
        - All forces are computed as negative gradients of the total energy.
        - Electric field forces include both direct dipole terms and overlap-derivative corrections.
        - SCC (self-consistent charge) and Pulay forces are included.
    """
    
    #Efield = 0.3*torch.tensor([-.3,0.4,0.0], device=Rx.device).T

    start_time1 = time.perf_counter()
    dtype = H.dtype
    device = H.device
    HDIM = H0.size(0)
    
    n_orbitals_per_atom = const.n_orb[TYPE] # Compute number of orbitals per atom. shape: (Nats,)
    atom_ids = torch.repeat_interleave(torch.arange(len(n_orbitals_per_atom), device=Rx.device), n_orbitals_per_atom) # Generate atom index for each orbital

    if dC is not None:
        if verbose: print('Doing Fcoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # Fcoul = -q_i * sum_j q_j * dCj/dRi
        Fcoul = (2*q - n) * (n @ dC)

        if verbose: print('Doing FScoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # FScoul
        CoulPot = C @ n
        FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
        factor = (U * n + CoulPot)*2
        dS_times_D = D*dS*factor[atom_ids].unsqueeze(-1)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim = 2) # sum of elements in each row
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)
    else:
        if verbose: print('Skipping Fcoul, done in PME.')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # Fcoul = -q_i * sum_j q_j * dCj/dRi
        Fcoul = torch.zeros((3, Nats), dtype=dtype, device=device)

        if verbose: print('Doing FScoul')
        # Ecoul = 0.5 * q @ (C @ q) + 0.5 * torch.sum(q**2 * U)
        # FScoul
        CoulPot = C
        FScoul = torch.zeros((3, Nats), dtype=dtype, device=device)
        factor = (U * n + CoulPot)*2
        dS_times_D = D*dS*factor[atom_ids].unsqueeze(-1)
        dDS_XYZ_row_sum = torch.sum(dS_times_D, dim = 2) # sum of elements in each row
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), dDS_XYZ_row_sum) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        dDS_XYZ_col_sum = torch.sum(dS_times_D, dim=1)
        FScoul.scatter_add_(1, atom_ids.expand(3, -1), -dDS_XYZ_col_sum)

    if verbose: print('Doing Fband0')
    # Eband0 = 2 * torch.trace(H0 @ (D))
    # Fband0 = -4 * Tr[D * dH0/dR]
    Fband0 = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = 4*(dH @ D).diagonal(offset=0, dim1=1, dim2=2)
    Fband0.scatter_add_(1, atom_ids.expand(3, -1), TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
        
    if verbose: print('Doing Pulay')
    # Pulay forces
    Z = fractional_matrix_power_symm(S, -0.5)
    SIHD = 4 * Z @ Z.T @ H @ D
    FPulay = torch.zeros((3, Nats), dtype=dtype, device=device)
    TMP = -(dS @ SIHD).diagonal(offset=0, dim1=1, dim2=2)
    FPulay.scatter_add_(1, atom_ids.expand(3, -1), TMP) # sums elements from DS into q based on number of AOs, e.g. x4 p orbs for carbon or x1 for hydrogen
    
    if verbose: print('Doing Fdipole') # $$$ ??? a bug in Efield calculations.
    # Fdipole = q_i * E
    Fdipole = q.unsqueeze(0) * Efield.view(3, 1)
        
    if verbose: print('Doing FSdipole')
    # FSdipole. $$$ ??? a bug in Efield calculations.
    D0 = torch.diag(D0)
    dotRE = Rx * Efield[0] + Ry * Efield[1] + Rz * Efield[2]
    FSdipole = torch.zeros((3, Nats), dtype=dtype, device=device)
    tmp1 = (D-D0)@dS
    tmp2 = -2*(tmp1).diagonal(offset=0, dim1=1, dim2=2)
    FSdipole.scatter_add_(1, atom_ids.expand(3, -1), tmp2)
    FSdipole *= dotRE

    D_diff = D - D0
    n_orb = dS.shape[1]
    a = dS*D_diff.permute(1,0).unsqueeze(0) # 3, n_ham, n_ham
    outs_by_atom = torch.zeros((3,n_orb,Nats),dtype=a.dtype,device=a.device)
    outs_by_atom=outs_by_atom.index_add(2, atom_ids,a)
    new_fs = outs_by_atom.permute(0,2,1) @ dotRE[atom_ids]
    FSdipole -= 2*new_fs

    if verbose: print('Doing Repulsion')
    Frep = dVr.sum(dim=2)
    
    # Total force
    Ftot = Fband0 + Fcoul + Fdipole + FPulay + FScoul + FSdipole + Frep
    print("Forces t {:.1f} s".format( time.perf_counter()-start_time1 ))

    return Ftot, Fcoul, Fband0, Fdipole, FPulay, FScoul, FSdipole, Frep