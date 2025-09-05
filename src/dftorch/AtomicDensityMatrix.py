import torch
def AtomicDensityMatrix(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc):
    """
    Vectorized construction of the atomic density matrix D_atomic.

    Parameters:
    - Nr_atoms (int): Number of atoms
    - H_INDEX_START (Tensor[int]): Start orbital indices per atom (length Nr_atoms)
    - H_INDEX_END (Tensor[int]): End orbital indices per atom (length Nr_atoms)
    - HDIM (int): Total dimension of the density matrix (number of orbitals)
    - Znuc (Tensor[float]): Nuclear charges per atom (length Nr_atoms)

    Returns:
    - D_atomic (Tensor[float]): Atomic density matrix as 1D tensor of length HDIM
    """
    # Initialize the atomic density matrix with zeros
    D_atomic = torch.zeros(HDIM, device=H_INDEX_START.device)

    # Start indexing
    INDEX = 0

    # Loop over all atoms
    for I in range(Nr_atoms):
        # Calculate the number of orbitals for the atom I
        N_orb = H_INDEX_END[I] - H_INDEX_START[I] + 1
        
        if N_orb == 1:
            INDEX += 1
            D_atomic[INDEX-1] = Znuc[I]  # PyTorch uses 0-based indexing
        else:
            if Znuc[I] <= 2:
                INDEX += 1
                D_atomic[INDEX-1] = Znuc[I]

                INDEX += 1
                D_atomic[INDEX-1] = 0
                INDEX += 1
                D_atomic[INDEX-1] = 0
                INDEX += 1
                D_atomic[INDEX-1] = 0
            else:
                INDEX += 1
                D_atomic[INDEX-1] = 2

                INDEX += 1
                OCC = (Znuc[I] - 2) / 3
                D_atomic[INDEX-1] = OCC
                INDEX += 1
                D_atomic[INDEX-1] = OCC
                INDEX += 1
                D_atomic[INDEX-1] = OCC

    return D_atomic


def AtomicDensityMatrix_vectorized(Nr_atoms, H_INDEX_START, H_INDEX_END, HDIM, Znuc):
    """
    Vectorized construction of the atomic density matrix D_atomic.

    Parameters:
    - Nr_atoms (int): Number of atoms
    - H_INDEX_START (Tensor[int]): Start orbital indices per atom (length Nr_atoms)
    - H_INDEX_END (Tensor[int]): End orbital indices per atom (length Nr_atoms)
    - HDIM (int): Total dimension of the density matrix (number of orbitals)
    - Znuc (Tensor[float]): Nuclear charges per atom (length Nr_atoms)

    Returns:
    - D_atomic (Tensor[float]): Atomic density matrix as 1D tensor of length HDIM
    """

    # Initialize D_atomic as float to allow fractional occupations
    D_atomic = torch.zeros(HDIM, device=Znuc.device)

    # Determine the number of orbitals per atom
    N_orb = H_INDEX_END - H_INDEX_START + 1

    # Build flat orbital indices per atom
    offsets = torch.zeros(Nr_atoms, dtype=torch.long, device=Znuc.device)
    offsets[1:] = torch.cumsum(N_orb[:-1], dim=0)
    flat_indices = torch.arange(HDIM, device=Znuc.device)

    # Atom ID for each orbital
    orbital_atom_idx = torch.repeat_interleave(torch.arange(Nr_atoms, device=Znuc.device), N_orb)

    # Identify which orbitals belong to which atom
    Znuc_expanded = Znuc[orbital_atom_idx]
    N_orb_expanded = N_orb[orbital_atom_idx]

    # Compute occupancy
    is_1s = (N_orb_expanded == 1)
    is_He = (Znuc_expanded <= 2) & (~is_1s)
    is_heavy = (Znuc_expanded > 2) & (~is_1s)

    # Assign occupations
    D_atomic[is_1s.nonzero().squeeze()] = 1.0*Znuc_expanded[is_1s]
    D_atomic[is_He.nonzero().squeeze()[0::4]] = 1.0*Znuc_expanded[is_He][0::4]  # first orbital
    # next 3 orbitals are zero by default

    occ_heavy = ((Znuc_expanded[is_heavy] - 2.0) / 3.0)
    
    D_atomic[is_heavy.nonzero().squeeze()[0::4]] = 2.0  # first orbital gets 2 electrons
    D_atomic[is_heavy.nonzero().squeeze()[1::4]] = occ_heavy
    D_atomic[is_heavy.nonzero().squeeze()[2::4]] = occ_heavy
    D_atomic[is_heavy.nonzero().squeeze()[3::4]] = occ_heavy

    return D_atomic
