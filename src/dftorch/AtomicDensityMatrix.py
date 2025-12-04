import torch
def atomic_density_matrix( H_INDEX_START, HDIM, TYPE, const, has_p, has_d):
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
    D_atomic = torch.zeros(HDIM, device=H_INDEX_START.device, dtype=torch.get_default_dtype())

    D_atomic[H_INDEX_START] = 1.0*const.n_s[TYPE]
    D_atomic[H_INDEX_START[has_p]+1] = 1.0*const.n_p[TYPE[has_p]]/3
    D_atomic[H_INDEX_START[has_p]+2] = 1.0*const.n_p[TYPE[has_p]]/3
    D_atomic[H_INDEX_START[has_p]+3] = 1.0*const.n_p[TYPE[has_p]]/3
    D_atomic[H_INDEX_START[has_d]+4] = 1.0*const.n_d[TYPE[has_d]]/5
    D_atomic[H_INDEX_START[has_d]+5] = 1.0*const.n_d[TYPE[has_d]]/5
    D_atomic[H_INDEX_START[has_d]+6] = 1.0*const.n_d[TYPE[has_d]]/5
    D_atomic[H_INDEX_START[has_d]+7] = 1.0*const.n_d[TYPE[has_d]]/5
    D_atomic[H_INDEX_START[has_d]+8] = 1.0*const.n_d[TYPE[has_d]]/5

    return D_atomic

def atomic_density_matrix_batch(batch_size, H_INDEX_START, HDIM, TYPE, const, has_p, has_d):
    """
    Vectorized batched atomic density matrix.

    Parameters:
    - batch_size (int)
    - H_INDEX_START: LongTensor [B, N_atoms]
    - HDIM (int)
    - TYPE: LongTensor [B, N_atoms]
    - const: object with n_s, n_p, n_d tensors indexed by TYPE
    - has_p: BoolTensor [B, N_atoms] atoms possessing p orbitals
    - has_d: BoolTensor [B, N_atoms] atoms possessing d orbitals

    Returns:
    - D_atomic: FloatTensor [B, HDIM]
    """
    device = H_INDEX_START.device
    B = H_INDEX_START.shape[0]
    assert B == batch_size, "batch_size mismatch"
    D_atomic = torch.zeros(B, HDIM, device=device, dtype=torch.get_default_dtype())  # force float dtype

    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    D_atomic[batch_idx, H_INDEX_START] = (1.0 * const.n_s[TYPE])

    # p occupations
    if has_p.any():
        b_p, a_p = has_p.nonzero(as_tuple=True)
        start_p = H_INDEX_START[b_p, a_p]
        type_p = TYPE[b_p, a_p]
        occ_p = (1.0 * const.n_p[type_p]) / 3.0
        for k in (1, 2, 3):
            idx_k = start_p + k
            if (idx_k >= HDIM).any():
                raise IndexError("p orbital index out of range")
            D_atomic[b_p, idx_k] = occ_p

    # d occupations
    if has_d.any():
        b_d, a_d = has_d.nonzero(as_tuple=True)
        start_d = H_INDEX_START[b_d, a_d]
        type_d = TYPE[b_d, a_d]
        occ_d = (1.0 * const.n_d[type_d]) / 5.0
        for k in (4, 5, 6, 7, 8):
            idx_k = start_d + k
            if (idx_k >= HDIM).any():
                raise IndexError("d orbital index out of range")
            D_atomic[b_d, idx_k] = occ_d

    return D_atomic
    return D_atomic
