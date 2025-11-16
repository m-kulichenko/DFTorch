import torch
from .Constants import Constants 

from sedacs.neighbor_list import NeighborState, calculate_displacement

# compile options you can tweak:
# - dynamic=True if matrix size can change between calls
# - mode="reduce-overhead" or "max-autotune" for more perf after warmup
@torch.compile
def fractional_matrix_power_symm(A: torch.Tensor, power: float = -0.5) -> torch.Tensor:
    """
    Compute A^power for (batch of) symmetric PSD matrices A using eigendecomposition.
    Shape: A (..., n, n) -> (..., n, n)
    """
    # eigh handles symmetric real matrices; returns real eigenpairs
    w, Q = torch.linalg.eigh(A)                   # w (..., n), Q (..., n, n)

    # clamp tiny/negative eigenvalues to keep things real/stable for negative powers
    eps = torch.finfo(A.dtype).eps
    w = torch.clamp(w, min=eps)

    # raise eigenvalues to the power; avoid torch.diag / torch.diag_embed to keep fusion
    d = w.pow(power)                              # (..., n)

    # Q @ diag(d) == column-scale Q by d
    Q_scaled = Q * d.unsqueeze(-2)               # (..., n, n), scales columns by d_j

    # A^p = Q @ diag(d) @ Q^T
    return Q_scaled @ Q.transpose(-2, -1)

# #@torch.compile
# def fractional_matrix_power_symm(A, power=-0.5):
    
#     # Symmetric fractional matrix power using eigendecomposition
#     eigvals, eigvecs = torch.linalg.eigh(A)

#     eigvals_clamped = torch.clamp(eigvals, min=1e-12)
#     D_power = torch.diag(eigvals_clamped ** power)
#     return eigvecs @ D_power @ eigvecs.T

def ordered_pairs_from_TYPE(TYPE: torch.Tensor, const: Constants = None):
    """
    Generate all ordered unique pairs from TYPE.
    A-B and B-A are treated as distinct pairs.

    Args:
      TYPE: torch.Tensor of atom type indices (1D or any shape).
      const: optional Constants() instance to map type index -> element label.

    Returns:
      pairs_tensor: torch.LongTensor, shape (P,2), all ordered pairs of unique types
      pairs_list: list of (intA, intB) tuples
      label_list: list of 'A-B' strings if const provided, else None
    """
    # Get unique type indices (sorted)
    unique = torch.unique(TYPE).to(torch.int64)

    # cartesian product: all ordered pairs (A,B) including A==B
    pairs_tensor = torch.cartesian_prod(unique, unique).to(torch.int64)

    # Python-friendly lists
    pairs_list = [(int(a.item()), int(b.item())) for a, b in pairs_tensor]

    label_list = None
    if const is not None:
        # Helper to get label string and strip whitespace/zeros
        def _lab(i):
            lab = str(const.label[int(i)]).strip()
            return lab if lab != '0' else str(int(i))
        label_list = [f"{_lab(a)}-{_lab(b)}" for a, b in pairs_list]

    return pairs_tensor, pairs_list, label_list
    
def list_global_tensors(ns=None):
    ns = globals() if ns is None else ns
    rows = []
    for name, obj in ns.items():
        t = None
        if torch.is_tensor(obj):
            t = obj
        elif hasattr(obj, "data") and torch.is_tensor(obj.data):
            t = obj.data
        if t is None:
            continue
        bytes_ = t.element_size() * t.numel()
        rows.append((bytes_, name, t))
    rows.sort(key=lambda x: x[0], reverse=True)  # descending by size
    total = 0
    for bytes_, name, t in rows:
        total += bytes_
        print(f"{name:>24}  size={bytes_/1e6:9.2f} MB  shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}  grad={t.requires_grad}")
    print(f"Total tensors: {len(rows)}  total size={total/1e6:.2f} MB")

def calculate_dist_dips(pos_T, long_nbr_state, cutoff):
    nbr_inds = long_nbr_state.nbr_inds
    disps = calculate_displacement(pos_T.to(torch.float64), nbr_inds,
                                long_nbr_state.lattice_lengths.to(torch.float64))
    dists = torch.linalg.norm(disps, dim=0)
    nbr_inds = torch.where((dists > cutoff) | (dists == 0.0), -1, nbr_inds)
    dists = torch.where(dists == 0, 1, dists)
    return disps.to(pos_T.dtype), dists.to(pos_T.dtype), nbr_inds