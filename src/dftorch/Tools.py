import torch
from .Elements import label
from typing import Any, Tuple, Optional, List

#from sedacs.neighbor_list import NeighborState, calculate_displacement
from .ewald_pme.neighbor_list import NeighborState, calculate_displacement


# compile options you can tweak:
# - dynamic=True if matrix size can change between calls
# - mode="reduce-overhead" or "max-autotune" for more perf after warmup
@torch.compile
def fractional_matrix_power_symm(A: torch.Tensor, power: float = -0.5) -> torch.Tensor:
    """
    Compute the fractional matrix power A**power for a (batch of) real symmetric
    positive (semi)‑definite matrices using eigendecomposition.

    Parameters
    ----------
    A : torch.Tensor, shape (..., n, n)
        Real symmetric (PSD) matrix or batch of matrices. For negative powers
        (e.g. -0.5 for inverse square root), very small / slightly negative
        eigenvalues are clamped to eps to maintain numerical stability.
    power : float, default -0.5
        Exponent applied to eigenvalues. Common cases:
          -0.5 : inverse square root
           0.5 : square root
           -1  : inverse (for well‑conditioned matrices)

    Returns
    -------
    A_power : torch.Tensor, shape (..., n, n)
        The reconstructed matrix Q diag(w**power) Q^T where w are eigenvalues and
        Q eigenvectors from torch.linalg.eigh(A).

    Notes
    -----
    - Uses torch.linalg.eigh (real symmetric) for stability.
    - Eigenvalues are clamped at machine epsilon of A.dtype to avoid inf/NaN
      when raising to a negative power.
    - Broadcasting supports arbitrary leading batch dimensions.
    - For poorly conditioned matrices, consider preconditioning before calling.
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

def ordered_pairs_from_TYPE(
    TYPE: torch.Tensor,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Optional[List[str]]]:
    """
    Generate all ordered pairs of unique atom types present in TYPE.

    Ordered means (A,B) and (B,A) are both included (even when A == B is allowed),
    so if there are U distinct types the result has U*U pairs.

    Parameters
    ----------
    TYPE : torch.Tensor
        Tensor containing atom type indices (any shape; flattened internally).
    const : Constants, optional
        If provided, const.label is used to map type indices to element symbols
        for the string labels. Whitespace and '0' placeholders are stripped.

    Returns
    -------
    pairs_tensor : torch.LongTensor, shape (P, 2)
        All ordered pairs of unique types; P = U*U with U distinct types.
    pairs_list : list[tuple[int, int]]
        Python list of integer tuples corresponding to pairs_tensor rows.
    label_list : list[str] | None
        If const is provided, list of "A-B" string labels per ordered pair;
        otherwise None.

    Examples
    --------
    Suppose TYPE = [6, 1, 6] (C, H, C):
      Unique types = {1, 6} -> U=2
      Ordered pairs = (1,1), (1,6), (6,1), (6,6) -> P=4

    Notes
    -----
    - Sorting of unique types follows torch.unique default (ascending).
    - All outputs reside on CPU/GPU matching the input tensor device.
    """
    # Get unique type indices (sorted)
    unique = torch.unique(TYPE).to(torch.int64)

    # cartesian product: all ordered pairs (A,B) including A==B
    pairs_tensor = torch.cartesian_prod(unique, unique).to(torch.int64)

    # Python-friendly lists
    pairs_list = [(int(a.item()), int(b.item())) for a, b in pairs_tensor]

    label_list = None
        # Helper to get label string and strip whitespace/zeros
    def _lab(i):
        lab = str(label[int(i)]).strip()
        return lab if lab != '0' else str(int(i))
    label_list = [f"{_lab(a)}-{_lab(b)}" for a, b in pairs_list]

    return pairs_tensor, pairs_list, label_list
    
def list_global_tensors(ns):
    """
    Print a summary of all torch.Tensor objects found in the given namespace.

    Parameters
    ----------
    ns : dict | None
        Namespace to inspect (e.g. globals()). If None, uses module globals().

    Prints
    ------
    For each tensor:
      name, size in MB, shape, dtype, device, requires_grad flag.
    Finally:
      total number of tensors and cumulative size in MB.

    Notes
    -----
    - Also includes objects whose .data attribute is a tensor.
    - Sorted by descending tensor size (bytes).
    - Does not return a value; purely side-effect printing.
    """
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

def calculate_dist_dips(
    pos_T: torch.Tensor,
    long_nbr_state: "NeighborState",
    cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute neighbor displacements, distances, and filtered neighbor indices.

    Parameters
    ----------
    pos_T : torch.Tensor, shape (3, N) or (D, N)
        Atomic positions (Å) used to build displacements; cast to float64 internally.
    long_nbr_state : NeighborState
        Object containing:
          - nbr_inds : torch.Tensor of neighbor indices
          - lattice_lengths : torch.Tensor of box lengths (Å)
    cutoff : float
        Distance threshold (Å); neighbors with distance > cutoff are masked out.

    Returns
    -------
    disps : torch.Tensor, same dtype as pos_T
        Displacement vectors for each neighbor pair.
    dists : torch.Tensor, same dtype as pos_T
        Euclidean distances corresponding to disps.
    nbr_inds : torch.Tensor
        Neighbor indices with invalid or beyond‑cutoff entries set to -1.

    Notes
    -----
    - Zero distance entries are replaced by 1 in dists to avoid division issues downstream.
    - Casting to float64 for displacement calculation improves numerical stability.
    """
    nbr_inds = long_nbr_state.nbr_inds
    disps = calculate_displacement(pos_T.to(torch.float64), nbr_inds,
                                long_nbr_state.lattice_lengths.to(torch.float64))
    dists = torch.linalg.norm(disps, dim=0)
    nbr_inds = torch.where((dists > cutoff) | (dists == 0.0), -1, nbr_inds)
    dists = torch.where(dists == 0, 1, dists)
    return disps.to(pos_T.dtype), dists.to(pos_T.dtype), nbr_inds