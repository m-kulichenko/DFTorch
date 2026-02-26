import torch
from ._elements import label, symbol_to_number
from typing import Tuple, Optional, List
import re


# from sedacs.neighbor_list import NeighborState, calculate_displacement
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
    w, Q = torch.linalg.eigh(A)  # w (..., n), Q (..., n, n)

    # clamp tiny/negative eigenvalues to keep things real/stable for negative powers
    eps = torch.finfo(A.dtype).eps
    w = torch.clamp(w, min=eps)

    # raise eigenvalues to the power; avoid torch.diag / torch.diag_embed to keep fusion
    d = w.pow(power)  # (..., n)

    # Q @ diag(d) == column-scale Q by d
    Q_scaled = Q * d.unsqueeze(-2)  # (..., n, n), scales columns by d_j

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
        return lab if lab != "0" else str(int(i))

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
        print(
            f"{name:>24}  size={bytes_ / 1e6:9.2f} MB  shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}  grad={t.requires_grad}"
        )
    print(f"Total tensors: {len(rows)}  total size={total / 1e6:.2f} MB")


def calculate_dist_dips(
    pos_T: torch.Tensor, long_nbr_state: "NeighborState", cutoff: float
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
    disps = calculate_displacement(
        pos_T.to(torch.float64),
        nbr_inds,
        long_nbr_state.lattice_lengths.to(torch.float64),
    )
    dists = torch.linalg.norm(disps, dim=0)
    nbr_inds = torch.where((dists > cutoff) | (dists == 0.0), -1, nbr_inds)
    dists = torch.where(dists == 0, 1, dists)
    return disps.to(pos_T.dtype), dists.to(pos_T.dtype), nbr_inds


def load_spinw_to_tensor(
    path: str, device: torch.device, max_Z: int = 120
) -> torch.Tensor:
    # Initialize tensor: [0] dummy row
    w_dict = torch.zeros(max_Z, 6, device=device, dtype=torch.float64)

    element = None
    matrix_rows = []

    # Regex: element header like "H:" or "C:" etc.
    header_re = re.compile(r"^\s*([A-Za-z]{1,2})\s*:\s*$")

    def flush_current():
        nonlocal element, matrix_rows
        if element is None or not matrix_rows:
            return
        sym = element
        Z = symbol_to_number.get(sym)
        if Z is None or Z >= max_Z:
            # Unknown or out of bounds; skip
            element = None
            matrix_rows = []
            return

        # Build matrix
        mat = torch.tensor(matrix_rows, dtype=torch.float64, device=device)
        # Expect symmetric square blocks of size 1x1 (s), 2x2 (s,p), or 3x3 (s,p,d)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            # Skip malformed entries
            element = None
            matrix_rows = []
            return

        n = mat.shape[0]
        # Map to [ss, sp, sd, pp, pd, dd]
        # Default zeros; fill what is available
        vals = torch.zeros(6, dtype=torch.float64, device=device)

        if n == 1:
            # [s] -> ss
            vals[0] = mat[0, 0]  # ss
        elif n == 2:
            # [[ss, sp],
            #  [sp, pp]]
            vals[0] = mat[0, 0]  # ss
            vals[1] = mat[0, 1]  # sp
            vals[3] = mat[1, 1]  # pp
            # sd, pd, dd remain 0
        elif n == 3:
            # [[ss, sp, sd],
            #  [sp, pp, pd],
            #  [sd, pd, dd]]
            vals[0] = mat[0, 0]  # ss
            vals[1] = mat[0, 1]  # sp
            vals[2] = mat[0, 2]  # sd
            vals[3] = mat[1, 1]  # pp
            vals[4] = mat[1, 2]  # pd
            vals[5] = mat[2, 2]  # dd
        else:
            # Larger blocks are not expected; take upper-tri entries up to 6 in s,p,d order
            # ss, sp, sd, pp, pd, dd from positions (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) if available
            idxs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            for k, (i, j) in enumerate(idxs):
                if i < n and j < n:
                    vals[k] = mat[i, j]

        w_dict[Z, :] = vals

        # Reset
        element = None
        matrix_rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            # Empty line -> possible separator
            if not line.strip():
                if matrix_rows:
                    flush_current()
                continue

            m = header_re.match(line)
            if m:
                # New element header; flush previous
                flush_current()
                element = m.group(1)
                matrix_rows = []
                continue

            # Data row: numbers separated by whitespace
            if element is not None:
                nums = line.split()
                # Convert to floats
                row = [float(x) for x in nums]
                matrix_rows.append(row)

    # Flush last element
    flush_current()

    return w_dict * 27.21138625


def load_spinw_to_matrix(
    path: str, device: torch.device, max_Z: int = 120
) -> torch.Tensor:
    """
    Parse spinw.txt and return per-element 3x3 symmetric matrices.
    The matrix layout is:
        [[ss, sp, sd],
         [sp, pp, pd],
         [sd, pd, dd]]
    For elements with smaller blocks:
      - 1x1: only ss set; others remain 0
      - 2x2: ss, sp, pp set; sd, pd, dd remain 0
      - 3x3: full matrix set
    Missing entries are left as zeros.

    Returns
    -------
    W : torch.Tensor, shape (max_Z, 3, 3)
        W[Z] is the 3x3 matrix for atomic number Z. Row 0 is dummy.
    """
    import re

    W = torch.zeros(max_Z, 3, 3, device=device, dtype=torch.float64)

    element = None
    matrix_rows = []

    header_re = re.compile(r"^\s*([A-Za-z]{1,2})\s*:\s*$")

    def flush_current():
        nonlocal element, matrix_rows
        if element is None or not matrix_rows:
            return
        sym = element
        Z = symbol_to_number.get(sym)
        # Skip unknown or out-of-range elements
        if Z is None or Z >= max_Z:
            element = None
            matrix_rows = []
            return

        mat = torch.tensor(matrix_rows, dtype=torch.float64, device=device)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            # Malformed block; skip
            element = None
            matrix_rows = []
            return

        n = mat.shape[0]
        # Fill into a 3x3 accumulator
        acc = torch.zeros(3, 3, dtype=torch.float64, device=device)

        if n >= 1:
            acc[0, 0] = mat[0, 0]  # ss
        if n >= 2:
            acc[0, 1] = mat[0, 1]  # sp
            acc[1, 0] = mat[0, 1]
            acc[1, 1] = mat[1, 1]  # pp
        if n >= 3:
            acc[0, 2] = mat[0, 2]  # sd
            acc[2, 0] = mat[0, 2]
            acc[1, 2] = mat[1, 2]  # pd
            acc[2, 1] = mat[1, 2]
            acc[2, 2] = mat[2, 2]  # dd

        W[Z] = acc

        # Reset
        element = None
        matrix_rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if matrix_rows:
                    flush_current()
                continue

            m = header_re.match(line)
            if m:
                flush_current()
                element = m.group(1)
                matrix_rows = []
                continue

            if element is not None:
                nums = line.split()
                row = [float(x) for x in nums]
                matrix_rows.append(row)

    flush_current()

    return W * 27.21138625
