import os
import re
from typing import List, Optional, Tuple

import torch

from ._elements import label, symbol_to_number


_COMPILE_ENABLED = os.environ.get("DFTORCH_ENABLE_COMPILE", "0") != "0"


def _maybe_compile(fn, fullgraph=False, dynamic=False):
    if not _COMPILE_ENABLED:
        return fn
    return torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic)


# IMPORTANT: do not import `dftorch.ewald_pme` (or neighbor_list) at module import
# time; it can try to load optional Triton backends and emit warnings.


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


fractional_matrix_power_symm_eager = fractional_matrix_power_symm
fractional_matrix_power_symm = _maybe_compile(
    fractional_matrix_power_symm,
)


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
    pos_T, long_nbr_state, cutoff
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
    # Local import to avoid pulling in optional Triton code (and related warnings)
    # during normal `import dftorch`.
    from .ewald_pme.neighbor_list import calculate_displacement

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


def normalize_coulomb_settings(dftorch_params: dict, cell, context: str = "DFTorch"):
    """Validate/sanitize Coulomb settings for periodic and non-periodic runs.

    Rules enforced:
    1) Non-periodic: force COULOMB_CUTOFF=10000.
    2) PME: require each cell-vector length >= 2 * COULOMB_CUTOFF.
    3) Non-periodic + PME: switch to FULL and print a message.
    """
    method = str(dftorch_params.get("COUL_METHOD", "FULL")).upper()

    if cell is None:
        dftorch_params["COULOMB_CUTOFF"] = 10000.0
        if method == "PME":
            print(
                f"[{context}] Non-periodic system with COUL_METHOD='PME' detected. "
                "Switching to COUL_METHOD='FULL'."
            )
            dftorch_params["COUL_METHOD"] = "FULL"
            method = "FULL"

    cutoff = float(dftorch_params.get("COULOMB_CUTOFF", 10.0))

    if method == "PME":
        cell_t = torch.as_tensor(cell)
        vec_lengths = torch.linalg.norm(cell_t, dim=-1)
        min_dim = float(vec_lengths.min().item())
        if min_dim < 2.0 * cutoff:
            raise ValueError(
                "PME requires each cell dimension (lattice-vector length) to be "
                f">= 2 * COULOMB_CUTOFF. Got min cell dimension {min_dim:.6f} "
                f"and COULOMB_CUTOFF {cutoff:.6f}."
            )

    return dftorch_params["COUL_METHOD"], cutoff


def load_spinw_to_matrix(
    path: str, device: torch.device, max_Z: int = 120
) -> torch.Tensor:
    """
    Parse spinw.txt in either mio-1-1 or 3ob-3-1 format and return
    per-element 3x3 symmetric spin-constant matrices.

    Supported formats
    -----------------
    mio-1-1:  plain ``Element:`` headers, no wrapper block, no # comments
    3ob-3-1:  ``SpinConstants { ... }`` wrapper, ``Element { ... }`` blocks,
              ``#`` comment lines

    Returns
    -------
    W : torch.Tensor, shape (max_Z, 3, 3)
        W[Z] is the 3x3 matrix for atomic number Z (row 0 is dummy).
        Values are converted from Hartree to eV (* 27.21138625).
    """
    import re

    W = torch.zeros(max_Z, 3, 3, device=device, dtype=torch.float64)

    element = None
    matrix_rows: list = []

    # mio format:  "H:"  or  "Zn:"
    header_plain = re.compile(r"^\s*([A-Za-z]{1,3})\s*:\s*$")
    # 3ob format:  "H {"  or  "Zn {"  (but NOT "SpinConstants {")
    header_block = re.compile(r"^\s*([A-Za-z]{1,3})\s*\{\s*$")
    # numeric data row (may have trailing # comment)
    data_re = re.compile(r"^\s*([-+]?\d[\d.eE+\-]*(?:\s+[-+]?\d[\d.eE+\-]*)*)")

    def flush():
        nonlocal element, matrix_rows
        if element is None or not matrix_rows:
            element = None
            matrix_rows = []
            return
        Z = symbol_to_number.get(element)
        if Z is None or Z >= max_Z:
            element = None
            matrix_rows = []
            return

        mat = torch.tensor(matrix_rows, dtype=torch.float64, device=device)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            element = None
            matrix_rows = []
            return

        n = mat.shape[0]
        acc = torch.zeros(3, 3, dtype=torch.float64, device=device)
        if n >= 1:
            acc[0, 0] = mat[0, 0]
        if n >= 2:
            acc[0, 1] = mat[0, 1]
            acc[1, 0] = mat[0, 1]
            acc[1, 1] = mat[1, 1]
        if n >= 3:
            acc[0, 2] = mat[0, 2]
            acc[2, 0] = mat[0, 2]
            acc[1, 2] = mat[1, 2]
            acc[2, 1] = mat[1, 2]
            acc[2, 2] = mat[2, 2]

        W[Z] = acc
        element = None
        matrix_rows = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # ── strip inline # comments ──────────────────────────────
            hash_pos = line.find("#")
            if hash_pos >= 0:
                line = line[:hash_pos]
            line_stripped = line.strip()

            # skip empty lines and block delimiters
            if not line_stripped or line_stripped in ("{", "}"):
                # "}" can close an element block → flush
                if line_stripped == "}":
                    flush()
                continue

            # skip wrapper keyword  "SpinConstants {"
            if re.match(r"^\s*SpinConstants\s*\{", line):
                continue

            # ── element header (3ob style):  "H {"  ──────────────────
            m = header_block.match(line)
            if m:
                flush()
                element = m.group(1)
                matrix_rows = []
                continue

            # ── element header (mio style):  "H:"  ───────────────────
            m = header_plain.match(line)
            if m:
                flush()
                element = m.group(1)
                matrix_rows = []
                continue

            # ── numeric data row ──────────────────────────────────────
            if element is not None:
                m = data_re.match(line)
                if m:
                    row = [float(x) for x in m.group(1).split()]
                    matrix_rows.append(row)

    flush()  # last element in file

    return W * 27.21138625


def load_hubbard_derivs(
    path: str, device: torch.device, max_Z: int = 120
) -> torch.Tensor:
    """
    Parse a hubbard_derivative.txt file and return per-element dU/dq tensor.

    Format (3ob-3-1 style):
        Br = -0.0573
         C = -0.1492
        ...

    Values are in Hartree/e, converted to eV/e (* 27.21138625).

    Returns
    -------
    dU_dq : torch.Tensor, shape (max_Z,)
        dU_dq[Z] is the Hubbard derivative for atomic number Z.
        Zero for elements not present in the file.
    """
    dU_dq = torch.zeros(max_Z, device=device, dtype=torch.float64)

    entry_re = re.compile(r"^\s*([A-Za-z]{1,3})\s*=\s*([-+]?\d[\d.eE+\-]*)\s*$")

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            # strip inline comments
            hash_pos = line.find("#")
            if hash_pos >= 0:
                line = line[:hash_pos].strip()
            if not line:
                continue
            m = entry_re.match(line)
            if m:
                sym = m.group(1).strip()
                val = float(m.group(2))
                Z = symbol_to_number.get(sym)
                if Z is not None and Z < max_Z:
                    dU_dq[Z] = val

    return dU_dq * 27.21138625
