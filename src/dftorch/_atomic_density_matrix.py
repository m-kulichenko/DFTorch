from __future__ import annotations

from typing import Protocol

import torch


class _ConstLike(Protocol):
    """Minimal protocol for the `const` argument used in this module."""

    n_s: torch.Tensor  # [n_types]
    n_p: torch.Tensor  # [n_types]
    n_d: torch.Tensor  # [n_types]


def atomic_density_matrix(
    H_INDEX_START: torch.Tensor,
    HDIM: int,
    TYPE: torch.Tensor,
    const: _ConstLike,
    has_p: torch.Tensor,
    has_d: torch.Tensor,
) -> torch.Tensor:
    """Build the per-orbital *atomic* occupation vector.

    This helper constructs a 1D tensor `D_atomic` of length `HDIM` containing
    default (initial) orbital occupations for each atom, based on the element/type
    occupations stored in `const`.

    Orbital layout per atom is assumed to be:

    - `s` orbital: offset `0`
    - `p` orbitals: offsets `1,2,3` (px, py, pz)
    - `d` orbitals: offsets `4,5,6,7,8` (five d functions)

    Occupations are distributed uniformly within a shell:
    - s: `const.n_s[type]`
    - p: `const.n_p[type] / 3`
    - d: `const.n_d[type] / 5`

    Parameters
    ----------
    H_INDEX_START:
        Long tensor of shape `[N_atoms]` with the starting orbital index for each atom.
    HDIM:
        Total number of orbitals (length of the returned vector).
    TYPE:
        Long tensor of shape `[N_atoms]` with the per-atom type index.
    const:
        Object providing `n_s`, `n_p`, `n_d` tensors indexable by `TYPE`.
    has_p:
        Bool tensor of shape `[N_atoms]`, `True` for atoms with p orbitals.
    has_d:
        Bool tensor of shape `[N_atoms]`, `True` for atoms with d orbitals.

    Returns
    -------
    torch.Tensor
        1D tensor of shape `[HDIM]` with floating dtype `torch.get_default_dtype()`.

    Raises
    ------
    IndexError
        If any computed orbital indices fall outside `[0, HDIM)`.
    ValueError
        If input shapes are inconsistent.
    TypeError
        If inputs have unexpected dtypes.
    """
    # Initialize the atomic density matrix with zeros
    D_atomic = torch.zeros(
        HDIM, device=H_INDEX_START.device, dtype=torch.get_default_dtype()
    )

    D_atomic[H_INDEX_START] = 1.0 * const.n_s[TYPE]
    D_atomic[H_INDEX_START[has_p] + 1] = 1.0 * const.n_p[TYPE[has_p]] / 3
    D_atomic[H_INDEX_START[has_p] + 2] = 1.0 * const.n_p[TYPE[has_p]] / 3
    D_atomic[H_INDEX_START[has_p] + 3] = 1.0 * const.n_p[TYPE[has_p]] / 3
    D_atomic[H_INDEX_START[has_d] + 4] = 1.0 * const.n_d[TYPE[has_d]] / 5
    D_atomic[H_INDEX_START[has_d] + 5] = 1.0 * const.n_d[TYPE[has_d]] / 5
    D_atomic[H_INDEX_START[has_d] + 6] = 1.0 * const.n_d[TYPE[has_d]] / 5
    D_atomic[H_INDEX_START[has_d] + 7] = 1.0 * const.n_d[TYPE[has_d]] / 5
    D_atomic[H_INDEX_START[has_d] + 8] = 1.0 * const.n_d[TYPE[has_d]] / 5

    return D_atomic


def atomic_density_matrix_batch(
    batch_size: int,
    H_INDEX_START: torch.Tensor,
    HDIM: int,
    TYPE: torch.Tensor,
    const: _ConstLike,
    has_p: torch.Tensor,
    has_d: torch.Tensor,
) -> torch.Tensor:
    """Batched version of :func:`atomic_density_matrix`.

    Parameters
    ----------
    batch_size:
        Expected batch size `B`. Must match `H_INDEX_START.shape[0]`.
    H_INDEX_START:
        Long tensor of shape `[B, N_atoms]` with starting orbital indices per atom.
    HDIM:
        Total number of orbitals (second dimension of the output).
    TYPE:
        Long tensor of shape `[B, N_atoms]` with per-atom type indices.
    const:
        Object providing `n_s`, `n_p`, `n_d` tensors indexable by `TYPE`.
    has_p:
        Bool tensor of shape `[B, N_atoms]`, `True` for atoms with p orbitals.
    has_d:
        Bool tensor of shape `[B, N_atoms]`, `True` for atoms with d orbitals.

    Returns
    -------
    torch.Tensor
        Float tensor of shape `[B, HDIM]` with dtype `torch.get_default_dtype()`.

    Raises
    ------
    IndexError
        If any computed orbital indices fall outside `[0, HDIM)`.
    ValueError
        If input shapes are inconsistent.
    TypeError
        If inputs have unexpected dtypes.
    """
    device = H_INDEX_START.device
    B = H_INDEX_START.shape[0]
    assert B == batch_size, "batch_size mismatch"
    D_atomic = torch.zeros(
        B, HDIM, device=device, dtype=torch.get_default_dtype()
    )  # force float dtype

    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    D_atomic[batch_idx, H_INDEX_START] = 1.0 * const.n_s[TYPE]

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
