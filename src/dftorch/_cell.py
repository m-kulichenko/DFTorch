from __future__ import annotations

from typing import Optional, Union

import torch


def normalize_cell(
    LBox: Optional[Union[torch.Tensor, list, tuple]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Convert a cell specification to a (3, 3) cell matrix.

    Parameters
    ----------
    LBox : array-like of shape (3,) or (3, 3), or None
        Cell specification.  A 1-D input of length 3 is treated as
        orthorhombic box lengths and expanded to a diagonal matrix.
        A (3, 3) input is returned as-is (cast to the requested dtype).
        ``None`` represents a non-periodic (gas-phase) system.
    device : torch.device, optional
        Target device for the output tensor.
    dtype : torch.dtype, optional
        Target dtype for the output tensor.

    Returns
    -------
    cell : torch.Tensor, shape (3, 3), or None
        Full cell matrix with lattice vectors as rows, or ``None`` for
        non-periodic systems.
    """
    if LBox is None:
        return None

    cell = torch.as_tensor(LBox, device=device, dtype=dtype)

    if cell.ndim == 1:
        if cell.shape[0] != 3:
            raise ValueError("LBox must have shape (3,) or (3,3)")
        return torch.diag(cell)

    if cell.shape == (3, 3):
        return cell

    raise ValueError("LBox must be None, shape (3,), or shape (3,3)")


def normalize_cell_batch(
    LBox: Optional[Union[torch.Tensor, list, tuple]],
    B: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Convert a batched cell specification to a (B, 3, 3) tensor.

    Accepts any of the following input shapes and broadcasts to (B, 3, 3):

    * ``None``      → returns ``None`` (non-periodic).
    * ``(3,)``      → orthorhombic lengths, expanded to diagonal and tiled.
    * ``(3, 3)``    → single cell matrix, tiled B times.
    * ``(B, 3)``    → per-structure orthorhombic lengths, converted to diagonal.
    * ``(B, 3, 3)`` → already in the target shape, returned as-is.

    Parameters
    ----------
    LBox : array-like or None
        Cell specification (see above).
    B : int
        Batch size.
    device : torch.device, optional
        Target device.
    dtype : torch.dtype, optional
        Target dtype.

    Returns
    -------
    cell : torch.Tensor, shape (B, 3, 3), or None
    """
    if LBox is None:
        return None

    cell = torch.as_tensor(LBox, device=device, dtype=dtype)

    if cell.ndim == 1:
        if cell.shape[0] != 3:
            raise ValueError("LBox must have shape (3,), (3,3), (B,3), or (B,3,3)")
        return torch.diag(cell).unsqueeze(0).expand(B, 3, 3)

    if cell.ndim == 2:
        if cell.shape == (3, 3):
            return cell.unsqueeze(0).expand(B, 3, 3)
        if cell.shape == (B, 3):
            return torch.diag_embed(cell)
        raise ValueError("2D LBox must have shape (3,3) or (B,3)")

    if cell.ndim == 3:
        if cell.shape == (B, 3, 3):
            return cell
        raise ValueError("3D LBox must have shape (B,3,3)")

    raise ValueError("Invalid batched LBox shape")


def cell_inverse(
    cell: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return the matrix inverse of a periodic cell, or ``None`` for gas phase.

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3), or None

    Returns
    -------
    cell_inv : torch.Tensor, shape (3, 3), or None
    """
    if cell is None:
        return None
    return torch.linalg.inv(cell)


def cart_to_frac(
    R: torch.Tensor,
    cell: Optional[torch.Tensor],
    cell_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    R : torch.Tensor, shape (..., 3)
        Cartesian positions in Å.
    cell : torch.Tensor, shape (3, 3), or None
        Cell matrix (lattice vectors as rows).  If ``None``, ``R`` is
        returned unchanged (no periodic boundary).
    cell_inv : torch.Tensor, shape (3, 3), optional
        Pre-computed inverse of ``cell``.  Computed on-the-fly if not
        provided.

    Returns
    -------
    S : torch.Tensor, shape (..., 3)
        Fractional coordinates in [0, 1).
    """
    if cell is None:
        return R
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    return R @ cell_inv


def frac_to_cart(
    S: torch.Tensor,
    cell: Optional[torch.Tensor],
) -> torch.Tensor:
    """Transform fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    S : torch.Tensor, shape (..., 3)
        Fractional coordinates.
    cell : torch.Tensor, shape (3, 3), or None
        Cell matrix (lattice vectors as rows).  If ``None``, ``S`` is
        returned unchanged.

    Returns
    -------
    R : torch.Tensor, shape (..., 3)
        Cartesian positions in Å.
    """
    if cell is None:
        return S
    return S @ cell


def wrap_positions(
    R: torch.Tensor,
    cell: Optional[torch.Tensor],
    cell_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wrap Cartesian positions into the primary unit cell.

    Positions are mapped to fractional coordinates, reduced to [0, 1) by
    ``floor``, and transformed back to Cartesian coordinates.

    Parameters
    ----------
    R : torch.Tensor, shape (..., 3)
        Cartesian positions in Å.
    cell : torch.Tensor, shape (3, 3), or None
        Cell matrix.  If ``None``, ``R`` is returned unchanged.
    cell_inv : torch.Tensor, shape (3, 3), optional
        Pre-computed ``cell⁻¹``.  Computed on-the-fly if not provided.

    Returns
    -------
    R_wrapped : torch.Tensor, shape (..., 3)
        Wrapped Cartesian positions.
    """
    if cell is None:
        return R
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    S = R @ cell_inv
    S = S - torch.floor(S)
    return S @ cell


def minimum_image_displacement(
    dR: torch.Tensor,
    cell: Optional[torch.Tensor],
    cell_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the minimum-image convention to a displacement vector.

    Fractional components are rounded to the nearest integer so that
    the returned displacement corresponds to the shortest image.

    Parameters
    ----------
    dR : torch.Tensor, shape (..., 3)
        Raw displacement vectors in Å.
    cell : torch.Tensor, shape (3, 3), or None
        Cell matrix.  If ``None``, ``dR`` is returned unchanged.
    cell_inv : torch.Tensor, shape (3, 3), optional
        Pre-computed ``cell⁻¹``.  Computed on-the-fly if not provided.

    Returns
    -------
    dR_mic : torch.Tensor, shape (..., 3)
        Minimum-image displacement vectors in Å.
    """
    if cell is None:
        return dR
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    dS = dR @ cell_inv
    dS = dS - torch.round(dS)
    return dS @ cell


def is_orthorhombic(
    cell: Optional[torch.Tensor],
    atol: float = 1e-12,
) -> bool:
    """Return ``True`` if the cell is orthorhombic (diagonal).

    Parameters
    ----------
    cell : torch.Tensor, shape (3, 3), or None
        Cell matrix.  ``None`` (gas phase) is treated as orthorhombic.
    atol : float, default 1e-12
        Absolute tolerance for the off-diagonal elements.

    Returns
    -------
    bool
    """
    if cell is None:
        return True
    offdiag = cell - torch.diag(torch.diagonal(cell))
    return bool(torch.all(torch.abs(offdiag) < atol))


def _cell_to_pdb_cryst1(
    LBox: Union[torch.Tensor, list, tuple],
) -> tuple[float, float, float, float, float, float]:
    """Convert a periodic cell specification into PDB CRYST1 parameters.

    Parameters
    ----------
    LBox : array-like, shape (3,) or (3, 3)
        - shape (3,)   → orthorhombic lengths [Lx, Ly, Lz] in Å
        - shape (3, 3) → full cell matrix with lattice vectors as rows

    Returns
    -------
    a, b, c : float
        Lattice lengths in Å.
    alpha, beta, gamma : float
        Lattice angles in degrees (alpha = b∧c, beta = a∧c, gamma = a∧b).
    """
    cell = torch.as_tensor(LBox, dtype=torch.get_default_dtype())

    if cell.ndim == 1:
        if cell.shape[0] != 3:
            raise ValueError("LBox must have shape (3,) or (3,3)")
        a, b, c = cell
        alpha = beta = gamma = torch.tensor(90.0, dtype=cell.dtype)
        return (
            float(a.item()),
            float(b.item()),
            float(c.item()),
            float(alpha.item()),
            float(beta.item()),
            float(gamma.item()),
        )

    if cell.shape != (3, 3):
        raise ValueError("LBox must have shape (3,) or (3,3)")

    va, vb, vc = cell[0], cell[1], cell[2]

    a = torch.linalg.norm(va)
    b = torch.linalg.norm(vb)
    c = torch.linalg.norm(vc)

    def angle_deg(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cosang = torch.dot(u, v) / (torch.linalg.norm(u) * torch.linalg.norm(v))
        cosang = torch.clamp(cosang, -1.0, 1.0)
        return torch.rad2deg(torch.arccos(cosang))

    alpha = angle_deg(vb, vc)
    beta = angle_deg(va, vc)
    gamma = angle_deg(va, vb)

    return (
        float(a.item()),
        float(b.item()),
        float(c.item()),
        float(alpha.item()),
        float(beta.item()),
        float(gamma.item()),
    )
