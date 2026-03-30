import torch


def normalize_cell(LBox, device=None, dtype=None):
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


def normalize_cell_batch(LBox, B, device=None, dtype=None):
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


def cell_inverse(cell):
    if cell is None:
        return None
    return torch.linalg.inv(cell)


def cart_to_frac(R, cell, cell_inv=None):
    if cell is None:
        return R
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    return R @ cell_inv


def frac_to_cart(S, cell):
    if cell is None:
        return S
    return S @ cell


def wrap_positions(R, cell, cell_inv=None):
    """
    Wrap Cartesian positions into the unit cell using fractional coordinates.

    Parameters
    ----------
    R : (..., 3) tensor
    cell : (3, 3) tensor or None
    cell_inv : (3, 3) tensor or None
    """
    if cell is None:
        return R
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    S = R @ cell_inv
    S = S - torch.floor(S)
    return S @ cell


def minimum_image_displacement(dR, cell, cell_inv=None):
    if cell is None:
        return dR
    if cell_inv is None:
        cell_inv = torch.linalg.inv(cell)
    dS = dR @ cell_inv
    dS = dS - torch.round(dS)
    return dS @ cell


def is_orthorhombic(cell, atol=1e-12):
    if cell is None:
        return True
    offdiag = cell - torch.diag(torch.diagonal(cell))
    return bool(torch.all(torch.abs(offdiag) < atol))


def _cell_to_pdb_cryst1(LBox):
    """
    Convert a periodic cell specification into PDB CRYST1 parameters.

    Accepts
    -------
    LBox : tensor-like
        - shape (3,)   -> orthorhombic lengths [Lx, Ly, Lz]
        - shape (3,3)  -> full cell matrix with lattice vectors as rows

    Returns
    -------
    a, b, c, alpha, beta, gamma : float
        Lengths in Angstrom and angles in degrees.
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

    def angle_deg(u, v):
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
