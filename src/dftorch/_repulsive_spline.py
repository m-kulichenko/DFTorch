from __future__ import annotations

import torch

from ._nearestneighborlist import (
    vectorized_nearestneighborlist,
    vectorized_nearestneighborlist_batch,
)


def get_repulsion_energy(
    R_rep_tensor: torch.Tensor,
    rep_splines_tensor: torch.Tensor,
    close_exp_tensor: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell,
    Rcut: float,
    Nats: int,
    const,
    verbose: bool,
    compute_stress: bool = True,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Compute the total repulsive energy and its Cartesian derivatives.

    Parameters
    ----------
    R_rep_tensor : (P, M) torch.Tensor
        Radial grid (Å) per pair type (P = number of pair types).
    rep_splines_tensor : (P, M, >=6) torch.Tensor
        Spline coefficients c0..c5 (Hartree) for 5th‑order local polynomial segments.
    close_exp_tensor : (P, 3) torch.Tensor
        Close exponential parameters for each pair type.
    TYPE : (Nats,) torch.Tensor
        Atom type indices (compatible with const.label).
    RX, RY, RZ : (Nats,) torch.Tensor
        Atomic Cartesian coordinates in Å.
    cell : sequence or tensor
        Periodic cell specification:
        - shape (3,) for orthorhombic box lengths
        - shape (3,3) for a triclinic cell matrix
    Rcut : float
        Cutoff (Å) used to build the neighbor list.
    Nats : int
        Number of atoms.
    const : object
        Contains chemical label mapping for pair typing.
    verbose : bool
        If True, neighbor list routine may emit timing info.
    compute_stress : bool
        If True, also compute and return the analytical pair-virial
        stress tensor (requires ``cell`` to be a (3,3) matrix).

    Returns
    -------
    Vr : torch.Tensor (scalar, eV)
        Total repulsive energy.
    dVr : torch.Tensor, shape (3, Nats, Nats), eV/Å
        Antisymmetric matrix of pairwise force contributions:
        dVr[:, i, j] = +∂E/∂r_i from pair (i,j); dVr[:, j, i] = −dVr[:, i, j].
    sigma_rep : torch.Tensor, shape (3, 3), eV/ų (only when ``compute_stress=True``)
        Symmetrised repulsive virial stress tensor:
        σ_{αβ} = (1/V) Σ_{ij} (dV/dR) R_{ij,α} R_{ij,β} / R_{ij}.

    Notes
    -----
    - Distances converted to Bohr internally (factor 0.52917721).
    - energy converted Hartree → eV using 27.21138625.
    - Neighbor list uses minimum image (min_image_only=True).
    """
    _, _, nnRx, nnRy, nnRz, nnType, _, _, neighbor_I, neighbor_J, IJ_pair_type, _ = (
        vectorized_nearestneighborlist(
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            Rcut,
            Nats,
            const,
            min_image_only=True,
            verbose=verbose,
        )
    )

    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)
    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    # ── per-pair starting distance ────────────────────────────────────
    R0 = R_rep_tensor[IJ_pair_type, 0]  # (P,) first spline knot
    use_exp = dR_mskd < R0  # (P,) bool mask

    # ── spline indices (only needed for spline region) ────────────────
    indices = (
        torch.searchsorted(
            R_rep_tensor[IJ_pair_type], dR_mskd.unsqueeze(-1), right=True
        ).squeeze(-1)
        - 1
    )
    # Optionally clamp to keep indices in bounds
    # K = R_rep_tensor.size(1)
    # indices = indices.clamp(min=0, max=K-1)

    dx = (dR_mskd - R_rep_tensor[IJ_pair_type, indices]) / 0.52917721

    # ── spline energy ─────────────────────────────────────────────────
    Vr = (
        rep_splines_tensor[IJ_pair_type, indices, 0]
        + rep_splines_tensor[IJ_pair_type, indices, 1] * dx
        + rep_splines_tensor[IJ_pair_type, indices, 2] * dx**2
        + rep_splines_tensor[IJ_pair_type, indices, 3] * dx**3
        + rep_splines_tensor[IJ_pair_type, indices, 4] * dx**4
        + rep_splines_tensor[IJ_pair_type, indices, 5] * dx**5
    )

    # ── exponential energy  E = exp(a0*r + a1) + a2 ──────────────────
    Vr_exp = (
        torch.exp(
            -close_exp_tensor[IJ_pair_type, 0] * dR_mskd / 0.52917721
            + close_exp_tensor[IJ_pair_type, 1]
        )
        + close_exp_tensor[IJ_pair_type, 2]
    )

    # ── select per pair ───────────────────────────────────────────────
    Vr = torch.where(use_exp, Vr_exp, Vr)
    Vr = Vr.sum() * 27.21138625  # eV

    # ── spline gradient  dV/dR (Ha/Bohr) ─────────────────────────────
    dVr = torch.zeros((3, Nats * Nats), device=RX.device)
    ind_start = torch.arange(Nats, device=RX.device)
    # now, it's Ha/Bohr
    dVr_dR = (
        rep_splines_tensor[IJ_pair_type, indices, 1]
        + 2 * rep_splines_tensor[IJ_pair_type, indices, 2] * dx
        + 3 * rep_splines_tensor[IJ_pair_type, indices, 3] * dx**2
        + 4 * rep_splines_tensor[IJ_pair_type, indices, 4] * dx**3
        + 5 * rep_splines_tensor[IJ_pair_type, indices, 5] * dx**4
    )

    # ── exponential gradient  dV/dR (Ha/Bohr) ────────────────────────
    dVr_exp_dR = -close_exp_tensor[IJ_pair_type, 0] * torch.exp(
        -close_exp_tensor[IJ_pair_type, 0] * dR_mskd / 0.52917721
        + close_exp_tensor[IJ_pair_type, 1]
    )

    # ── select per pair ───────────────────────────────────────────────
    dVr_dR = torch.where(use_exp, dVr_exp_dR, dVr_dR)

    # ── stress tensor (pair virial) ──────────────────────────────────
    sigma_rep = None
    if compute_stress and cell is not None:
        Rab = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1)[nn_mask]  # (P, 3)
        dVr_dR_eV = dVr_dR * 27.21138625 / 0.52917721  # eV/Å
        Vcell = torch.abs(torch.det(cell))
        sigma_rep = (
            dVr_dR_eV[:, None, None]
            * (Rab.unsqueeze(-1) * Rab.unsqueeze(-2))
            / dR_mskd[:, None, None]
        ).sum(dim=0) / Vcell
        sigma_rep = 0.5 * (sigma_rep + sigma_rep.T)

    # ── accumulate forces ───────────────────
    dR_dxyz = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=0)[:, nn_mask] / dR_mskd
    dVr.index_add_(
        1, ind_start[neighbor_I] * Nats + ind_start[neighbor_J], dVr_dR * dR_dxyz
    )
    # now, it's eV/A
    dVr = dVr.reshape(3, Nats, Nats) * 27.21138625 / 0.52917721
    dVr = dVr - torch.transpose(dVr, 1, 2)
    del nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, _

    if sigma_rep is not None:
        return Vr, dVr, sigma_rep
    return Vr, dVr, None


def get_repulsion_energy_batch(
    R_rep_tensor: torch.Tensor,
    rep_splines_tensor: torch.Tensor,
    close_exp_tensor: torch.Tensor,
    TYPE: torch.Tensor,
    RX: torch.Tensor,
    RY: torch.Tensor,
    RZ: torch.Tensor,
    cell: torch.Tensor,
    Rcut: float,
    Nats: int,
    const,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the total repulsive energy and its Cartesian derivatives.
    Notes
    -----
    - Distances converted to Bohr internally (factor 0.52917721).
    - energy converted Hartree → eV using 27.21138625.
    - Neighbor list uses minimum image (min_image_only=True).
    """
    batch_size = RX.shape[0]
    _, _, nnRx, nnRy, nnRz, nnType, _, _, neighbor_I, neighbor_J, IJ_pair_type, _ = (
        vectorized_nearestneighborlist_batch(
            TYPE,
            RX,
            RY,
            RZ,
            cell,
            Rcut,
            Nats,
            const,
            upper_tri_only=True,
            remove_self_neigh=False,
            min_image_only=True,
            verbose=verbose,
        )
    )
    Rab_X = nnRx - RX.unsqueeze(-1)
    Rab_Y = nnRy - RY.unsqueeze(-1)
    Rab_Z = nnRz - RZ.unsqueeze(-1)
    dR = torch.norm(torch.stack((Rab_X, Rab_Y, Rab_Z), dim=-1), dim=-1)

    valid_pairs = (neighbor_I >= 0) & (
        neighbor_J >= 0
    )  # $$$ maybe '& (neighbor_J >= 0)' is not necessary???
    nn_mask = nnType != -1  # mask to exclude zero padding from the neigh list
    dR_mskd = dR[nn_mask]

    safe_IJ = IJ_pair_type.clamp(min=0)
    R_rep_valid = R_rep_tensor[safe_IJ][valid_pairs]

    # ── per-pair starting distance ────────────────────────────────────
    R0 = R_rep_valid[:, 0]  # (P,) first spline knot
    use_exp = dR_mskd < R0  # (P,) bool mask

    K = R_rep_valid.size(1)
    indices = (
        torch.searchsorted(R_rep_valid, dR_mskd.unsqueeze(-1), right=True).squeeze(-1)
        - 1
    ) % K  # ← wraps -1 → K-1 (last segment), same as non-batched behavior
    # Optionally clamp to keep indices in bounds
    # K = R_rep_tensor.size(1)
    # indices = indices.clamp(min=0, max=K-1)

    R_selected = torch.gather(R_rep_valid, 1, indices.unsqueeze(1)).squeeze(1)  # (P,)
    dx = (dR_mskd - R_selected) / 0.52917721

    sel_IJ = IJ_pair_type[valid_pairs]

    Vr = torch.zeros(batch_size, device=RX.device)
    batch_ids = (
        torch.arange(batch_size, device=RX.device)
        .unsqueeze(1)
        .expand_as(safe_IJ)[valid_pairs]
    )

    # ── spline energy (Horner) ────────────────────────────────────────
    coeffs = rep_splines_tensor[sel_IJ, indices]  # (P, 6)
    # pair_interactions = sum_{n=0..5} a_n * dx^n
    # Horner: ((((a5*dx + a4)*dx + a3)*dx + a2)*dx + a1)*dx + a0
    poly = coeffs[:, 5]
    poly = poly * dx + coeffs[:, 4]
    poly = poly * dx + coeffs[:, 3]
    poly = poly * dx + coeffs[:, 2]
    poly = poly * dx + coeffs[:, 1]
    Vr_spline = poly * dx + coeffs[:, 0]

    # ── exponential energy ────────────────────────────────────────────
    a = close_exp_tensor[sel_IJ]  # (P, 3)
    exp_arg = -a[:, 0] * dR_mskd / 0.52917721 + a[:, 1]
    Vr_exp = torch.exp(exp_arg) + a[:, 2]

    # ── select per pair ───────────────────────────────────────────────
    pair_interactions = torch.where(use_exp, Vr_exp, Vr_spline)
    Vr.index_add_(0, batch_ids, pair_interactions * 27.21138625)

    # ── spline gradient (Horner, Ha/Bohr) ────────────────────────────
    # now, it's Ha/Bohr
    dpoly = 5.0 * coeffs[:, 5]
    dpoly = dpoly * dx + 4.0 * coeffs[:, 4]
    dpoly = dpoly * dx + 3.0 * coeffs[:, 3]
    dpoly = dpoly * dx + 2.0 * coeffs[:, 2]
    dVr_spline = dpoly * dx + coeffs[:, 1]  # Ha/Bohr

    # ── exponential gradient (Ha/Bohr) ───────────────────────────────
    a_exp = close_exp_tensor[sel_IJ]  # (P, 3)
    dVr_exp = -a_exp[:, 0] * torch.exp(exp_arg)

    # ── select per pair ───────────────────────────────────────────────
    dVr_dR = torch.where(use_exp, dVr_exp, dVr_spline)

    # ── accumulate forces ─────────────────────────────────────────────
    dR_dxyz = torch.stack((Rab_X, Rab_Y, Rab_Z), dim=0)[:, nn_mask] / dR_mskd
    dVr = torch.zeros((3, batch_size * Nats * Nats), device=RX.device)
    ind_start = torch.arange(Nats, device=RX.device)
    dVr.index_add_(
        1,
        ind_start[neighbor_I[valid_pairs]] * Nats
        + ind_start[neighbor_J[valid_pairs]]
        + batch_ids * Nats * Nats,
        dVr_dR * dR_dxyz,
    )
    # now, it's eV/A
    dVr = dVr.view(3, batch_size, Nats, Nats) * 27.21138625 / 0.52917721
    dVr = dVr.permute(1, 0, 2, 3).contiguous()
    dVr = dVr - torch.transpose(dVr, 2, 3)
    del nnRx, nnRy, nnRz, nnType, neighbor_I, neighbor_J, IJ_pair_type, _
    return Vr, dVr
