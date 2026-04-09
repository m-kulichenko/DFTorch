# ruff: noqa
import torch
import math
from . import ewald_real, CONV_FACTOR, ewald_self_energy, ewald_real_screening
from .ewald_torch import ewald_real_screening_stress, h_damp_h5_correction
from typing import Optional, Dict, Tuple, List, Union


def b(m, order=4):
    k = torch.arange(order - 1, device=m.device).reshape(1, 1, 1, order - 1)
    M = compute_spline_coefficients(torch.tensor(1.0), order)[1:].flip(dims=(0,))
    M = M.reshape(1, 1, 1, order - 1)
    M = M.to(m.dtype).to(m.device)

    prefix = torch.exp(2 * torch.pi * 1j * (order - 1) * m)
    m = m[..., None]

    bot = torch.sum(M * torch.exp(2 * torch.pi * 1j * m * k), dim=-1)

    res = prefix / bot
    return res.real * res.real + res.imag * res.imag


def B(mx, my, mz, order=4):
    b_x = b(mx, order)
    b_y = b(my, order)
    b_z = b(mz, order)
    return b_x * b_y * b_z


def compute_spline_coefficients(w, order: int = 4):
    shape = w.shape
    w = w.flatten()
    scale = 1.0 / (order - 1)
    data = [torch.empty_like(w) for _ in range(order)]
    # data = torch.zeros(order, *shape, dtype=w.dtype, device=w.device)
    data[order - 1] = torch.zeros_like(w)
    data[1] = w
    data[0] = 1.0 - w

    for j in range(3, order):
        div = 1.0 / (j - 1)
        data[j - 1] = div * w * data[j - 2]

        for k in range(1, j - 1):
            data[j - k - 1] = div * (
                (w + k) * data[j - k - 2] + (j - k - w) * data[j - k - 1]
            )

        data[0] = div * (1.0 - w) * data[0]

    data[order - 1] = scale * w * data[order - 2]

    for j in range(1, order - 1):
        data[order - j - 1] = scale * (
            (w + j) * data[order - j - 2] + (order - j - w) * data[order - j - 1]
        )

    data[0] = scale * (1.0 - w) * data[0]
    data = torch.stack(data, dim=0)
    return data.reshape(order, *shape)


def grid_position(u, K, order=4):
    base_grid = torch.floor(u).to(torch.int32)
    base_grid = base_grid.unsqueeze(0)  # shape: [1, 3, N]
    offsets = torch.arange(0, order, device=u.device).view(order, 1, 1)
    grid = base_grid + offsets  # shape: [order, 3, N]
    return grid % K.view(1, 3, 1)


def map_charges_to_grid(position, charge, inv_box, grid_dimensions, order=4):
    """
    Maps particle charges to a grid via B-spline spreading.
    """
    grid_dimensions_torch = torch.tensor(
        grid_dimensions, device=position.device, dtype=torch.int32
    )
    Q = torch.zeros(
        tuple(grid_dimensions), device=position.device, dtype=position.dtype
    )

    u = (inv_box @ position) * grid_dimensions_torch.view(3, 1)

    # Compute the fractional part
    w = u - torch.floor(u)  # shape: [3, N]

    coeffs = compute_spline_coefficients(w, order)

    # [order, 3, N].
    grid_pos = grid_position(u, grid_dimensions_torch, order)

    # Multiply by particle charge.
    accum = charge.reshape(1, 1, 1, -1) * (
        coeffs[:, 0, None, None, :]
        * coeffs[None, :, 1, None, :]
        * coeffs[None, None, :, 2, :]
    )

    gp_x = grid_pos[:, 0, None, None, :]  # [order, 1, 1, N]
    gp_y = grid_pos[None, :, 1, None, :]  # [1, order, 1, N]
    gp_z = grid_pos[None, None, :, 2, :]  # [1, 1, order, N]

    grid_positions = torch.stack(
        (
            gp_x.expand(order, order, order, -1),
            gp_y.expand(order, order, order, -1),
            gp_z.expand(order, order, order, -1),
        ),
        dim=0,
    )

    # Flatten over particles and all order dimensions.
    gp_new = grid_positions.view(3, -1)  # [N * order^3, 3]
    ac_new = accum.view(-1)  # [N * order^3]

    gp_new_flat = (
        gp_new[0] * (grid_dimensions[1] * grid_dimensions[2])
        + gp_new[1] * grid_dimensions[2]
        + gp_new[2]
    )

    Q_flat = Q.flatten()
    Q_flat.index_add_(0, gp_new_flat, ac_new)
    Q = Q_flat.reshape(tuple(grid_dimensions))
    return Q


def init_PME_data(grid_dimensions, box, alpha, order):
    inverse_box = torch.linalg.inv(box)

    res = torch.meshgrid(
        *[torch.fft.fftfreq(g) for g in grid_dimensions], indexing="ij"
    )
    mx, my, mz = [r.to(box.dtype).to(box.device) for r in res]
    m = (
        inverse_box[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0]
        + inverse_box[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1]
        + inverse_box[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2]
    )
    m_2 = torch.sum(m**2, dim=-1)
    V = torch.linalg.det(box)
    mask = m_2 != 0
    m_2 = torch.where(mask == 1, m_2, 1)

    exp_m = 1 / (2 * torch.pi * V) * torch.exp(-(torch.pi**2) * m_2 / alpha**2) / m_2
    grid_multip = mask * exp_m * B(mx, my, mz, order)
    return (grid_dimensions, grid_multip, inverse_box, order)


@torch.compile(dynamic=False, fullgraph=True, options={"max_autotune": True})
def calculate_PME_energy(position, charge, box, alpha: float, PME_init_data: tuple):
    grid_dimensions, grid_multip, inverse_box, order = PME_init_data

    grid_new = map_charges_to_grid(
        position, charge, inverse_box, grid_dimensions, order
    )
    Fgrid = torch.fft.fftn(grid_new)

    e = torch.sum(grid_multip * (Fgrid.real**2 + Fgrid.imag**2))
    return e


def calculate_PME_kspace_stress(
    positions: torch.Tensor,
    charges: torch.Tensor,
    box: torch.Tensor,
    alpha: float,
    PME_init_data: tuple,
) -> torch.Tensor:
    """Reciprocal-space Ewald metric stress via PME.

    For each reciprocal-space grid point **m** with wavevector
    G = h^{-T} m  the k-space energy density is

        E_G = grid_multip(G) · |F[Q](G)|^2

    Under a homogeneous cell strain at fixed fractional coordinates the
    metric derivative gives the stress contribution

        σ_{αβ}^{kspace} = (1/V) Σ_G  E_G · (-δ_{αβ} + 2 G_α G_β / G²
                                              + G_α G_β / (2 α²))

    This uses the same FFT grid and B-spline interpolation as
    ``calculate_PME_energy`` so no additional Fourier transforms are needed
    beyond the one already done for the energy.

    Parameters
    ----------
    positions : (3, N) atomic coordinates
    charges : (N,) atomic charges
    box : (3, 3) lattice vectors
    alpha : Ewald splitting parameter
    PME_init_data : tuple from ``init_PME_data``

    Returns
    -------
    sigma : (3, 3) k-space metric stress, same internal units as the energy
            (before CONV_FACTOR).
    """
    grid_dimensions, grid_multip, inverse_box, order = PME_init_data

    grid_new = map_charges_to_grid(
        positions, charges, inverse_box, grid_dimensions, order
    )
    Fgrid = torch.fft.fftn(grid_new)
    # Per-G energy density (real, same grid as grid_multip)
    E_G = grid_multip * (Fgrid.real**2 + Fgrid.imag**2)  # (K1, K2, K3)

    V = torch.abs(torch.linalg.det(box))
    alpha2 = alpha * alpha

    # Build G-vectors on the grid
    # m-vectors in fractional reciprocal coordinates
    freq0 = torch.fft.fftfreq(grid_dimensions[0], device=box.device, dtype=box.dtype)
    freq1 = torch.fft.fftfreq(grid_dimensions[1], device=box.device, dtype=box.dtype)
    freq2 = torch.fft.fftfreq(grid_dimensions[2], device=box.device, dtype=box.dtype)
    m0, m1, m2 = torch.meshgrid(freq0, freq1, freq2, indexing="ij")

    # G = h^{-T} · (K1*m0, K2*m1, K3*m2)  — but init_PME_data already built
    # m = inv_box * (K_i * m_i), so we rebuild G in Cartesian:
    # G_cart = 2π * (inv_box^T @ n) where n = (K1*m0, K2*m1, K3*m2) / K_i = (m0,m1,m2)
    # Actually the m-vectors in init_PME_data are:
    #   m = inv_box[0]*m0*K0 + inv_box[1]*m1*K1 + inv_box[2]*m2*K2
    # and m_2 = |m|^2. The actual G-vector is 2π·m.  But in the energy formula
    # the 2π factors are already absorbed into the prefactor.
    # We just need the direction for the metric tensor.

    # Reconstruct the Cartesian m-vectors (matching init_PME_data)
    # m = inv_box[:,0]*m0*K0 + inv_box[:,1]*m1*K1 + inv_box[:,2]*m2*K2
    m_vec = (
        inverse_box[None, None, None, 0] * m0[:, :, :, None] * grid_dimensions[0]
        + inverse_box[None, None, None, 1] * m1[:, :, :, None] * grid_dimensions[1]
        + inverse_box[None, None, None, 2] * m2[:, :, :, None] * grid_dimensions[2]
    )  # (K1, K2, K3, 3)

    m_2 = torch.sum(m_vec**2, dim=-1)  # (K1, K2, K3)
    # Avoid division by zero at G=0
    m_2_safe = torch.where(m_2 > 0, m_2, torch.ones_like(m_2))

    eye = torch.eye(3, dtype=box.dtype, device=box.device)

    # Metric tensor per G-point:
    #   T_{αβ}(G) = -δ_{αβ} + 2 G_α G_β / G² + G_α G_β / (2α²)
    # where G = 2π·m, G² = 4π²·m², G_αG_β = 4π² m_α m_β
    # So:  2 G_αG_β/G² = 2 m_α m_β / m²
    #      G_αG_β/(2α²) = 4π² m_α m_β / (2α²) = 2π² m_α m_β / α²
    # But wait — in init_PME_data the exponential is exp(-π² m² / α²),
    # and the prefactor is 1/(2πV) · exp(…) / m².
    # The actual G = 2π·m, so G² = 4π²m², and the metric correction is:
    #   -δ + 2·m⊗m/m² + 2π²·m⊗m/α²

    mm = m_vec.unsqueeze(-1) * m_vec.unsqueeze(-2)  # (K1,K2,K3,3,3)
    metric = (
        -eye[None, None, None, :, :]
        + 2.0 * mm / m_2_safe[:, :, :, None, None]
        + 2.0 * (math.pi**2) * mm / (alpha2)
    )  # (K1,K2,K3,3,3)

    # Zero out the G=0 contribution (m_2 == 0)
    g_mask = (m_2 > 0).float()  # (K1,K2,K3)

    # σ_{αβ} = (1/V) Σ_G  E_G · T_{αβ}(G)
    sigma = torch.einsum("ijk,ijk,ijkab->ab", E_G, g_mask, metric) / V
    sigma = 0.5 * (sigma + sigma.T)
    return sigma


def calculate_PME_ewald(
    positions: torch.Tensor,
    charges: torch.Tensor,
    box: torch.Tensor,
    nbr_inds: torch.Tensor,
    nbr_disp_vecs: torch.Tensor,
    nbr_dists: torch.Tensor,
    alpha: float,
    cutoff: float,
    PME_init_data: Tuple[Union[List, Tuple], torch.Tensor, torch.Tensor, int],
    hubbard_u: torch.Tensor = None,
    atomtypes: torch.Tensor = None,
    calculate_forces: int = 0,
    calculate_dq: int = 0,
    screening: int = 0,
    calculate_stress: int = 0,
    h_damp_exp: Optional[float] = None,
    h5_params: Optional[Dict] = None,
) -> Tuple[
    float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """
    Computes the total Ewald sum energy using Particle Mesh Ewald (PME).

    This function calculates the real-space and PME-based reciprocal-space
    contributions to the Ewald summation. It also optionally computes forces,
    charge derivatives, and the stress tensor.

    Args:
        positions (torch.Tensor): Atomic positions. Shape: `(N, 3)` or  `(3, N)`, where:
            - `N` is the number of atoms.
            - `3` represents the x, y, and z coordinates.
        charges (torch.Tensor): Atomic charge values. Shape: `(N,)`.
        box (torch.Tensor): Simulation box dimensions. Shape: `(3, 3)`.
        nbr_inds (torch.Tensor): Indices of neighboring atoms. Shape: `(N, K)`, where:
            - `K` is the maximum number of neighbors per atom.
        nbr_disp_vecs (torch.Tensor): Displacement vectors to neighbors. Shape: `(3, N, K)` or `(N, K, 3)`.
        nbr_dists (torch.Tensor): Distances to neighboring atoms. Shape: `(N, K)`.
        alpha (float): Ewald screening parameter (scalar).
        cutoff (float): Cutoff distance for real-space interactions (scalar).
        PME_init_data (tuple): Precomputed PME initialization data.
        calculate_forces (int): Flag to compute forces (`1` for True, `0` for False).
        calculate_dq (int, optional): Flag to compute charge derivatives (`1` for True, `0` for False`). Defaults to `0`.
        screening (int, optional): Flag to use Hubbard-U screening. Defaults to `0`.
        calculate_stress (int, optional): Flag to compute stress tensor (`1` for True, `0` for False). Defaults to `0`.

    Returns:
        Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            - **(float)** Total Ewald energy contribution.
            - **(torch.Tensor, shape `(3, N)`, optional)** Computed forces if `calculate_forces` is enabled, otherwise `None`.
                If the positions are provided as `(N, 3)`, the forces will be also  `(N, 3)`.
            - **(torch.Tensor, shape `(N,)`, optional)** Charge derivatives if `calculate_dq` is enabled, otherwise `None`.
            - **(torch.Tensor, shape `(3, 3)`, optional)** Stress tensor in eV/Å³ if `calculate_stress` is enabled, otherwise `None`.

    Notes:
        - Forces and charge derivatives for PME are computed via automatic differentiation.
        - Stress is computed analytically: real-space virial + k-space metric tensor.
    """
    N = len(charges)
    # As the internal functions expects (3, N), transpose the position tensor as needed
    # Enforce canonical layout
    assert positions.dim() == 2 and positions.shape[0] == 3, "positions must be (3,N)"

    transpose = False
    if positions.shape[1] == 3:
        transpose = True
        positions = positions.T.contiguous()

    # transpose the disp. vectors as needed
    if nbr_disp_vecs.shape[2] == 3:
        nbr_disp_vecs = nbr_disp_vecs.permute(2, 0, 1).contiguous()

    dq = None
    forces = None
    if screening:
        my_e_real, my_f_real, my_dq_real = ewald_real_screening(
            nbr_inds,
            nbr_disp_vecs,
            nbr_dists,
            charges,
            hubbard_u,
            atomtypes,
            alpha,
            cutoff,
            calculate_forces,
            calculate_dq,
        )

        # if my_f_real is not None and my_f_real.device.type == 'cuda':
        #     my_f_real = my_f_real.T.contiguous()

    else:
        my_e_real, my_f_real, my_dq_real = ewald_real(
            nbr_inds,
            nbr_disp_vecs,
            nbr_dists,
            charges,
            alpha,
            cutoff,
            calculate_forces,
            calculate_dq,
        )
    if calculate_dq:
        charges.grad = None
        charges.requires_grad = True
        # charges = charges.detach().requires_grad_(True)

    if calculate_forces:
        positions.grad = None
        positions.requires_grad = True

    with torch.enable_grad():
        pme_e = calculate_PME_energy(positions, charges, box, alpha, PME_init_data)
        self_e, self_dq = ewald_self_energy(charges, alpha, calculate_dq)

        if calculate_dq or calculate_forces:
            pme_e.backward()

        if calculate_forces:
            if my_f_real.shape[1] == 3:
                my_f_real = my_f_real.T.contiguous()

            forces = (-1.0 * positions.grad + my_f_real).detach() * CONV_FACTOR
            # positions.grad.zero_()
            if transpose:
                forces = forces.T
            positions.requires_grad = False
        if calculate_dq:
            dq = (charges.grad + my_dq_real + self_dq).detach() * CONV_FACTOR
            # charges.grad.zero_()
            charges.requires_grad = False

    total_ewald_e = (my_e_real + pme_e + self_e) * CONV_FACTOR

    # ── H-damping / H5 correction to real-space screening ────────────────
    if screening and (h_damp_exp is not None or h5_params is not None):
        # nbr_disp_vecs may have been transposed above; need (3,N,K) or (N,K,3)
        de_corr, df_corr, dq_corr = h_damp_h5_correction(
            nbr_inds,
            nbr_disp_vecs,
            nbr_dists,
            charges,
            hubbard_u,
            atomtypes,
            cutoff,
            calculate_forces,
            calculate_dq,
            h_damp_exp=h_damp_exp,
            h5_params=h5_params,
        )
        total_ewald_e = total_ewald_e + de_corr
        if calculate_forces and df_corr is not None:
            if transpose:
                forces = forces + df_corr.T
            else:
                forces = forces + df_corr
        if calculate_dq and dq_corr is not None:
            dq = dq + dq_corr

    # --- Stress tensor (analytical) ---
    if calculate_stress:
        # Real-space virial (uses same damping logic as ewald_real_screening)
        real_stress = ewald_real_screening_stress(
            nbr_inds,
            nbr_disp_vecs,
            nbr_dists,
            charges,
            hubbard_u,
            atomtypes,
            alpha,
            cutoff,
            box,
        )
        # K-space metric tensor stress
        kspace_stress = calculate_PME_kspace_stress(
            positions, charges, box, alpha, PME_init_data
        )
        # Self-energy has no cell dependence → zero stress contribution
        stress = (real_stress + kspace_stress) * CONV_FACTOR
        return total_ewald_e, forces, dq, stress

    return total_ewald_e, forces, dq
