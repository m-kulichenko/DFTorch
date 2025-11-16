import torch

def initialize_velocities(structure, temperature_K, masses_amu=None, remove_com=True, rescale_to_T=True, remove_angmom=False, generator=None):
    """
    Initialize atomic velocities at a target temperature using a Maxwell–Boltzmann distribution.
    - Velocities are in Å/fs (consistent with EKIN = 0.5 * MVV2KE * sum(m_i * v_i^2) in eV).
    - Ensures zero center-of-mass momentum.
    - Optionally removes net angular momentum (about the COM) without changing coordinates.
    - Rescales to match the exact target temperature.

    Args:
        structure: object with RX (device/dtype) and Mnuc (amu) or provide masses_amu.
        temperature_K (float): target temperature in Kelvin.
        masses_amu (torch.Tensor, optional): shape (N,), atomic masses in amu. Defaults to structure.Mnuc.
        remove_com (bool): enforce zero total momentum.
        rescale_to_T (bool): rescale to match exact temperature after constraints.
        remove_angmom (bool): remove net angular momentum (after COM removal).
        generator (torch.Generator, optional): for reproducibility.

    Returns:
        VX, VY, VZ: tensors of shape (N,) in Å/fs.
    """
    device = structure.RX.device
    dtype = structure.RX.dtype
    N = structure.Nats

    if masses_amu is None:
        masses_amu = structure.Mnuc.to(device=device, dtype=dtype)
    else:
        masses_amu = masses_amu.to(device=device, dtype=dtype)

    # Positive-mass mask
    mpos = masses_amu > 0
    npos = int(mpos.sum().item())

    # Physical constants
    kB_eV_per_K = 8.617333262145e-5  # eV/K
    amu_kg = 1.66053906660e-27
    ang2m = 1e-10
    fs2s = 1e-15
    eC = 1.602176634e-19
    MVV2KE = (amu_kg * (ang2m / fs2s) ** 2) / eC  # ≈ 103.642691 eV per (amu * (Å/fs)^2)

    # Sampling stddev (0 for zero-mass)
    kT = torch.as_tensor(kB_eV_per_K * temperature_K, device=device, dtype=dtype)
    inv_m = torch.where(mpos, 1.0 / masses_amu, torch.zeros_like(masses_amu))
    std = torch.sqrt(torch.clamp(kT * inv_m / MVV2KE, min=0)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = torch.randn((N, 3), device=device, dtype=dtype, generator=g) * std[:, None]

    # Remove center-of-mass velocity
    if remove_com and npos > 1:
        Mtot = masses_amu[mpos].sum()
        v_cm = (masses_amu[:, None] * V).sum(dim=0) / Mtot
        V = V - v_cm[None, :]

    # Optionally remove net angular momentum (about COM, without changing coordinates)
    if remove_angmom and npos > 1:
        # Positions
        R = torch.stack((structure.RX.to(dtype), structure.RY.to(dtype), structure.RZ.to(dtype)), dim=1)  # (N,3)
        Mtot = masses_amu[mpos].sum()
        r_com = (masses_amu[:, None] * R).sum(dim=0) / Mtot
        r_rel = R - r_com[None, :]

        # Angular momentum L = sum m r x v  (after COM removed)
        L = (masses_amu[:, None] * torch.cross(r_rel, V, dim=1))[mpos].sum(dim=0)  # (3,)

        # Inertia tensor I = sum m (r^2 I - r r^T)
        eye3 = torch.eye(3, dtype=dtype, device=device)
        r2 = (r_rel[mpos] * r_rel[mpos]).sum(dim=1)  # (npos,)
        I = (masses_amu[mpos][:, None, None] * (r2[:, None, None] * eye3 - r_rel[mpos][:, :, None] * r_rel[mpos][:, None, :])).sum(dim=0)  # (3,3)

        # Solve I * omega = L; use pinv for robustness
        # If I is near-singular (e.g., colinear atoms), pinv safely handles it.
        omega = torch.linalg.pinv(I) @ L  # (3,)

        # Remove rotation: v <- v + r x omega  (since r x omega = - omega x r)
        if npos > 0:
            deltaV = torch.cross(r_rel[mpos], omega.expand_as(r_rel[mpos]), dim=1)  # (npos,3)
            V[mpos] = V[mpos] + deltaV

    # Rescale to target temperature using DOF of massive atoms minus constraints
    if rescale_to_T and npos > 0:
        dof = 3 * npos
        if remove_com and npos > 1:
            dof -= 3
        if remove_angmom and npos > 2:
            dof -= 3
        if dof > 0:
            EKIN = 0.5 * MVV2KE * (masses_amu[:, None] * (V * V)).sum()
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if T_cur > 0:
                scale = torch.sqrt(torch.as_tensor(temperature_K, device=device, dtype=dtype) / T_cur)
                V = V * scale

    return V[:, 0].contiguous(), V[:, 1].contiguous(), V[:, 2].contiguous()

