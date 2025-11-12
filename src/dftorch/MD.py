import torch

def initialize_velocities(structure, temperature_K, masses_amu=None, remove_com=True, rescale_to_T=True, generator=None):
    """
    Initialize atomic velocities at a target temperature using a Maxwell–Boltzmann distribution.
    - Velocities are in Å/fs (consistent with EKIN = 0.5 * MVV2KE * sum(m_i * v_i^2) in eV).
    - Ensures zero center-of-mass momentum.
    - Rescales to match the exact target temperature.

    Args:
        structure: object with RX (device/dtype) and Mnuc (amu) or provide masses_amu.
        temperature_K (float): target temperature in Kelvin.
        masses_amu (torch.Tensor, optional): shape (N,), atomic masses in amu. Defaults to structure.Mnuc.
        remove_com (bool): enforce zero total momentum.
        rescale_to_T (bool): rescale to match exact temperature after COM removal.
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

    # Physical constants
    kB_eV_per_K = 8.617333262145e-5  # eV/K
    amu_kg = 1.66053906660e-27
    ang2m = 1e-10
    fs2s = 1e-15
    eC = 1.602176634e-19
    MVV2KE = (amu_kg * (ang2m / fs2s) ** 2) / eC  # ≈ 103.642691 eV per (amu * (Å/fs)^2)

    # Per-atom stddev for each Cartesian component so that
    # 0.5 * MVV2KE * m * <v_x^2> = 0.5 * kB * T  =>  std = sqrt(kB*T / (MVV2KE*m))
    std = torch.sqrt(kB_eV_per_K * temperature_K / (MVV2KE * masses_amu)).to(dtype)

    # Sample velocities
    g = generator if generator is not None else None
    V = torch.randn((N, 3), device=device, dtype=dtype, generator=g) * std[:, None]

    if remove_com:
        # Remove center-of-mass velocity: v_cm = sum(m v)/sum(m)
        Mtot = masses_amu.sum()
        v_cm = (masses_amu[:, None] * V).sum(dim=0) / Mtot
        V = V - v_cm[None, :]

    if rescale_to_T and N > 0:
        # Rescale to match exact target temperature after COM removal
        dof = 3 * N - (3 if remove_com and N > 1 else 0)
        if dof > 0:
            EKIN = 0.5 * MVV2KE * (masses_amu[:, None] * (V * V)).sum()
            T_cur = (2.0 / dof) * (EKIN / kB_eV_per_K)
            if T_cur > 0:
                scale = torch.sqrt(torch.tensor(temperature_K, device=device, dtype=dtype) / T_cur)
                V = V * scale

    return V[:, 0].contiguous(), V[:, 1].contiguous(), V[:, 2].contiguous()

