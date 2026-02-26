import torch


# @torch.compile
def get_h_spin(TYPE, net_spin, w, n_shells_per_atom, shell_types):

    n_orb_per_shell = torch.tensor([0, 1, 3, 5], device=net_spin.device)
    n_orb_per_shell_global = n_orb_per_shell[shell_types]
    mu = 0.0 * torch.zeros_like(n_shells_per_atom.repeat_interleave(n_shells_per_atom))

    # s atoms
    mask_tmp1 = n_shells_per_atom == 1  #
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (1, 1), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:1, 0:1]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # map atom inds onto orbitals
    q_tmp = net_spin[mask_tmp].view(-1, 1)
    spin_term_tmp = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()
    mu[mask_tmp] = spin_term_tmp

    # sp atoms
    mask_tmp1 = n_shells_per_atom == 2
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (2, 2), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:2, 0:2]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp = net_spin[mask_tmp].view(-1, 2)
    spin_term_tmp = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()
    mu[mask_tmp] = spin_term_tmp

    # spd atoms
    mask_tmp1 = n_shells_per_atom == 3
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (3, 3), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:3, 0:3]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp = net_spin[mask_tmp].view(-1, 3)
    spin_term_tmp = (w_tmp * q_tmp.unsqueeze(1)).sum(-1).flatten()
    mu[mask_tmp] = spin_term_tmp

    mu = mu.repeat_interleave(n_orb_per_shell_global)
    MU = mu.unsqueeze(0).expand(len(mu), -1)
    NU = mu.unsqueeze(1).expand(-1, len(mu))
    # H_spin = 0.5 * S * (MU + NU)
    H_spin = MU + NU

    return H_spin


# @torch.compile
def get_spin_energy(TYPE, net_spin, w, n_shells_per_atom):

    # s atoms
    mask_tmp1 = n_shells_per_atom == 1  #
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (1, 1), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:1, 0:1]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # map atom inds onto orbitals
    q_tmp = net_spin[mask_tmp].view(-1, 1)
    e_spin = (q_tmp.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp.unsqueeze(1)).sum()

    # sp atoms
    mask_tmp1 = n_shells_per_atom == 2
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (2, 2), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:2, 0:2]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp = net_spin[mask_tmp].view(-1, 2)
    e_spin = (
        e_spin
        + (q_tmp.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp.unsqueeze(1)).sum()
    )

    # spd atoms
    mask_tmp1 = n_shells_per_atom == 3
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (3, 3), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:3, 0:3]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp = net_spin[mask_tmp].view(-1, 3)
    e_spin = (
        e_spin
        + (q_tmp.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp.unsqueeze(1)).sum()
    )

    return 0.5 * e_spin


# @torch.compile
def get_spin_energy_shadow(TYPE, net_spin, n_net_spin, w, n_shells_per_atom):

    diff = 2 * net_spin - n_net_spin
    # s atoms
    mask_tmp1 = n_shells_per_atom == 1  #
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (1, 1), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:1, 0:1]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # map atom inds onto orbitals
    q_tmp1 = diff[mask_tmp].view(-1, 1)
    q_tmp2 = n_net_spin[mask_tmp].view(-1, 1)
    e_spin = (q_tmp1.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp2.unsqueeze(1)).sum()

    # sp atoms
    mask_tmp1 = n_shells_per_atom == 2
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (2, 2), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:2, 0:2]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp1 = diff[mask_tmp].view(-1, 2)
    q_tmp2 = n_net_spin[mask_tmp].view(-1, 2)
    e_spin = (
        e_spin
        + (q_tmp1.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp2.unsqueeze(1)).sum()
    )

    # spd atoms
    mask_tmp1 = n_shells_per_atom == 3
    if w.dim() == 1:
        w_tmp = w[TYPE[mask_tmp1]].unsqueeze(-1).unsqueeze(-1) * torch.ones(
            (3, 3), device=net_spin.device
        )
    else:
        w_tmp = w[TYPE[mask_tmp1]][:, 0:3, 0:3]
    mask_tmp = mask_tmp1.repeat_interleave(
        n_shells_per_atom
    )  # Generate atom index for each orbital
    q_tmp1 = diff[mask_tmp].view(-1, 3)
    q_tmp2 = n_net_spin[mask_tmp].view(-1, 3)
    e_spin = (
        e_spin
        + (q_tmp1.unsqueeze(1).transpose(-1, -2) * w_tmp * q_tmp2.unsqueeze(1)).sum()
    )

    return 0.5 * e_spin
