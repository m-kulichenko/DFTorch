import torch

from ._tools import _maybe_compile


def _get_shell_spin_potential(TYPE, net_spin, w, n_shells_per_atom):

    n_atoms = TYPE.shape[0]
    total_shells = net_spin.shape[0]
    shell_ids = torch.arange(
        total_shells, device=net_spin.device, dtype=n_shells_per_atom.dtype
    )
    atom_ids = torch.arange(
        n_atoms, device=net_spin.device, dtype=n_shells_per_atom.dtype
    )
    shell_to_atom = torch.repeat_interleave(atom_ids, n_shells_per_atom)
    shell_starts = torch.cumsum(n_shells_per_atom, dim=0) - n_shells_per_atom
    local_shell_ids = shell_ids - torch.repeat_interleave(
        shell_starts, n_shells_per_atom
    )
    flat_shell_ids = (shell_to_atom * 3 + local_shell_ids).long()

    net_spin_padded = torch.zeros(
        n_atoms * 3, dtype=net_spin.dtype, device=net_spin.device
    )
    net_spin_padded.scatter_add_(0, flat_shell_ids, net_spin)
    net_spin_padded = net_spin_padded.view(n_atoms, 3)

    if w.dim() == 1:
        mu_by_atom = w[TYPE] * net_spin_padded.sum(dim=1)
        return mu_by_atom.gather(0, shell_to_atom.long())
    else:
        mu_by_atom = torch.matmul(w[TYPE], net_spin_padded.unsqueeze(-1)).squeeze(-1)

    return mu_by_atom.reshape(-1).gather(0, flat_shell_ids)


def _get_shell_spin_energy(lhs_spin, rhs_spin, TYPE, w, n_shells_per_atom):
    rhs_potential = _get_shell_spin_potential(TYPE, rhs_spin, w, n_shells_per_atom)
    return 0.5 * torch.sum(lhs_spin * rhs_potential)


def get_h_spin(TYPE, net_spin, w, n_shells_per_atom, shell_types):

    n_orb_per_shell = torch.tensor([0, 1, 3, 5], device=net_spin.device)
    n_orb_per_shell_global = n_orb_per_shell[shell_types]
    mu = _get_shell_spin_potential(TYPE, net_spin, w, n_shells_per_atom)
    mu = mu.repeat_interleave(n_orb_per_shell_global)
    return mu.unsqueeze(0) + mu.unsqueeze(1)


def get_h_spin_diag(TYPE, net_spin, w, n_shells_per_atom, shell_types):
    """Return the spin potential as a vector (diagonal), avoiding the full n_orb×n_orb matrix.

    The full H_spin matrix (MU + NU) has the property that when used as
    0.5 * S * (MU + NU), it equals 0.5 * (mu[:,None]*S + S*mu[None,:]),
    i.e. the same diagonal-broadcast form as the Coulomb shift. So we only
    need the vector mu, not the outer sum.
    """
    n_orb_per_shell = torch.tensor([0, 1, 3, 5], device=net_spin.device)
    n_orb_per_shell_global = n_orb_per_shell[shell_types]
    mu = _get_shell_spin_potential(TYPE, net_spin, w, n_shells_per_atom)
    mu = mu.repeat_interleave(n_orb_per_shell_global)
    return mu  # (n_orb,) vector


get_h_spin_eager = get_h_spin
get_h_spin_diag_eager = get_h_spin_diag
get_h_spin = _maybe_compile(get_h_spin)
get_h_spin_diag = _maybe_compile(get_h_spin_diag)


def get_spin_energy(TYPE, net_spin, w, n_shells_per_atom):
    return _get_shell_spin_energy(net_spin, net_spin, TYPE, w, n_shells_per_atom)


def get_spin_energy_shadow(TYPE, net_spin, n_net_spin, w, n_shells_per_atom):

    diff = 2 * net_spin - n_net_spin
    return _get_shell_spin_energy(diff, n_net_spin, TYPE, w, n_shells_per_atom)


get_spin_energy_eager = get_spin_energy
get_spin_energy_shadow_eager = get_spin_energy_shadow
get_spin_energy = _maybe_compile(get_spin_energy)
get_spin_energy_shadow = _maybe_compile(get_spin_energy_shadow)
