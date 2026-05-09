from __future__ import annotations

import torch

from ._tools import _maybe_compile


def _get_shell_spin_potential(
    TYPE: torch.Tensor,
    net_spin: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
) -> torch.Tensor:
    """Compute the shell-resolved spin potential μ_s per shell.

    For each shell s belonging to atom A, the potential is

        μ_s = ∑_{t ∈ shells(A)} W_{A, l_s, l_t} · m_t

    where W is the spin-Hamiltonian matrix (atom-resolved, l-dependent) and
    m_t is the net spin population of shell t.

    Parameters
    ----------
    TYPE : torch.Tensor, shape (Nats,), dtype int
        Atom type indices (0-based atomic numbers used for ``w`` lookup).
    net_spin : torch.Tensor, shape (n_shells_total,)
        Net spin population per shell (α − β electrons per shell).
    w : torch.Tensor
        Spin constants W.  May be:
        * shape (max_Z,)    — atom-resolved isotropic constants (no l dependence).
        * shape (max_Z, 3, 3) — atom-resolved shell-dependent (l × l) matrices.
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int
        Number of shells contributed by each atom.

    Returns
    -------
    mu : torch.Tensor, shape (n_shells_total,)
        Spin potential per shell in eV.
    """
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


def _get_shell_spin_energy(
    lhs_spin: torch.Tensor,
    rhs_spin: torch.Tensor,
    TYPE: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
) -> torch.Tensor:
    """Compute the bilinear spin-interaction energy ½ ⟨m_lhs | W | m_rhs⟩.

    Parameters
    ----------
    lhs_spin : torch.Tensor, shape (n_shells_total,)
        Left net-spin vector.
    rhs_spin : torch.Tensor, shape (n_shells_total,)
        Right net-spin vector.
    TYPE : torch.Tensor, shape (Nats,), dtype int
        Atom type indices.
    w : torch.Tensor
        Spin constants (see :func:`_get_shell_spin_potential`).
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int
        Number of shells per atom.

    Returns
    -------
    E_spin : torch.Tensor, scalar
        Spin-interaction energy in eV.
    """
    rhs_potential = _get_shell_spin_potential(TYPE, rhs_spin, w, n_shells_per_atom)
    return 0.5 * torch.sum(lhs_spin * rhs_potential)


def get_h_spin(
    TYPE: torch.Tensor,
    net_spin: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
    shell_types: torch.Tensor,
) -> torch.Tensor:
    """Build the full (n_orb × n_orb) spin-Hamiltonian matrix.

    The spin contribution to the Hamiltonian is diagonal in atom–shell space
    and acts as an outer sum: H_spin_{μν} = μ_μ + μ_ν, where μ is the
    spin potential vector expanded from shells to AOs.

    Parameters
    ----------
    TYPE : torch.Tensor, shape (Nats,), dtype int
        Atom type indices.
    net_spin : torch.Tensor, shape (n_shells_total,)
        Net spin population per shell.
    w : torch.Tensor
        Spin constants (see :func:`_get_shell_spin_potential`).
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int
        Number of shells per atom.
    shell_types : torch.Tensor, shape (n_shells_total,), dtype int
        Angular momentum type per shell: 1 = s, 2 = p, 3 = d.

    Returns
    -------
    H_spin : torch.Tensor, shape (n_orb, n_orb)
        Full spin-Hamiltonian matrix in AO basis.
    """
    n_orb_per_shell = torch.tensor([0, 1, 3, 5], device=net_spin.device)
    n_orb_per_shell_global = n_orb_per_shell[shell_types]
    mu = _get_shell_spin_potential(TYPE, net_spin, w, n_shells_per_atom)
    mu = mu.repeat_interleave(n_orb_per_shell_global)
    return mu.unsqueeze(0) + mu.unsqueeze(1)


def get_h_spin_diag(
    TYPE: torch.Tensor,
    net_spin: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
    shell_types: torch.Tensor,
) -> torch.Tensor:
    """Return the spin potential as a 1-D AO vector (diagonal of H_spin).

    The full H_spin matrix (MU + NU) has the property that when used as
    ``0.5 * S * (MU + NU)``, it equals ``0.5 * (mu[:,None]*S + S*mu[None,:])``,
    i.e. the same diagonal-broadcast form as the Coulomb shift.  So only
    the vector ``mu`` is needed, not the outer sum.

    Parameters
    ----------
    TYPE : torch.Tensor, shape (Nats,), dtype int
    net_spin : torch.Tensor, shape (n_shells_total,)
    w : torch.Tensor
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int
    shell_types : torch.Tensor, shape (n_shells_total,), dtype int

    Returns
    -------
    mu : torch.Tensor, shape (n_orb,)
        Spin potential per AO in eV.
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


def get_spin_energy(
    TYPE: torch.Tensor,
    net_spin: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
) -> torch.Tensor:
    """Compute the spin-interaction energy ½ ⟨m | W | m⟩.

    Parameters
    ----------
    TYPE : torch.Tensor, shape (Nats,), dtype int
    net_spin : torch.Tensor, shape (n_shells_total,)
    w : torch.Tensor
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int

    Returns
    -------
    E_spin : torch.Tensor, scalar, eV
    """
    return _get_shell_spin_energy(net_spin, net_spin, TYPE, w, n_shells_per_atom)


def get_spin_energy_shadow(
    TYPE: torch.Tensor,
    net_spin: torch.Tensor,
    n_net_spin: torch.Tensor,
    w: torch.Tensor,
    n_shells_per_atom: torch.Tensor,
) -> torch.Tensor:
    """Compute the shadow spin-interaction energy for XL-BOMD.

    Uses the extended-Lagrangian shadow energy formula::

        E_shadow = ½ ⟨(2m - n) | W | n⟩

    where ``m`` is the current net spin and ``n`` is the shadow spin
    (the extended variable from the previous step).

    Parameters
    ----------
    TYPE : torch.Tensor, shape (Nats,), dtype int
    net_spin : torch.Tensor, shape (n_shells_total,)
        Current SCC net spin m.
    n_net_spin : torch.Tensor, shape (n_shells_total,)
        Shadow net spin n (extended Lagrangian variable).
    w : torch.Tensor
    n_shells_per_atom : torch.Tensor, shape (Nats,), dtype int

    Returns
    -------
    E_shadow_spin : torch.Tensor, scalar, eV
    """
    diff = 2 * net_spin - n_net_spin
    return _get_shell_spin_energy(diff, n_net_spin, TYPE, w, n_shells_per_atom)


get_spin_energy_eager = get_spin_energy
get_spin_energy_shadow_eager = get_spin_energy_shadow
get_spin_energy = _maybe_compile(get_spin_energy)
get_spin_energy_shadow = _maybe_compile(get_spin_energy_shadow)
