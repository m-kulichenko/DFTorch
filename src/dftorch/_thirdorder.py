"""
Full off-diagonal DFTB3 third-order correction.

Implements the full third-order extension to the SCC-DFTB energy, shifts, and
gradient contributions following Gaus, Cui & Elstner, JCTC 7, 931 (2011) and
the DFTB+ ``thirdorder.F90`` implementation.

In the non-shell-resolved case the third-order energy reads

    E₃ = (1/3) ∑_{A,B} Γ³_AB · Δq_B · Δq_A²
       + (1/3) ∑_{A,B} Γ³_BA · Δq_B² · Δq_A

where Γ³_AB(r) = (dγ_AB/dU_A) · (dU_A/dq_A) is asymmetric in A↔B.

The Hamiltonian shift that enters the SCF (per atom A) is:

    σ₃(A) = (1/3) [2 ∑_B Γ³_AB · q_B · q_A  +  ∑_B Γ³_BA · q_B²]

The gradient contribution (from dΓ³/dr) is handled by ``add_gradient_dc``.

This module provides:

* ``compute_gamma3_matrices``  – build Γ³_ab and Γ³_ba (Nats×Nats) and their
  radial derivatives (3×Nats×Nats), plus energy/shift helpers.
* ``ThirdOrder`` – a container that stores the matrices and provides
  ``get_shifts``, ``get_energy``, ``get_dshifts_dq``, and ``add_gradient_dc``
  methods, following the same pattern as ``_gbsa.py``.
"""

from __future__ import annotations

from typing import Optional

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Low-level helper functions (translated from DFTB+ thirdorder.F90)
# All inputs/outputs are in *atomic units* (Hartree, Bohr) unless noted.
# ═══════════════════════════════════════════════════════════════════════════

_TOL_SAME_DIST = 1.0e-10  # Bohr
_MIN_HUB_DIFF = 1.0e-6  # Hartree


# ── f(τ_A, τ_B, r) and derivatives ──────────────────────────────────────
def _ff(tauA, tauB, r):
    """f(τ_A, τ_B, r) – auxiliary for short-range γ when τ_A ≠ τ_B."""
    t2 = tauA**2 - tauB**2
    return 0.5 * tauA * tauB**4 / t2**2 - (tauB**6 - 3.0 * tauA**2 * tauB**4) / (
        t2**3 * r
    )


def _fpT1(tauA, tauB, r):
    """df/dτ_A."""
    t2 = tauA**2 - tauB**2
    return -0.5 * (
        tauB**6 + 3.0 * tauA**2 * tauB**4
    ) / t2**3 - 12.0 * tauA**3 * tauB**4 / (t2**4 * r)


def _fpT2(tauB, tauA, r):
    """df(τ_B,τ_A,r)/dτ_A – note argument ordering."""
    t2 = tauB**2 - tauA**2
    return 2.0 * tauB**3 * tauA**3 / t2**3 + 12.0 * tauB**4 * tauA**3 / (t2**4 * r)


# ── g(τ, r) and derivatives ─────────────────────────────────────────────
def _gg(tau, r):
    """g(τ, r) – auxiliary for short-range γ when τ_A = τ_B = τ."""
    return (48.0 + 33.0 * tau * r + 9.0 * tau**2 * r**2 + tau**3 * r**3) / (48.0 * r)


def _gpT(tau, r):
    """dg/dτ."""
    return (33.0 + 18.0 * tau * r + 3.0 * tau**2 * r**2) / 48.0


# ── short-range functions S₁, S₂ ────────────────────────────────────────
def _short_1(tauA, tauB, r):
    """S₁(τ_A,τ_B,r): short-range SCC when τ_A ≠ τ_B and r ≠ 0."""
    return torch.exp(-tauA * r) * _ff(tauA, tauB, r) + torch.exp(-tauB * r) * _ff(
        tauB, tauA, r
    )


def _short_2(tau, r):
    """S₂(τ,r): short-range SCC when τ_A = τ_B = τ and r ≠ 0."""
    return torch.exp(-tau * r) * _gg(tau, r)


def _shortpT_1(tauA, tauB, r):
    """dS₁/dτ_A."""
    return torch.exp(-tauA * r) * (
        _fpT1(tauA, tauB, r) - r * _ff(tauA, tauB, r)
    ) + torch.exp(-tauB * r) * _fpT2(tauB, tauA, r)


def _shortpT_2(tau, r):
    """dS₂/dτ."""
    return torch.exp(-tau * r) * (_gpT(tau, r) - r * _gg(tau, r))


# ── Short-range radial derivatives (for gradient) ───────────────────────
def _fpR(tauA, tauB, r):
    """df/dr."""
    t2 = tauA**2 - tauB**2
    return (tauB**6 - 3.0 * tauB**4 * tauA**2) / (r**2 * t2**3)


def _gpR(tau, r):
    """dg/dr."""
    return (-48.0 + 9.0 * (tau * r) ** 2 + 2.0 * (tau * r) ** 3) / (48.0 * r**2)


def _shortpR_1(tauA, tauB, r):
    """dS₁/dr."""
    return torch.exp(-tauA * r) * (
        _fpR(tauA, tauB, r) - tauA * _ff(tauA, tauB, r)
    ) + torch.exp(-tauB * r) * (_fpR(tauB, tauA, r) - tauB * _ff(tauB, tauA, r))


def _shortpR_2(tau, r):
    """dS₂/dr."""
    return torch.exp(-tau * r) * (_gpR(tau, r) - tau * _gg(tau, r))


# ── dτ/dU derivatives for gamma3 ────────────────────────────────────────
def _fpT1pR(tauA, tauB, r):
    """d²f/dτ_A·dr."""
    t2 = tauA**2 - tauB**2
    return 12.0 * tauA**3 * tauB**4 / (r**2 * t2**4)


def _fpT2pR(tauB, tauA, r):
    """d²f(τ_B,τ_A,r)/dτ_A·dr."""
    t2 = tauB**2 - tauA**2
    return -12.0 * tauA**3 * tauB**4 / (r**2 * t2**4)


def _shortpTpR_1(tauA, tauB, r):
    """d²S₁/dτ_A·dr."""
    return torch.exp(-tauA * r) * (
        _ff(tauA, tauB, r) * (tauA * r - 1.0)
        - tauA * _fpT1(tauA, tauB, r)
        + _fpT1pR(tauA, tauB, r)
        - r * _fpR(tauA, tauB, r)
    ) + torch.exp(-tauB * r) * (_fpT2pR(tauB, tauA, r) - tauB * _fpT2(tauB, tauA, r))


def _shortpTpR_2(tau, r):
    """d²S₂/dτ·dr."""
    return torch.exp(-tau * r) * (
        (tau * r - 1.0) * _gg(tau, r)
        - tau * _gpT(tau, r)
        + _gpTpR(tau, r)
        - r * _gpR(tau, r)
    )


def _gpTpR(tau, r):
    """d²g/dτ·dr."""
    return (3.0 * tau + tau**2 * r) / 8.0


# ── Damping functions h(U_A, U_B, r, ξ) and derivatives ─────────────────
def _hh(Ua, Ub, r, xi):
    """Damping function h = exp(−((U_A+U_B)/2)^ξ · r²)."""
    return torch.exp(-((0.5 * (Ua + Ub)) ** xi) * r**2)


def _hpU(Ua, Ub, r, xi):
    """dh/dU_A."""
    return -0.5 * xi * r**2 * (0.5 * (Ua + Ub)) ** (xi - 1.0) * _hh(Ua, Ub, r, xi)


def _hpR(Ua, Ub, r, xi):
    """dh/dr."""
    return -2.0 * r * (0.5 * (Ua + Ub)) ** xi * _hh(Ua, Ub, r, xi)


def _hpUpR(Ua, Ub, r, xi):
    """d²h/dU_A·dr."""
    avg = 0.5 * (Ua + Ub)
    return xi * r * avg ** (xi - 1.0) * (r**2 * avg**xi - 1.0) * _hh(Ua, Ub, r, xi)


# ── dγ/dU_A (on-site limit r=0 with U_A ≠ U_B) ─────────────────────────
def _dGdUr0(tauA, tauB):
    """dγ/dU for r = 0 when τ_A ≠ τ_B (Eq. S7 in Gaus et al. 2015)."""
    invS = 1.0 / (tauA + tauB)
    return (
        1.6
        * invS
        * (
            tauB
            + invS
            * (
                -tauA * tauB
                + invS * (2.0 * tauA * tauB**2 + invS * (-3.0 * tauA**2 * tauB**2))
            )
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core gamma3 functions (vectorized)
# ═══════════════════════════════════════════════════════════════════════════


def _gamma2pU_vec(Ua, Ub, r_bohr, damped_mask, xi):
    """
    Vectorized dγ_AB/dU_A for an array of pairs.

    Parameters
    ----------
    Ua, Ub : (N,) tensors – Hubbard U in Hartree
    r_bohr : (N,) tensor – interatomic distance in Bohr
    damped_mask : (N,) bool tensor – True for pairs that need H-damping
    xi : float – damping exponent (e.g. 4.0)

    Returns
    -------
    result : (N,) tensor – dγ/dU_A in a.u.
    """
    tauA = 3.2 * Ua
    tauB = 3.2 * Ub
    result = torch.zeros_like(r_bohr)

    # ── r ≈ 0 (on-site): only appears for A == B in the pair list ────
    on_site = r_bohr < _TOL_SAME_DIST
    same_hub = torch.abs(Ua - Ub) < _MIN_HUB_DIFF

    # Case 1: r=0, Ua=Ub  →  0.5
    m1 = on_site & same_hub
    if m1.any():
        result[m1] = 0.5

    # Case 2: r=0, Ua≠Ub  →  dGdUr0
    m2 = on_site & (~same_hub)
    if m2.any():
        result[m2] = _dGdUr0(tauA[m2], tauB[m2])

    # Finite-r masks
    finite = ~on_site

    # Case 3: r>0, Ua≈Ub
    m3 = finite & same_hub
    if m3.any():
        tau = 0.5 * (tauA[m3] + tauB[m3])
        r = r_bohr[m3]
        val = -3.2 * _shortpT_2(tau, r)
        if damped_mask.any():
            dm = damped_mask[m3]
            if dm.any():
                uu = 0.5 * (Ua[m3][dm] + Ub[m3][dm])
                rd = r[dm]
                val[dm] = val[dm] * _hh(uu, uu, rd, xi) - _short_2(tau[dm], rd) * _hpU(
                    uu, uu, rd, xi
                )
        result[m3] = val

    # Case 4: r>0, Ua≠Ub
    m4 = finite & (~same_hub)
    if m4.any():
        tA = tauA[m4]
        tB = tauB[m4]
        r = r_bohr[m4]
        val = -3.2 * _shortpT_1(tA, tB, r)
        if damped_mask.any():
            dm = damped_mask[m4]
            if dm.any():
                uA = Ua[m4][dm]
                uB = Ub[m4][dm]
                rd = r[dm]
                val[dm] = val[dm] * _hh(uA, uB, rd, xi) - _short_1(
                    tA[dm], tB[dm], rd
                ) * _hpU(uA, uB, rd, xi)
        result[m4] = val

    return result


def _gamma2pUpR_vec(Ua, Ub, r_bohr, damped_mask, xi):
    """
    Vectorized d²γ_AB/(dU_A·dr) for an array of pairs (radial derivative
    of gamma2pU, needed for gradient contribution).

    Returns
    -------
    result : (N,) tensor in a.u. (1/Bohr²)
    """
    tauA = 3.2 * Ua
    tauB = 3.2 * Ub
    result = torch.zeros_like(r_bohr)

    finite = r_bohr >= _TOL_SAME_DIST
    same_hub = torch.abs(Ua - Ub) < _MIN_HUB_DIFF

    # Case: r>0, Ua≈Ub
    m3 = finite & same_hub
    if m3.any():
        tau = 0.5 * (tauA[m3] + tauB[m3])
        r = r_bohr[m3]
        val = -3.2 * _shortpTpR_2(tau, r)
        if damped_mask.any():
            dm = damped_mask[m3]
            if dm.any():
                uu = 0.5 * (Ua[m3][dm] + Ub[m3][dm])
                rd = r[dm]
                tau_d = tau[dm]
                val[dm] = (
                    val[dm] * _hh(uu, uu, rd, xi)
                    - 3.2 * _shortpT_2(tau_d, rd) * _hpR(uu, uu, rd, xi)
                    - _shortpR_2(tau_d, rd) * _hpU(uu, uu, rd, xi)
                    - _short_2(tau_d, rd) * _hpUpR(uu, uu, rd, xi)
                )
        result[m3] = val

    # Case: r>0, Ua≠Ub
    m4 = finite & (~same_hub)
    if m4.any():
        tA = tauA[m4]
        tB = tauB[m4]
        r = r_bohr[m4]
        val = -3.2 * _shortpTpR_1(tA, tB, r)
        if damped_mask.any():
            dm = damped_mask[m4]
            if dm.any():
                uA = Ua[m4][dm]
                uB = Ub[m4][dm]
                tAd = tA[dm]
                tBd = tB[dm]
                rd = r[dm]
                val[dm] = (
                    val[dm] * _hh(uA, uB, rd, xi)
                    - 3.2 * _shortpT_1(tAd, tBd, rd) * _hpR(uA, uB, rd, xi)
                    - _shortpR_1(tAd, tBd, rd) * _hpU(uA, uB, rd, xi)
                    - _short_1(tAd, tBd, rd) * _hpUpR(uA, uB, rd, xi)
                )
        result[m4] = val

    return result


# ═══════════════════════════════════════════════════════════════════════════
# ThirdOrder class
# ═══════════════════════════════════════════════════════════════════════════


class ThirdOrder:
    """Full off-diagonal DFTB3 third-order container.

    Mirrors the ``TThirdOrder`` type from DFTB+ ``thirdorder.F90`` but in
    a non-shell-resolved, vectorized PyTorch form.

    Usage (single geometry, non-batch)
    -----------------------------------
    1. ``to = ThirdOrder(Hubbard_U, dU_dq, TYPE, h_damp_exp)``
    2. ``to.update_coords(RX, RY, RZ, cell, cutoff, ...)`` – builds Γ³ matrices
    3. During SCF:
        a. ``shift = to.get_shifts(q)`` – add to Coulomb potential
        b. ``dshift = to.get_dshifts_dq(q, v)`` – linearised shift for Krylov
    4. After SCF:
        a. ``E3 = to.get_energy(q)`` – third-order energy
        b. ``grad3 = to.get_gradient_dc(q)`` – (3, Nats) gradient from dΓ³/dr
    """

    # Unit conversion constants
    EV_TO_HA = 1.0 / 27.211386245988
    HA_TO_EV = 27.211386245988
    ANG_TO_BOHR = 1.0 / 0.52917721067
    BOHR_TO_ANG = 0.52917721067

    def __init__(
        self,
        Hubbard_U: torch.Tensor,
        dU_dq: torch.Tensor,
        TYPE: torch.Tensor,
        h_damp_exp: float = 4.0,
    ):
        """
        Parameters
        ----------
        Hubbard_U : (Nats,) tensor – per-atom Hubbard U in eV.
        dU_dq : (Nats,) tensor – per-atom dU/dq in eV/e.
        TYPE : (Nats,) long tensor – atomic numbers.
        h_damp_exp : float – ζ exponent for H-damping (typ. 4.0 for mio, 4.05 for 3ob).
        """
        self.Nats = TYPE.shape[0]
        self.device = TYPE.device
        self.dtype = Hubbard_U.dtype

        # Store in eV — will convert to a.u. when computing
        self.Hubbard_U = Hubbard_U  # (Nats,) eV
        self.dU_dq = dU_dq  # (Nats,) eV/e
        self.TYPE = TYPE  # (Nats,)
        self.h_damp_exp = h_damp_exp

        # Determine which atoms are "damped" (hydrogen, Z=1)
        self.damped = TYPE == 1  # (Nats,) bool

        # Matrices to be set by update_coords
        self.gamma3ab = None  # (Nats, Nats) – Γ³(U_A, U_B, dU_A, r_AB)
        self.gamma3ba = None  # (Nats, Nats) – Γ³(U_B, U_A, dU_B, r_AB)
        self.gamma3ab_pR = (
            None  # (Nats, Nats) – dΓ³_ab/dr (a.u. values, to be used with unit vecs)
        )
        self.gamma3ba_pR = None  # (Nats, Nats) – dΓ³_ba/dr
        self.diff_xyz = (
            None  # (3, Nats, Nats) – unit vectors * (1/r), for gradient assembly
        )

    def update_coords(
        self,
        RX: torch.Tensor,
        RY: torch.Tensor,
        RZ: torch.Tensor,
        cell: Optional[torch.Tensor],
        neighbor_I: torch.Tensor,
        neighbor_J: torch.Tensor,
        dR_ang: torch.Tensor,
        dR_dxyz: torch.Tensor,
    ):
        """Build Γ³_ab and Γ³_ba matrices from current coordinates.

        Parameters
        ----------
        RX, RY, RZ : (Nats,) – Cartesian coordinates in Angstrom.
        cell : (3,3) or None – lattice vectors.
        neighbor_I, neighbor_J : (Npairs,) – pair indices from the neighbor list
            (same ones used for the Coulomb matrix).
        dR_ang : (Npairs,) – pairwise distances in Angstrom.
        dR_dxyz : (Npairs, 3) – unit displacement vectors (Å).
        """
        Nats = self.Nats
        device = self.device
        dtype = self.dtype

        # Convert to atomic units
        Ua_eV = self.Hubbard_U[neighbor_I]  # (Npairs,) eV
        Ub_eV = self.Hubbard_U[neighbor_J]  # (Npairs,) eV
        Ua_au = Ua_eV * self.EV_TO_HA  # (Npairs,) Hartree
        Ub_au = Ub_eV * self.EV_TO_HA  # (Npairs,) Hartree
        dUa_au = self.dU_dq[neighbor_I] * self.EV_TO_HA  # (Npairs,) Ha/e
        dUb_au = self.dU_dq[neighbor_J] * self.EV_TO_HA  # (Npairs,) Ha/e
        r_bohr = dR_ang * self.ANG_TO_BOHR  # (Npairs,) Bohr

        # Damping mask: pair is damped if either atom is hydrogen
        damped_mask = self.damped[neighbor_I] | self.damped[neighbor_J]  # (Npairs,)
        xi = self.h_damp_exp

        # ── Compute γ₂' = dγ/dU for each pair ──────────────────────────
        g2pU_ab = _gamma2pU_vec(Ua_au, Ub_au, r_bohr, damped_mask, xi)  # dγ/dU_A
        g2pU_ba = _gamma2pU_vec(Ub_au, Ua_au, r_bohr, damped_mask, xi)  # dγ/dU_B

        # Γ³_ab = (dγ/dU_A) * (dU_A/dq_A)   in Hartree
        gamma3ab_vals = g2pU_ab * dUa_au  # (Npairs,) Hartree
        gamma3ba_vals = g2pU_ba * dUb_au  # (Npairs,) Hartree

        # Convert to eV (since charges are in electrons and we want shifts in eV)
        gamma3ab_vals = gamma3ab_vals * self.HA_TO_EV
        gamma3ba_vals = gamma3ba_vals * self.HA_TO_EV

        # Assemble into (Nats, Nats) matrices
        self.gamma3ab = torch.zeros((Nats, Nats), device=device, dtype=dtype)
        self.gamma3ba = torch.zeros((Nats, Nats), device=device, dtype=dtype)
        flat_idx = neighbor_I * Nats + neighbor_J
        self.gamma3ab.view(-1).index_add_(0, flat_idx, gamma3ab_vals)
        self.gamma3ba.view(-1).index_add_(0, flat_idx, gamma3ba_vals)

        # ── On-site diagonal (A == A, r = 0) ───────────────────────────
        # For the on-site term: dγ_AA/dU_A = 0.5 when U_A == U_B
        # So Γ³_aa = 0.5 * dU_A/dq_A (in eV)
        # These may already be in the matrix from pairs with iNeigh=0 (self),
        # but let's ensure they are correct.
        diag_val = 0.5 * self.dU_dq  # (Nats,) eV/e · e = eV
        self.gamma3ab.diagonal().copy_(diag_val)
        self.gamma3ba.diagonal().copy_(diag_val)

        # ── Radial derivatives for gradient contribution ────────────────
        g2pUpR_ab = _gamma2pUpR_vec(Ua_au, Ub_au, r_bohr, damped_mask, xi)
        g2pUpR_ba = _gamma2pUpR_vec(Ub_au, Ua_au, r_bohr, damped_mask, xi)

        # Γ³_ab'(r) = d²γ/(dU_A dr) * dU_A/dq_A   in Hartree/Bohr
        gamma3ab_pR_vals = g2pUpR_ab * dUa_au  # Ha/Bohr
        gamma3ba_pR_vals = g2pUpR_ba * dUb_au  # Ha/Bohr

        # Convert: Ha/Bohr → eV/Å  (multiply by HA_TO_EV / BOHR_TO_ANG = HA_TO_EV * ANG_TO_BOHR)
        conv = self.HA_TO_EV * self.ANG_TO_BOHR
        gamma3ab_pR_vals = gamma3ab_pR_vals * conv
        gamma3ba_pR_vals = gamma3ba_pR_vals * conv

        self.gamma3ab_pR = torch.zeros((Nats, Nats), device=device, dtype=dtype)
        self.gamma3ba_pR = torch.zeros((Nats, Nats), device=device, dtype=dtype)
        self.gamma3ab_pR.view(-1).index_add_(0, flat_idx, gamma3ab_pR_vals)
        self.gamma3ba_pR.view(-1).index_add_(0, flat_idx, gamma3ba_pR_vals)

        # Store displacement info for gradient assembly  (3, Npairs)
        # diff_xyz[k, pair] = (R_A - R_B)[k] / |R_A - R_B|  (unit vector, Å)
        # We need to store full (3, Nats, Nats) for gradient computation
        self.diff_xyz = torch.zeros((3, Nats, Nats), device=device, dtype=dtype)
        for k in range(3):
            self.diff_xyz[k].view(-1).index_add_(0, flat_idx, dR_dxyz[:, k])

    # ── SCF interface ────────────────────────────────────────────────────

    def get_shifts(self, q: torch.Tensor) -> torch.Tensor:
        """Third-order Hamiltonian shift per atom.

        Returns σ₃(A) to be **added** to the second-order Coulomb potential
        before forming the Hamiltonian diagonal.

        σ₃(A) = (1/3) [2 ∑_B Γ³_AB · q_B · q_A  +  ∑_B Γ³_BA · q_B²]

        This is the derivative of the third-order energy with respect to q_A.

        Parameters
        ----------
        q : (Nats,) tensor – Mulliken charges (Δq).

        Returns
        -------
        shift : (Nats,) tensor in eV.
        """
        # Term 1:  2 * (Γ³_ab @ q) * q_A  (element-wise)
        # Term 2:  Γ³_ba @ (q²)
        t1 = 2.0 * (self.gamma3ab @ q) * q  # (Nats,)
        t2 = self.gamma3ba @ (q**2)  # (Nats,)
        return (1.0 / 3.0) * (t1 + t2)

    def get_dshifts_dq(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Linearized third-order shift response for Krylov SCF acceleration.

        Computes dσ₃/dλ|_{λ=0} for q → q + λ·v.

        dσ₃(A)/dλ = (1/3) [2 * ((Γ³_ab @ v) * q_A + (Γ³_ab @ q) * v_A)
                          + 2 * (Γ³_ba @ (q * v))]

        Parameters
        ----------
        q : (Nats,) – current charges.
        v : (Nats,) – perturbation direction.

        Returns
        -------
        d_shift : (Nats,) tensor in eV.
        """
        dt1 = 2.0 * ((self.gamma3ab @ v) * q + (self.gamma3ab @ q) * v)
        dt2 = 2.0 * (self.gamma3ba @ (q * v))
        return (1.0 / 3.0) * (dt1 + dt2)

    def get_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Third-order energy (scalar).

        Using the Euler relation for cubic functionals:

            E₃ = (1/3) ∑_A σ₃(A) · q_A

        where σ₃ is the Hamiltonian shift returned by ``get_shifts``.

        Parameters
        ----------
        q : (Nats,) charges.

        Returns
        -------
        E3 : scalar tensor in eV.
        """
        shift = self.get_shifts(q)
        return (1.0 / 3.0) * torch.sum(shift * q)

    def get_energy_xlbomd(
        self, q_in: torch.Tensor, q_out: torch.Tensor
    ) -> torch.Tensor:
        """Third-order energy for XL-BOMD shadow potential.

        Uses shifts built from q_in (shadow charges) and output charges q_out.
        Following DFTB+ ``getEnergyPerAtomXlbomd`` for non-shell-resolved:

            E₃ = Σ_A [shift1_A * qOut_A
                     + shift2_A * (qOut_A − qIn_A)
                     + shift3_A * (qOut_A − qIn_A)]

        where (non-shell-resolved):
            shift1(A) = shift3(A) = (1/3) Σ_B Γ³_AB · qIn_B · qIn_A
            shift2(A) = (1/3) Σ_B Γ³_BA · qIn_B²

        Parameters
        ----------
        q_in : (Nats,) – shadow/input charges (used to build shifts).
        q_out : (Nats,) – SCF output charges.

        Returns
        -------
        E3 : scalar tensor in eV.
        """
        s1 = (1.0 / 3.0) * (self.gamma3ab @ q_in) * q_in  # shift1 = shift3
        s2 = (1.0 / 3.0) * (self.gamma3ba @ (q_in**2))  # shift2
        dq = q_out - q_in
        return torch.sum(s1 * q_out + s2 * dq + s1 * dq)

    def get_gradient_dc(self, q: torch.Tensor) -> torch.Tensor:
        """Gradient from dΓ³/dr (the "dc" contribution to forces).

        Following DFTB+ ``addGradientDc``, for non-shell-resolved case:

        F₃(A) = -(1/3) ∑_{B≠A} q_A · q_B · [q_A · Γ³_AB'(r) + q_B · Γ³_BA'(r)]
                         · (R_A - R_B) / r_AB

        Parameters
        ----------
        q : (Nats,) charges.

        Returns
        -------
        gradient : (3, Nats) tensor in eV/Å.
        """
        Nats = self.Nats

        # tmp(A,B) = q_A * q_B * [q_A * gamma3ab_pR(A,B) + q_B * gamma3ba_pR(A,B)]
        qq = q.unsqueeze(1) * q.unsqueeze(0)  # (Nats, Nats)
        tmp = qq * (
            q.unsqueeze(1) * self.gamma3ab_pR + q.unsqueeze(0) * self.gamma3ba_pR
        )
        # Zero diagonal (no self-interaction gradient)
        tmp.fill_diagonal_(0.0)
        tmp = (1.0 / 3.0) * tmp

        # F(k, A) = sum_B  tmp(A,B) * diff_xyz(k, A, B)
        # The diff_xyz already contains the unit vector * sign
        grad = torch.zeros((3, Nats), device=self.device, dtype=self.dtype)
        for k in range(3):
            # grad[k, A] = sum_B tmp[A,B] * diff_xyz[k,A,B]
            grad[k] = (tmp * self.diff_xyz[k]).sum(dim=1)

        return grad

    def get_gradient_dc_xlbomd(
        self, q_in: torch.Tensor, q_out: torch.Tensor
    ) -> torch.Tensor:
        """Gradient for XL-BOMD case where shifts are built from q_in but
        forces use q_out.

        Following DFTB+ ``addGradientDcXlbomd`` for non-shell-resolved:

        tmp(A,B) = Γ³ab'(A,B) * (qIn_A * qIn_B * qOut_A
                       + qIn_A² * dq_B + qIn_A * qIn_B * dq_A)
                 + Γ³ba'(A,B) * (qIn_B * qIn_A * qOut_B
                       + qIn_B² * dq_A + qIn_B * qIn_A * dq_B)

        where dq = qOut − qIn.

        Parameters
        ----------
        q_in : (Nats,) – shadow/input charges.
        q_out : (Nats,) – SCF output charges.

        Returns
        -------
        gradient : (3, Nats) tensor in eV/Å.
        """
        Nats = self.Nats
        dq = q_out - q_in

        # Build (Nats, Nats) outer products
        qI_A = q_in.unsqueeze(1)  # (Nats, 1) – rows = atom A
        qI_B = q_in.unsqueeze(0)  # (1, Nats) – cols = atom B
        qO_A = q_out.unsqueeze(1)
        qO_B = q_out.unsqueeze(0)
        dq_A = dq.unsqueeze(1)
        dq_B = dq.unsqueeze(0)

        part1 = self.gamma3ab_pR * (
            qI_A * qI_B * qO_A + qI_A**2 * dq_B + qI_A * qI_B * dq_A
        )
        part2 = self.gamma3ba_pR * (
            qI_B * qI_A * qO_B + qI_B**2 * dq_A + qI_B * qI_A * dq_B
        )
        tmp = (1.0 / 3.0) * (part1 + part2)
        tmp.fill_diagonal_(0.0)

        grad = torch.zeros((3, Nats), device=self.device, dtype=self.dtype)
        for k in range(3):
            grad[k] = (tmp * self.diff_xyz[k]).sum(dim=1)

        return grad


# ═══════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════


def create_thirdorder(
    Hubbard_U: torch.Tensor,
    dU_dq: torch.Tensor,
    TYPE: torch.Tensor,
    h_damp_exp: float = 4.0,
) -> ThirdOrder:
    """Convenience constructor.

    Parameters
    ----------
    Hubbard_U : (Nats,) – per-atom Hubbard U in eV.
    dU_dq : (Nats,) – per-atom dU/dq in eV/e.
    TYPE : (Nats,) – atomic numbers (long tensor).
    h_damp_exp : float – ζ for H-damping.

    Returns
    -------
    ThirdOrder instance (call ``update_coords`` before use).
    """
    return ThirdOrder(Hubbard_U, dU_dq, TYPE, h_damp_exp)
