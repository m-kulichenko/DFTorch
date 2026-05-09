"""
DFT-D3(BJ) dispersion correction  —  vectorized PyTorch implementation.

Implements the SimpleDftD3 model from DFTB+ with Becke-Johnson rational
damping, following the Fortran source exactly:
  - src/dftbp/dftb/simpledftd3.F90       (energy / gradients)
  - src/dftbp/dftb/coordnumber.F90       (fractional coordination numbers)
  - src/dftbp/dftb/dftd3param.F90        (reference C6 data)
  - src/dftbp/dftb/dftd4param.F90        (sqrtZr4r2)

Reference: S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
           J. Chem. Phys. 132, 154104 (2010).
           S. Grimme, S. Ehrlich, L. Goerigk,
           J. Comput. Chem. 32, 1456 (2011).

All internal calculations in atomic units (Bohr, Hartree).
Public API accepts coordinates in **Angstrom** (DFTorch convention) and
returns energies in **eV** and forces in **eV/Angstrom**.

Two force paths are provided:
  * **analytical** (default) — fully vectorized closed-form gradients following
    the Fortran implementation.  The energy graph is kept differentiable so
    that ``torch.autograd`` can back-propagate through D3 energy into
    upstream parameters (DFTorch philosophy).
  * **autograd** — forces via ``torch.autograd.grad`` on the energy; useful
    as a reference / sanity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ._cell import normalize_cell, normalize_cell_batch
from ._tools import _maybe_compile

# ---------------------------------------------------------------------------
# Unit conversions  (DFTorch internal: eV / Angstrom)
# ---------------------------------------------------------------------------
BOHR_TO_ANG: float = 0.529177249
ANG_TO_BOHR: float = 1.0 / BOHR_TO_ANG
HA_TO_EV: float = 27.211386245988
EV_TO_HA: float = 1.0 / HA_TO_EV

# ---------------------------------------------------------------------------
# Load reference data (generated from DFTB+ Fortran sources)
# ---------------------------------------------------------------------------
_PARAMS_DIR = Path(__file__).resolve().parent / "params"
_REF_FILE = _PARAMS_DIR / "dftd3_reference.npz"

_ref_data = np.load(_REF_FILE, allow_pickle=False)

# c6ab[pair_idx, iref, jref]  — pair_idx is 0-based triangular index
#   pair(a,b) with a<=b: idx = (a-1) + b*(b-1)/2    (1-based Z → 0-based idx)
_C6AB_NP: np.ndarray = _ref_data["c6ab"]  # (4465, 5, 5)
_NUM_REF_NP: np.ndarray = _ref_data["numberOfReferences"]  # (94,)
_REF_CN_NP: np.ndarray = _ref_data["referenceCN"]  # (94, 5)
_SQRT_ZR4R2_NP: np.ndarray = _ref_data["sqrtZr4r2"]  # (118,)
_COV_RAD_BOHR_NP: np.ndarray = _ref_data["covRadii_bohr"]  # (118,)

MAX_ELEM: int = 94
MAX_REF: int = 5

# ---------------------------------------------------------------------------
# D3 constants (matching simpledftd3.F90)
# ---------------------------------------------------------------------------
_WEIGHTING_FACTOR: float = 4.0  # Gaussian exponent for CN weighting
_KCN: float = 16.0  # steepness of exponential CN counting func
_D3_SCALING: float = 4.0 / 3.0  # covalent radii scaling for D3 CN
_CUTOFF_CN: float = 40.0  # Bohr — CN counting cutoff
_CUTOFF_DISP: float = 64.0  # Bohr — dispersion interaction cutoff


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


class SimpleDftD3:
    """
    DFT-D3(BJ) dispersion calculator.

    Parameters
    ----------
    atomic_numbers : list[int] or (N,) array
        Atomic numbers for each atom.
    s6, s8, a1, a2 : float
        BJ damping parameters.  ``a2`` is in **Bohr**.
    s10 : float, optional
        Coefficient for the r⁻¹⁰ term (default 0).
    cutoff_cn : float, optional
        CN counting cutoff in Bohr (default 40).
    cutoff_disp : float, optional
        Dispersion interaction cutoff in Bohr (default 64).
    device, dtype : torch device / dtype
        Tensor placement.
    """

    def __init__(
        self,
        atomic_numbers: list[int] | np.ndarray | torch.Tensor,
        s6: float = 1.0,
        s8: float = 0.5883,
        a1: float = 0.5719,
        a2: float = 3.6017,
        s10: float = 0.0,
        cutoff_cn: float = _CUTOFF_CN,
        cutoff_disp: float = _CUTOFF_DISP,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float64

        self.device = device
        self.dtype = dtype

        Z = np.asarray(atomic_numbers, dtype=np.int64)
        self.N = len(Z)

        if np.any(Z < 1) or np.any(Z > MAX_ELEM):
            raise ValueError(f"Atomic numbers must be in [1, {MAX_ELEM}]; got {Z}")

        # --- damping parameters ---
        self.s6 = s6
        self.s8 = s8
        self.s10 = s10
        self.a1 = a1
        self.a2 = a2
        self.cutoff_cn = cutoff_cn
        self.cutoff_disp = cutoff_disp

        # --- per-atom data ---
        # sqrtZr4r2 indexed by Z  (1-based → 0-based)
        self._sqrtZr4r2 = torch.tensor(
            _SQRT_ZR4R2_NP[Z - 1], device=device, dtype=dtype
        )  # (N,)

        # D3-scaled covalent radii in Bohr
        self._covrad = torch.tensor(
            _D3_SCALING * _COV_RAD_BOHR_NP[Z - 1], device=device, dtype=dtype
        )  # (N,)

        # number of reference systems per atom
        nref = _NUM_REF_NP[Z - 1]  # (N,) int
        self._nref = torch.tensor(nref, device=device, dtype=torch.int64)

        # reference CN for each atom's references   (N, maxRef)
        self._refcn = torch.tensor(_REF_CN_NP[Z - 1], device=device, dtype=dtype)

        # --- pair reference C6 data ---
        # Build (N, N, maxRef, maxRef) tensor of reference C6 values.
        # Fortran indexing: for ati <= atj,
        #   ic = ati + atj*(atj-1)/2          (1-based Z)
        #   c6 = referenceC6(jref, iref, ic)  (note transpose!)
        # For ati > atj, swap iref/jref.
        c6_pair = np.zeros((self.N, self.N, MAX_REF, MAX_REF), dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                zi, zj = int(Z[i]), int(Z[j])
                zmin, zmax = min(zi, zj), max(zi, zj)
                ic = (zmin - 1) + zmax * (zmax - 1) // 2  # 0-based pair index
                if zi <= zj:
                    # c6ab[ic, jref, iref] (Fortran column-major: first idx = jref)
                    c6_pair[i, j] = _C6AB_NP[ic]  # (maxRef_j, maxRef_i) but symmetric
                else:
                    c6_pair[i, j] = _C6AB_NP[ic].T

        self._c6ref = torch.tensor(c6_pair, device=device, dtype=dtype)

        # Pre-compute rc = 3 * sqrtZr4r2[i] * sqrtZr4r2[j] for all pairs
        self._rc = 3.0 * self._sqrtZr4r2.unsqueeze(1) * self._sqrtZr4r2.unsqueeze(0)

        # BJ damping radius: a1*sqrt(rc) + a2
        self._rc1 = self.a1 * torch.sqrt(self._rc) + self.a2  # (N, N)

        # Reference mask for Gaussian weighting
        self._ref_mask = (
            torch.arange(MAX_REF, device=device).unsqueeze(0) < self._nref.unsqueeze(1)
        ).to(dtype)

    def _pair_diff(
        self, coords_bohr: torch.Tensor, cell_bohr: torch.Tensor = None
    ) -> torch.Tensor:
        diff = coords_bohr.unsqueeze(0) - coords_bohr.unsqueeze(1)  # (N, N, 3)
        if cell_bohr is None:
            return diff

        cell_bohr = normalize_cell(cell_bohr, device=diff.device, dtype=diff.dtype)
        cell_inv = torch.linalg.inv(cell_bohr)
        diff_frac = diff @ cell_inv
        diff_frac = diff_frac - torch.round(diff_frac)
        return diff_frac @ cell_bohr

    def _pair_diff_batch(
        self, coords_bohr: torch.Tensor, cell_bohr: torch.Tensor = None
    ) -> torch.Tensor:
        diff = coords_bohr.unsqueeze(1) - coords_bohr.unsqueeze(2)  # (B, N, N, 3)
        if cell_bohr is None:
            return diff

        cell_bohr = normalize_cell_batch(
            cell_bohr,
            coords_bohr.shape[0],
            device=diff.device,
            dtype=diff.dtype,
        )
        cell_inv = torch.linalg.inv(cell_bohr)
        diff_frac = torch.einsum("bnmi,bij->bnmj", diff, cell_inv)
        diff_frac = diff_frac - torch.round(diff_frac)
        return torch.einsum("bnmi,bij->bnmj", diff_frac, cell_bohr)

    # -------------------------------------------------------------------
    # Core energy computation (Bohr / Hartree, differentiable)
    # -------------------------------------------------------------------

    def _compute_energy_ha(
        self, coords_bohr: torch.Tensor, cell_bohr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        D3(BJ) dispersion energy in Hartree.

        Fully differentiable w.r.t. ``coords_bohr`` via autograd.
        """
        diff = self._pair_diff(coords_bohr, cell_bohr)
        r2 = (diff * diff).sum(dim=-1)  # (N, N)
        diag_mask = torch.eye(self.N, device=self.device, dtype=self.dtype) * 1e30
        r2 = r2 + diag_mask
        r = torch.sqrt(r2)

        # --- coordination numbers ---
        cn = self._compute_cn(r)  # (N,)

        # --- C6 interpolation ---
        c6 = self._weight_references(cn)  # (N, N)

        # --- BJ-damped energy ---
        r6 = r2 * r2 * r2
        r8 = r6 * r2
        rc1 = self._rc1
        rc = self._rc

        f6 = 1.0 / (r6 + rc1**6)
        f8 = 1.0 / (r8 + rc1**8)

        dEr = self.s6 * f6 + self.s8 * f8 * rc
        if self.s10 != 0.0:
            r10 = r8 * r2
            f10 = 1.0 / (r10 + rc1**10)
            dEr = dEr + self.s10 * (49.0 / 40.0) * rc * rc * f10

        pair_mask = (r < self.cutoff_disp).to(self.dtype)
        pair_mask = pair_mask * (
            1.0 - torch.eye(self.N, device=self.device, dtype=self.dtype)
        )

        e_disp_ha = -0.5 * (c6 * dEr * pair_mask).sum()
        return e_disp_ha

    # -------------------------------------------------------------------
    # Analytical forces (Bohr / Hartree) — matches Fortran exactly
    # -------------------------------------------------------------------

    def _compute_analytical_forces_ha(
        self,
        coords_bohr: torch.Tensor,
        cell_bohr: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        D3(BJ) energy (Hartree) and analytical forces (Hartree/Bohr).

        Returns
        -------
        e_ha : scalar Tensor
        f_ha : (3, N) Tensor  — forces = −dE/dR
        """
        N = self.N
        dev = self.device
        dt = self.dtype

        # ── pair geometry ────────────────────────────────────────────
        diff = self._pair_diff(coords_bohr, cell_bohr)
        r2 = (diff * diff).sum(dim=-1)  # (N,N)
        diag_inf = torch.eye(N, device=dev, dtype=dt) * 1e30
        r2safe = r2 + diag_inf
        r = torch.sqrt(r2safe)

        off_diag = 1.0 - torch.eye(N, device=dev, dtype=dt)
        disp_mask = (r < self.cutoff_disp).to(dt) * off_diag  # (N,N)
        cn_mask = (r < self.cutoff_cn).to(dt) * off_diag  # (N,N)

        # ── 1) coordination numbers + dcn/dr ─────────────────────────
        #  CN(i) = Σ_j  1 / (1 + exp(-kcn*(rc0_ij/r_ij - 1)))
        #  dCN(i)/dr_ij = -kcn * rc0_ij / r_ij^2 * exp(-kcn*(rc0_ij/r_ij-1))
        #                  / (1 + exp(-kcn*(rc0_ij/r_ij-1)))^2
        rc0 = self._covrad.unsqueeze(0) + self._covrad.unsqueeze(1)  # (N,N)
        arg = -_KCN * (rc0 / r - 1.0)
        expterm = torch.exp(arg.clamp(max=50.0))
        cn_count = cn_mask / (1.0 + expterm)  # (N,N)
        cn = cn_count.sum(dim=1)  # (N,)

        # scalar derivative  d(count_ij)/d(r_ij)  — Fortran dexpCount
        #   = -kcn * rc0 * exp(-kcn*(rc0/r-1)) / (r^2 * (1+exp(-..))^2)
        dcount_dr = (-_KCN * rc0 * expterm) / (r2safe * (1.0 + expterm) ** 2)
        dcount_dr = dcount_dr * cn_mask  # (N,N)

        # ── 2) Gaussian weights + d(gw)/d(CN) ───────────────────────
        wf = _WEIGHTING_FACTOR
        dcn = cn.unsqueeze(1) - self._refcn  # (N, maxRef)
        gw_raw = torch.exp(-wf * dcn * dcn) * self._ref_mask  # (N, maxRef)
        gw_sum = gw_raw.sum(dim=1, keepdim=True).clamp(min=1e-30)
        gw = gw_raw / gw_sum  # (N, maxRef)

        # derivative of un-normalised weight:  d(gw_raw)/dCN = 2*wf*(cnref-cn)*gw_raw
        dgw_raw = 2.0 * wf * (-dcn) * gw_raw  # (N, maxRef)
        dgw_sum = dgw_raw.sum(dim=1, keepdim=True)
        # quotient rule: d(gw_k)/dCN = (dgw_raw_k * sum - gw_raw_k * dgw_sum) / sum^2
        gwdcn = (dgw_raw * gw_sum - gw_raw * dgw_sum) / (gw_sum * gw_sum)  # (N, maxRef)

        # ── 3) C6 interpolation + dc6/dCN ───────────────────────────
        # c6(i,j) = Σ_rs  gw(i,r) * c6ref(i,j,r,s) * gw(j,s)
        c6 = torch.einsum("ir,ijrs,js->ij", gw, self._c6ref, gw)  # (N,N)

        # dc6dcn(i,j) = dC6(i,j)/dCN(i) = Σ_rs gwdcn(i,r) * c6ref * gw(j,s)
        dc6dcn = torch.einsum("ir,ijrs,js->ij", gwdcn, self._c6ref, gw)  # (N,N)
        # note: dc6/dCN(j) for pair (i,j) = dc6dcn(j,i)  (swap roles)

        # ── 4) BJ-damped dispersion energy + direct dE/dr ───────────
        r4 = r2safe * r2safe
        r5 = r4 * r
        r6 = r4 * r2safe
        r8 = r6 * r2safe
        rc1 = self._rc1
        rc = self._rc

        f6 = 1.0 / (r6 + rc1**6)
        f8 = 1.0 / (r8 + rc1**8)

        # df/dr  (Fortran: df6 = -6*r5*f6^2,  df8 = -8*r2*r5*f8^2)
        df6 = -6.0 * r5 * f6 * f6
        df8 = -8.0 * r2safe * r5 * f8 * f8

        dEr = self.s6 * f6 + self.s8 * f8 * rc
        dGr = self.s6 * df6 + self.s8 * df8 * rc

        if self.s10 != 0.0:
            r10 = r8 * r2safe
            f10 = 1.0 / (r10 + rc1**10)
            df10 = -10.0 * r4 * r5 * f10 * f10
            c10 = self.s10 * (49.0 / 40.0) * rc * rc
            dEr = dEr + c10 * f10
            dGr = dGr + c10 * df10

        # energy
        e_disp_ha = -0.5 * (c6 * dEr * disp_mask).sum()

        # ── 5) assemble forces ───────────────────────────────────────
        # 5a. Direct gradient (dE/dR at constant C6):
        #     Fortran: grad = -dGr * c6 * vec/r,  vec = R_i - R_j
        #     Our diff[i,j] = R_j - R_i, so R_i - R_j = -diff[i,j].
        coeff_direct = (-dGr * c6 * disp_mask) / r  # (N,N)
        # f_direct[i] = Σ_j coeff[i,j] * (R_i-R_j) = -Σ_j coeff[i,j] * diff[i,j]
        f_direct = -torch.einsum(
            "ij,ijc->ic", coeff_direct, diff
        )  # (N, 3)  [= gradient]

        # 5b. CN chain-rule:  dEdcn[i] = -Σ_j dc6dcn[i,j] * dEr[i,j]
        dEdcn = -(dc6dcn * dEr * disp_mask).sum(dim=1)  # (N,)

        # 5c. CN-coordinate chain rule (coordnumber.F90 → gemv):
        #     For pair (i,j):
        #       gradient(:,i) += countd * (dEdcn[i] + dEdcn[j])
        #       gradient(:,j) -= countd * (dEdcn[i] + dEdcn[j])
        #     where countd = dcount_dr * vec/r, vec = R_i - R_j = -diff[i,j].
        #     Vectorised as: B[i,j] = (dEdcn[i]+dEdcn[j]) * dcount_dr[i,j] / r[i,j]
        #     gradient(i) = -Σ_j B[i,j] * diff[i,j]
        #     force(i)    =  Σ_j B[i,j] * diff[i,j]
        A = dcount_dr / r  # (N,N)
        coeff_cn = (dEdcn.unsqueeze(0) + dEdcn.unsqueeze(1)) * A  # (N,N)
        f_cn = torch.einsum("ij,ijc->ic", coeff_cn, diff)  # (N, 3)  [= force]

        # ── total force (Hartree / Bohr) ─────────────────────────────
        # f_direct = gradient (dE/dR);  f_cn = force (−dE/dR from CN chain rule)
        f_ha = (-f_direct + f_cn).T  # (3, N)

        return e_disp_ha, f_ha

    # -------------------------------------------------------------------
    # Public: energy + forces  (Angstrom / eV)
    # -------------------------------------------------------------------

    def _get_dispersion_analytical(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytical D3(BJ) dispersion energy and forces in DFTorch units."""
        coords_bohr = coords.to(self.dtype) * ANG_TO_BOHR
        cell_bohr = None if cell is None else cell.to(self.dtype) * ANG_TO_BOHR
        e_ha, f_ha = self._compute_analytical_forces_ha(coords_bohr, cell_bohr)
        e_ev = e_ha * HA_TO_EV
        f_ev_ang = f_ha * (HA_TO_EV * ANG_TO_BOHR)
        return e_ev, f_ev_ang

    def get_dispersion(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
        use_autograd: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute D3(BJ) dispersion energy and atomic forces.

        Parameters
        ----------
        coords : (N, 3) torch.Tensor
            Cartesian coordinates in **Angstrom**.
        use_autograd : bool
            If *True*, forces are computed via ``torch.autograd.grad``
            (useful for testing).  Default *False* → analytical forces.

        Returns
        -------
        e_disp : scalar torch.Tensor
            Dispersion energy in **eV**.
        f_disp : (3, N) torch.Tensor
            Dispersion forces (= −dE/dR) in **eV / Angstrom**.
            Shape (3, N) matches DFTorch convention.
        """
        if use_autograd:
            return self._get_dispersion_autograd(coords, cell)

        return self._get_dispersion_analytical(coords, cell)

    def _get_dispersion_autograd(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forces via ``torch.autograd`` (reference implementation)."""
        coords_ang = coords.to(self.dtype).detach().requires_grad_(True)
        coords_bohr = coords_ang * ANG_TO_BOHR
        cell_bohr = None if cell is None else cell.to(self.dtype) * ANG_TO_BOHR

        with torch.enable_grad():
            e_ha = self._compute_energy_ha(coords_bohr, cell_bohr)
            e_ev = e_ha * HA_TO_EV
            (grad_ev_ang,) = torch.autograd.grad(
                e_ev,
                coords_ang,
                create_graph=False,
            )

        f_disp = (-grad_ev_ang).T  # (3, N)
        return e_ev.detach(), f_disp.detach()

    def get_energy(
        self, coords: torch.Tensor, cell: torch.Tensor = None
    ) -> torch.Tensor:
        """Return dispersion energy (eV), differentiable w.r.t. *coords*."""
        coords_bohr = coords.to(self.dtype) * ANG_TO_BOHR
        cell_bohr = None if cell is None else cell.to(self.dtype) * ANG_TO_BOHR
        e_ha = self._compute_energy_ha(coords_bohr, cell_bohr)
        return e_ha * HA_TO_EV

    def _get_forces_analytical(
        self, coords: torch.Tensor, cell: torch.Tensor = None
    ) -> torch.Tensor:
        """Analytical dispersion forces (3, N) in eV/Å."""
        _, f = self._get_dispersion_analytical(coords, cell)
        return f

    def get_forces(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
        use_autograd: bool = False,
    ) -> torch.Tensor:
        """Return dispersion forces (3, N) in eV/Å."""
        if use_autograd:
            _, f = self.get_dispersion(coords, cell, use_autograd=True)
            return f
        return self._get_forces_analytical(coords, cell)

    # -------------------------------------------------------------------
    # Batched API  — coords (B, N, 3) in Angstrom
    # -------------------------------------------------------------------

    def get_energy_batch(
        self, coords: torch.Tensor, cell: torch.Tensor = None
    ) -> torch.Tensor:
        """Dispersion energy for a batch of geometries.

        Parameters
        ----------
        coords : (B, N, 3) in Angstrom.

        Returns
        -------
        e_disp : (B,) in eV, differentiable w.r.t. *coords*.
        """
        coords_bohr = coords.to(self.dtype) * ANG_TO_BOHR  # (B,N,3)
        cell_bohr = None if cell is None else cell.to(self.dtype) * ANG_TO_BOHR

        diff = self._pair_diff_batch(coords_bohr, cell_bohr)
        r2 = (diff * diff).sum(dim=-1)  # (B,N,N)
        diag_mask = torch.eye(self.N, device=self.device, dtype=self.dtype) * 1e30
        r2 = r2 + diag_mask.unsqueeze(0)
        r = torch.sqrt(r2)

        # CN  (B,N)
        rc0 = self._covrad.unsqueeze(0) + self._covrad.unsqueeze(1)  # (N,N)
        mask = (r < self.cutoff_cn).to(self.dtype)
        off_diag = 1.0 - torch.eye(self.N, device=self.device, dtype=self.dtype)
        mask = mask * off_diag.unsqueeze(0)
        arg = -_KCN * (rc0.unsqueeze(0) / r - 1.0)
        expterm = torch.exp(arg.clamp(max=50.0))
        cn = (mask / (1.0 + expterm)).sum(dim=2)  # (B,N)

        # C6 interpolation  (B,N,N)
        dcn = cn.unsqueeze(2) - self._refcn.unsqueeze(0)  # (B,N,maxRef)
        gw_raw = torch.exp(-_WEIGHTING_FACTOR * dcn * dcn) * self._ref_mask.unsqueeze(0)
        gw_sum = gw_raw.sum(dim=2, keepdim=True).clamp(min=1e-30)
        gw = gw_raw / gw_sum  # (B,N,maxRef)
        c6 = torch.einsum("bir,ijrs,bjs->bij", gw, self._c6ref, gw)  # (B,N,N)

        # BJ-damped energy
        r6 = r2 * r2 * r2
        r8 = r6 * r2
        rc1 = self._rc1.unsqueeze(0)  # (1,N,N)
        rc = self._rc.unsqueeze(0)  # (1,N,N)

        f6 = 1.0 / (r6 + rc1**6)
        f8 = 1.0 / (r8 + rc1**8)
        dEr = self.s6 * f6 + self.s8 * f8 * rc

        if self.s10 != 0.0:
            r10 = r8 * r2
            f10 = 1.0 / (r10 + rc1**10)
            dEr = dEr + self.s10 * (49.0 / 40.0) * rc * rc * f10

        pair_mask = (r < self.cutoff_disp).to(self.dtype)
        pair_mask = pair_mask * off_diag.unsqueeze(0)

        e_disp_ha = -0.5 * (c6 * dEr * pair_mask).sum(dim=(1, 2))  # (B,)
        return e_disp_ha * HA_TO_EV

    def _get_forces_batch_analytical(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
    ) -> torch.Tensor:
        """Analytical batched dispersion forces (B, 3, N) in eV/Å."""
        # Analytical — loop-free over batch via full vectorised path
        N = self.N
        dev = self.device
        dt = self.dtype

        coords_bohr = coords.to(dt) * ANG_TO_BOHR
        cell_bohr = None if cell is None else cell.to(dt) * ANG_TO_BOHR

        diff = self._pair_diff_batch(coords_bohr, cell_bohr)
        r2 = (diff * diff).sum(dim=-1)  # (B,N,N)
        diag_inf = torch.eye(N, device=dev, dtype=dt).unsqueeze(0) * 1e30
        r2safe = r2 + diag_inf
        r = torch.sqrt(r2safe)

        off_diag = (1.0 - torch.eye(N, device=dev, dtype=dt)).unsqueeze(0)
        disp_mask = (r < self.cutoff_disp).to(dt) * off_diag
        cn_mask = (r < self.cutoff_cn).to(dt) * off_diag

        # 1) CN + dcount/dr
        rc0 = (self._covrad.unsqueeze(0) + self._covrad.unsqueeze(1)).unsqueeze(0)
        arg = -_KCN * (rc0 / r - 1.0)
        expterm = torch.exp(arg.clamp(max=50.0))
        cn_count = cn_mask / (1.0 + expterm)
        cn = cn_count.sum(dim=2)  # (B,N)

        dcount_dr = (-_KCN * rc0 * expterm) / (r2safe * (1.0 + expterm) ** 2)
        dcount_dr = dcount_dr * cn_mask

        # 2) Gaussian weights + d(gw)/d(CN)
        wf = _WEIGHTING_FACTOR
        dcn = cn.unsqueeze(2) - self._refcn.unsqueeze(0)  # (B,N,maxRef)
        gw_raw = torch.exp(-wf * dcn * dcn) * self._ref_mask.unsqueeze(0)
        gw_sum = gw_raw.sum(dim=2, keepdim=True).clamp(min=1e-30)
        gw = gw_raw / gw_sum

        dgw_raw = 2.0 * wf * (-dcn) * gw_raw
        dgw_sum = dgw_raw.sum(dim=2, keepdim=True)
        gwdcn = (dgw_raw * gw_sum - gw_raw * dgw_sum) / (gw_sum * gw_sum)

        # 3) C6 + dc6/dCN
        c6 = torch.einsum("bir,ijrs,bjs->bij", gw, self._c6ref, gw)
        dc6dcn = torch.einsum("bir,ijrs,bjs->bij", gwdcn, self._c6ref, gw)

        # 4) BJ-damped energy + direct dE/dr
        r4 = r2safe * r2safe
        r5 = r4 * r
        r6 = r4 * r2safe
        r8 = r6 * r2safe
        rc1 = self._rc1.unsqueeze(0)
        rc = self._rc.unsqueeze(0)

        f6 = 1.0 / (r6 + rc1**6)
        f8 = 1.0 / (r8 + rc1**8)
        df6 = -6.0 * r5 * f6 * f6
        df8 = -8.0 * r2safe * r5 * f8 * f8

        dEr = self.s6 * f6 + self.s8 * f8 * rc
        dGr = self.s6 * df6 + self.s8 * df8 * rc

        if self.s10 != 0.0:
            r10 = r8 * r2safe
            f10 = 1.0 / (r10 + rc1**10)
            df10 = -10.0 * r4 * r5 * f10 * f10
            c10 = self.s10 * (49.0 / 40.0) * rc * rc
            dEr = dEr + c10 * f10
            dGr = dGr + c10 * df10

        # 5) Forces
        coeff_direct = (-dGr * c6 * disp_mask) / r
        f_direct = -torch.einsum("bij,bijc->bic", coeff_direct, diff)  # (B,N,3)

        dEdcn = -(dc6dcn * dEr * disp_mask).sum(dim=2)  # (B,N)
        A = dcount_dr / r
        coeff_cn = (dEdcn.unsqueeze(1) + dEdcn.unsqueeze(2)) * A
        f_cn = torch.einsum("bij,bijc->bic", coeff_cn, diff)  # (B,N,3)

        f_ha = (-f_direct + f_cn).permute(0, 2, 1)  # (B,3,N)

        # Convert Ha/Bohr → eV/Å
        return f_ha * (HA_TO_EV * ANG_TO_BOHR)

    def get_forces_batch(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor = None,
        use_autograd: bool = False,
    ) -> torch.Tensor:
        """Dispersion forces for a batch of geometries.

        Parameters
        ----------
        coords : (B, N, 3) in Angstrom.

        Returns
        -------
        f_disp : (B, 3, N) in eV/Å.
        """
        if use_autograd:
            coords_ag = coords.detach().requires_grad_(True)
            e = self.get_energy_batch(coords_ag, cell)  # (B,)
            (grad,) = torch.autograd.grad(e.sum(), coords_ag, create_graph=False)
            return (-grad).permute(0, 2, 1)  # (B,3,N)
        return self._get_forces_batch_analytical(coords, cell)

    # -------------------------------------------------------------------
    # Coordination number (exponential counting function, kcn = 16)
    # -------------------------------------------------------------------

    def _compute_cn(self, r: torch.Tensor) -> torch.Tensor:
        """
        Fractional coordination numbers.

        Parameters
        ----------
        r : (N, N) pair distances in Bohr (diagonal = large sentinel).

        Returns
        -------
        cn : (N,)
        """
        rc0 = self._covrad.unsqueeze(0) + self._covrad.unsqueeze(1)

        mask = (r < self.cutoff_cn).to(self.dtype)
        mask = mask * (1.0 - torch.eye(self.N, device=self.device, dtype=self.dtype))

        arg = -_KCN * (rc0 / r - 1.0)
        expterm = torch.exp(arg.clamp(max=50.0))
        count = mask / (1.0 + expterm)

        return count.sum(dim=1)

    # -------------------------------------------------------------------
    # C6 interpolation via Gaussian weighting
    # -------------------------------------------------------------------

    def _weight_references(self, cn: torch.Tensor) -> torch.Tensor:
        """
        Interpolate C6 from reference data with Gaussian weights on CN.

        Parameters
        ----------
        cn : (N,) coordination numbers

        Returns
        -------
        c6 : (N, N) interpolated C6 for each pair
        """
        wf = _WEIGHTING_FACTOR

        dcn = cn.unsqueeze(1) - self._refcn  # (N, maxRef)
        gw_raw = torch.exp(-wf * dcn * dcn) * self._ref_mask

        gw_sum = gw_raw.sum(dim=1, keepdim=True).clamp(min=1e-30)
        gw = gw_raw / gw_sum  # (N, maxRef)

        c6 = torch.einsum("ir,ijrs,js->ij", gw, self._c6ref, gw)
        return c6


SimpleDftD3._compute_energy_ha_eager = SimpleDftD3._compute_energy_ha
SimpleDftD3._compute_analytical_forces_ha_eager = (
    SimpleDftD3._compute_analytical_forces_ha
)
SimpleDftD3._get_dispersion_analytical_eager = SimpleDftD3._get_dispersion_analytical
SimpleDftD3.get_dispersion_eager = SimpleDftD3.get_dispersion
SimpleDftD3.get_energy_eager = SimpleDftD3.get_energy
SimpleDftD3._get_forces_analytical_eager = SimpleDftD3._get_forces_analytical
SimpleDftD3.get_forces_eager = SimpleDftD3.get_forces
SimpleDftD3.get_energy_batch_eager = SimpleDftD3.get_energy_batch
SimpleDftD3._get_forces_batch_analytical_eager = (
    SimpleDftD3._get_forces_batch_analytical
)
SimpleDftD3.get_forces_batch_eager = SimpleDftD3.get_forces_batch

SimpleDftD3._compute_energy_ha = _maybe_compile(SimpleDftD3._compute_energy_ha)
SimpleDftD3._compute_analytical_forces_ha = _maybe_compile(
    SimpleDftD3._compute_analytical_forces_ha
)
SimpleDftD3._get_dispersion_analytical = _maybe_compile(
    SimpleDftD3._get_dispersion_analytical
)
SimpleDftD3.get_energy = _maybe_compile(SimpleDftD3.get_energy)
SimpleDftD3._get_forces_analytical = _maybe_compile(SimpleDftD3._get_forces_analytical)
SimpleDftD3.get_energy_batch = _maybe_compile(SimpleDftD3.get_energy_batch)
SimpleDftD3._get_forces_batch_analytical = _maybe_compile(
    SimpleDftD3._get_forces_batch_analytical
)


# ═══════════════════════════════════════════════════════════════════════════
# Factory function (mirroring create_gbsa pattern)
# ═══════════════════════════════════════════════════════════════════════════


def create_dftd3(
    atomic_numbers: list[int] | np.ndarray | torch.Tensor,
    s6: float = 1.0,
    s8: float = 0.5883,
    a1: float = 0.5719,
    a2: float = 3.6017,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SimpleDftD3:
    """
    Create a SimpleDftD3 dispersion calculator.

    Parameters
    ----------
    atomic_numbers : list[int] or array-like
        Atomic numbers of each atom.
    s6, s8, a1, a2 : float
        BJ-damping parameters.  ``a2`` in **Bohr**.
    device : torch.device, optional
        Device used for the internal tensors.
    dtype : torch.dtype, optional
        Floating-point dtype used for the internal tensors.

    Returns
    -------
    SimpleDftD3
        Dispersion calculator configured for the supplied species.
    """
    return SimpleDftD3(
        atomic_numbers=atomic_numbers,
        s6=s6,
        s8=s8,
        a1=a1,
        a2=a2,
        device=device,
        dtype=dtype,
    )
