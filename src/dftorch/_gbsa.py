"""
ALPB / Generalized Born / Solvent Accessible Surface Area solvation model.

Implements the Analytical Linearized Poisson-Boltzmann (ALPB) solvation model
following the DFTB+ source code exactly:
  - src/dftbp/solvation/born.F90
  - src/dftbp/solvation/sasa.F90

Vectorized implementation -- all O(N^2) pair loops replaced by tensor ops.
"""

from __future__ import annotations

import math
import os

import torch

# ---------------------------------------------------------------------------
# Unit conversions  (DFTorch internal: eV / angstrom)
# ---------------------------------------------------------------------------
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HA_TO_EV = 27.211386245988
EV_TO_HA = 1.0 / HA_TO_EV
KCAL_TO_HA = 1.0 / 627.509474
KCAL_TO_EV = KCAL_TO_HA * HA_TO_EV

# Coulomb constant in eV*angstrom / e^2
KE = HA_TO_EV * BOHR_TO_ANG  # 14.39964651...

# ---------------------------------------------------------------------------
# DFT-D3 van der Waals radii (ANGSTROM).  Index 0 = dummy.
# ---------------------------------------------------------------------------
VDW_D3_ANG = [
    0.0000,
    1.09155,
    0.86735,
    1.74780,
    1.54910,
    1.60800,
    1.45515,
    1.31125,
    1.24085,
    1.14980,
    1.06870,
    1.85410,
    1.74195,
    2.00530,
    1.89585,
    1.75085,
    1.65535,
    1.55230,
    1.45740,
    2.12055,
    2.05175,
    1.94515,
    1.88210,
    1.86055,
    1.72070,
    1.77310,
    1.72105,
    1.71635,
    1.67310,
    1.65040,
    1.61545,
    1.97895,
    1.93095,
    1.83125,
    1.76340,
    1.68310,
    1.60480,
    2.30880,
    2.23820,
    2.10980,
    2.02985,
    1.92980,
    1.87715,
    1.78450,
    1.73115,
    1.69875,
    1.67625,
    1.66540,
    1.73100,
    2.13115,
    2.09370,
    2.00750,
    1.94505,
    1.86900,
    1.79445,
    2.52835,
    2.59070,
    2.31305,
    2.31005,
    2.28510,
    2.26355,
    2.24480,
    2.22575,
    2.21170,
    2.06215,
    2.12135,
    2.07705,
    2.13970,
    2.12250,
    2.11040,
    2.09930,
    2.00650,
    2.12250,
    2.04900,
    1.99275,
    1.94775,
    1.87450,
    1.72280,
    1.67625,
    1.62820,
    1.67995,
    2.15635,
    2.13820,
    2.05875,
    2.00270,
    1.93220,
    1.86080,
    2.53980,
    2.46470,
    2.35215,
    2.21260,
    2.22970,
    2.19785,
    2.17695,
    2.21705,
]

# Pre-build lookup tensor (created once, on CPU; moved to device in __init__)
_VDW_TABLE = torch.tensor(VDW_D3_ANG, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Lebedev-Laikov 230-point angular grid
# ---------------------------------------------------------------------------
def _lebedev_230():
    """230-point Lebedev-Laikov grid. Weights sum to 1.0 (DFTB+ convention)."""
    import numpy as np

    z = 0.0
    pts, wts = [], []

    def _oh1(v):
        a = 1.0
        for c in [[a, z, z], [-a, z, z], [z, a, z], [z, -a, z], [z, z, a], [z, z, -a]]:
            pts.append(c)
            wts.append(v)

    def _oh3(v):
        a = 1.0 / math.sqrt(3.0)
        for sx in (a, -a):
            for sy in (a, -a):
                for sz in (a, -a):
                    pts.append([sx, sy, sz])
                    wts.append(v)

    def _oh4(a, v):
        b = math.sqrt(1.0 - 2.0 * a * a)
        for x, y, zz in [
            (a, a, b),
            (-a, a, b),
            (a, -a, b),
            (-a, -a, b),
            (a, a, -b),
            (-a, a, -b),
            (a, -a, -b),
            (-a, -a, -b),
            (a, b, a),
            (-a, b, a),
            (a, -b, a),
            (-a, -b, a),
            (a, b, -a),
            (-a, b, -a),
            (a, -b, -a),
            (-a, -b, -a),
            (b, a, a),
            (-b, a, a),
            (b, -a, a),
            (-b, -a, a),
            (b, a, -a),
            (-b, a, -a),
            (b, -a, -a),
            (-b, -a, -a),
        ]:
            pts.append([x, y, zz])
            wts.append(v)

    def _oh5(a, v):
        b = math.sqrt(1.0 - a * a)
        for x, y, zz in [
            (a, b, z),
            (-a, b, z),
            (a, -b, z),
            (-a, -b, z),
            (a, z, b),
            (-a, z, b),
            (a, z, -b),
            (-a, z, -b),
            (b, a, z),
            (-b, a, z),
            (b, -a, z),
            (-b, -a, z),
            (b, z, a),
            (-b, z, a),
            (b, z, -a),
            (-b, z, -a),
            (z, a, b),
            (z, -a, b),
            (z, a, -b),
            (z, -a, -b),
            (z, b, a),
            (z, -b, a),
            (z, b, -a),
            (z, -b, -a),
        ]:
            pts.append([x, y, zz])
            wts.append(v)

    def _oh6(a, b, v):
        c = math.sqrt(1.0 - a * a - b * b)
        for p, q, r in [
            (a, b, c),
            (-a, b, c),
            (a, -b, c),
            (-a, -b, c),
            (a, b, -c),
            (-a, b, -c),
            (a, -b, -c),
            (-a, -b, -c),
            (a, c, b),
            (-a, c, b),
            (a, -c, b),
            (-a, -c, b),
            (a, c, -b),
            (-a, c, -b),
            (a, -c, -b),
            (-a, -c, -b),
            (b, a, c),
            (-b, a, c),
            (b, -a, c),
            (-b, -a, c),
            (b, a, -c),
            (-b, a, -c),
            (b, -a, -c),
            (-b, -a, -c),
            (b, c, a),
            (-b, c, a),
            (b, -c, a),
            (-b, -c, a),
            (b, c, -a),
            (-b, c, -a),
            (b, -c, -a),
            (-b, -c, -a),
            (c, a, b),
            (-c, a, b),
            (c, -a, b),
            (-c, -a, b),
            (c, a, -b),
            (-c, a, -b),
            (c, -a, -b),
            (-c, -a, -b),
            (c, b, a),
            (-c, b, a),
            (c, -b, a),
            (-c, -b, a),
            (c, b, -a),
            (-c, b, -a),
            (c, -b, -a),
            (-c, -b, -a),
        ]:
            pts.append([p, q, r])
            wts.append(v)

    _oh1(-0.5522639919727325e-1)
    _oh3(0.4450274607445226e-2)
    _oh4(0.4492044687397611e0, 0.4496841067921404e-2)
    _oh4(0.2520419490210201e0, 0.5049153450478750e-2)
    _oh4(0.6981906658447242e0, 0.3976408018051883e-2)
    _oh4(0.6587405243460960e0, 0.4401400650381014e-2)
    _oh4(0.4038544050097660e-1, 0.1724544350544401e-1)
    _oh5(0.5823842309715585e0, 0.4231083095357343e-2)
    _oh5(0.3545877390518688e0, 0.5198069864064399e-2)
    _oh6(0.2272181808998187e0, 0.4864661535886647e0, 0.4695720972568883e-2)

    pts_arr = np.array(pts, dtype=np.float64)
    wts_arr = np.array(wts, dtype=np.float64)
    assert pts_arr.shape == (230, 3)
    return (
        torch.tensor(pts_arr, dtype=torch.float64),
        torch.tensor(wts_arr, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Parameter file reader  (gbsafile.F90 :: readParamGBSA)
# ---------------------------------------------------------------------------
def read_param_file(filepath):
    """
    Read a DFTB+ GBSA/ALPB parameter file (e.g. param_alpb_h2o.txt).

    File format (gbsafile.F90):
      Lines 1-8: global floats
      Lines 9-102: 94 elements (surfaceTension, descreening, hBondPar)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    vals = [float(lines[i].strip()) for i in range(8)]
    elem_st, elem_ds, elem_hb = {}, {}, {}
    for i in range(94):
        parts = lines[8 + i].split()
        Z = i + 1
        elem_st[Z] = float(parts[0])
        elem_ds[Z] = float(parts[1])
        elem_hb[Z] = float(parts[2])

    return {
        "epsilon": vals[0],
        "born_scale": vals[3],
        "probe_radius_ang": vals[4],
        "free_energy_shift_kcal": vals[5],
        "born_offset_ang": vals[6] * 0.1,
        "element_surf_tension": elem_st,
        "element_descreening": elem_ds,
        "element_hbond_par": elem_hb,
    }


# ===================================================================
# Vectorised helper: "standard" pair integral  (no overlap)
# ===================================================================
def _std(dist, rho_j):
    """
    Standard (non-overlapping) volume integral g_i and derivative dg_i/dr.

    dist  : (P,) pairwise distances
    rho_j : (P,) effective radii rho[j] for each pair

    Returns g, dg  each (P,).
    """
    r1 = 1.0 / dist
    ap = dist + rho_j
    am = dist - rho_j
    ab = ap * am
    rhab = rho_j / ab
    lnab = 0.5 * torch.log(am / ap) * r1
    g = rhab + lnab
    dg = -2.0 * rhab / ab + (rhab - lnab) * r1 * r1
    return g, dg


# ===================================================================
# Vectorised helper: "overlap" pair integral
# ===================================================================
def _ovl(dist, rho_j, vdw_i):
    """
    Overlapping volume integral g_i and dg_i for the case
    dist + rho_j > vdw_i  (otherwise g=0).

    dist  : (P,)
    rho_j : (P,)   effective radii of the *other* atom
    vdw_i : (P,)   vdW radius of the *receiving* atom

    Returns g, dg  each (P,).
    """
    r1 = 1.0 / dist
    r12 = 0.5 * r1
    r24 = r12 * r12
    ap = dist + rho_j
    am = dist - rho_j
    rh1 = 1.0 / vdw_i
    rhr1 = 1.0 / ap
    aprh1 = ap * rh1
    lnab = torch.log(aprh1)
    g = rh1 - rhr1 + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - lnab)
    dg = (
        rhr1 * rhr1 * (1.0 - 0.25 * am * r1 * (1.0 + aprh1 * aprh1))
        + rho_j * r24 * (rhr1 - rh1 * aprh1)
        + r12 * (r1 * lnab - rhr1)
    )
    dg = dg * r1
    return g, dg


# ===================================================================
# Differentiable helpers (value only — autograd handles derivatives)
# ===================================================================
def _std_g(dist, rho_j):
    """Standard (non-overlapping) volume integral g — differentiable.

    Safe for all dist > 0: clamps (dist - rho_j) to avoid log of
    negative values when called on overlap pairs (the result on those
    pairs is discarded by torch.where, but autograd still needs finite
    values to avoid NaN gradient pollution).
    """
    ap = dist + rho_j
    am = (dist - rho_j).clamp(min=1e-30)
    ab = ap * am
    return rho_j / ab + 0.5 * torch.log(am / ap) / dist


def _ovl_g(dist, rho_j, vdw_i):
    """Overlapping volume integral g — differentiable.

    Safe for all dist > 0: clamps (dist + rho_j) away from zero.
    """
    r12 = 0.5 / dist
    ap = (dist + rho_j).clamp(min=1e-30)
    am = dist - rho_j
    rh1 = 1.0 / vdw_i.clamp(min=1e-30)
    rhr1 = 1.0 / ap
    aprh1 = ap * rh1
    return (
        rh1
        - rhr1
        + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - torch.log(aprh1.clamp(min=1e-30)))
    )


# ===================================================================
# Main GBSA / ALPB class
# ===================================================================
class GBSA:
    """
    Generalized Born / ALPB / SASA solvation model (vectorised).

    Parameters
    ----------
    coords  : (N, 3) Cartesian coordinates in angstrom.
    species : (N,) Atomic numbers (Z).
    device  : torch.device
    param_file : str   Path to DFTB+ GBSA parameter file.
    alpb       : bool  Enable ALPB correction (default True).
    alpha_alpb : float ALPB alpha parameter (default 0.571412).
    """

    def __init__(
        self, coords, species, device, param_file, alpb=True, alpha_alpb=0.571412
    ):
        self.device = device
        self.nAtom = coords.shape[0]
        self.coords = coords.to(device)
        self.species = species.to(device)

        # Read parameter file
        params = read_param_file(param_file)
        eps = params["epsilon"]

        # ALPB keps and alpbet
        if alpb:
            self.alpbet = alpha_alpb / eps
        else:
            self.alpbet = 0.0
        self.keps = (1.0 / eps - 1.0) / (1.0 + self.alpbet)

        # Global parameters
        self.born_scale = params["born_scale"]
        self.born_offset = params["born_offset_ang"]
        self.obc = [1.0, 0.8, 4.85]
        self.probe_radius = params["probe_radius_ang"]
        self.free_energy_shift = params["free_energy_shift_kcal"] * KCAL_TO_EV
        self.smoothing_w = 0.3

        # ------ Per-atom parameters (vectorised table lookup) ------
        Z = species.long()
        Zc = Z.clamp(max=len(VDW_D3_ANG) - 1)

        vdw_table = _VDW_TABLE.to(device)
        vdw_rad = vdw_table[Zc]

        # Build element-to-value tensors (94 elements, index 0 unused)
        nElem = 95
        st_table = torch.zeros(nElem, dtype=torch.float64, device=device)
        ds_table = torch.zeros(nElem, dtype=torch.float64, device=device)
        hb_table = torch.zeros(nElem, dtype=torch.float64, device=device)
        for zz in range(1, nElem):
            st_table[zz] = params["element_surf_tension"].get(zz, 0.0)
            ds_table[zz] = params["element_descreening"].get(zz, 1.0)
            hb_table[zz] = params["element_hbond_par"].get(zz, 0.0)

        surf_tension = st_table[Zc] * 4.0e-5 * math.pi * HA_TO_EV / BOHR_TO_ANG**2
        descreening = ds_table[Zc]
        hbond_str = -(hb_table[Zc] ** 2) * KCAL_TO_EV

        self.vdw_rad = vdw_rad
        self.surf_tension = surf_tension
        self.descreening = descreening
        self.hbond_str = hbond_str
        self.rho = vdw_rad * descreening

        # Alias used by differentiable path
        self.N_atoms = self.nAtom

        # Pre-compute Lebedev grid (geometry-independent)
        ang_pts, ang_wts = _lebedev_230()
        self._ang_pts = ang_pts.to(device)  # (230, 3)
        self._ang_wts = ang_wts.to(device)  # (230,)

        # Pre-compute wrp per atom (depends only on vdw radii, not coords)
        w = self.smoothing_w
        rad = vdw_rad + self.probe_radius
        rm = rad - w
        rp = rad + w
        ah5 = -1.0 / (4.0 * w**3)
        self._wrp = (
            0.25 / w + 3.0 * ah5 * (0.2 * rp**2 - 0.5 * rp * rad + rad**2 / 3.0)
        ) * rp**3 - (
            0.25 / w + 3.0 * ah5 * (0.2 * rm**2 - 0.5 * rm * rad + rad**2 / 3.0)
        ) * rm**3  # (N,)

        # Geometry-dependent quantities
        self.born_rad = None
        self.dbrdr = None
        self.born_mat = None
        self.sasa = None
        self.dsasa_dr = None

        self._compute_born_radii()
        self._compute_born_matrix()
        self._compute_sasa()

    # -------------------------------------------------------------------
    # aDet -- molecular shape descriptor  (vectorised)
    # -------------------------------------------------------------------
    def _compute_aDet(self):
        """aDet = sqrt(det(inertia)^(1/3) / (2*totRad3) * 5)  (in Bohr)."""
        coords_bohr = self.coords * ANG_TO_BOHR
        rad_bohr = self.vdw_rad * ANG_TO_BOHR

        rad3 = rad_bohr**3
        totRad3 = rad3.sum()
        center = (coords_bohr * rad3.unsqueeze(1)).sum(dim=0) / totRad3
        vec = coords_bohr - center.unsqueeze(0)  # (N,3)

        rr = (vec * vec).sum(dim=1)  # (N,)
        tof_r2 = 0.4 * rad_bohr**2  # 2/5 * r^2
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)

        diag_part = ((rr + tof_r2) * rad3).sum() * eye3
        outer_part = torch.einsum("i,ij,ik->jk", rad3, vec, vec)
        inertia = diag_part - outer_part

        det_val = torch.det(inertia).item()
        aDet = math.sqrt(det_val ** (1.0 / 3.0) / (2.0 * totRad3.item()) * 5.0)
        return aDet

    # -------------------------------------------------------------------
    # Born radii (OBC corrected) -- vectorised psi
    # -------------------------------------------------------------------
    def _compute_psi(self):
        """Pairwise volume integrals psi_i (vectorised)."""
        N = self.nAtom
        coords = self.coords
        vdw = self.vdw_rad
        rho = self.rho

        # upper-triangle pair indices
        I, J = torch.triu_indices(N, N, offset=1, device=self.device)  # noqa: E741
        P = I.shape[0]

        vec = coords[I] - coords[J]  # (P,3)
        dist = vec.norm(dim=1)  # (P,)

        rhoi = rho[I]
        rhoj = rho[J]
        vdwi = vdw[I]
        vdwj = vdw[J]

        # Overlap flags  (bool masks over P pairs)
        tOvij = dist < (vdwi + rhoj)  # i overlaps j's rho
        tOvji = dist < (rhoi + vdwj)  # j overlaps i's rho

        # Allocate per-pair contributions
        gi = torch.zeros(P, dtype=torch.float64, device=self.device)
        dgi = torch.zeros(P, dtype=torch.float64, device=self.device)
        gj = torch.zeros(P, dtype=torch.float64, device=self.device)
        dgj = torch.zeros(P, dtype=torch.float64, device=self.device)

        # ---- Case 1: no overlap on either side ----
        m1 = (~tOvij) & (~tOvji)
        if m1.any():
            g1i, dg1i = _std(dist[m1], rhoj[m1])
            g1j, dg1j = _std(dist[m1], rhoi[m1])
            gi[m1] = g1i
            dgi[m1] = dg1i
            gj[m1] = g1j
            dgj[m1] = dg1j

        # ---- Case 2: j overlaps i  (tOvji only) ----
        m2 = (~tOvij) & tOvji
        if m2.any():
            g2i, dg2i = _std(dist[m2], rhoj[m2])
            gi[m2] = g2i
            dgi[m2] = dg2i
            m2b = m2.clone()
            m2b[m2] = (dist[m2] + rhoi[m2]) > vdwj[m2]
            if m2b.any():
                g2j, dg2j = _ovl(dist[m2b], rhoi[m2b], vdwj[m2b])
                gj[m2b] = g2j
                dgj[m2b] = dg2j

        # ---- Case 3: i overlaps j  (tOvij only) ----
        m3 = tOvij & (~tOvji)
        if m3.any():
            m3b = m3.clone()
            m3b[m3] = (dist[m3] + rhoj[m3]) > vdwi[m3]
            if m3b.any():
                g3i, dg3i = _ovl(dist[m3b], rhoj[m3b], vdwi[m3b])
                gi[m3b] = g3i
                dgi[m3b] = dg3i
            g3j, dg3j = _std(dist[m3], rhoi[m3])
            gj[m3] = g3j
            dgj[m3] = dg3j

        # ---- Case 4: both sides overlap ----
        m4 = tOvij & tOvji
        if m4.any():
            m4i = m4.clone()
            m4i[m4] = (dist[m4] + rhoj[m4]) > vdwi[m4]
            if m4i.any():
                g4i, dg4i = _ovl(dist[m4i], rhoj[m4i], vdwi[m4i])
                gi[m4i] = g4i
                dgi[m4i] = dg4i
            m4j = m4.clone()
            m4j[m4] = (dist[m4] + rhoi[m4]) > vdwj[m4]
            if m4j.any():
                g4j, dg4j = _ovl(dist[m4j], rhoi[m4j], vdwj[m4j])
                gj[m4j] = g4j
                dgj[m4j] = dg4j

        # ---- Scatter into per-atom psi and dpsi_dr ----
        psi = torch.zeros(N, dtype=torch.float64, device=self.device)
        psi.index_add_(0, I, gi)
        psi.index_add_(0, J, gj)

        dGi = dgi.unsqueeze(1) * vec  # (P,3)
        dGj = dgj.unsqueeze(1) * vec  # (P,3)

        dpsi_dr = torch.zeros(N, N, 3, dtype=torch.float64, device=self.device)
        for c in range(3):
            dpsi_dr[:, :, c].index_put_((I, J), -dGi[:, c], accumulate=True)
            dpsi_dr[:, :, c].index_put_((J, I), dGj[:, c], accumulate=True)

        # Diagonal: dpsi_dr[i,i] = dpsitr[i]
        dpsitr = torch.zeros(N, 3, dtype=torch.float64, device=self.device)
        dpsitr.index_add_(0, I, dGi)
        dpsitr.index_add_(0, J, -dGj)
        idx = torch.arange(N, device=self.device)
        dpsi_dr[idx, idx] = dpsitr

        return psi, dpsi_dr

    def _compute_born_radii(self):
        """Born radii via OBC correction (vectorised over atoms)."""
        psi, dpsi_dr = self._compute_psi()

        svdw = self.vdw_rad - self.born_offset
        s1 = 1.0 / svdw
        v1 = 1.0 / self.vdw_rad
        s2 = 0.5 * svdw

        br = psi * s2
        arg2 = br * (self.obc[2] * br - self.obc[1])
        arg = br * (self.obc[0] + arg2)
        arg2_deriv = 2.0 * arg2 + self.obc[0] + self.obc[2] * br * br

        th = torch.tanh(arg)
        ch = torch.cosh(arg)

        born_rad = self.born_scale / (s1 - v1 * th)
        dpsi_chain = s2 * v1 * arg2_deriv / (ch * (s1 - v1 * th)) ** 2
        dpsi_chain = self.born_scale * dpsi_chain

        # dbrdr[i,k,c] = dpsi_chain[i] * dpsi_dr[i,k,c]
        dbrdr = dpsi_chain[:, None, None] * dpsi_dr

        self.born_rad = born_rad
        self.dbrdr = dbrdr

    # -------------------------------------------------------------------
    # Born matrix (Still kernel + ALPB correction) -- vectorised
    # -------------------------------------------------------------------
    def _compute_born_matrix(self):
        N = self.nAtom
        a = self.born_rad
        coords = self.coords
        ke = KE

        # Diagonal
        born_mat = torch.diag(self.keps * ke / a)

        # Off-diagonal: upper triangle
        I, J = torch.triu_indices(N, N, offset=1, device=self.device)  # noqa: E741
        vec = coords[I] - coords[J]
        r2 = (vec * vec).sum(dim=1)
        aa = a[I] * a[J]
        dd = 0.25 * r2 / aa
        expd = torch.exp(-dd)
        fgb = torch.sqrt(r2 + aa * expd)
        bij = self.keps * ke / fgb

        born_mat[I, J] = bij
        born_mat[J, I] = bij

        # ALPB correction
        if self.alpbet > 0.0:
            aDet_bohr = self._compute_aDet()
            alpb_corr = self.keps * self.alpbet * HA_TO_EV / aDet_bohr
            born_mat = born_mat + alpb_corr

        self.born_mat = born_mat

    # -------------------------------------------------------------------
    # SASA  (one atom loop retained -- diagonal exclusion + product chain)
    # -------------------------------------------------------------------
    def _compute_sasa(self):
        """SASA via 230-point Lebedev grid + smooth switching."""
        N = self.nAtom
        coords = self.coords
        vdw = self.vdw_rad
        w = self.smoothing_w
        probe = self.probe_radius

        ang_pts, ang_wts = _lebedev_230()
        ang_pts = ang_pts.to(self.device)  # (230,3)
        ang_wts = ang_wts.to(self.device)  # (230,)

        rad = vdw + probe  # (N,)

        ah1 = 0.5
        ah3 = 3.0 / (4.0 * w)
        ah5 = -1.0 / (4.0 * w**3)

        sasa = torch.zeros(N, dtype=torch.float64, device=self.device)
        dsasa_dr = torch.zeros(N, N, 3, dtype=torch.float64, device=self.device)

        for i in range(N):
            ri = rad[i]
            rm_i = ri - w
            rp_i = ri + w
            wrp = (
                0.25 / w + 3.0 * ah5 * (0.2 * rp_i**2 - 0.5 * rp_i * ri + ri**2 / 3.0)
            ) * rp_i**3 - (
                0.25 / w + 3.0 * ah5 * (0.2 * rm_i**2 - 0.5 * rm_i * ri + ri**2 / 3.0)
            ) * rm_i**3

            # Grid points for atom i: (n_ang, 3)
            gp = coords[i].unsqueeze(0) + ri * ang_pts

            # Distances from all grid points to all OTHER atoms
            diff = gp.unsqueeze(1) - coords.unsqueeze(0)  # (230, N, 3)
            dist_gp = diff.norm(dim=2)  # (230, N)

            rj = rad.unsqueeze(0)  # (1, N)
            rm_j = rj - w
            rp_j = rj + w

            # h switching function for all (grid_pt, atom) pairs
            dr = dist_gp - rj
            h_val = ah1 + ah3 * dr + ah5 * dr**3
            dh_val = ah3 + 3.0 * ah5 * dr**2

            # Clamp: h=0 if dist <= rm_j;  h=1 if dist >= rp_j
            mask_zero = dist_gp <= rm_j
            mask_one = dist_gp >= rp_j
            h_val = torch.where(mask_zero, torch.zeros_like(h_val), h_val)
            h_val = torch.where(mask_one, torch.ones_like(h_val), h_val)
            dh_val = torch.where(mask_zero | mask_one, torch.zeros_like(dh_val), dh_val)

            # Exclude self (j == i)
            h_val[:, i] = 1.0
            dh_val[:, i] = 0.0

            # Product over neighbours for each grid point
            prod_h = h_val.prod(dim=1)  # (230,)

            sasa[i] = wrp * (ang_wts * prod_h).sum()

            # Gradient: d(prod)/d(h_j) = prod/h_j * dh_j
            nonzero = prod_h.abs() > 1e-30  # (230,)
            if nonzero.any():
                ph = prod_h[nonzero].unsqueeze(1)  # (K,1)
                hv = h_val[nonzero]  # (K,N)
                dhv = dh_val[nonzero]  # (K,N)
                dv = dist_gp[nonzero]  # (K,N)
                df = diff[nonzero]  # (K,N,3)
                wk = ang_wts[nonzero]  # (K,)

                safe_h = hv.clamp(min=1e-30)
                factor = (
                    wrp * wk.unsqueeze(1) * (ph / safe_h) * dhv / dv.clamp(min=1e-30)
                )
                grad_all = -factor.unsqueeze(2) * df  # (K, N, 3)
                grad_sum = grad_all.sum(dim=0)  # (N, 3)

                dsasa_dr[i] += grad_sum
                dsasa_dr[i, i] -= grad_sum.sum(dim=0)

        self.sasa = sasa
        self.dsasa_dr = dsasa_dr

    # -------------------------------------------------------------------
    # Shifts, energies, gradients
    # -------------------------------------------------------------------
    def get_shifts(self, q):
        """Born shift for SCC: shift = 2*sasa*hBondStrength*q + bornMat @ q."""
        sasa_rad2 = (self.vdw_rad + self.probe_radius) ** 2
        hbond_strength = self.hbond_str / sasa_rad2
        shift = 2.0 * self.sasa * hbond_strength * q
        shift = shift + self.born_mat @ q
        return shift

    def get_energies(self, q):
        """Returns e_gb, e_sasa (eV)."""
        shift = self.get_shifts(q)
        e_gb = 0.5 * torch.sum(shift * q)
        e_sasa = torch.sum(self.surf_tension * self.sasa) + self.free_energy_shift
        return e_gb, e_sasa

    # -------------------------------------------------------------------
    # Differentiable energy  (backpropable through coordinates)
    # -------------------------------------------------------------------
    def get_energy_differentiable(self, coords, q):
        """
        Total solvation energy (eV), differentiable w.r.t. *coords*.

        Recomputes all geometry-dependent quantities (Born radii, Born
        matrix, SASA) using autograd-compatible operations so that
        ``e_solv.backward()`` propagates gradients to *coords*.

        Parameters
        ----------
        coords : (N, 3) Tensor — Cartesian coordinates in Angstrom.
        q      : (N,) Tensor   — Mulliken charges (may or may not carry grad).

        Returns
        -------
        e_solv : scalar Tensor = e_gb + e_sasa  (eV), with grad_fn.
        """
        N = self.N_atoms
        dev = self.device
        dt = torch.float64

        # ── 1) pairwise distances ────────────────────────────────────
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
        r2 = (diff * diff).sum(dim=-1)  # (N, N)
        diag_inf = torch.eye(N, device=dev, dtype=dt) * 1e30
        r = torch.sqrt(r2 + diag_inf)  # (N, N)

        vdw = self.vdw_rad  # (N,)
        rho = self.rho  # (N,)

        # ── 2) psi integrals (branchless via torch.where) ────────────
        #  For pair (i,j):
        #   - "standard" integral:  g_std(r_ij, rho_j)
        #   - "overlap"  integral:  g_ovl(r_ij, rho_j, vdw_i)
        #  Select overlap when r_ij < vdw_i + rho_j
        #  Additionally: g_ovl is valid only when r_ij + rho_j > vdw_i

        rho_j = rho.unsqueeze(0).expand(N, N)  # (N, N)
        rho_i = rho.unsqueeze(1).expand(N, N)  # (N, N)
        vdw_i = vdw.unsqueeze(1).expand(N, N)  # (N, N)
        vdw_j = vdw.unsqueeze(0).expand(N, N)  # (N, N)

        # Standard integral  (safe — uses |am|, |ap| > 0 for off-diag)
        g_std_ij = _std_g(r, rho_j)  # g(i←j)
        g_std_ji = _std_g(r, rho_i)  # g(j←i)

        # Overlap integral  (safe denominators via clamped r)
        g_ovl_ij = _ovl_g(r, rho_j, vdw_i)  # g(i←j) overlap
        g_ovl_ji = _ovl_g(r, rho_i, vdw_j)  # g(j←i) overlap

        # Select which integral to use for each pair
        overlap_ij = r < (vdw_i + rho_j)  # i overlaps j's rho
        overlap_ji = r < (vdw_j + rho_i)  # j overlaps i's rho
        ovl_valid_ij = r + rho_j > vdw_i  # overlap integral valid
        ovl_valid_ji = r + rho_i > vdw_j

        # g_ij: contribution to psi[i] from atom j
        g_ij = torch.where(
            overlap_ij & ovl_valid_ij,
            g_ovl_ij,
            torch.where(overlap_ij, torch.zeros_like(r), g_std_ij),
        )
        # g_ji: contribution to psi[j] from atom i
        g_ji = torch.where(
            overlap_ji & ovl_valid_ji,
            g_ovl_ji,
            torch.where(overlap_ji, torch.zeros_like(r), g_std_ji),
        )

        # Zero diagonal
        off_diag = 1.0 - torch.eye(N, device=dev, dtype=dt)
        g_ij = g_ij * off_diag
        g_ji = g_ji * off_diag

        # psi[i] = Σ_j g_ij  (but g_ij already has the right (i,j) layout)
        # Note: g_ij[i,j] = contribution to atom i from atom j
        # g_ji[i,j] = contribution to atom j from atom i  → need to transpose
        psi = g_ij.sum(dim=1) + g_ji.T.sum(dim=1)
        # Actually g_ij already computes psi[i] from all j (row sum).
        # g_ji[i,j] means "overlap at j from i" so psi[j] += g_ji[i,j] → psi += g_ji.sum(dim=0)
        # But we're double-counting: g_ij covers ALL (i,j) pairs, not just upper triangle.
        # Let's redo: psi[i] = Σ_{j≠i} g(r_ij, rho_j, vdw_i) where g selects std/ovl.
        # That's just g_ij summed over j:
        psi = g_ij.sum(dim=1)

        # ── 3) OBC-corrected Born radii ─────────────────────────────
        svdw = vdw - self.born_offset
        s2 = 0.5 * svdw
        s1 = 1.0 / svdw
        v1 = 1.0 / vdw

        br = psi * s2
        arg2 = br * (self.obc[2] * br - self.obc[1])
        arg = br * (self.obc[0] + arg2)
        th = torch.tanh(arg)
        born_rad = self.born_scale / (s1 - v1 * th)  # (N,)

        # ── 4) Born matrix (Still kernel + ALPB) ────────────────────
        ke = KE
        # Diagonal: keps * ke / a_i
        born_diag = self.keps * ke / born_rad  # (N,)

        # Off-diagonal
        aa = born_rad.unsqueeze(0) * born_rad.unsqueeze(1)  # (N, N)
        dd = 0.25 * r2 / (aa + diag_inf)
        expd = torch.exp(-dd) * off_diag
        fgb = torch.sqrt(r2 * off_diag + aa * expd + diag_inf)
        bij = self.keps * ke / fgb * off_diag  # (N, N)

        # GB energy: 0.5 * Σ_ij born_mat[i,j] * q[i] * q[j]
        e_gb_offdiag = 0.5 * (bij * q.unsqueeze(0) * q.unsqueeze(1)).sum()
        e_gb_diag = 0.5 * (born_diag * q * q).sum()

        # ALPB correction
        if self.alpbet > 0.0:
            # aDet (differentiable)
            coords_bohr = coords * ANG_TO_BOHR
            rad_bohr = vdw * ANG_TO_BOHR
            rad3 = rad_bohr**3
            totRad3 = rad3.sum()
            center = (coords_bohr * rad3.unsqueeze(1)).sum(dim=0) / totRad3
            vec = coords_bohr - center.unsqueeze(0)
            rr = (vec * vec).sum(dim=1)
            tof_r2 = 0.4 * rad_bohr**2
            eye3 = torch.eye(3, dtype=dt, device=dev)
            diag_part = ((rr + tof_r2) * rad3).sum() * eye3
            outer_part = torch.einsum("i,ij,ik->jk", rad3, vec, vec)
            inertia = diag_part - outer_part
            det_val = torch.det(inertia)
            aDet = torch.sqrt(det_val ** (1.0 / 3.0) / (2.0 * totRad3) * 5.0)
            alpb_corr_per_pair = self.keps * self.alpbet * HA_TO_EV / aDet
            e_alpb = 0.5 * alpb_corr_per_pair * q.sum() ** 2
        else:
            e_alpb = torch.tensor(0.0, dtype=dt, device=dev)

        e_gb = e_gb_offdiag + e_gb_diag + e_alpb

        # ── 5) SASA energy (differentiable) ──────────────────────────
        w = self.smoothing_w
        probe = self.probe_radius
        rad = vdw + probe  # (N,)

        ang_pts = self._ang_pts  # (G, 3)
        ang_wts = self._ang_wts  # (G,)

        ah1 = 0.5
        ah3 = 3.0 / (4.0 * w)
        ah5 = -1.0 / (4.0 * w**3)

        # wrp per atom (geometry-independent, pre-computed)
        wrp = self._wrp  # (N,)

        # Grid points for ALL atoms: (N, G, 3)
        gp = coords.unsqueeze(1) + rad.unsqueeze(1).unsqueeze(2) * ang_pts.unsqueeze(0)

        # Distances from each grid point to each atom: (N_owner, G, N_other, 3)
        diff_gp = gp.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)  # (N, G, N, 3)
        dist_gp = torch.sqrt(
            (diff_gp * diff_gp).sum(dim=-1).clamp(min=1e-30)
        )  # (N, G, N)

        # Switching function h(r)
        rj = rad.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        rm_j = rj - w
        rp_j = rj + w
        dr = dist_gp - rj
        h_val = ah1 + ah3 * dr + ah5 * dr**3

        # Clamp: h=0 if dist <= rm_j;  h=1 if dist >= rp_j
        h_val = torch.where(dist_gp <= rm_j, torch.zeros_like(h_val), h_val)
        h_val = torch.where(dist_gp >= rp_j, torch.ones_like(h_val), h_val)

        # Exclude self: h[i, :, i] = 1 for all grid points
        self_mask = torch.eye(N, device=dev, dtype=dt).unsqueeze(1)  # (N, 1, N)
        h_val = h_val * (1.0 - self_mask) + self_mask

        # Product over all atoms for each grid point
        prod_h = h_val.prod(dim=2)  # (N, G)

        # SASA per atom
        sasa = wrp.unsqueeze(1) * ang_wts.unsqueeze(0) * prod_h  # (N, G)
        sasa_per_atom = sasa.sum(dim=1)  # (N,)

        # SASA energy = Σ_i surfTension[i] * sasa[i]  +  hBond term
        sasa_rad2 = rad**2
        hbond_strength = self.hbond_str / sasa_rad2
        e_sasa = (
            torch.sum(self.surf_tension * sasa_per_atom)
            + torch.sum(hbond_strength * sasa_per_atom * q * q)
            + self.free_energy_shift
        )

        return e_gb + e_sasa

    # -------------------------------------------------------------------
    # Shadow (XL-BOMD) variants  —  q = SCF charges, n = extrapolated
    # -------------------------------------------------------------------
    def get_shadow_shifts(self, n):
        """Born shift for shadow Hamiltonian (built from extrapolated charges n).

        Identical to get_shifts(n) — the Hamiltonian sees n, not q.
        """
        return self.get_shifts(n)

    def get_shadow_energies(self, q, n):
        """Shadow solvation energies: e_gb uses (2q-n)·shift(n)/2.

        Returns e_gb, e_sasa (eV).
        """
        shift = self.get_shifts(n)
        e_gb = 0.5 * torch.sum((2.0 * q - n) * shift)
        e_sasa = torch.sum(self.surf_tension * self.sasa) + self.free_energy_shift
        return e_gb, e_sasa

    def get_shadow_sasa_gradients(self, q, n):
        """Shadow SASA force (surfTension + hBond). Returns (N,3) eV/A.

        The hBond energy has charge dependence q^2 -> (2q-n)*n in the shadow.
        """
        sasa_rad2 = (self.vdw_rad + self.probe_radius) ** 2
        weight = self.surf_tension + self.hbond_str / sasa_rad2 * (2.0 * q - n) * n
        return -torch.einsum("i,ijk->jk", weight, self.dsasa_dr)

    def get_shadow_born_gradients(self, q, n):
        """Shadow Born gradient: q_i*q_j -> (2q_i-n_i)*n_j symmetrised.

        Returns (N,3) eV/A.
        """
        N = self.nAtom
        coords = self.coords
        a = self.born_rad
        ke = KE
        p = 2.0 * q - n  # shadow prefactor

        # Diagonal: dE/d(born_rad) diagonal — 0.5*keps*ke*q^2/a^2 -> 0.5*keps*ke*p*n/a^2
        dEdbr = -0.5 * self.keps * ke * p * n / a**2

        I, J = torch.triu_indices(N, N, offset=1, device=self.device)  # noqa: E741
        vec = coords[I] - coords[J]
        r2 = (vec * vec).sum(dim=1)
        # Shadow: q_i*q_j -> 0.5*((2q_i-n_i)*n_j + n_i*(2q_j-n_j))
        pn_sym = 0.5 * (p[I] * n[J] + n[I] * p[J])
        aa = a[I] * a[J]
        dd = 0.25 * r2 / aa
        expd = torch.exp(-dd)
        fgb2 = r2 + aa * expd
        dfgb = torch.sqrt(fgb2)
        dfgb3 = 1.0 / (dfgb * fgb2)

        ap_coeff = (1.0 - 0.25 * expd) * dfgb3 * self.keps * ke
        dG = (ap_coeff * pn_sym).unsqueeze(1) * vec

        grad = torch.zeros(N, 3, dtype=torch.float64, device=self.device)
        grad.index_add_(0, I, -dG)
        grad.index_add_(0, J, dG)

        bp_coeff = -0.5 * expd * (1.0 + dd) * dfgb3 * self.keps * ke
        dEdbr.index_add_(0, I, a[J] * bp_coeff * pn_sym)
        dEdbr.index_add_(0, J, a[I] * bp_coeff * pn_sym)

        grad += torch.einsum("i,ikc->kc", dEdbr, self.dbrdr)

        if self.alpbet > 0.0:
            self._add_shadow_aDet_gradient(q, n, grad)

        return -grad

    def _add_shadow_aDet_gradient(self, q, n, gradient):
        """Shadow ALPB aDet gradient: q_total^2 -> (2*q_total - n_total)*n_total."""
        coords_bohr = self.coords * ANG_TO_BOHR
        rad_bohr = self.vdw_rad * ANG_TO_BOHR

        qtotal = q.sum().item()
        ntotal = n.sum().item()
        shadow_q2 = (2.0 * qtotal - ntotal) * ntotal

        rad3 = rad_bohr**3
        totRad3 = rad3.sum()
        center = (coords_bohr * rad3.unsqueeze(1)).sum(dim=0) / totRad3
        vec = coords_bohr - center.unsqueeze(0)

        rr = (vec * vec).sum(dim=1)
        tof_r2 = 0.4 * rad_bohr**2
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)

        diag_part = ((rr + tof_r2) * rad3).sum() * eye3
        outer_part = torch.einsum("i,ij,ik->jk", rad3, vec, vec)
        inertia = diag_part - outer_part

        det_val = torch.det(inertia).item()
        aDet = math.sqrt(det_val ** (1.0 / 3.0) / (2.0 * totRad3.item()) * 5.0)

        II = inertia
        aDeriv = torch.zeros(3, 3, dtype=torch.float64, device=self.device)
        aDeriv[0, 0] = II[0, 0] * (II[1, 1] + II[2, 2]) - II[0, 1] ** 2 - II[0, 2] ** 2
        aDeriv[1, 0] = II[0, 1] * II[2, 2] - II[0, 2] * II[1, 2]
        aDeriv[2, 0] = II[0, 2] * II[1, 1] - II[0, 1] * II[2, 1]
        aDeriv[0, 1] = II[0, 1] * II[2, 2] - II[0, 2] * II[1, 2]
        aDeriv[1, 1] = II[1, 1] * (II[0, 0] + II[2, 2]) - II[0, 1] ** 2 - II[1, 2] ** 2
        aDeriv[2, 1] = II[0, 0] * II[1, 2] - II[0, 1] * II[0, 2]
        aDeriv[0, 2] = II[0, 2] * II[1, 1] - II[0, 1] * II[2, 1]
        aDeriv[1, 2] = II[0, 0] * II[1, 2] - II[0, 1] * II[0, 2]
        aDeriv[2, 2] = II[2, 2] * (II[0, 0] + II[1, 1]) - II[0, 2] ** 2 - II[1, 2] ** 2

        scale = 250.0 / (48.0 * totRad3.item() ** 3 * aDet**5)
        energy_factor = -0.5 * self.keps * self.alpbet * shadow_q2 / aDet**2
        ha_bohr_to_ev_ang = HA_TO_EV * ANG_TO_BOHR
        aDeriv = aDeriv * (scale * energy_factor * ha_bohr_to_ev_ang)

        gradient += rad3.unsqueeze(1) * (vec @ aDeriv.T)

    # -------------------------------------------------------------------
    # Standard SCF gradients
    # -------------------------------------------------------------------
    def get_sasa_gradients(self, q):
        """SASA force (surfTension + hBond). Returns (N,3) eV/A."""
        sasa_rad2 = (self.vdw_rad + self.probe_radius) ** 2
        weight = self.surf_tension + self.hbond_str / sasa_rad2 * q**2
        return -torch.einsum("i,ijk->jk", weight, self.dsasa_dr)

    def get_born_gradients(self, q):
        """Born energy force (-dE/dr) w.r.t. coords. Returns (N,3) eV/A."""
        N = self.nAtom
        coords = self.coords
        a = self.born_rad
        ke = KE

        # Diagonal contribution to dE/d(born_rad)
        dEdbr = -0.5 * self.keps * ke * q**2 / a**2

        # Upper-triangle pairs
        I, J = torch.triu_indices(N, N, offset=1, device=self.device)  # noqa: E741
        vec = coords[I] - coords[J]
        r2 = (vec * vec).sum(dim=1)
        qq = q[I] * q[J]
        aa = a[I] * a[J]
        dd = 0.25 * r2 / aa
        expd = torch.exp(-dd)
        fgb2 = r2 + aa * expd
        dfgb = torch.sqrt(fgb2)
        dfgb3 = 1.0 / (dfgb * fgb2)

        # Direct gradient  dE/dr
        ap_coeff = (1.0 - 0.25 * expd) * dfgb3 * self.keps * ke
        dG = (ap_coeff * qq).unsqueeze(1) * vec  # (P,3)

        grad = torch.zeros(N, 3, dtype=torch.float64, device=self.device)
        grad.index_add_(0, I, -dG)
        grad.index_add_(0, J, dG)

        # Chain rule through born radii
        bp_coeff = -0.5 * expd * (1.0 + dd) * dfgb3 * self.keps * ke
        dEdbr.index_add_(0, I, a[J] * bp_coeff * qq)
        dEdbr.index_add_(0, J, a[I] * bp_coeff * qq)

        # grad[k] += sum_i dEdbr[i] * dbrdr[i,k]
        grad += torch.einsum("i,ikc->kc", dEdbr, self.dbrdr)

        if self.alpbet > 0.0:
            self._add_aDet_gradient(q, grad)

        return -grad

    def _add_aDet_gradient(self, q, gradient):
        """ALPB aDet gradient (born.F90::getADetDeriv) -- vectorised."""
        coords_bohr = self.coords * ANG_TO_BOHR
        rad_bohr = self.vdw_rad * ANG_TO_BOHR

        qtotal = q.sum().item()
        rad3 = rad_bohr**3
        totRad3 = rad3.sum()
        center = (coords_bohr * rad3.unsqueeze(1)).sum(dim=0) / totRad3
        vec = coords_bohr - center.unsqueeze(0)

        rr = (vec * vec).sum(dim=1)
        tof_r2 = 0.4 * rad_bohr**2
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)

        diag_part = ((rr + tof_r2) * rad3).sum() * eye3
        outer_part = torch.einsum("i,ij,ik->jk", rad3, vec, vec)
        inertia = diag_part - outer_part

        det_val = torch.det(inertia).item()
        aDet = math.sqrt(det_val ** (1.0 / 3.0) / (2.0 * totRad3.item()) * 5.0)

        # Cofactor-derived matrix
        II = inertia
        aDeriv = torch.zeros(3, 3, dtype=torch.float64, device=self.device)
        aDeriv[0, 0] = II[0, 0] * (II[1, 1] + II[2, 2]) - II[0, 1] ** 2 - II[0, 2] ** 2
        aDeriv[1, 0] = II[0, 1] * II[2, 2] - II[0, 2] * II[1, 2]
        aDeriv[2, 0] = II[0, 2] * II[1, 1] - II[0, 1] * II[2, 1]
        aDeriv[0, 1] = II[0, 1] * II[2, 2] - II[0, 2] * II[1, 2]
        aDeriv[1, 1] = II[1, 1] * (II[0, 0] + II[2, 2]) - II[0, 1] ** 2 - II[1, 2] ** 2
        aDeriv[2, 1] = II[0, 0] * II[1, 2] - II[0, 1] * II[0, 2]
        aDeriv[0, 2] = II[0, 2] * II[1, 1] - II[0, 1] * II[2, 1]
        aDeriv[1, 2] = II[0, 0] * II[1, 2] - II[0, 1] * II[0, 2]
        aDeriv[2, 2] = II[2, 2] * (II[0, 0] + II[1, 1]) - II[0, 2] ** 2 - II[1, 2] ** 2

        scale = 250.0 / (48.0 * totRad3.item() ** 3 * aDet**5)
        energy_factor = -0.5 * self.keps * self.alpbet * qtotal**2 / aDet**2
        ha_bohr_to_ev_ang = HA_TO_EV * ANG_TO_BOHR
        aDeriv = aDeriv * (scale * energy_factor * ha_bohr_to_ev_ang)

        # gradient[i] += rad3[i] * (aDeriv @ vec[i])
        gradient += rad3.unsqueeze(1) * (vec @ aDeriv.T)


# ---------------------------------------------------------------------------
# Convenience function for ESDriver
# ---------------------------------------------------------------------------
def create_gbsa(
    structure, device, solvent="water", param_file=None, solvation_model="alpb"
):
    """Create a GBSA/ALPB object from a DFTorch Structure.

    Parameters
    ----------
    structure : Structure
        DFTorch Structure object.
    device : torch.device
        Device for tensors.
    solvent : str
        Solvent name (used for auto-discovery of parameter files).
    param_file : str or None
        Explicit path to parameter file.  If None, auto-discovered from
        the SK directory using *solvation_model* and *solvent*.
    solvation_model : str
        ``"alpb"`` (default) — Analytical Linearized Poisson-Boltzmann.
        ``"gbsa"`` — plain Generalized Born + SASA (no ALPB correction).
        The choice determines (a) which parameter file is loaded when
        *param_file* is None, and (b) whether the ALPB correction term
        is applied.  The two models use **different** fitted parameters
        and must not be mixed.
    """
    solvation_model = solvation_model.lower()
    if solvation_model not in ("alpb", "gbsa"):
        raise ValueError(
            f"solvation_model must be 'alpb' or 'gbsa', got '{solvation_model}'"
        )
    use_alpb = solvation_model == "alpb"

    coords = torch.stack([structure.RX, structure.RY, structure.RZ], dim=-1)
    species = structure.TYPE

    if param_file is None:
        if hasattr(structure, "SK_path"):
            sk_dir = structure.SK_path
        else:
            sk_dir = None
        if sk_dir is not None:
            # Search for the matching parameter file (alpb or gbsa)
            candidate = os.path.join(
                sk_dir,
                "solvation",
                f"param_{solvation_model}_{solvent.lower()}.txt",
            )
            if os.path.isfile(candidate):
                param_file = candidate

    if param_file is None:
        raise ValueError(
            f"Could not find {solvation_model.upper()} parameter file for "
            f"solvent '{solvent}'. Please provide param_file explicitly via "
            f"dftorch_params['solvent_param_file']."
        )

    return GBSA(coords, species, device, param_file=param_file, alpb=use_alpb)
