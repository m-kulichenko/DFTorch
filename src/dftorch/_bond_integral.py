from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch

from ._tools import ordered_pairs_from_TYPE

symbol_to_number: Final[dict[str, int]] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}

_CHANNELS: Final[list[str]] = [
    "Hdd0",
    "Hdd1",
    "Hdd2",
    "Hpd0",
    "Hpd1",
    "Hpp0",
    "Hpp1",
    "Hsd0",
    "Hsp0",
    "Hss0",
    "Sdd0",
    "Sdd1",
    "Sdd2",
    "Spd0",
    "Spd1",
    "Spp0",
    "Spp1",
    "Ssd0",
    "Ssp0",
    "Sss0",
]


def load_bond_integral_parameters(
    neighbor_I: torch.Tensor,
    neighbor_J: torch.Tensor,
    TYPE: torch.Tensor,
    fname: str,
) -> torch.Tensor:
    """Load bond-integral parameters for neighbor pairs from a CSV-like table.

    The input file is expected to be comma-separated with a header row containing
    `id1,id2,...`. Each subsequent row should contain:

    - `id1` (int): type id of the first element
    - `id2` (int): type id of the second element
    - 14 floats: parameter vector for the pair `(id1, id2)`

    A dense lookup tensor `q` of shape `(m+1, m+1, 14)` is constructed, where
    `m = max(TYPE)`. The output parameters for each neighbor pair are gathered as
    `q[type_I, type_J]`.

    Parameters
    ----------
    neighbor_I:
        1D integer tensor with atom indices for the first atom in each pair.
    neighbor_J:
        1D integer tensor with atom indices for the second atom in each pair.
    TYPE:
        1D integer tensor mapping each atom index to a type id.
    fname:
        Path to the parameter file.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(len(neighbor_I), 14)` containing the parameter vector for
        each neighbor pair.

    Notes
    -----
    This function intentionally preserves the original behavior:
    - `m = max(TYPE)` uses Python's `max` over the tensor.
    - The lookup tensor is allocated on `neighbor_I.device`.
    - Only rows whose `id1` and `id2` appear in `TYPE` are loaded.
    """

    type_I = TYPE[neighbor_I]
    type_J = TYPE[neighbor_J]
    m = max(TYPE)
    q = torch.zeros((m + 1, m + 1, 14), device=neighbor_I.device)
    import os

    f = open(os.path.abspath(fname))
    TYPE_set = set(TYPE.cpu().numpy())
    for l in f:  # noqa: E741
        t = l.strip().replace(" ", "").split(",")
        if t[0] == "id1":
            continue

        id1 = int(t[0])
        id2 = int(t[1])
        if id1 in TYPE_set and id2 in TYPE_set:
            q[id1, id2] = torch.tensor(list(map(float, t[2:16])), dtype=q.dtype)

    fss_sigma = q[type_I, type_J]
    f.close()
    return fss_sigma


def bond_integral_vectorized(dR: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Compute bond integrals for many pairs in a vectorized piecewise form.

    Parameters
    ----------
    dR:
        Tensor of shape `(N,)` with interatomic distances.
    f:
        Tensor of shape `(N, 14)` with bond-integral parameters per pair.

        Layout (by column index):
        - `f[:,0]`: prefactor
        - `f[:,5]`: R0 (shift for region-1 polynomial)
        - `f[:,6]`: R1 (region-1/2 boundary)
        - `f[:,7]`: R2 (cutoff boundary)
        - `f[:,1:5]`: coefficients for the quartic polynomial (region 1)
        - `f[:,8:14]`: coefficients for the quintic polynomial (region 2)

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N,)` with bond-integral values.
    """
    # Masks
    region1 = (dR > 1e-12) & (dR <= f[:, 6])
    region2 = (dR > f[:, 6]) & (dR < f[:, 7])
    # region3 = dR >= f[:, 7]

    # Output tensor
    X = torch.zeros_like(dR, dtype=f.dtype)

    # Region 1: Polynomial + exp
    RMOD = dR[region1] - f[region1, 5]
    POLYNOM = RMOD * (
        f[region1, 1]
        + RMOD * (f[region1, 2] + RMOD * (f[region1, 3] + f[region1, 4] * RMOD))
    )
    X[region1] = torch.exp(POLYNOM)

    # Region 2: Quintic polynomial
    RMINUSR1 = dR[region2] - f[region2, 6]

    X[region2] = f[region2, 8] + RMINUSR1 * (
        f[region2, 9]
        + RMINUSR1
        * (
            f[region2, 10]
            + RMINUSR1
            * (f[region2, 11] + RMINUSR1 * (f[region2, 12] + RMINUSR1 * f[region2, 13]))
        )
    )
    # Region 3 stays zero
    return f[:, 0] * X


def bond_integral_with_grad_vectorized(
    dR: torch.Tensor, f: torch.Tensor
) -> torch.Tensor:
    """Compute radial derivative of the bond integral (dX/dr), vectorized.

    This matches :func:`bond_integral_vectorized` but returns the derivative
    with respect to `dR`, and applies the same prefactor `f[:,0]`.

    Parameters
    ----------
    dR:
        Tensor of shape `(N,)` with interatomic distances.
    f:
        Tensor of shape `(N, 14)` with bond-integral parameters per pair.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(N,)` with d(bond_integral)/dr values.
    """
    # Masks
    region1 = (dR > 1e-12) & (dR <= f[:, 6])
    region2 = (dR > f[:, 6]) & (dR < f[:, 7])
    # region3 = dR >= f[:, 7]

    # Output tensor
    X = torch.zeros_like(dR, dtype=f.dtype)
    dSx = torch.zeros_like(dR, dtype=f.dtype)

    # Region 1: Polynomial + exp
    RMOD = dR[region1] - f[region1, 5]
    POLYNOM = RMOD * (
        f[region1, 1]
        + RMOD * (f[region1, 2] + RMOD * (f[region1, 3] + f[region1, 4] * RMOD))
    )

    X[region1] = torch.exp(POLYNOM)

    dSx[region1] = X[region1] * (
        f[region1, 1]
        + 2 * RMOD * f[region1, 2]
        + 3 * (RMOD**2) * f[region1, 3]
        + 4 * (RMOD**3) * f[region1, 4]
    )

    # Region 2: Quintic polynomial
    RMINUSR1 = dR[region2] - f[region2, 6]

    X[region2] = f[region2, 8] + RMINUSR1 * (
        f[region2, 9]
        + RMINUSR1
        * (
            f[region2, 10]
            + RMINUSR1
            * (f[region2, 11] + RMINUSR1 * (f[region2, 12] + RMINUSR1 * f[region2, 13]))
        )
    )

    dSx[region2] = (
        f[region2, 9]
        + 2 * RMINUSR1 * f[region2, 10]
        + 3 * (RMINUSR1**2) * f[region2, 11]
        + 4 * (RMINUSR1**3) * f[region2, 12]
        + 5 * (RMINUSR1**4) * f[region2, 13]
    )

    # Region 3 stays zero
    return f[:, 0] * dSx


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Expand Fortran-style repetition tokens.

    Examples
    --------
    `"3*0.0"` becomes `"0.0", "0.0", "0.0"`.

    Parameters
    ----------
    tokens:
        List of string tokens.

    Returns
    -------
    list[str]
        Expanded token list.
    """
    out = []
    for t in tokens:
        if "*" in t:
            num, val = t.split("*")
            out.extend([val] * int(num))
        else:
            out.append(t)
    return out


def read_skf_table(
    path: str,
    N_ORB: torch.Tensor,
    MAX_ANG: torch.Tensor,
    MAX_ANG_OCC: torch.Tensor,
    TORE: torch.Tensor,
    N_S: torch.Tensor,
    N_P: torch.Tensor,
    N_D: torch.Tensor,
    ES: torch.Tensor,
    EP: torch.Tensor,
    ED: torch.Tensor,
    US: torch.Tensor,
    UP: torch.Tensor,
    UD: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Read a DFTB+ `.skf` file and extract integral channels + repulsion spline data.

    Parameters
    ----------
    path:
        Path to the `.skf` file. Filename should resemble `ElemA-ElemB.skf`.
    N_ORB, MAX_ANG, MAX_ANG_OCC, TORE:
        Integer tensors updated in-place for homonuclear files.
    N_S, N_P, N_D:
        Integer tensors updated in-place for homonuclear files.
    ES, EP, ED, US, UP, UD:
        Float tensors updated in-place for homonuclear files.

    Returns
    -------
    tuple
        `(R, channels, R_rep, rep_splines)` where:
        - `R`: `(1001,)` padded orbital radial grid in Angstrom
        - `channels`: dict[str, Tensor] channel_name -> `(npts, )` values in eV
        - `R_rep`: repulsive radial grid in Angstrom
        - `rep_splines`: spline polynomial coefficients
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    data_lines = [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]

    # First line: step and number of points
    first = data_lines[0].replace(",", " ").split()
    step, npts_read = float(first[0]), int(first[1])
    npts_pad = 1001
    if "mio" in path:
        npts_read = 519

    # Decide where the table starts
    base = os.path.basename(path)  # 'C-Ni.skf'
    name, _ext = os.path.splitext(base)  # ('C-Ni', '.skf')
    elemA, elemB = name.split("-", 1)  # split only on first '-'
    homonuclear = elemA == elemB
    start_idx = 3 if homonuclear else 2

    if homonuclear:
        tokens = data_lines[1].replace(",", " ").split()
        (
            Ed,
            Ep,
            Es,
            SPE,  # exception for unused var # noqa: F841
            Ud,
            Up,
            Us,
            fd,
            fp,
            fs,
        ) = (
            float(tokens[0]),
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            float(tokens[4]),
            float(tokens[5]),
            float(tokens[6]),
            float(tokens[7]),
            float(tokens[8]),
            float(tokens[9]),
        )
        el_num = symbol_to_number[elemA]

        tmp = data_lines[40].split()[0]
        N_ORB[el_num] = 1 if "9*" in tmp else (4 if "5*" in tmp else 9)
        # N_ORB[el_num] = 1*(Es != 0) + 3*(Ep != 0) + 5*(Ed != 0)
        MAX_ANG[el_num] = 1 if "9*" in tmp else (2 if "5*" in tmp else 3)
        # MAX_ANG[el_num] = 3 if Ed != 0 else (2 if Ep != 0 else 1)
        MAX_ANG_OCC[el_num] = 3 if fd != 0 else (2 if fp != 0 else 1)
        TORE[el_num] = fs + fp + fd
        N_S[el_num] = fs
        N_P[el_num] = fp
        N_D[el_num] = fd
        ES[el_num] = Es * 27.21138625
        EP[el_num] = Ep * 27.21138625
        ED[el_num] = Ed * 27.21138625
        US[el_num] = Us * 27.21138625
        UP[el_num] = Up * 27.21138625
        UD[el_num] = Ud * 27.21138625

    rows = []
    # print(path)
    for ln in data_lines[start_idx : start_idx + npts_read - 1]:
        tokens = _expand_tokens(ln.replace(",", " ").split())
        if len(tokens) != 20:
            raise ValueError(f"Expected 20 values, got {len(tokens)} in line: {ln}")
        rows.append([float(x) for x in tokens])
    rows.append([float(x) * 0.0 for x in tokens])

    mat = torch.tensor(rows) * 27.21138625  # (npts,20)
    R = torch.arange(1, npts_pad + 1) * step * 0.52917721  # Convert to Angstrom
    channels = {ch: mat[:, j] for j, ch in enumerate(_CHANNELS)}

    for spline_start, line in enumerate(data_lines):
        if spline_start == len(data_lines) - 1:  # skip blanks and comment lines
            print("Spline not found")
        if "Spline" in line:  # use s.casefold()=="spline" for case-insensitive
            break

    ### Do repulsion splines
    first = data_lines[spline_start + 1].replace(",", " ").split()
    npts = int(first[0])
    # close_exp = torch.tensor([float(x) for x in data_lines[spline_start+2].replace(',', ' ').split()])
    rows = []
    rows_R = []
    for ln in data_lines[spline_start + 3 : spline_start + 3 + npts - 1]:
        tokens = _expand_tokens(ln.replace(",", " ").split())
        if len(tokens) != 6:
            raise ValueError(f"Expected 6 values, got {len(tokens)} in line: {ln}")
        rows_R.append(float(tokens[0]))
        rows.append(
            [float(x) for x in tokens[2:]] + [0.0] * 2
        )  # pad woth two zeros to satisfy dimensions of the last polyniomial tail

    # add last polyniomial tail
    ln = data_lines[spline_start + 3 + npts - 1]
    tokens = _expand_tokens(ln.replace(",", " ").split())
    rows_R.append(float(tokens[0]))
    rows.append([float(x) for x in tokens[2:]])

    # add zero for r > Rcut
    rows_R.append(float(tokens[1]))
    rows.append([0.0] * 6)

    rep_splines = torch.tensor(rows)  # *27.21138625
    R_rep = torch.tensor(rows_R) * 0.52917721  # Convert to Angstrom

    return R, channels, R_rep, rep_splines


def channels_to_matrix(
    channels: dict[str, torch.Tensor],
    order: list[str] = _CHANNELS,
) -> torch.Tensor:
    """Convert channel dict into a dense matrix of shape `(npts, len(order))`."""
    return torch.stack([channels[ch] for ch in order], dim=1)


def cubic_spline_coeffs(R: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Compute cubic spline coefficients for all channels.

    Parameters
    ----------
    R:
        Knot positions, shape `(n,)`.
    M:
        Values at knots, shape `(n, m)`.

    Returns
    -------
    torch.Tensor
        Coefficients of shape `(n-1, m, 4)` with `[a, b, c, d]` per interval.
    """
    n, m = M.shape
    h = (R[1:] - R[:-1]).unsqueeze(1)  # (n-1,1)

    # Build A system (n×n) shared across channels
    A = torch.zeros((n, n), dtype=R.dtype, device=R.device)
    rhs = torch.zeros((n, m), dtype=R.dtype, device=R.device)

    # Left BC: natural (c0=0)
    A[0, 0] = 1.0

    # Interior equations
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * ((M[i + 1] - M[i]) / h[i] - (M[i] - M[i - 1]) / h[i - 1])

    # Right BC: clamped to zero
    A[-1, -2] = h[-1]
    A[-1, -1] = 2 * h[-1]
    rhs[-1] = 3 * ((0 - M[-1]) / h[-1] - (M[-1] - M[-2]) / h[-1])

    # Solve for c (n×m)
    c = torch.linalg.solve(A, rhs)  # (n,m)

    # Back substitution
    a = M[:-1].clone()  # (n-1,m)
    b = torch.zeros((n - 1, m), dtype=R.dtype, device=R.device)
    d = torch.zeros((n - 1, m), dtype=R.dtype, device=R.device)

    for i in range(n - 1):
        b[i] = (M[i + 1] - M[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    coeffs = torch.stack([a, b, c[:-1], d], dim=2)  # (n-1,m,4)
    return coeffs


def get_skf_tensors(
    TYPE: torch.Tensor, skfpath: str
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load SKF tensors for all unique element pairs present in `TYPE`.

    Returns a tuple matching the historical layout used elsewhere in the project.
    """
    _, _, label_list = ordered_pairs_from_TYPE(TYPE)

    # Allocate padded tensors
    n_pairs = len(label_list)
    npts = 1000  # padded # old 518
    coeffs_tensor = torch.zeros((n_pairs, npts, 20, 4), device=TYPE.device)
    R_tensor = torch.zeros(
        (n_pairs, npts + 1), device=TYPE.device
    )  # not necessarily if all R are the same. Makes sense to use zero padding if not.

    rep_splines_tensor = torch.zeros((n_pairs, 500, 6), device=TYPE.device)  # old 120
    R_rep_tensor = torch.zeros((n_pairs, 500), device=TYPE.device) + 1e8

    N_ORB = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    MAX_ANG = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    MAX_ANG_OCC = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    TORE = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    N_S = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    N_P = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    N_D = torch.zeros(120, dtype=torch.int64, device=TYPE.device)
    ES = torch.zeros(120, device=TYPE.device)
    EP = torch.zeros(120, device=TYPE.device)
    ED = torch.zeros(120, device=TYPE.device)
    US = torch.zeros(120, device=TYPE.device)
    UP = torch.zeros(120, device=TYPE.device)
    UD = torch.zeros(120, device=TYPE.device)

    for i in range(len(label_list)):
        R_orb, channels, R_rep, rep_splines = read_skf_table(
            skfpath + "{}.skf".format(label_list[i]),
            N_ORB,
            MAX_ANG,
            MAX_ANG_OCC,
            TORE,
            N_S,
            N_P,
            N_D,
            ES,
            EP,
            ED,
            US,
            UP,
            UD,
        )

        channels_matrix = channels_to_matrix(channels)
        coeffs = cubic_spline_coeffs(R_orb, channels_matrix)
        R_tensor[i, : len(R_orb)] = R_orb
        coeffs_tensor[i, : len(coeffs)] = coeffs

        R_rep_tensor[i, : len(R_rep)] = R_rep
        rep_splines_tensor[i, : len(rep_splines)] = rep_splines

    R_orb = R_orb.to(device=TYPE.device)

    coeffs_tensor = torch.cat(
        (
            coeffs_tensor,
            torch.zeros(
                coeffs_tensor.shape[0],
                1,
                coeffs_tensor.shape[2],
                coeffs_tensor.shape[3],
                device=TYPE.device,
            ),
        ),
        dim=1,
    )  # pad last channel with zeros
    return (
        R_tensor,
        R_orb,
        coeffs_tensor,
        R_rep_tensor,
        rep_splines_tensor,
        N_ORB,
        MAX_ANG,
        MAX_ANG_OCC,
        TORE,
        N_S,
        N_P,
        N_D,
        ES,
        EP,
        ED,
        US,
        UP,
        UD,
    )
