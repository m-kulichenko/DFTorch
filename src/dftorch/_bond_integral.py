import torch
from pathlib import Path
import os
from ._tools import ordered_pairs_from_TYPE

symbol_to_number = {
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


def LoadBondIntegralParameters(neighbor_I, neighbor_J, TYPE, fname):
    """
    Loads bond integral parameters for given atomic pairs from a file.

    Parameters:
    - neighbor_I (Tensor[int]): Tensor of atom indices representing the first atom in each bonded pair.
    - neighbor_J (Tensor[int]): Tensor of atom indices representing the second atom in each bonded pair.
    - TYPE (Tensor[int]): Tensor mapping atom indices to atomic types (e.g., H=0, C=1, etc.).
    - fname (str): Path to the CSV-like text file containing bond integral parameters.

    File Format:
    - The file must be comma-separated with a header row containing "id1,id2,...".
    - Each subsequent row must contain:
        - id1 (int): Atomic type of the first atom.
        - id2 (int): Atomic type of the second atom.
        - 14 float values: The bond integral parameters associated with the (id1, id2) pair.

    Returns:
    - fss_sigma (Tensor[float]): Tensor of shape `(len(neighbor_I), 14)` containing
      bond integral parameters for each pair of neighboring atoms defined by `neighbor_I` and `neighbor_J`.

    Notes:
    - The function constructs a (m+1, m+1, 14) lookup tensor, where m is the highest atomic type in TYPE.
    - Only pairs present in the provided TYPE set are considered when loading data from the file.
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
    """
    Computes bond integrals in a fully vectorized form using distance-dependent
    piecewise functions defined by bond integral parameters.

    Parameters:
    - dR (Tensor): Tensor of shape (N,) containing interatomic distances.
    - f (Tensor): Tensor of shape (N, 14) where each row contains the 14 bond
      integral parameters for a corresponding pair of atoms.

    Returns:
    - Tensor of shape (N,): The computed bond integral values for each distance in dR.

    Description:
    The function evaluates the bond integral between atomic pairs using three
    regions defined by the parameter thresholds `f[:,6]` (R1) and `f[:,7]` (R2):

    Region 1 (`dR <= f[:,6]`):
        The bond integral is computed using an exponential of a quartic polynomial:
        X = exp((dR - f[:,5]) * (f[:,1] + ... + f[:,4]*(dR - f[:,5])^3))

    Region 2 (`f[:,6] < dR < f[:,7]`):
        A smooth quintic polynomial interpolation is used.

    Region 3 (`dR >= f[:,7]`):
        The bond integral is set to zero (no interaction beyond this range).

    Finally, the bond integral is scaled by `f[:,0]` as a prefactor.
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
    """
    Computes derivatives of bond integrals (dX/dr) in a fully vectorized form using distance-dependent
    piecewise functions defined by bond integral parameters.

    Parameters:
    - dR (Tensor): Tensor of shape (N,) containing interatomic distances.
    - f (Tensor): Tensor of shape (N, 14) where each row contains the 14 bond
      integral parameters for a corresponding pair of atoms.

    Returns:
    - Tensor of shape (N,): The computed derivatives of bond integral values for each distance in dR.

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


_CHANNELS = [
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


def _expand_tokens(tokens):
    """Expand compressed tokens like '3*0.0' -> ['0.0','0.0','0.0']"""
    out = []
    for t in tokens:
        if "*" in t:
            num, val = t.split("*")
            out.extend([val] * int(num))
        else:
            out.append(t)
    return out


def read_skf_table(
    path,
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
):
    """
    Read an SKF file and extract R grid + 20 channel integrals as tensors.

    Args:
        path (str): path to .skf file
        elemA, elemB (str): element symbols (to detect homo/hetero)

    Returns:
        R (Tensor): shape (npts,)
        channels (dict): channel_name -> Tensor (npts,)
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


def channels_to_matrix(channels, order=_CHANNELS):
    """
    Convert dict of channel tensors into a (n, m) matrix.
    Args:
        channels (dict): channel_name -> Tensor(n,)
        order (list): list of channel names in fixed order
    Returns:
        M (Tensor): shape (n, m)
    """
    return torch.stack([channels[ch] for ch in order], dim=1)


def cubic_spline_coeffs(R, M):
    """
    Vectorized cubic spline with:
        - natural left boundary
        - clamped right boundary: y(R[-1])=0, y'(R[-1])=0

    Args:
        R (Tensor): (n,) knot positions
        M (Tensor): (n,m) values for m channels

    Returns:
        coeffs (Tensor): (n-1, m, 4), spline coeffs [a,b,c,d] per interval
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


def spline_eval_all_and_search_coef(R, coeffs, x):
    """
    Evaluate all-channel splines + derivatives at x.
    Beyond R[-1] → force y=0, dy=0.

    Args:
        R (Tensor): (n,)
        coeffs (Tensor): (n-1,m,4)
        x (Tensor): (k,)

    Returns:
        y  (Tensor): (k,m)
        dy (Tensor): (k,m)
    """
    n = len(R)
    idx = torch.searchsorted(R, x, right=True) - 1
    idx = torch.clamp(idx, 0, n - 2)  # which interval each x falls in

    dx = (x - R[idx]).unsqueeze(1)  # (k,1)
    a, b, c, d = [coeffs[idx, :, j] for j in range(4)]  # each (k,m)

    y = a + b * dx + c * dx**2 + d * dx**3
    dy = b + 2 * c * dx + 3 * d * dx**2

    # Enforce cutoff
    mask_hi = x >= R[-1]
    y[mask_hi] = 0.0
    dy[mask_hi] = 0.0
    return y, dy


def spline_eval_all(R, coeffs, x):
    """
    Evaluate all-channel splines + derivatives at x.
    Beyond R[-1] → force y=0, dy=0.

    Args:
        R (Tensor): (n,)
        coeffs (Tensor): (n-1,m,4)
        x (Tensor): (k,)

    Returns:
        y  (Tensor): (k,m)
        dy (Tensor): (k,m)
    """
    n = len(R)
    idx = torch.searchsorted(R, x, right=True) - 1
    idx = torch.clamp(idx, 0, n - 2)  # which interval each x falls in

    dx = x - R

    y = coeffs[:, 0] + coeffs[:, 1] * dx + coeffs[:, 2] * dx**2 + coeffs[:, 3] * dx**3
    dy = coeffs[:, 1] + 2 * coeffs[:, 2] * dx + 3 * coeffs[:, 3] * dx**2

    # Enforce cutoff
    mask_hi = x >= R[-1]
    y[mask_hi] = 0.0
    dy[mask_hi] = 0.0
    return y, dy


def get_skf_tensors(TYPE, skfpath):
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
