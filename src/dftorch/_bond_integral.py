from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

import torch

from ._tools import ordered_pairs_from_TYPE

symbol_to_number: Final[dict[str, int]] = { #XConverts tthe first number in the skf file header to the number of protons, NOTE should add more elements
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
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
}

_CHANNELS: Final[list[str]] = [ #Aryan NOTE -> keep an internal order to minimize structure changes, or skf standard order?
    "Hff0",
    "Hff1",
    "Hff2",
    "Hff3",         
    "Hdf0",
    "Hdf1",
    "Hdf2",
    "Hdd0",
    "Hdd1",
    "Hdd2",
    "Hpf0",
    "Hpf1",
    "Hpd0",
    "Hpd1",
    "Hpp0",
    "Hpp1",
    "Hsf0",
    "Hsd0",
    "Hsp0",
    "Hss0",
    "Sff0",
    "Sff1",
    "Sff2",
    "Sff3",
    "Sdf0",
    "Sdf1",
    "Sdf2",
    "Sdd0",
    "Sdd1",
    "Sdd2",
    "Spf0",
    "Spf1",
    "Spd0",
    "Spd1",
    "Spp0",
    "Spp1",
    "Ssf0",
    "Ssd0",
    "Ssp0",
    "Sss0",
]

_SIMPLE_CHANNELS: Final[list[str]] = [
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

_SIMPLE_TO_EXTENDED: Final[list[int]] = [_CHANNELS.index(ch) for ch in _SIMPLE_CHANNELS]


SK_BLOCK_SIZE: Final[int] = 20
N_SK_CHANNELS: Final[int] = len(_CHANNELS)
MAX_SHELLS: Final[int] = 4
EV_PER_HARTREE: Final[float] = 27.21138625
BOHR_TO_ANGSTROM: Final[float] = 0.52917721



def load_bond_integral_parameters( #old function, not for skf files
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


def bond_integral_vectorized(dR: torch.Tensor, f: torch.Tensor) -> torch.Tensor: #Old function, not for skf files
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


def bond_integral_with_grad_vectorized( #old function, not for skf files
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


def _expand_tokens(tokens: list[str]) -> list[str]: #Skf files use 8* 0.0 a lot, useful to expand this
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


def _normalize_skf_row(tokens: list[str], path: str, line: str) -> list[float]:
    """Return one electronic SKF row in the 40-column extended order.

    SKF files in this code path may use either the older 20-column electronic
    table or the extended 40-column table with f-shell channels. The rest of
    DFTorch should see one consistent layout, so simple-format rows are copied
    into their matching official extended positions while all f-related columns
    are left as zero.

    Parameters
    ----------
    tokens:
        Expanded string tokens from one electronic table row.
    path:
        File path used only for a helpful error message.
    line:
        Original row text used only for a helpful error message.

    Returns
    -------
    list[float]
        Row values in the official 40-channel order named by ``_CHANNELS``.
    """
    values = [float(x) for x in tokens]
    if len(values) == len(_CHANNELS):
        return values
    if len(values) == len(_SIMPLE_CHANNELS):
        row = [0.0] * len(_CHANNELS)
        for old_idx, new_idx in enumerate(_SIMPLE_TO_EXTENDED):
            row[new_idx] = values[old_idx]
        return row
    raise ValueError(
        f"Expected 20 or 40 electronic values in {path}, got {len(values)} in line: {line}"
    )


def _resolve_skf_path(skfpath: str, label_name: str) -> str:
    """Resolve an SKF pair label to either dashed or undashed filenames.

    Existing DFTB parameter directories commonly use dashed names such as
    ``C-N.skf``. The f-orbital test data in this repository uses compact names
    such as ``EuN.skf`` and ``NN.skf``. This helper lets ``get_skf_tensors`` keep
    using ordered labels like ``Eu-N`` while supporting both file naming styles.

    Parameters
    ----------
    skfpath:
        Directory containing SKF files.
    label_name:
        Ordered pair label, usually with a dash, e.g. ``"Eu-N"``.

    Returns
    -------
    str
        Existing matching path when found, otherwise the dashed path so the
        eventual file-open error names the conventional target.
    """
    dashed = os.path.join(skfpath, f"{label_name}.skf")
    if os.path.isfile(dashed):
        return dashed

    undashed = os.path.join(skfpath, f"{label_name.replace('-', '')}.skf")
    if os.path.isfile(undashed):
        return undashed

    return dashed


def _split_skf_pair_name(name: str) -> tuple[str, str]:
    """Split an SKF basename into its two element symbols.

    Handles both conventional dashed names, e.g. ``"C-N"``, and compact names,
    e.g. ``"EuGa"``. For compact names it tries longer symbols first so
    two-letter symbols are not accidentally split as one-letter elements.

    Parameters
    ----------
    name:
        SKF basename without the ``.skf`` suffix.

    Returns
    -------
    tuple[str, str]
        The left and right element symbols encoded by the file name.
    """
    if "-" in name:
        return name.split("-", 1)

    symbols = sorted(symbol_to_number, key=len, reverse=True)
    for elem_a in symbols:
        if not name.startswith(elem_a):
            continue
        elem_b = name[len(elem_a) :]
        if elem_b in symbol_to_number:
            return elem_a, elem_b

    raise ValueError(f"Could not parse SKF pair name: {name}")


def _validate_nested_shells(
    elem: str,
    has_s: bool,
    has_p: bool,
    has_d: bool,
    has_f: bool,
    path: str,
) -> None:
    """Require supported SKF bases to be contiguous from s upward.

    DFTorch's AO ordering assumes nested shells: an f-shell basis is ``spdf``,
    a d-shell basis is ``spd``, and a p-shell basis is ``sp``. Parameter sets
    that skip an intermediate virtual shell would need a different shell-offset
    model, so they are rejected here instead of being loaded into an ambiguous
    basis layout.

    Parameters
    ----------
    elem:
        Element symbol whose homonuclear SKF header is being parsed.
    has_s, has_p, has_d, has_f:
        Shell-presence flags inferred from onsite energies or reference
        occupations.
    path:
        SKF path used in the error message.
    """
    if has_f and not (has_s and has_p and has_d):
        raise ValueError(f"{path}: {elem} f-shell basis requires nested s/p/d/f shells")
    if has_d and not (has_s and has_p):
        raise ValueError(f"{path}: {elem} d-shell basis requires nested s/p/d shells")
    if has_p and not has_s:
        raise ValueError(f"{path}: {elem} p-shell basis requires an s shell")


def _shell_metadata_from_presence(
    shell_presence: tuple[bool, bool, bool, bool],
    shell_occ: tuple[float, float, float, float],
) -> tuple[int, int, int]:
    """Return ``(n_orb, max_ang, max_ang_occ)`` for s/p/d/f shell metadata."""
    n_orb = sum((2 * l + 1) for l, present in enumerate(shell_presence) if present)
    max_ang = max(
        (l + 1 for l, present in enumerate(shell_presence) if present),
        default=0,
    )
    max_ang_occ = max(
        (l + 1 for l, occ in enumerate(shell_occ) if occ != 0.0),
        default=0,
    )
    return n_orb, max_ang, max_ang_occ


def read_skf_table(
    path: str,
    N_ORB: torch.Tensor,
    MAX_ANG: torch.Tensor,
    MAX_ANG_OCC: torch.Tensor,
    TORE: torch.Tensor,
    N_S: torch.Tensor,
    N_P: torch.Tensor,
    N_D: torch.Tensor,
    N_F: torch.Tensor,
    ES: torch.Tensor,
    EP: torch.Tensor,
    ED: torch.Tensor,
    EF: torch.Tensor,
    US: torch.Tensor,
    UP: torch.Tensor,
    UD: torch.Tensor,
    UF: torch.Tensor,
    SHELL_PRESENT: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Read a DFTB+ ``.skf`` file and normalize electronic channels to 40 columns.

    The parser accepts both simple 20-column SKF files and extended 40-column
    SKF files. Simple electronic rows are expanded into the official extended
    order in ``_CHANNELS`` by placing the old s/p/d values into their matching
    positions and filling all f-related channels with zero.

    Homonuclear files also update atomic metadata in-place. Simple homonuclear
    headers do not contain f-shell metadata, so ``Ef``, ``Uf``, and ``ff`` are
    filled as zero and ``SHELL_PRESENT[Z, 3]`` remains false unless later
    overridden by ``wfc.hsd``.
    """
    device = torch.device("cpu") if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype
    lines = Path(path).read_text(errors="ignore").splitlines()
    data_lines = [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]
    if not data_lines:
        raise ValueError(f"Empty or comment-only SKF file: {path}")

    extended = data_lines[0].startswith("@")
    grid_idx = 1 if extended else 0
    if grid_idx >= len(data_lines):
        raise ValueError(f"Missing grid line in SKF file: {path}")

    # First numerical line: electronic grid spacing and number of grid points.
    first = data_lines[grid_idx].replace(",", " ").split()
    if len(first) < 2:
        raise ValueError(f"Malformed SKF grid line in {path}: {data_lines[grid_idx]}")
    step = float(first[0])
    npts_read = int(first[1])
    npts_pad = npts_read + 50

    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    elemA, elemB = _split_skf_pair_name(name)
    homonuclear = elemA == elemB

    # Layout after the optional extended marker:
    #   homonuclear: grid, atomic header, mass/poly line, electronic table
    #   heteronuclear: grid, mass/poly line, electronic table
    start_idx = grid_idx + 3 if homonuclear else grid_idx + 2

    if homonuclear:
        header_tokens = data_lines[grid_idx + 1].replace(",", " ").split()
        if extended:
            if len(header_tokens) < 13:
                raise ValueError(
                    f"Expected 13 extended homonuclear header values in {path}, "
                    f"got {len(header_tokens)}"
                )
            (
                Ef,
                Ed,
                Ep,
                Es,
                SPE,  # noqa: F841
                Uf,
                Ud,
                Up,
                Us,
                ff,
                fd,
                fp,
                fs,
            ) = (float(token) for token in header_tokens[:13])
        else:
            if len(header_tokens) < 10:
                raise ValueError(
                    f"Expected 10 simple homonuclear header values in {path}, "
                    f"got {len(header_tokens)}"
                )
            (
                Ed,
                Ep,
                Es,
                SPE,  # noqa: F841
                Ud,
                Up,
                Us,
                fd,
                fp,
                fs,
            ) = (float(token) for token in header_tokens[:10])
            Ef = 0.0
            Uf = 0.0
            ff = 0.0

        el_num = symbol_to_number[elemA]

        # This is still a parser-level inference. If wfc.hsd exists, it can
        # override shell presence below in get_skf_tensors().
        has_s = Es != 0.0 or fs != 0.0
        has_p = Ep != 0.0 or fp != 0.0
        has_d = Ed != 0.0 or fd != 0.0
        has_f = Ef != 0.0 or ff != 0.0
        _validate_nested_shells(elemA, has_s, has_p, has_d, has_f, path)

        shell_presence = (has_s, has_p, has_d, has_f)
        shell_occ = (fs, fp, fd, ff)
        n_orb, max_ang, max_ang_occ = _shell_metadata_from_presence(
            shell_presence,
            shell_occ,
        )

        N_ORB[el_num] = n_orb
        MAX_ANG[el_num] = max_ang
        MAX_ANG_OCC[el_num] = max_ang_occ
        TORE[el_num] = fs + fp + fd + ff
        N_S[el_num] = fs
        N_P[el_num] = fp
        N_D[el_num] = fd
        N_F[el_num] = ff
        ES[el_num] = Es * EV_PER_HARTREE
        EP[el_num] = Ep * EV_PER_HARTREE
        ED[el_num] = Ed * EV_PER_HARTREE
        EF[el_num] = Ef * EV_PER_HARTREE
        US[el_num] = Us * EV_PER_HARTREE
        UP[el_num] = Up * EV_PER_HARTREE
        UD[el_num] = Ud * EV_PER_HARTREE
        UF[el_num] = Uf * EV_PER_HARTREE
        SHELL_PRESENT[el_num] = torch.tensor(
            shell_presence,
            dtype=torch.bool,
            device=SHELL_PRESENT.device,
        )

    if start_idx + npts_read - 1 > len(data_lines):
        raise ValueError(
            f"Electronic table in {path} is shorter than expected: "
            f"need {npts_read - 1} rows from index {start_idx}, "
            f"have {max(len(data_lines) - start_idx, 0)} candidate rows"
        )

    rows: list[list[float]] = []
    for ln in data_lines[start_idx : start_idx + npts_read - 1]:
        tokens = _expand_tokens(ln.replace(",", " ").split())
        rows.append(_normalize_skf_row(tokens, path, ln))

    # Append one zero knot at the tabulated cutoff, then zero-pad the tail.
    zero_row = [0.0] * len(_CHANNELS)
    rows.append(zero_row)
    if len(rows) < npts_pad:
        rows.extend([zero_row.copy() for _ in range(npts_pad - len(rows))])

    mat = torch.tensor(rows, dtype=dtype, device=device) * EV_PER_HARTREE
    R = (
        torch.arange(1, npts_pad + 1, dtype=dtype, device=device)
        * step
        * BOHR_TO_ANGSTROM
    )
    channels = {ch: mat[:, j] for j, ch in enumerate(_CHANNELS)}

    spline_start = None
    for idx, line in enumerate(data_lines):
        if line.casefold() == "spline":
            spline_start = idx
            break
    if spline_start is None:
        raise ValueError(f"No Spline block found in {path}")

    first = data_lines[spline_start + 1].replace(",", " ").split()
    if len(first) < 2:
        raise ValueError(f"Malformed repulsive Spline header in {path}: {data_lines[spline_start + 1]}")
    npts = int(first[0])

    close_exp = torch.tensor(
        [float(x) for x in data_lines[spline_start + 2].replace(",", " ").split()],
        dtype=dtype,
        device=device,
    )

    rows_rep: list[list[float]] = []
    rows_R: list[float] = []
    for ln in data_lines[spline_start + 3 : spline_start + 3 + npts - 1]:
        tokens = _expand_tokens(ln.replace(",", " ").split())
        if len(tokens) != 6:
            raise ValueError(f"Expected 6 repulsive spline values in {path}, got {len(tokens)} in line: {ln}")
        rows_R.append(float(tokens[0]))
        rows_rep.append([float(x) for x in tokens[2:]] + [0.0] * 2)

    ln = data_lines[spline_start + 3 + npts - 1]
    tokens = _expand_tokens(ln.replace(",", " ").split())
    if len(tokens) != 8:
        raise ValueError(f"Expected 8 final repulsive spline values in {path}, got {len(tokens)} in line: {ln}")
    rows_R.append(float(tokens[0]))
    rows_rep.append([float(x) for x in tokens[2:]])

    rows_R.append(float(tokens[1]))
    rows_rep.append([0.0] * 6)

    rep_splines = torch.tensor(rows_rep, dtype=dtype, device=device)
    R_rep = torch.tensor(rows_R, dtype=dtype, device=device) * BOHR_TO_ANGSTROM

    return R, channels, R_rep, rep_splines, close_exp

def channels_to_matrix(
    channels: dict[str, torch.Tensor],
    order: list[str] = _CHANNELS,
) -> torch.Tensor:
    """Convert channel dict into a dense matrix of shape `(npts, len(order))`."""
    return torch.stack([channels[ch] for ch in order], dim=1)


def cubic_spline_coeffs(R: torch.Tensor, M: torch.Tensor) -> torch.Tensor: #Probably have to compute for f orbitals too
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


def _extract_blocks(text: str, keyword: str) -> list[str]:
    """
    Extract all top-level blocks matching:
        keyword = { ... }
    Handles nested braces correctly by counting depth.
    Returns list of inner content strings (without outer braces).
    """
    results = []
    search_str = keyword
    pos = 0
    while True:
        # find next occurrence of keyword followed by '='  and '{'
        idx = text.find(search_str, pos)
        if idx == -1:
            break
        # skip to the opening brace
        brace_start = text.find("{", idx + len(search_str))
        if brace_start == -1:
            break
        # walk forward counting depth
        depth = 0
        i = brace_start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[brace_start + 1 : i])
                    pos = i + 1
                    break
            i += 1
        else:
            break  # unterminated block
    return results


def read_wfc_hsd(
    path: str,
    N_ORB: torch.Tensor,
    MAX_ANG: torch.Tensor,
    MAX_ANG_OCC: torch.Tensor,
    SHELL_PRESENT: torch.Tensor,
) -> None:
    """Parse ``wfc.hsd`` and override shell-presence basis metadata in-place.

    ``wfc.hsd`` is treated as authoritative for which angular-momentum shells
    are present. It updates ``SHELL_PRESENT``, ``N_ORB``, ``MAX_ANG``, and
    ``MAX_ANG_OCC`` consistently. Occupation totals ``N_S``/``N_P``/``N_D``/
    ``N_F`` still come from the homonuclear SKF headers.
    """
    text = Path(path).read_text(errors="ignore")

    ang_re = re.compile(r"AngularMomentum\s*=\s*(\d+)")
    occ_re = re.compile(r"Occupation\s*=\s*([\d.eE+\-]+)")
    sym_re = re.compile(r"^([A-Z][a-z]{0,2})\s*=?\s*\{", re.MULTILINE)

    for sym_m in sym_re.finditer(text):
        sym = sym_m.group(1).strip()
        Z = symbol_to_number.get(sym)
        if Z is None:
            continue

        brace_start = text.find("{", sym_m.start())
        if brace_start == -1:
            continue
        depth = 0
        elem_body = None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    elem_body = text[brace_start + 1 : i]
                    break
        if elem_body is None:
            continue

        shell_presence = [False] * MAX_SHELLS
        shell_occ = [0.0] * MAX_SHELLS

        for orb_body in _extract_blocks(elem_body, "Orbital"):
            am = ang_re.search(orb_body)
            oc = occ_re.search(orb_body)
            if am is None or oc is None:
                continue

            l = int(am.group(1))  # noqa: E741
            occ = float(oc.group(1))
            if l < 0 or l >= MAX_SHELLS:
                raise ValueError(f"{path}: unsupported angular momentum l={l} for {sym}")

            shell_presence[l] = True
            shell_occ[l] = occ

        if not any(shell_presence):
            continue

        _validate_nested_shells(sym, *shell_presence, path)
        n_orb, max_ang, max_ang_occ = _shell_metadata_from_presence(
            tuple(shell_presence),
            tuple(shell_occ),
        )

        SHELL_PRESENT[Z] = torch.tensor(
            shell_presence,
            dtype=torch.bool,
            device=SHELL_PRESENT.device,
        )
        N_ORB[Z] = n_orb
        MAX_ANG[Z] = max_ang
        MAX_ANG_OCC[Z] = max_ang_occ


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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load SKF tensors for all ordered element pairs present in ``TYPE``.

    All electronic SK tables are normalized to the official 40-channel extended
    order. The return tuple extends the historical layout with f-shell metadata:
    ``N_F``, ``EF``, ``UF``, and ``SHELL_PRESENT``.
    """
    _, _, label_list = ordered_pairs_from_TYPE(TYPE)

    n_pairs = len(label_list)
    npts = 1300
    dtype = torch.get_default_dtype()
    device = TYPE.device

    coeffs_tensor = torch.zeros(
        (n_pairs, npts, len(_CHANNELS), 4),
        dtype=dtype,
        device=device,
    )
    R_tensor = torch.zeros((n_pairs, npts + 1), dtype=dtype, device=device)

    rep_splines_tensor = torch.zeros((n_pairs, 500, 6), dtype=dtype, device=device)
    R_rep_tensor = torch.zeros((n_pairs, 500), dtype=dtype, device=device) + 1e8
    close_exp_tensor = torch.zeros((n_pairs, 3), dtype=dtype, device=device)

    N_ORB = torch.zeros(120, dtype=torch.int64, device=device)
    MAX_ANG = torch.zeros(120, dtype=torch.int64, device=device)
    MAX_ANG_OCC = torch.zeros(120, dtype=torch.int64, device=device)
    SHELL_PRESENT = torch.zeros((120, MAX_SHELLS), dtype=torch.bool, device=device)

    TORE = torch.zeros(120, dtype=dtype, device=device)
    N_S = torch.zeros(120, dtype=dtype, device=device)
    N_P = torch.zeros(120, dtype=dtype, device=device)
    N_D = torch.zeros(120, dtype=dtype, device=device)
    N_F = torch.zeros(120, dtype=dtype, device=device)

    ES = torch.zeros(120, dtype=dtype, device=device)
    EP = torch.zeros(120, dtype=dtype, device=device)
    ED = torch.zeros(120, dtype=dtype, device=device)
    EF = torch.zeros(120, dtype=dtype, device=device)
    US = torch.zeros(120, dtype=dtype, device=device)
    UP = torch.zeros(120, dtype=dtype, device=device)
    UD = torch.zeros(120, dtype=dtype, device=device)
    UF = torch.zeros(120, dtype=dtype, device=device)

    R_orb_master = None

    for i, label in enumerate(label_list):
        R_orb_i, channels, R_rep, rep_splines, close_exp = read_skf_table(
            _resolve_skf_path(skfpath, label),
            N_ORB,
            MAX_ANG,
            MAX_ANG_OCC,
            TORE,
            N_S,
            N_P,
            N_D,
            N_F,
            ES,
            EP,
            ED,
            EF,
            US,
            UP,
            UD,
            UF,
            SHELL_PRESENT,
            device=device,
            dtype=dtype,
        )

        channels_matrix = channels_to_matrix(channels)
        coeffs = cubic_spline_coeffs(R_orb_i, channels_matrix)
        zero_row_idx = torch.nonzero(channels_matrix.eq(0).all(dim=1), as_tuple=False)
        if zero_row_idx.numel() > 0:
            coeffs[int(zero_row_idx[0].item()) :] = 0

        R_tensor[i, : len(R_orb_i)] = R_orb_i
        coeffs_tensor[i, : len(coeffs)] = coeffs

        if R_orb_master is None or len(R_orb_i) > len(R_orb_master):
            R_orb_master = R_orb_i

        R_rep_tensor[i, : len(R_rep)] = R_rep
        rep_splines_tensor[i, : len(rep_splines)] = rep_splines
        close_exp_tensor[i] = close_exp

    wfc_path = os.path.join(skfpath, "wfc.hsd")
    if os.path.isfile(wfc_path):
        read_wfc_hsd(wfc_path, N_ORB, MAX_ANG, MAX_ANG_OCC, SHELL_PRESENT)

    if R_orb_master is None:
        raise ValueError(f"No SKF files were loaded from {skfpath}")
    R_orb = R_orb_master.to(device=device, dtype=dtype)

    coeffs_tensor = torch.cat(
        (
            coeffs_tensor,
            torch.zeros(
                coeffs_tensor.shape[0],
                1,
                coeffs_tensor.shape[2],
                coeffs_tensor.shape[3],
                dtype=coeffs_tensor.dtype,
                device=device,
            ),
        ),
        dim=1,
    )

    return (
        R_tensor,
        R_orb,
        coeffs_tensor,
        R_rep_tensor,
        rep_splines_tensor,
        close_exp_tensor,
        N_ORB,
        MAX_ANG,
        MAX_ANG_OCC,
        TORE,
        N_S,
        N_P,
        N_D,
        N_F,
        ES,
        EP,
        ED,
        EF,
        US,
        UP,
        UD,
        UF,
        SHELL_PRESENT,
    )