# ruff: noqa
import numpy as np
import re

from dftorch._cell import _cell_to_pdb_cryst1


def write_XYZ_trajectory(filename, structure, comment, step=0, Ftot=None):
    with open(filename, "a+") as f:
        num_atoms = structure.Nats
        f.write(f"{structure.Nats}\n")
        f.write(f"Step {step}, {comment}\n")
        for i in range(num_atoms):
            x = structure.RX[i].item()
            y = structure.RY[i].item()
            z = structure.RZ[i].item()
            symbol = structure.TYPE[i].item()
            if Ftot is not None:
                fx = Ftot[i, 0].item()
                fy = Ftot[i, 1].item()
                fz = Ftot[i, 2].item()
                f.write(
                    f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n"
                )
            else:
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


SYMBOL_TO_NUMBER = {
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
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
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
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}
# Reverse lookup for the PDB/XYZ writer: atomic number → element symbol
NUMBER_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_NUMBER.items()}


def write_pdb_frame(filename, structure, cell=None, step=0, comment="", mode="a"):
    """
    Append one MD frame to a PDB trajectory file.
    Each frame is wrapped in MODEL/ENDMDL records — readable by PyMOL, VMD, Chimera.

    Parameters
    ----------
    filename  : str            output .pdb path
    structure : Structure      DFTorch structure object
    cell      : tensor|None
        Periodic cell specification in Angstrom:
        - shape (3,)   for orthorhombic box lengths [Lx, Ly, Lz]
        - shape (3,3)  for triclinic lattice vectors
        Written as CRYST1 if given.
    step      : int            MD step number
    comment   : str            free-form metadata string — written as REMARK,
                               identical to the XYZ comment line content
    mode      : str            'w' for first frame, 'a' to append
    """
    with open(filename, mode) as f:
        # CRYST1 — unit cell (required for periodic systems in PyMOL)
        if cell is not None:
            a, b, c, alpha, beta, gamma = _cell_to_pdb_cryst1(cell)

            f.write(
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
            )

        # REMARK — same content as the XYZ comment line: "Step {step}, {comment}"
        f.write(f"REMARK  Step {step}, {comment}\n")

        # MODEL record
        f.write(f"MODEL     {step:>8d}\n")

        for i in range(structure.Nats):
            serial = (i + 1) % 100000
            x = structure.RX[i].item()
            y = structure.RY[i].item()
            z = structure.RZ[i].item()
            raw = structure.TYPE[i].item()
            symbol = NUMBER_TO_SYMBOL.get(int(raw), "X")

            # strict PDB ATOM fixed-format columns
            f.write(
                f"{'ATOM':<6}{serial:>5d}  "
                f"{symbol:<4s}{'MOL':>3s} {'A':1s}{1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"  1.00  0.00          "
                f"{symbol:>2s}\n"
            )

        f.write("ENDMDL\n")


def finalise_pdb(filename):
    """Write terminal END record. Call once after all frames are written."""
    with open(filename, "a") as f:
        f.write("END\n")


def write_XYZ(filename, structure, comment, step=0, Ftot=None):
    with open(filename, "a+") as f:
        num_atoms = structure.Nats
        f.write(f"{structure.Nats}\n")
        f.write(f"Step {step}, {comment}\n")
        for i in range(num_atoms):
            x = structure.RX[i].item()
            y = structure.RY[i].item()
            z = structure.RZ[i].item()
            symbol = structure.TYPE[i].item()
            if Ftot is not None:
                fx = Ftot[i, 0].item()
                fy = Ftot[i, 1].item()
                fz = Ftot[i, 2].item()
                f.write(
                    f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n"
                )
            else:
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def write_xyz_from_xyz(filename, TYPE, RX, RY, RZ, comment, step=0, Ftot=None):
    with open(filename, "a+") as f:
        num_atoms = len(TYPE)
        f.write(f"{num_atoms}\n")
        f.write(f"Step {step}, {comment}\n")
        for i in range(num_atoms):
            x = RX[i].item()
            y = RY[i].item()
            z = RZ[i].item()
            symbol = TYPE[i].item()
            if Ftot is not None:
                fx = Ftot[i, 0].item()
                fy = Ftot[i, 1].item()
                fz = Ftot[i, 2].item()
                f.write(
                    f"{symbol} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n"
                )
            else:
                f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def read_xyz(files, sort=True):
    """
    reads xyz structure from a list (files) of files names
    """

    COORDINATES = []
    for file in files:
        f = open(file)
        lines = f.readlines()
        f.close()
        Natoms = int(lines[0])
        coords = []
        try:
            int(lines[2].split()[0])
            atoms_are_number = True
        except:
            atoms_are_number = False
        for i in range(2, 2 + Natoms):
            # species.append(int(lines[i].split()[0]))
            if atoms_are_number:
                coords.append(
                    [
                        int(lines[i].split()[0]),
                        float(lines[i].split()[1]),
                        float(lines[i].split()[2]),
                        float(lines[i].split()[3]),
                    ]
                )
            else:
                coords.append(
                    [
                        SYMBOL_TO_NUMBER[lines[i].split()[0]],
                        float(lines[i].split()[1]),
                        float(lines[i].split()[2]),
                        float(lines[i].split()[3]),
                    ]
                )
        COORDINATES.append(coords)
    COORDINATES = np.array(COORDINATES)
    if sort:
        COORDINATES = np.array([x[(-1 * x[:, 0]).argsort()] for x in COORDINATES])

    SPECIES = COORDINATES[:, :, 0].astype(int)
    COORDINATES = COORDINATES[:, :, 1:4]

    return SPECIES, COORDINATES


# ── Element-symbol → atomic-number lookup (covers Z = 1–54) ──────────


def read_pdb(files, sort=True):
    """
    Read geometry from PDB files.

    Parses ATOM / HETATM records for coordinates and element symbols.
    Returns arrays in the same format as :func:`read_xyz` so that
    ``Structure`` can consume them interchangeably::

        species, coordinates = read_pdb(["system.pdb"])

    Parameters
    ----------
    files : list[str]
        List of PDB file paths (one frame per file).
    sort : bool, optional
        If *True* (default), atoms are sorted by descending atomic number
        (same convention as ``read_xyz``).

    Returns
    -------
    SPECIES : ndarray, shape (nfiles, natoms), int
        Atomic numbers.
    COORDINATES : ndarray, shape (nfiles, natoms, 3), float64
        Cartesian coordinates in **Angstrom**.

    Notes
    -----
    *   Element is read from **columns 77-78** (standard PDB).  If that
        field is blank the element is inferred from the atom-name field
        (columns 13-16) by stripping digits and whitespace.
    *   REMARK lines are silently skipped.  To extract charge or
        selection metadata from REMARK headers use :func:`read_pdb_remarks`.
    """
    COORDINATES = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()

        coords = []
        for line in lines:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            # --- coordinates (columns 31-54, 8.3f each) ---
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            # --- element symbol ---
            # Prefer columns 77-78 (right-justified element)
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if not elem:
                # Fallback: infer from atom name (columns 13-16)
                atom_name = line[12:16].strip()
                elem = re.sub(r"[0-9]", "", atom_name).strip()
                # atom names like "CA", "CB" are carbon, not calcium —
                # use only the first character if len > 1 and second char is lowercase
                if len(elem) > 1 and elem[1].islower():
                    pass  # e.g. "Fe", "Cl", "Br" — keep as-is
                elif len(elem) > 1:
                    elem = elem[0]

            # Capitalize properly: first upper, rest lower
            elem = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()

            z_num = SYMBOL_TO_NUMBER.get(elem)
            if z_num is None:
                raise ValueError(
                    f"Unknown element '{elem}' in {file}, line: {line.rstrip()}"
                )

            coords.append([z_num, x, y, z])

        if not coords:
            raise ValueError(f"No ATOM/HETATM records found in {file}")

        COORDINATES.append(coords)

    COORDINATES = np.array(COORDINATES, dtype=np.float64)
    if sort:
        COORDINATES = np.array([x[(-1 * x[:, 0]).argsort()] for x in COORDINATES])

    SPECIES = COORDINATES[:, :, 0].astype(int)
    COORDINATES = COORDINATES[:, :, 1:4]

    return SPECIES, COORDINATES


def read_pdb_remarks(file):
    """
    Extract REMARK metadata from a PDB file.

    Returns a dict of ``{key: value}`` for REMARK lines of the form
    ``REMARK key value``, e.g.::

        REMARK charge -1
        REMARK selection_a %not(:unk)

    Parameters
    ----------
    file : str
        Path to PDB file.

    Returns
    -------
    dict[str, str]
        Keys and values (as strings).
    """
    remarks = {}
    with open(file) as f:
        for line in f:
            if not line.startswith("REMARK"):
                continue
            parts = line.split(maxsplit=2)
            if len(parts) >= 3:
                remarks[parts[1]] = parts[2].strip()
    return remarks


def read_xyz_traj_data(file):
    """
    Reads data from xyz trajectory comment lines at each step
    e.g. "Step 0, Etot = -145495.406250 eV, Epot = -145702.171875 eV, Ekin = 206.764328 eV, T = 399.90 K, Res = 0.000000, mu = -2.5353 eV"
    Returns a dict with 'step' (int array) and any parsed named quantities as numpy arrays.
    Missing values are filled with NaN.
    """

    steps = []
    data = {}  # key -> list of values (aligned with steps)
    with open(file, "r") as fh:
        lines = fh.readlines()

    i = 0
    nlines = len(lines)
    while i < nlines:
        line = lines[i].strip()
        if line == "":
            i += 1
            continue
        # expect number-of-atoms line
        try:
            natoms = int(line)
        except ValueError:
            # skip malformed line
            i += 1
            continue
        # comment line should be next
        if i + 1 >= nlines:
            break
        comment = lines[i + 1].strip()

        # extract step number if present
        m = re.search(r"\bStep\b\s*([0-9]+)", comment, re.IGNORECASE)
        step = int(m.group(1)) if m else None
        steps.append(step)

        # extract key=value pairs (value is a float, may have units after)
        found = {}
        for key, val in re.findall(
            r"([A-Za-z_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*[A-Za-z%°K]*)?",
            comment,
        ):
            try:
                found[key] = float(val)
            except Exception:
                found[key] = np.nan

        # update existing keys with either found value or NaN
        for k in list(data.keys()):
            data[k].append(found.get(k, np.nan))
        # add any newly seen keys, backfilling previous steps with NaN
        for k, v in found.items():
            if k not in data:
                data[k] = [np.nan] * (len(steps) - 1)  # previous steps
                data[k].append(v)
        # advance to next frame
        i += natoms + 2

    result = {"step": np.array(steps, dtype=float)}
    for k, lst in data.items():
        result[k] = np.array(lst, dtype=float)

    return result
