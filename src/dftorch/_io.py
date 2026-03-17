# ruff: noqa
import numpy as np
import re


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


ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    35: "Br",
    53: "I",
}


def write_pdb_frame(
    filename, structure, LBox=None, step=0, etot=None, temp=None, mode="a"
):
    """
    Append one MD frame to a PDB trajectory file.
    Each frame is wrapped in MODEL/ENDMDL records — readable by PyMOL, VMD, Chimera.

    Parameters
    ----------
    filename  : str            output .pdb path
    structure : Structure      DFTorch structure object
    LBox      : tensor|None    [Lx, Ly, Lz] in Angstrom — written as CRYST1 if given
    step      : int            MD step number
    etot      : float|None     total energy in eV — written as REMARK
    temp      : float|None     temperature in K   — written as REMARK
    mode      : str            'w' for first frame, 'a' to append
    """
    with open(filename, mode) as f:
        # CRYST1 — unit cell (required for periodic systems in PyMOL)
        if LBox is not None:
            Lx = LBox[0].item()
            Ly = LBox[1].item()
            Lz = LBox[2].item()
            # CRYST1 format: a b c alpha beta gamma space_group Z
            f.write(
                f"CRYST1{Lx:9.3f}{Ly:9.3f}{Lz:9.3f}"
                f"  90.00  90.00  90.00 P 1           1\n"
            )

        # REMARK — metadata
        remark = f"Step={step}"
        if etot is not None:
            remark += f"  Etot={etot:.6f} eV"
        if temp is not None:
            remark += f"  T={temp:.2f} K"
        f.write(f"REMARK  {remark}\n")

        # MODEL record
        f.write(f"MODEL     {step:>8d}\n")

        for i in range(structure.Nats):
            serial = (i + 1) % 100000
            x = structure.RX[i].item()
            y = structure.RY[i].item()
            z = structure.RZ[i].item()
            raw = structure.TYPE[i].item()
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(raw), "X")

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
    element_dict = {
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
    }

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
                        element_dict[lines[i].split()[0]],
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
