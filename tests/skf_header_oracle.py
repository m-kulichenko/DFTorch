"""An SKF header parser that knows nothing about dftorch.

WHY THIS MODULE EXISTS
----------------------
Every other SKF metadata check in this suite reads its expected values through
`dftorch._bond_integral`.  That makes the production parser its own oracle: if
`read_skf_table` mis-reads the extended header, both the value under test and
the value it is compared against move together and the test still passes.

`src/dftorch/script.py` was the only place in the repository that did not have
that flaw.  Its `parse_expected_homonuclear_metadata()` re-read the SKF header
text itself and computed onsite energies, Hubbard values and reference
occupations from the raw tokens.  Decision D-03 in
`.planning/phases/05-regression-safety-and-support-policy-cleanup/05-CONTEXT.md`
names exactly that property as the reason to port the file rather than delete
it, and this module is where the property now lives.

THE INVARIANT THIS FILE MUST KEEP
---------------------------------
This module MUST NOT import dftorch, and MUST NOT reference any dftorch symbol,
in any form, ever.  It reaches the same numbers as the production loader by a
different route, using only the standard library.  That is not a style
preference; it is the entire reason the file is worth having.

`tests/test_skf_metadata_oracle.py::test_oracle_does_not_use_production_parser`
enforces this by reading this file's own source text and failing if the string
"dftorch" or any production parsing symbol appears in it.  A future edit that
"simplifies" a helper here by calling `_split_skf_pair_name` or `read_skf_table`
will be caught by that test, not by a reviewer's memory.

Two element tables and two AO layout tables are duplicated here on purpose.
A single-sourced constant cannot detect a one-sided edit; two written sources
can.

ASCII ONLY.  Phase 4 recorded that non-ASCII characters render as replacement
characters on a cp1252 console and destroy the diagnosability of pytest output.
"""

from __future__ import annotations

from pathlib import Path


# Hartree -> eV.  Written independently of `dftorch.Constants`; if the two ever
# disagree the metadata tests below fail, which is the point.
EV_PER_HARTREE = 27.21138625

# Absolute tolerance for every float comparison built on this oracle.  The SKF
# header carries at most 8 decimal digits, so 1e-9 eV is a tight but achievable
# band once the Hartree conversion is applied.
ATOL = 1.0e-9


# Element symbols in atomic-number order, index 0 unused.  A second written
# source for the periodic table, so that a corrupted entry in the production
# `symbol_to_number` map shows up as a failing metadata comparison instead of
# cancelling out on both sides.
ELEMENT_SYMBOLS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
)

_SYMBOL_TO_NUMBER = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS) if sym}


# The local AO layout, written a second time.  `dftorch.Structure` defines
# SHELL_DIMS / SHELL_LOCAL_STARTS / SHELL_TYPE_IDS / AO_LABEL_TEMPLATE /
# AO_SHELL_TEMPLATE; these are the independent copies that
# `test_basis_tables_match_expected_tables` compares them against.  A one-sided
# edit to either copy fails that test, which is the only way a silent
# renumbering of the f block gets caught.
EXPECTED_SHELL_DIMS = [1, 3, 5, 7]
EXPECTED_SHELL_LOCAL_STARTS = [0, 1, 4, 9]
EXPECTED_SHELL_TYPE_IDS = [1, 2, 3, 4]
EXPECTED_AO_LABEL_TEMPLATE = [
    "s",
    "px",
    "py",
    "pz",
    "dxy",
    "dyz",
    "dzx",
    "dx2_y2",
    "dz2",
    "fx3",
    "fy3",
    "fz3",
    "fx_y2_z2",
    "fy_z2_x2",
    "fz_x2_y2",
    "fxyz",
]
EXPECTED_AO_SHELL_TEMPLATE = [
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]


def element_number(symbol: str) -> int:
    """Return the atomic number of ``symbol`` from this module's own table."""
    try:
        return _SYMBOL_TO_NUMBER[symbol]
    except KeyError:
        raise ValueError(f"Unknown element symbol: {symbol!r}") from None


def is_known_element(symbol: str) -> bool:
    return symbol in _SYMBOL_TO_NUMBER


def read_data_lines(skf_path: Path) -> list[str]:
    """Return the non-blank, non-comment lines of an SKF file, stripped."""
    lines = Path(skf_path).read_text(errors="ignore").splitlines()
    return [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.lstrip().startswith(("#", "!", ";"))
    ]


def split_skf_pair_name_independently(name: str) -> tuple[str, str]:
    """Split an SKF basename into two element symbols without the production splitter.

    Handles the two forms this repository's fixtures use: dashed (``"Eu-Ga"``,
    ``"C-C"``) and compact (``"EuGa"``, ``"NN"``).  Compact names are matched
    two-letter-symbol-first so ``"NN"`` does not resolve as ``("N", "N")`` only
    by luck and ``"CCl"`` cannot be mis-split as ``("C", "Cl")`` versus
    ``("Cc", "l")``.

    Reimplemented here rather than delegating to
    `dftorch._bond_integral._split_skf_pair_name` because the element identity
    is part of what the oracle asserts: a splitter bug that attributed Eu's
    header to Ga would be invisible if both sides used the same splitter.
    """
    if "-" in name:
        elem_a, elem_b = name.split("-", 1)
        return elem_a, elem_b

    for length in (2, 1):
        elem_a = name[:length]
        elem_b = name[length:]
        if is_known_element(elem_a) and is_known_element(elem_b):
            return elem_a, elem_b

    raise ValueError(f"Could not parse SKF pair name: {name!r}")


def resolve_homonuclear_skf_independently(skf_dir: Path, symbol: str) -> Path:
    """Return the homonuclear SKF path for ``symbol``, dashed form preferred.

    A local stand-in for `dftorch._bond_integral._resolve_skf_path`, for the
    same reason the pair-name split is local.  The returned path may not exist;
    callers decide whether that is a skip or a failure.
    """
    skf_dir = Path(skf_dir)
    dashed = skf_dir / f"{symbol}-{symbol}.skf"
    if dashed.is_file():
        return dashed
    compact = skf_dir / f"{symbol}{symbol}.skf"
    if compact.is_file():
        return compact
    return dashed


def collect_elements_independently(skf_dir: Path) -> list[str]:
    """Return every element named by a *.skf basename in ``skf_dir``, Z-ordered."""
    elements = set()
    for skf_path in sorted(Path(skf_dir).glob("*.skf")):
        elem_a, elem_b = split_skf_pair_name_independently(skf_path.stem)
        for sym in (elem_a, elem_b):
            if not is_known_element(sym):
                raise ValueError(f"Unknown element {sym!r} in {skf_path.name}")
        elements.add(elem_a)
        elements.add(elem_b)
    return sorted(elements, key=element_number)


def expected_shell_metadata(
    shell_present: list[bool], shell_occ: list[float]
) -> tuple[int, int, int]:
    """Return expected (N_ORB, MAX_ANG, MAX_ANG_OCC) from shell presence and occupation.

    Derived from angular momentum alone: a present shell of angular momentum l
    contributes 2l+1 orbitals.  The production code stores MAX_ANG as l + 1,
    not l, so this returns the same convention.
    """
    n_orb = 0
    max_ang = 0
    max_ang_occ = 0

    for l, present in enumerate(shell_present):
        if present:
            n_orb += 2 * l + 1
            max_ang = max(max_ang, l + 1)

    for l, occ in enumerate(shell_occ):
        if occ != 0.0:
            max_ang_occ = max(max_ang_occ, l + 1)

    return n_orb, max_ang, max_ang_occ


def parse_expected_homonuclear_metadata(skf_path: Path) -> dict[str, object] | None:
    """Independently parse a homonuclear SKF header and compute expected metadata.

    This is the function decision D-03 exists to preserve.  It reads the raw
    header tokens and computes onsite energies, Hubbard values and reference
    occupations from them, so nothing it returns has passed through
    `read_skf_table` or `get_skf_tensors`.

    Header layouts, as they appear in this repository's fixtures:

    * extended (first data line starts with ``@``, e.g. `f_orbital_data/Eu-Eu.skf`):
      13 values, ``Ef Ed Ep Es SPE Uf Ud Up Us ff fd fp fs``.
    * simple (e.g. `data_skf_mio-1-1/C-C.skf`):
      10 values, ``Ed Ep Es SPE Ud Up Us fd fp fs``; f is absent, so Ef/Uf/ff
      are zero and the f shell is not present.

    Energies and Hubbard values are stored in the SKF file in Hartree and are
    returned here in eV, matching what `Constants` exposes.

    Heteronuclear files return ``None``: they carry no element-level
    onsite/Hubbard/occupation block at all.
    """
    skf_path = Path(skf_path)
    elem_a, elem_b = split_skf_pair_name_independently(skf_path.stem)
    if elem_a != elem_b:
        return None

    data_lines = read_data_lines(skf_path)
    if not data_lines:
        raise ValueError(f"Empty SKF file: {skf_path}")

    extended = data_lines[0].startswith("@")
    grid_idx = 1 if extended else 0

    # Grid line, e.g. ``0.02, 500,1`` (s-only hydrogen) or ``0.02, 500 ,2`` (sp carbon).
    # The optional third field is the declared shell count. Parsed here independently of
    # the production parser; see the shell-presence block below for why it is preferred.
    grid_tokens = data_lines[grid_idx].replace(",", " ").split()
    n_shells_declared = int(grid_tokens[2]) if len(grid_tokens) >= 3 else None

    header_tokens = data_lines[grid_idx + 1].replace(",", " ").split()

    if extended:
        if len(header_tokens) < 13:
            raise ValueError(f"Expected 13 extended header values in {skf_path}")
        (
            Ef,
            Ed,
            Ep,
            Es,
            _SPE,
            Uf,
            Ud,
            Up,
            Us,
            ff,
            fd,
            fp,
            fs,
        ) = (float(x) for x in header_tokens[:13])
        header_value_count = 13
    else:
        if len(header_tokens) < 10:
            raise ValueError(f"Expected 10 simple header values in {skf_path}")
        (
            Ed,
            Ep,
            Es,
            _SPE,
            Ud,
            Up,
            Us,
            fd,
            fp,
            fs,
        ) = (float(x) for x in header_tokens[:10])
        Ef = 0.0
        Uf = 0.0
        ff = 0.0
        header_value_count = 10

    # Shell presence. The declared shell count on the grid line is authoritative when the
    # file supplies it; the energy/occupation inference is only a fallback.
    #
    # The inference alone is not safe, and this oracle previously got it wrong in exactly
    # the way the production parser did -- which is why it confirmed a real defect instead
    # of catching it. mio-1-1's H-H.skf carries ``Ep = 0.000039`` Hartree (~0.001 eV) as a
    # placeholder even though hydrogen is s-only, so ``Ep != 0.0`` gave hydrogen a phantom
    # p shell. An oracle that re-reads the file but reuses the production parser's decision
    # rule is an independent transcription, not an independent derivation.
    #
    # The fallback's own reasoning stays intact and is still needed for files that omit the
    # count (the f fixtures, mio-1-1's Zn-Zn): a shell exists if it carries either an onsite
    # energy or a reference occupation, and neither alone suffices. Ga's d shell has zero
    # occupation but a real onsite energy; lanthanum's f shell likewise carries an onsite
    # energy with no f electrons. That is why the codebase separates ``max_ang`` (highest
    # shell available) from ``max_ang_occ`` (highest shell actually occupied).
    if n_shells_declared is not None:
        if not 1 <= n_shells_declared <= 4:
            raise ValueError(
                f"{skf_path}: declared shell count {n_shells_declared} outside 1..4"
            )
        shell_present = [i < n_shells_declared for i in range(4)]
    else:
        shell_present = [
            Es != 0.0 or fs != 0.0,
            Ep != 0.0 or fp != 0.0,
            Ed != 0.0 or fd != 0.0,
            Ef != 0.0 or ff != 0.0,
        ]
    shell_occ = [fs, fp, fd, ff]
    n_orb, max_ang, max_ang_occ = expected_shell_metadata(shell_present, shell_occ)

    return {
        "element": elem_a,
        "Z": element_number(elem_a),
        "EXTENDED": extended,
        "HEADER_VALUE_COUNT": header_value_count,
        "N_ORB": n_orb,
        "MAX_ANG": max_ang,
        "MAX_ANG_OCC": max_ang_occ,
        "TORE": fs + fp + fd + ff,
        "N_S": fs,
        "N_P": fp,
        "N_D": fd,
        "N_F": ff,
        "ES": Es * EV_PER_HARTREE,
        "EP": Ep * EV_PER_HARTREE,
        "ED": Ed * EV_PER_HARTREE,
        "EF": Ef * EV_PER_HARTREE,
        "US": Us * EV_PER_HARTREE,
        "UP": Up * EV_PER_HARTREE,
        "UD": Ud * EV_PER_HARTREE,
        "UF": Uf * EV_PER_HARTREE,
        "SHELL_PRESENT": shell_present,
    }


def original_electronic_row_count(skf_path: Path) -> int:
    """Return the number of electronic table rows actually present in the file.

    The SKF header gives nGridPoints.  The production parser reads
    nGridPoints - 1 electronic rows and then appends one zero cutoff row of its
    own.  This counts only the rows the file really carries.
    """
    data_lines = read_data_lines(skf_path)
    if not data_lines:
        raise ValueError(f"Empty SKF file: {skf_path}")

    extended = data_lines[0].startswith("@")
    grid_idx = 1 if extended else 0

    first = data_lines[grid_idx].replace(",", " ").split()
    if len(first) < 2:
        raise ValueError(f"Malformed grid line in {skf_path}: {data_lines[grid_idx]}")

    npts_read = int(first[1])
    return npts_read - 1


# --- AO layout derivations, built from the EXPECTED_* tables above -----------


def expected_local_starts(shell_present: list[bool]) -> list[int]:
    return [
        start if present else -1
        for start, present in zip(EXPECTED_SHELL_LOCAL_STARTS, shell_present)
    ]


def expected_local_ends(shell_present: list[bool]) -> list[int]:
    return [
        start + dim - 1 if present else -1
        for start, dim, present in zip(
            EXPECTED_SHELL_LOCAL_STARTS, EXPECTED_SHELL_DIMS, shell_present
        )
    ]


def expected_ao_labels(shell_present: list[bool]) -> list[str]:
    labels: list[str] = []
    cursor = 0
    for present, dim in zip(shell_present, EXPECTED_SHELL_DIMS):
        if present:
            labels.extend(EXPECTED_AO_LABEL_TEMPLATE[cursor : cursor + dim])
        cursor += dim
    return labels


def expected_ao_shell_types(shell_present: list[bool]) -> list[int]:
    shell_types: list[int] = []
    cursor = 0
    for present, dim in zip(shell_present, EXPECTED_SHELL_DIMS):
        if present:
            shell_types.extend(EXPECTED_AO_SHELL_TEMPLATE[cursor : cursor + dim])
        cursor += dim
    return shell_types


def expected_diagonal_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    energies = [float(md["ES"]), float(md["EP"]), float(md["ED"]), float(md["EF"])]
    vals: list[float] = []
    for present, energy, dim in zip(shell_present, energies, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([energy] * dim)
    return vals


def expected_d0_values(md: dict[str, object]) -> list[float]:
    """Expected Structure.D0 values after the 0.5 closed-shell factor."""
    shell_present = list(md["SHELL_PRESENT"])
    occs = [float(md["N_S"]), float(md["N_P"]), float(md["N_D"]), float(md["N_F"])]
    vals: list[float] = []
    for present, occ, dim in zip(shell_present, occs, EXPECTED_SHELL_DIMS):
        if present:
            vals.extend([0.5 * occ / float(dim)] * dim)
    return vals


def expected_shell_hubbard_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    vals = [float(md["US"]), float(md["UP"]), float(md["UD"]), float(md["UF"])]
    return [v for present, v in zip(shell_present, vals) if present]


def expected_el_per_shell_values(md: dict[str, object]) -> list[float]:
    shell_present = list(md["SHELL_PRESENT"])
    vals = [float(md["N_S"]), float(md["N_P"]), float(md["N_D"]), float(md["N_F"])]
    return [v for present, v in zip(shell_present, vals) if present]


def expected_shell_type_values(md: dict[str, object]) -> list[int]:
    shell_present = list(md["SHELL_PRESENT"])
    return [
        sid for present, sid in zip(shell_present, EXPECTED_SHELL_TYPE_IDS) if present
    ]
