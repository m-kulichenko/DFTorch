"""SKF metadata checked against an independent re-parse of the same file.

WHY THIS MODULE EXISTS
----------------------
Every other SKF metadata assertion in this suite obtains its expected values
through `dftorch._bond_integral`.  That makes the production parser its own
oracle: if `read_skf_table` mis-reads the extended header, the value under test
and the value it is compared against move together and the test still passes.

`src/dftorch/script.py` was the only code in this repository without that flaw,
and decision D-03 in
`.planning/phases/05-regression-safety-and-support-policy-cleanup/05-CONTEXT.md`
names that single property as the reason the file was ported rather than simply
deleted.  The independent parse now lives in `tests/skf_header_oracle.py`, which
imports nothing from `dftorch` at all; this module drives it against the real
loader.

WHAT MAKES THIS DIFFERENT FROM A NORMAL REGRESSION TEST
-------------------------------------------------------
A regression test pins today's output so tomorrow's change has to justify
itself.  These tests do something stronger: they compute what the answer OUGHT
to be from the file's raw bytes, by a second route, and demand the loader agree.
A regression pin cannot tell you the pinned number was wrong on the day it was
recorded.  This can.

`test_oracle_does_not_use_production_parser` is what keeps that claim honest
over time.  It is not a comment asking future maintainers to be careful; it
reads `tests/skf_header_oracle.py` and fails if any production parsing symbol
appears in its executable code.  Without it, one convenient
`from dftorch._bond_integral import _split_skf_pair_name` would quietly turn
this whole module back into a tautology while every test stayed green.

ASCII ONLY.  Phase 4 recorded that non-ASCII characters render as replacement
characters on a cp1252 console and destroy the diagnosability of pytest output.
"""

import os

# Keep TorchDynamo/Inductor out of the tests, matching tests/test_scf.py and
# tests/test_radial_grid.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import skf_header_oracle as oracle


# --- Harness -----------------------------------------------------------------


def _purge_dftorch_modules() -> None:
    for name in [
        name
        for name in sys.modules
        if name == "dftorch" or name.startswith("dftorch.")
    ]:
        sys.modules.pop(name, None)


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults with a freshly imported dftorch.

    Copied from `tests/test_f_orbital_skf.py`, following this suite's convention
    that each module carries its own harness rather than importing one.  The
    purge on entry is what makes a plain `importlib` import inside ``fn`` pick
    up float64 module-level state regardless of which test module ran first.
    """
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    _purge_dftorch_modules()
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        _purge_dftorch_modules()
        sys.modules.update(previous_modules)


# --- Fixture locations -------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _f_dir() -> Path:
    return _repo_root() / "tests" / "f_orbital_data"


def _mio_dir() -> Path:
    return _repo_root() / "tests" / "data_skf_mio-1-1"


# Every homonuclear SKF file in both fixture directories, as
# (directory label, directory, element).  `f_orbital_data` carries the extended
# 13-value header format, `data_skf_mio-1-1` the simple 10-value one; both
# shapes have to round-trip or the oracle is only checking half the problem.
def _homonuclear_cases() -> list[tuple[str, Path, str]]:
    cases: list[tuple[str, Path, str]] = []
    for label, directory in (("f_orbital_data", _f_dir()), ("mio-1-1", _mio_dir())):
        for element in oracle.collect_elements_independently(directory):
            path = oracle.resolve_homonuclear_skf_independently(directory, element)
            if path.is_file():
                cases.append((label, directory, element))
    return cases


HOMONUCLEAR_CASES = _homonuclear_cases()
HOMONUCLEAR_IDS = [f"{label}-{element}" for label, _dir, element in HOMONUCLEAR_CASES]


def _write_single_element_xyz(path: Path, element: str) -> None:
    """One atom at the origin.  Constants only needs the species list."""
    path.write_text(f"1\noracle metadata check\n{element} 0.00000000 0.00000000 0.00000000\n")


def _build_constants(skf_dir: Path, xyz_path: Path):
    """Build the production Constants object whose metadata is under test."""
    from dftorch.Constants import Constants

    return Constants(
        {
            "SKFPATH": str(skf_dir),
            "FILENAME": str(xyz_path),
            "DFTB3": False,
            "MAGNETIC_HUBBARD_LDEP": False,
            "GRAD_PARAM": False,
        }
    )


def _constants_for(skf_dir: Path, element: str, tmp_path: Path):
    xyz_path = tmp_path / f"{element}_oracle.xyz"
    _write_single_element_xyz(xyz_path, element)
    return _build_constants(skf_dir, xyz_path)


def _oracle_metadata(skf_dir: Path, element: str) -> dict:
    path = oracle.resolve_homonuclear_skf_independently(skf_dir, element)
    metadata = oracle.parse_expected_homonuclear_metadata(path)
    assert metadata is not None, f"{path.name} did not parse as homonuclear"
    return metadata


def _compare_float_attrs(const, metadata: dict, attr_to_key: dict, element: str) -> None:
    Z = int(metadata["Z"])
    for attr, key in attr_to_key.items():
        got = float(getattr(const, attr)[Z].item())
        expected = float(metadata[key])
        assert abs(got - expected) <= oracle.ATOL, (
            f"{element}: Constants.{attr}[{Z}] is {got!r}, the independent "
            f"header parse says {expected!r} (delta {got - expected!r})"
        )


# --- The independence guarantee ----------------------------------------------
#
# Every name a ported oracle helper is forbidden to mention in executable code.
# Kept in one module-level tuple so the names appear exactly once in this file
# and so adding a newly discovered production entry point is a one-line change.
FORBIDDEN_PRODUCTION_SYMBOLS = (
    "dftorch",
    "get_skf_tensors",
    "read_skf_table",
    "channels_to_matrix",
    "cubic_spline_coeffs",
    "symbol_to_number",
    "_split_skf_pair_name",
    "_resolve_skf_path",
    "_validate_nested_shells",
)

ORACLE_HELPERS = (
    oracle.read_data_lines,
    oracle.split_skf_pair_name_independently,
    oracle.expected_shell_metadata,
    oracle.parse_expected_homonuclear_metadata,
    oracle.resolve_homonuclear_skf_independently,
    oracle.original_electronic_row_count,
)


def _code_without_docstring(func) -> str:
    """Return ``func``'s body as code text, with its docstring and comments gone.

    Round-tripping through the AST is what removes them.  A plain substring scan
    over `inspect.getsource` would flag the docstrings, which deliberately NAME
    the production functions they refuse to call, and the test would be
    unfalsifiable in the wrong direction: it would fail on honest prose and pass
    on a real violation hidden behind an alias.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    definition = tree.body[0]
    body = definition.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return "\n".join(ast.unparse(node) for node in body)


def test_oracle_does_not_use_production_parser():
    """The independent parse must be unable to reach the production parser.

    Three checks, weakest to strongest:

    1. No helper's executable code mentions a production parsing symbol.
       This is the check decision D-03 asks for by name.
    2. `tests/skf_header_oracle.py` contains no import of dftorch, at module
       level or inside any function.  A helper cannot call what its module never
       imports, which closes the alias loophole check 1 alone would leave open.
    3. No identifier anywhere in that module's executable code is a forbidden
       name, which catches an indirection through a second local helper.
    """
    oracle_source = Path(oracle.__file__).read_text(encoding="ascii")
    tree = ast.parse(oracle_source)

    # 1. Per-helper source inspection.
    for helper in ORACLE_HELPERS:
        code = _code_without_docstring(helper)
        for symbol in FORBIDDEN_PRODUCTION_SYMBOLS:
            assert symbol not in code, (
                f"{helper.__name__} reaches its expected values through "
                f"{symbol!r}.  The independent-oracle property (D-03) is the "
                f"only reason these checks were kept when src/dftorch/script.py "
                f"was deleted; building expectations with the code under test "
                f"destroys it while leaving every test green."
            )

    # 2. No dftorch import anywhere in the oracle module.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "dftorch", (
                    f"skf_header_oracle imports {alias.name}; it must import "
                    f"nothing from the package it exists to disagree with"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.split(".")[0] == "dftorch", (
                f"skf_header_oracle imports from {module}; it must import "
                f"nothing from the package it exists to disagree with"
            )

    # 3. No forbidden identifier anywhere in the module's executable code.
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    offenders = sorted(identifiers.intersection(FORBIDDEN_PRODUCTION_SYMBOLS))
    assert not offenders, (
        f"skf_header_oracle names production parsing symbols in its code: {offenders}"
    )


# --- Header parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "label,skf_dir,element", HOMONUCLEAR_CASES, ids=HOMONUCLEAR_IDS
)
def test_oracle_parses_both_header_formats(label, skf_dir, element):
    """Both header shapes parse, and each to its own width.

    The extended format (first data line starts with ``@``) carries 13 values
    because it adds Ef, Uf and ff for the f shell.  The simple format carries
    10 and has no f block at all.  Asserting the COUNT, not merely that parsing
    succeeded, is what catches an extended file being read with the simple
    layout: that misalignment silently assigns Ef's value to Ed and still
    produces a well-formed dict.
    """
    metadata = _oracle_metadata(skf_dir, element)
    extended = bool(metadata["EXTENDED"])

    assert metadata["HEADER_VALUE_COUNT"] == (13 if extended else 10), (
        f"{element} in {label}: extended={extended} but the header carried "
        f"{metadata['HEADER_VALUE_COUNT']} values"
    )
    assert metadata["element"] == element
    assert metadata["N_ORB"] in (1, 4, 9, 16), (
        f"{element}: N_ORB {metadata['N_ORB']} is not a nested s/p/d/f basis size"
    )

    if not extended:
        # The simple format has no f block, so the f shell must be absent and
        # its onsite energy, Hubbard value and occupation must all be zero.
        assert metadata["SHELL_PRESENT"][3] is False
        assert metadata["EF"] == 0.0
        assert metadata["UF"] == 0.0
        assert metadata["N_F"] == 0.0


def test_both_header_formats_are_actually_exercised():
    """Both header widths really occur in the fixtures, and it is not by directory.

    Measured 2026-07-31: `tests/f_orbital_data` is NOT uniformly extended.  Only
    `Eu-Eu.skf` carries the ``@`` marker and the 13-value header; `N-N.skf` and
    `Ga-Ga.skf` are ordinary 10-value simple-format files sitting in the same
    directory.  That is worth pinning rather than assuming, because it means the
    loader has to switch layouts per FILE, not per directory, and a test that
    inferred the format from the directory would pass while testing nothing.
    """
    widths = {}
    for label, skf_dir, element in HOMONUCLEAR_CASES:
        metadata = _oracle_metadata(skf_dir, element)
        widths[(label, element)] = int(metadata["HEADER_VALUE_COUNT"])

    assert widths[("f_orbital_data", "Eu")] == 13
    assert widths[("f_orbital_data", "N")] == 10
    assert widths[("f_orbital_data", "Ga")] == 10
    assert widths[("mio-1-1", "C")] == 10

    assert set(widths.values()) == {10, 13}, (
        f"fixture drift: header widths present are {sorted(set(widths.values()))}, "
        f"so one of the two formats is no longer being exercised"
    )


def test_heteronuclear_files_have_no_element_metadata():
    """Heteronuclear SKF files carry no element block, and the oracle says so.

    A parser that invented metadata for, say, Eu-N would associate Eu's Z with
    numbers read from the wrong line.  Returning None is the behaviour every
    caller in this suite branches on.
    """
    checked = 0
    for skf_path in sorted(_f_dir().glob("*.skf")):
        elem_a, elem_b = oracle.split_skf_pair_name_independently(skf_path.stem)
        if elem_a == elem_b:
            continue
        assert oracle.parse_expected_homonuclear_metadata(skf_path) is None, (
            f"{skf_path.name} is heteronuclear but produced element metadata"
        )
        checked += 1
    assert checked >= 6, f"expected the six heteronuclear f fixtures, saw {checked}"


# --- Constants against the independent parse ---------------------------------


@pytest.mark.parametrize(
    "label,skf_dir,element", HOMONUCLEAR_CASES, ids=HOMONUCLEAR_IDS
)
def test_onsite_energies_match_independent_parse(label, skf_dir, element, tmp_path):
    """Constants' onsite energies equal the header values converted to eV."""

    def check():
        const = _constants_for(skf_dir, element, tmp_path)
        metadata = _oracle_metadata(skf_dir, element)
        _compare_float_attrs(
            const,
            metadata,
            {"Es": "ES", "Ep": "EP", "Ed": "ED", "Ef": "EF"},
            element,
        )

    run_with_float64(check)


@pytest.mark.parametrize(
    "label,skf_dir,element", HOMONUCLEAR_CASES, ids=HOMONUCLEAR_IDS
)
def test_hubbard_values_match_independent_parse(label, skf_dir, element, tmp_path):
    """Constants' Hubbard U values equal the header values converted to eV.

    Note the attribute naming asymmetry the production code carries: the s-shell
    Hubbard value is `const.U`, not `const.Us`, while p/d/f follow the expected
    pattern.  This test is the only place that pins that mapping against the
    header token it is supposed to come from.
    """

    def check():
        const = _constants_for(skf_dir, element, tmp_path)
        metadata = _oracle_metadata(skf_dir, element)
        _compare_float_attrs(
            const,
            metadata,
            {"U": "US", "Up": "UP", "Ud": "UD", "Uf": "UF"},
            element,
        )

    run_with_float64(check)


@pytest.mark.parametrize(
    "label,skf_dir,element", HOMONUCLEAR_CASES, ids=HOMONUCLEAR_IDS
)
def test_reference_occupations_match_independent_parse(
    label, skf_dir, element, tmp_path
):
    """Constants' reference occupations equal the header occupations, and sum to TORE."""

    def check():
        const = _constants_for(skf_dir, element, tmp_path)
        metadata = _oracle_metadata(skf_dir, element)
        _compare_float_attrs(
            const,
            metadata,
            {"n_s": "N_S", "n_p": "N_P", "n_d": "N_D", "n_f": "N_F", "tore": "TORE"},
            element,
        )

    run_with_float64(check)


@pytest.mark.parametrize(
    "label,skf_dir,element", HOMONUCLEAR_CASES, ids=HOMONUCLEAR_IDS
)
def test_shell_metadata_matches_independent_derivation(
    label, skf_dir, element, tmp_path
):
    """n_orb, max_ang and max_ang_occ follow from shell presence and occupation alone.

    `expected_shell_metadata` derives them by counting 2l+1 over the present
    shells, with no reference to the loader's own bookkeeping.  It is
    re-evaluated here from the parsed presence/occupation rather than read back
    out of the metadata dict, so the derivation itself is exercised and not just
    its cached result.
    """

    def check():
        const = _constants_for(skf_dir, element, tmp_path)
        metadata = _oracle_metadata(skf_dir, element)
        Z = int(metadata["Z"])

        shell_present = list(metadata["SHELL_PRESENT"])
        shell_occ = [
            float(metadata["N_S"]),
            float(metadata["N_P"]),
            float(metadata["N_D"]),
            float(metadata["N_F"]),
        ]
        n_orb, max_ang, max_ang_occ = oracle.expected_shell_metadata(
            shell_present, shell_occ
        )

        assert int(const.n_orb[Z].item()) == n_orb, element
        assert int(const.max_ang[Z].item()) == max_ang, element
        assert int(const.max_ang_occ[Z].item()) == max_ang_occ, element

        got_present = [bool(x) for x in const.shell_present[Z].detach().cpu().tolist()]
        assert got_present == shell_present, element

        # The presence flags and the orbital count have to agree with each other,
        # which rules out a loader that gets both wrong in the same direction.
        assert n_orb == sum(
            2 * l + 1 for l, present in enumerate(shell_present) if present
        )

    run_with_float64(check)


# --- Basis layout tables against a second written source ---------------------


def test_basis_tables_match_expected_tables():
    """`dftorch.Structure`'s AO layout tables equal the copies in the oracle module.

    The point is that there are TWO written sources.  `Structure.py` defines the
    local AO layout once; `skf_header_oracle.py` writes it out again by hand.
    A single-sourced constant cannot detect a one-sided edit, so if someone
    renumbers the f block in `Structure.py` -- or renames `fx_y2_z2` -- this
    test fails and the change has to be made deliberately in both places.

    That matters here more than it usually would: the f AO ordering is the
    subject of the Phase 3 source lock
    (`03-SOURCE-LOCK.md`), and a silent reordering would leave every f angular
    block subtly wrong while the matrices stayed the right shape.
    """

    def check():
        # `dftorch/__init__.py` re-exports the Structure CLASS under the name
        # `Structure`, which shadows the module of the same name, so the module
        # has to be reached explicitly.
        import importlib

        structure_mod = importlib.import_module("dftorch.Structure")

        assert list(structure_mod.SHELL_DIMS) == oracle.EXPECTED_SHELL_DIMS
        assert (
            list(structure_mod.SHELL_LOCAL_STARTS) == oracle.EXPECTED_SHELL_LOCAL_STARTS
        )
        assert list(structure_mod.SHELL_TYPE_IDS) == oracle.EXPECTED_SHELL_TYPE_IDS

        production_labels = list(structure_mod.AO_LABEL_TEMPLATE)
        assert len(production_labels) == len(oracle.EXPECTED_AO_LABEL_TEMPLATE)
        for index, (got, expected) in enumerate(
            zip(production_labels, oracle.EXPECTED_AO_LABEL_TEMPLATE)
        ):
            assert got == expected, (
                f"AO_LABEL_TEMPLATE[{index}] is {got!r}, the independent table "
                f"says {expected!r}"
            )

        production_shells = list(structure_mod.AO_SHELL_TEMPLATE)
        assert len(production_shells) == len(oracle.EXPECTED_AO_SHELL_TEMPLATE)
        for index, (got, expected) in enumerate(
            zip(production_shells, oracle.EXPECTED_AO_SHELL_TEMPLATE)
        ):
            assert got == expected, (
                f"AO_SHELL_TEMPLATE[{index}] is {got!r}, the independent table "
                f"says {expected!r}"
            )

        # The local starts must be the running sum of the dims, which is the
        # arithmetic relationship both tables encode separately.
        cursor = 0
        for dim, start in zip(
            oracle.EXPECTED_SHELL_DIMS, oracle.EXPECTED_SHELL_LOCAL_STARTS
        ):
            assert start == cursor
            cursor += dim
        assert cursor == len(oracle.EXPECTED_AO_LABEL_TEMPLATE) == 16

    run_with_float64(check)


# --- Structure AO layout -----------------------------------------------------


EU_N_ELEMENTS = ["Eu", "N"]


def _write_eu_n_xyz(path: Path) -> None:
    """Eu at the origin, N displaced along +x at the Phase 4 validation separation."""
    path.write_text(
        "2\noracle AO layout check\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        "N 2.65500000 0.00000000 0.00000000\n"
    )


def _f_structure_params(xyz_path: Path) -> dict:
    return {
        "SKFPATH": str(_f_dir()),
        "FILENAME": str(xyz_path),
        "T_ELECTRONIC": 1000.0,
        "CHARGE": 0,
        "GRAD_XYZ": False,
        "GRAD_CELL": False,
    }


def _expected_eu_n_layout() -> dict:
    """Derive the whole Eu-N AO layout from the oracle tables and shell presence."""
    metadata = {sym: _oracle_metadata(_f_dir(), sym) for sym in EU_N_ELEMENTS}

    n_orb = [int(metadata[sym]["N_ORB"]) for sym in EU_N_ELEMENTS]
    h_start = [0]
    for count in n_orb[:-1]:
        h_start.append(h_start[-1] + count)

    labels: list[str] = []
    shell_types: list[int] = []
    local_start: list[list[int]] = []
    local_end: list[list[int]] = []
    ao_start: list[list[int]] = []
    ao_end: list[list[int]] = []
    for index, sym in enumerate(EU_N_ELEMENTS):
        present = list(metadata[sym]["SHELL_PRESENT"])
        labels.extend(oracle.expected_ao_labels(present))
        shell_types.extend(oracle.expected_ao_shell_types(present))
        starts = oracle.expected_local_starts(present)
        ends = oracle.expected_local_ends(present)
        local_start.append(starts)
        local_end.append(ends)
        ao_start.append([s + h_start[index] if s >= 0 else -1 for s in starts])
        ao_end.append([e + h_start[index] if e >= 0 else -1 for e in ends])

    return {
        "metadata": metadata,
        "n_orb": n_orb,
        "h_start": h_start,
        "hdim": sum(n_orb),
        "labels": labels,
        "shell_types": shell_types,
        "local_start": local_start,
        "local_end": local_end,
        "ao_start": ao_start,
        "ao_end": ao_end,
    }


def test_structure_ao_layout_matches_expected_derivation(tmp_path):
    """Structure's Eu-N AO layout equals what the oracle tables derive for it.

    Eu is the only spdf element in the fixtures, so this is the case where the
    f block's seven orbitals actually have to be placed: Eu contributes 16 AOs
    with its f shell starting at local index 9, and N's four AOs start after all
    sixteen.  An off-by-one in the f offset would put N's s orbital inside Eu's
    f block, which is exactly the class of silent misplacement Phases 3 and 4
    kept finding.
    """

    def check():
        from dftorch.Structure import Structure

        xyz_path = tmp_path / "eu_n_layout.xyz"
        _write_eu_n_xyz(xyz_path)
        params = _f_structure_params(xyz_path)
        const = _build_constants(_f_dir(), xyz_path)
        struct = Structure(params, const, device="cpu", ignore_spin=True)

        expected = _expected_eu_n_layout()

        assert expected["n_orb"] == [16, 4], (
            f"fixture drift: Eu/N basis sizes are {expected['n_orb']}, so this "
            f"test is no longer exercising an f-containing layout"
        )
        assert int(struct.HDIM) == expected["hdim"]
        assert struct.n_orbitals_per_atom.tolist() == expected["n_orb"]
        assert struct.H_INDEX_START.tolist() == expected["h_start"]

        assert list(struct.ao_labels) == expected["labels"]
        assert struct.ao_shell_types.tolist() == expected["shell_types"]
        assert struct.shell_local_start.tolist() == expected["local_start"]
        assert struct.shell_local_end.tolist() == expected["local_end"]
        assert struct.shell_ao_start.tolist() == expected["ao_start"]
        assert struct.shell_ao_end.tolist() == expected["ao_end"]

        # The f shell really is present and really is placed at local 9.
        assert expected["local_start"][0][3] == 9
        assert expected["local_end"][0][3] == 15
        assert struct.ao_labels[9:16] == oracle.EXPECTED_AO_LABEL_TEMPLATE[9:16]

    run_with_float64(check)


def test_batch_structure_layout_matches_single_structure(tmp_path):
    """StructureBatch's per-member AO layout equals the single-Structure layout.

    This is the one check ported from `script.py` that had no counterpart
    anywhere in `tests/`: no test file exercised `StructureBatch` at all before
    this one.  The batched path pads every member to the widest row, so the risk
    it carries is that padding leaks into the real AO region or that per-member
    offsets are computed against the padded width instead of the member's own.

    Two identical Eu-N copies are used so that any difference between the rows,
    or between a row and the single-Structure result, is a bug rather than a
    property of the input.  `StructureBatch` accepts the f-containing system, so
    no narrowing to an f-free case was needed (the plan allowed for one).
    """

    def check():
        from dftorch.Structure import Structure, StructureBatch

        xyz_a = tmp_path / "eu_n_batch_a.xyz"
        xyz_b = tmp_path / "eu_n_batch_b.xyz"
        _write_eu_n_xyz(xyz_a)
        _write_eu_n_xyz(xyz_b)

        const = _build_constants(_f_dir(), xyz_a)
        single = Structure(
            _f_structure_params(xyz_a), const, device="cpu", ignore_spin=True
        )

        batch_params = _f_structure_params(xyz_a)
        batch_params["FILENAME"] = [str(xyz_a), str(xyz_b)]
        batch = StructureBatch(batch_params, const, device="cpu", ignore_spin=True)

        expected = _expected_eu_n_layout()

        assert int(batch.batch_size) == 2
        assert batch.HDIM_struct.tolist() == [expected["hdim"], expected["hdim"]]

        for member in range(2):
            assert list(batch.ao_labels[member]) == expected["labels"], member
            assert list(batch.ao_labels[member]) == list(single.ao_labels), member
            assert batch.H_INDEX_START[member].tolist() == expected["h_start"], member
            assert (
                batch.H_INDEX_START[member].tolist()
                == single.H_INDEX_START.tolist()
            ), member
            assert (
                batch.n_orbitals_per_atom[member].tolist() == expected["n_orb"]
            ), member
            assert batch.shell_ao_start[member].tolist() == expected["ao_start"], member
            assert batch.shell_ao_end[member].tolist() == expected["ao_end"], member
            assert (
                batch.shell_ao_start[member].tolist()
                == single.shell_ao_start.tolist()
            ), member

            # The onsite diagonal over the real AO region matches the single
            # structure, so padding has not displaced anything.
            hdim = expected["hdim"]
            row = batch.diagonal[member, :hdim].detach().cpu().tolist()
            for index, (got, want) in enumerate(
                zip(row, single.diagonal.detach().cpu().tolist())
            ):
                assert abs(got - want) <= oracle.ATOL, (member, index, got, want)

    run_with_float64(check)
