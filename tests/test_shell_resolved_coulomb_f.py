"""All sixteen angular blocks of the shell-resolved Coulomb matrix are built.

Plain-language orientation
--------------------------
Electrons repel each other.  This code models that with a matrix whose entries
say how strongly a unit of charge sitting on one atom pushes on a unit of charge
sitting on another.  There are two resolutions available.  The *per-atom*
version tracks one number per atom.  The *shell-resolved* version tracks one
number per orbital group per atom - separately for the s group, the p group, the
d group and the f group.  "Shell" here means one of those groups.

``_coulomb_matrix.ewald_real_space_vectorized_sr`` builds the shell-resolved
one.  It writes one block per ordered pair of orbital groups, so a complete
matrix needs sixteen: s-s, s-p, s-d, s-f, p-s, ... , f-f.  Before requirement
SCC-02 (Phase 6) it wrote nine, and the seven involving f were absent.  Worse,
the six existing *off-diagonal* blocks selected their pairs by testing each
atom's ``max_ang`` against the values 1, 2 and 3 only, so a 16-orbital f atom
(``max_ang == 4``) matched none of them and fell straight through.  The recorded
evidence is in ``tests/test_shell_resolved_u.py``: an Eu-N molecule came back
with row sums ``[0.473581, 0, 0, 0, 0.473581, 0]`` - finite, correctly shaped,
and zero in every row except the two s rows, because only the s-s block (which
had no mask at all) had filled.  Zero is a legal-looking repulsion value, which
is why that defect was invisible to a shape or finiteness check.

Why it matters numerically: the per-atom matrix charges every element the
repulsion strength (the Hubbard U) of its *s* group unconditionally.  For
europium that is 0.21 Ha while its f group is 0.50 Ha - 2.4 times larger - and
seven of europium's nine outer electrons live in the f group.

How this module validates, and what it deliberately does not do
---------------------------------------------------------------
**No entry of the matrix is written down here as a reference number.**  The main
correctness gate is a *derived identity*: if every orbital group of every element
is given the same repulsion strength, then all sixteen blocks collapse to one and
the same expression, so the shell-resolved matrix must reproduce the per-atom
matrix entry for entry.  That identity follows from the formula rather than from
a recorded run, so it validates the seven new blocks, their masks, their row and
column offsets and their strength selection all at once - and it stays true if
the parameter files ever change.

The supporting gates are non-emptiness (a block that never matches is silently
zero), symmetry (the short-range formula is unchanged when the two atoms'
strengths are swapped), and an f-free control on methane.

ASCII only, per the Phase 4 rule: a failure must be diagnosable from pytest
output on a cp1252 console.  "A" means Angstrom and "->" means an arrow.

Covers SCC-02.
"""

import math
import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic
# and avoids requiring a C++ toolchain).  Mirrors tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import sys
from pathlib import Path

import torch


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.

    Copied from ``tests/test_shell_resolved_u.py`` - the established harness for
    every f-orbital test module in this project.  Copied rather than imported
    across test modules, matching the existing convention.
    """
    previous_dtype = torch.get_default_dtype()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "dftorch" or name.startswith("dftorch.")
    }
    torch.set_default_dtype(torch.float64)
    try:
        return fn()
    finally:
        torch.set_default_dtype(previous_dtype)
        for name in [
            name
            for name in sys.modules
            if name == "dftorch" or name.startswith("dftorch.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


TESTS_DIR = Path(__file__).resolve().parent

#: Orbital-group names in the fixed order the builder uses.  The offset of a
#: group inside an atom's block of shell rows is its index here, which is why
#: ``test_orbital_groups_are_a_contiguous_run_from_s`` exists.
SHELL_NAMES = ("s", "p", "d", "f")

#: The seven ordered group pairs that involve f - the blocks SCC-02 adds.
F_BLOCKS = (
    (0, 3),  # s-f
    (3, 0),  # f-s
    (1, 3),  # p-f
    (3, 1),  # f-p
    (2, 3),  # d-f
    (3, 2),  # f-d
    (3, 3),  # f-f
)

#: Eu-Eu at 3.0 A.  This is the only two-atom arrangement in the fixture set
#: that reaches all seven f blocks: f-f, d-f and f-d each need an f group on
#: *both* sides of the pair, and nitrogen has none.
EU_EU_SEPARATION = 3.0

#: Eu-N at the Phase 4 target separation (decision D-24).
EU_N_SEPARATION = 2.655

BASE_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}

#: The Coulomb real-space cutoff both builders are driven with.
COULCUT = 10.0


def _skf_dir() -> Path:
    return TESTS_DIR / "f_orbital_data"


def _mio_skf_dir() -> Path:
    return TESTS_DIR / "data_skf_mio-1-1"


def _ch4_xyz() -> Path:
    return TESTS_DIR / "ch4.xyz"


def _elements_in(skf_dir: Path):
    """Every element symbol a fixture directory carries a homonuclear file for.

    Discovered from the ``A-B.skf`` file names rather than hardcoded, so an
    element added to a fixture set is checked without anyone remembering to
    extend a list here.
    """
    symbols = sorted(
        {path.name.split("-")[0] for path in skf_dir.glob("*.skf")}
    )
    return [s for s in symbols if (skf_dir / f"{s}-{s}.skf").exists()]


def _write_xyz(path: Path, elements, spacing: float = 1.5) -> None:
    """Write atoms evenly spaced along x.  Copied from test_shell_resolved_u.py."""
    lines = [str(len(elements)), "shell-resolved Coulomb fixture"]
    for idx, symbol in enumerate(elements):
        lines.append(f"{symbol} {spacing * idx:.8f} 0.00000000 0.00000000")
    path.write_text("\n".join(lines) + "\n")


def _write_eu_eu_xyz(tmp_path: Path) -> Path:
    """The Eu-Eu pair: the only fixture arrangement reaching all seven f blocks."""
    xyz_path = tmp_path / "eu_eu.xyz"
    _write_xyz(xyz_path, ["Eu", "Eu"], spacing=EU_EU_SEPARATION)
    return xyz_path


def _build(xyz_path: Path, skf_dir: Path, magnetic_hubbard_ldep: bool = True):
    """Build ``(const, structure)`` for a geometry against one SKF fixture set."""
    from dftorch.Constants import Constants
    from dftorch.Structure import Structure

    params = dict(BASE_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(skf_dir) + os.sep
    params["MAGNETIC_HUBBARD_LDEP"] = magnetic_hubbard_ldep

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    return const, structure


def _build_eu_eu(tmp_path: Path):
    return _build(_write_eu_eu_xyz(tmp_path), _skf_dir())


def _build_eu_n(tmp_path: Path):
    xyz_path = tmp_path / "eu_n.xyz"
    _write_xyz(xyz_path, ["Eu", "N"], spacing=EU_N_SEPARATION)
    return _build(xyz_path, _skf_dir())


def _build_ch4(tmp_path: Path):
    return _build(_ch4_xyz(), _mio_skf_dir())


def _neighbour_data(const, structure, coulcut: float = COULCUT):
    """Build the neighbour list and pair geometry both builders consume.

    Reproduces the argument derivation ``ESDriver.forward`` performs for the
    Coulomb matrices: a real-space neighbour list at the Coulomb cutoff, then
    ``dR`` / ``dR_dxyz`` from the neighbour coordinates.  Returned as one bundle
    so the shell-resolved and the per-atom builder can be driven from *exactly*
    the same pair list - the reduction identity means nothing otherwise.
    """
    from dftorch._nearestneighborlist import vectorized_nearestneighborlist

    coulomb_acc = 1e-5
    calpha = math.sqrt(-math.log(coulomb_acc)) / coulcut

    (
        _,
        _,
        nnRx,
        nnRy,
        nnRz,
        nnType,
        _,
        _,
        neighbor_I,
        neighbor_J,
        _,
        _,
    ) = vectorized_nearestneighborlist(
        structure.TYPE,
        structure.RX,
        structure.RY,
        structure.RZ,
        structure.cell,
        coulcut,
        structure.Nats,
        const,
        upper_tri_only=False,
        verbose=False,
    )

    Ra = torch.stack(
        (
            structure.RX.unsqueeze(-1),
            structure.RY.unsqueeze(-1),
            structure.RZ.unsqueeze(-1),
        ),
        dim=-1,
    )
    Rb = torch.stack((nnRx, nnRy, nnRz), dim=-1)
    Rab = Rb - Ra
    dR = torch.norm(Rab, dim=-1)
    dR_dxyz = Rab / dR.unsqueeze(-1).clamp(min=1e-30)

    return {
        "dR": dR,
        "dR_dxyz": dR_dxyz,
        "nnType": nnType,
        "neighbor_I": neighbor_I,
        "neighbor_J": neighbor_J,
        "calpha": calpha,
    }


def _call_shell_resolved_coulomb(const, structure, nbr=None):
    """Call ``ewald_real_space_vectorized_sr`` directly."""
    from dftorch._coulomb_matrix import ewald_real_space_vectorized_sr

    if nbr is None:
        nbr = _neighbour_data(const, structure)
    return ewald_real_space_vectorized_sr(
        structure,
        nbr["dR"],
        nbr["dR_dxyz"],
        structure.TYPE,
        nbr["nnType"],
        nbr["neighbor_I"],
        nbr["neighbor_J"],
        nbr["calpha"],
    )


def _call_per_atom_coulomb(structure, nbr):
    """Call the per-atom sibling builder on the *same* pair list.

    ``ewald_real_space_vectorized_eager`` is the uncompiled alias, so the two
    builders are compared as plain Python rather than through TorchDynamo.
    """
    from dftorch._coulomb_matrix import ewald_real_space_vectorized_eager

    return ewald_real_space_vectorized_eager(
        structure.Hubbard_U,
        structure.TYPE,
        nbr["dR"],
        nbr["dR_dxyz"],
        nbr["nnType"],
        nbr["neighbor_I"],
        nbr["neighbor_J"],
        nbr["calpha"],
        True,  # use_ewald: the shell-resolved builder always screens with erfc
    )


def _equal_shell_u_override(const) -> None:
    """Give every orbital group of every element its own s-group strength.

    This is what makes the reduction identity bite: with one strength shared by
    all four groups, every one of the sixteen blocks evaluates the same
    expression, so the shell-resolved matrix must reproduce the per-atom matrix
    exactly.  The override is done in memory on a freshly built ``Constants``,
    never on a file.
    """
    with torch.no_grad():
        const.Up.copy_(const.U)
        const.Ud.copy_(const.U)
        const.Uf.copy_(const.U)


def _shell_index(structure, atom: int, shell: int) -> int:
    """Row/column of one orbital group, addressed the way the builder writes it.

    ``H_INDEX_START_U[atom] + shell`` with shell 0/1/2/3 for s/p/d/f.  Never a
    hand-counted index: a hand-counted one would still "pass" if the builder
    wrote into the neighbouring atom's rows.
    """
    return int(structure.H_INDEX_START_U[atom]) + shell


# ---------------------------------------------------------------------------
# The seven new blocks
# ---------------------------------------------------------------------------


def test_eu_eu_matrix_has_no_empty_row_or_column(tmp_path):
    """Eu-Eu gets a complete matrix: no empty row, no empty column, all finite.

    This is the exact defect the retired refusal existed to prevent, now
    asserted to be gone from the other direction: an unfilled block leaves its
    row and its column at exactly zero inside a matrix that is otherwise finite
    and correctly shaped.

    The shape is derived from ``n_shells_per_atom`` rather than hardcoded.  A
    literal shape previously encoded a parser defect that gave hydrogen a
    phantom p shell (see ``tests/test_shell_count_parsing.py``).
    """

    def check():
        const, structure = _build_eu_eu(tmp_path)
        CC, dCC = _call_shell_resolved_coulomb(const, structure)

        n_sh = int(structure.n_shells_per_atom.sum())
        assert n_sh == 8, (
            f"Eu-Eu must have 8 shells total (4 per Eu: s, p, d, f), got {n_sh} "
            f"from n_shells_per_atom={structure.n_shells_per_atom.tolist()}"
        )
        assert CC.shape == (n_sh, n_sh), f"expected ({n_sh}, {n_sh}), got {CC.shape}"
        assert dCC.shape == (3, n_sh, n_sh), (
            f"expected (3, {n_sh}, {n_sh}), got {dCC.shape}"
        )
        assert bool(torch.isfinite(CC).all()), (
            f"non-finite entries in the Eu-Eu matrix: {CC}"
        )
        assert bool(torch.isfinite(dCC).all()), (
            "non-finite entries in the Eu-Eu matrix derivative"
        )

        row_sums = CC.sum(dim=1)
        col_sums = CC.sum(dim=0)
        assert not bool((row_sums == 0.0).any()), (
            f"an all-zero row means a silently unfilled block: row sums {row_sums}"
        )
        assert not bool((col_sums == 0.0).any()), (
            f"an all-zero column means a silently unfilled block: column sums "
            f"{col_sums}"
        )

    run_with_float64(check)


def test_all_seven_f_blocks_are_populated(tmp_path):
    """Each of s-f, f-s, p-f, f-p, d-f, f-d and f-f is addressed individually.

    Threat T-06-07: a block that is added but whose mask never matches is
    silently empty while the code reads as complete.  A row-sum check can be
    satisfied by a neighbouring block, so each of the seven positions is
    addressed on its own and named on failure.
    """

    def check():
        const, structure = _build_eu_eu(tmp_path)
        CC, _ = _call_shell_resolved_coulomb(const, structure)

        empty = []
        for shell_i, shell_j in F_BLOCKS:
            row = _shell_index(structure, 0, shell_i)
            col = _shell_index(structure, 1, shell_j)
            if float(CC[row, col]) == 0.0:
                empty.append(
                    f"{SHELL_NAMES[shell_i]}-{SHELL_NAMES[shell_j]} "
                    f"(CC[{row}, {col}])"
                )
        assert not empty, (
            "these f blocks came back exactly zero on an Eu-Eu pair, which means "
            "their mask never matched: " + ", ".join(empty)
        )

    run_with_float64(check)


def test_matrix_equals_its_own_transpose(tmp_path):
    """The finished matrix is symmetric, for Eu-Eu and for Eu-N.

    Justified by the algebra, not by a recorded run: in
    ``coul_diff_elem_and_ang`` swapping the two strength arguments swaps SB with
    SE and SC with SF, leaving the result unchanged, and the screened Coulomb
    term depends only on the separation.  So the entry for (group a on I, group
    b on J) must equal the entry for (group b on J, group a on I).  An asymmetry
    means a block wrote into the wrong row or column.
    """

    def check():
        for name, build in (
            ("Eu-Eu", _build_eu_eu),
            ("Eu-N", _build_eu_n),
        ):
            const, structure = build(tmp_path / name)
            CC, _ = _call_shell_resolved_coulomb(const, structure)
            worst = float((CC - CC.T).abs().max())
            assert worst < 1e-12, (
                f"{name}: the shell-resolved matrix is not its own transpose; "
                f"largest |CC - CC.T| is {worst:.3e}"
            )

    for name in ("Eu-Eu", "Eu-N"):
        (tmp_path / name).mkdir()
    run_with_float64(check)


def test_equal_group_strengths_reproduce_the_per_atom_matrix(tmp_path):
    """The main correctness gate, and it freezes no number.

    Give every orbital group of every element the repulsion strength of its s
    group.  Every one of the sixteen blocks then evaluates the same expression,
    which is exactly the expression the per-atom builder evaluates.  So every
    shell-resolved entry whose row group sits on atom I and whose column group
    sits on a *different* atom J must equal the per-atom entry for (I, J).

    That single identity covers the seven new blocks, their masks, their row and
    column offsets and their strength selection at once (threat T-06-09: an
    offset that lands on the neighbouring atom's shell produces a plausible but
    wrong matrix, and it cannot survive an entry-for-entry comparison).

    Entries with I == J are excluded because the neighbour list carries no
    self-pairs, so both builders leave them at zero.
    """

    def check():
        for name, build in (
            ("Eu-Eu", _build_eu_eu),
            ("Eu-N", _build_eu_n),
            ("CH4", _build_ch4),
        ):
            const, structure = build(tmp_path / name)
            _equal_shell_u_override(const)

            nbr = _neighbour_data(const, structure)
            CC_sr, _ = _call_shell_resolved_coulomb(const, structure, nbr)
            CC_atom, _ = _call_per_atom_coulomb(structure, nbr)

            n_shells = structure.n_shells_per_atom.tolist()
            worst = 0.0
            worst_where = ""
            for atom_i in range(structure.Nats):
                for atom_j in range(structure.Nats):
                    if atom_i == atom_j:
                        continue
                    expected = float(CC_atom[atom_i, atom_j])
                    for shell_i in range(n_shells[atom_i]):
                        for shell_j in range(n_shells[atom_j]):
                            row = _shell_index(structure, atom_i, shell_i)
                            col = _shell_index(structure, atom_j, shell_j)
                            gap = abs(float(CC_sr[row, col]) - expected)
                            if gap > worst:
                                worst = gap
                                worst_where = (
                                    f"block {SHELL_NAMES[shell_i]}-"
                                    f"{SHELL_NAMES[shell_j]} for atoms "
                                    f"({atom_i}, {atom_j}): shell-resolved "
                                    f"{float(CC_sr[row, col])!r} against per-atom "
                                    f"{expected!r}"
                                )
            assert worst < 1e-10, (
                f"{name}: with every orbital group carrying its element's s "
                f"strength the shell-resolved matrix must reproduce the per-atom "
                f"matrix, but it differs by {worst:.3e} at {worst_where}"
            )

    for name in ("Eu-Eu", "Eu-N", "CH4"):
        (tmp_path / name).mkdir()
    run_with_float64(check)


# ---------------------------------------------------------------------------
# The structural assumption the row and column offsets rest on
# ---------------------------------------------------------------------------


def test_orbital_groups_are_a_contiguous_run_from_s(tmp_path):
    """No shipped element skips a group, so a fixed offset is a valid address.

    The builder addresses group ``l`` of atom I as ``H_INDEX_START_U[I] + l``
    with l = 0, 1, 2, 3 for s, p, d, f.  That arithmetic is only correct if an
    atom's present groups are a contiguous run starting at s - an element with,
    say, s and d but no p would make the offset for d land on nothing.  No
    element in either shipped fixture set violates it.  This test says so out
    loud instead of leaving it as an unstated assumption.
    """

    def check():
        from dftorch.Constants import Constants

        checked = []
        for name, skf_dir in (
            ("f_orbital_data", _skf_dir()),
            ("mio-1-1", _mio_skf_dir()),
        ):
            for symbol in _elements_in(skf_dir):
                # One element at a time, on its own homonuclear pair file.  The
                # mio-1-1 set has no P-Zn.skf, so a single geometry naming every
                # element at once cannot be built from it.
                xyz_path = tmp_path / f"{name}-{symbol}.xyz"
                _write_xyz(xyz_path, [symbol], spacing=3.0)
                params = dict(BASE_PARAMS)
                params["FILENAME"] = str(xyz_path)
                params["SKFPATH"] = str(skf_dir) + os.sep
                const = Constants(params).to("cpu")

                number = const.symbol_to_number[symbol]
                present = [bool(v) for v in const.shell_present[number].tolist()]
                n_present = sum(present)
                assert present[:n_present] == [True] * n_present and not any(
                    present[n_present:]
                ), (
                    f"{name}/{symbol}: shell_present is {present}, which is not a "
                    "contiguous run of present groups starting at s. The builder "
                    "addresses group l as H_INDEX_START_U[atom] + l, so a gap "
                    "would make every offset past the gap land on the wrong row."
                )
                assert present[0], (
                    f"{name}/{symbol}: has no s group at all, so offset 0 is not "
                    "its s row"
                )
                checked.append(f"{name}/{symbol}")

        assert len(checked) >= 9, (
            "the element discovery found only "
            f"{len(checked)} elements ({checked}); both fixture sets together "
            "carry at least 9, so a smaller number means the .skf name scan "
            "stopped matching and this test is no longer checking anything"
        )

    run_with_float64(check)


# ---------------------------------------------------------------------------
# Controls: nothing on the f-free path moved
# ---------------------------------------------------------------------------


def test_f_free_matrix_still_has_no_empty_row(tmp_path):
    """Methane on mio-1-1 still gets a fully populated matrix.

    Threat T-06-08: the uniform mask rule replaced eight hand-enumerated pair
    masks, and it must reproduce the nine pre-existing blocks exactly.  CH4 is
    C(s+p) plus four H(s) = 6 shells; the shape is derived, never a literal.
    """

    def check():
        const, structure = _build_ch4(tmp_path)
        CC, dCC = _call_shell_resolved_coulomb(const, structure)

        n_sh = int(structure.n_shells_per_atom.sum())
        assert n_sh == 6, (
            f"CH4/mio-1-1 must have 6 shells total (C: s+p, 4x H: s), got {n_sh} "
            f"from n_shells_per_atom={structure.n_shells_per_atom.tolist()}"
        )
        assert CC.shape == (n_sh, n_sh)
        assert dCC.shape == (3, n_sh, n_sh)
        assert bool(torch.isfinite(CC).all())

        row_sums = CC.sum(dim=1)
        assert not bool((row_sums == 0.0).any()), (
            f"an all-zero row would mean a silently dropped shell: {row_sums}"
        )

    run_with_float64(check)
