"""The charge loop tracks charge one orbital group at a time, and uses it.

Phase 6 plan 06-03, requirement SCC-03.

Plain-language orientation
--------------------------
During the repeat-until-settled loop the code has to answer "how much charge is
sitting on each part of the molecule, and what does it cost to put it there?".
Until this plan it answered per **atom**: one charge number for europium, one
for nitrogen, and one electron-repulsion strength each.  The finer answer is per
**orbital group** - separately for the s, p, d and f groups of each atom.  This
project calls that *shell-resolved*; "shell" is just its word for one of those
groups.

Why it matters here is concrete.  ``Constants.py:232`` loads the per-atom
repulsion strength from the **s** column for every element.  For nitrogen that
is harmless (its s, p and d values coincide).  For europium it is not: the s
group costs about 5.7 eV per unit of charge and the f group about 13.6 eV, and
seven of europium's nine outer electrons live in the f group.  The per-atom path
therefore charges most of europium's electrons at roughly 42 percent of the
right strength.  The shell-resolved path is the remedy a human ruling of
2026-08-04 named (``06-RESEARCH.md``, "Scope rulings", item 3).

What is deliberately NOT here
-----------------------------
No converged per-group charge, per-atom charge or energy produced by this plan
is written down as a reference number.  Decision D-6.08 forbids freezing a value
from a path this phase is still building.  Every assertion below is a shape, a
sum, an inequality, a table equality or a *difference* between two runs.

The switch is the parameter key that already existed for it,
``MAGNETIC_HUBBARD_LDEP`` (Phase 4 decision D-23).  No new key is introduced.

ASCII only, per the Phase 4 rule: a failure must be diagnosable from pytest
output on a cp1252 console.  Write "A" for Angstrom and "->" for an arrow.
"""

import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic
# and avoids requiring a C++ toolchain).  Mirrors tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import sys
from pathlib import Path

import pytest
import torch


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.

    Copied from ``tests/test_scf_convergence_f.py`` -- the established harness
    for every f-orbital test module in this project.
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

#: Eu-N at the Phase 4 target separation (decision D-24).
EU_N_SEPARATION = 2.655

#: The library default in ``_scf.py``: ``dftorch_params.get("SCF_MAX_ITER", 100)``.
#: Named so a failure message can quote the cap it was measured against.
SCF_ITERATION_CAP = 100

#: The sentinel a loop returns when it exhausted its cap without reaching
#: tolerance (plan 06-01).
DID_NOT_CONVERGE = -1

#: The Fermi-level bisection floor already used throughout the code
#: (``eps=1e-9`` per call; worst accumulation observed in plan 06-01 was
#: 1.3e-09).  This is a conservation floor, not a pinned value.
CHARGE_CONSERVATION_TOL = 1e-6

#: Two views of one answer must agree to round-off, not merely to plotting
#: accuracy.
RESOLUTION_AGREEMENT_TOL = 1e-10

#: The smallest difference between the two runs this module is willing to call
#: a real change rather than round-off.  It is a floor on a *difference*, never
#: a claim about either value.
ANSWER_CHANGED_TOL = 1e-6

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}

#: The f-free control: methane on mio-1-1.  ``COUL_METHOD`` is "FULL" rather
#: than the "PME" that ``tests/test_scf.py`` uses, because the shell-resolved
#: matrix is only ever built on the real-space branch - which is exactly what
#: ``test_per_group_request_refuses_when_no_matrix_can_be_built`` pins.
CH4_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}


def _skf_dir() -> Path:
    return TESTS_DIR / "f_orbital_data"


def _mio_skf_dir() -> Path:
    return TESTS_DIR / "data_skf_mio-1-1"


def _ch4_xyz() -> Path:
    return TESTS_DIR / "ch4.xyz"


def _write_eu_n_xyz(path: Path, separation: float = EU_N_SEPARATION) -> None:
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 6 shell-resolved charge case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _drive(params):
    """Build ``(const, structure)`` from a parameter dict and run the SCF loop."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    driver(structure, const, do_scf=True)
    return const, structure


def _eu_n_params(tmp_path: Path, shell_resolved: bool, **overrides):
    xyz_path = tmp_path / "eu_n.xyz"
    _write_eu_n_xyz(xyz_path)
    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_skf_dir()) + os.sep
    params["MAGNETIC_HUBBARD_LDEP"] = shell_resolved
    params.update(overrides)
    return params


def _run_eu_n(tmp_path: Path, shell_resolved: bool = True, **overrides):
    """Drive the Eu-N diatomic through ``forward(do_scf=True)``."""
    return _drive(_eu_n_params(tmp_path, shell_resolved, **overrides))


def _run_ch4(shell_resolved: bool = True, **overrides):
    """Drive the f-free methane control through ``forward(do_scf=True)``."""
    params = dict(CH4_PARAMS)
    params["FILENAME"] = str(_ch4_xyz())
    params["SKFPATH"] = str(_mio_skf_dir()) + os.sep
    params["MAGNETIC_HUBBARD_LDEP"] = shell_resolved
    params.update(overrides)
    return _drive(params)


def _shell_to_atom(structure) -> torch.Tensor:
    """Map each orbital group to the atom it belongs to.

    Built the same way the loop builds it, from ``n_shells_per_atom``, so the
    test cannot agree with the implementation by accident of a hand-typed list.
    """
    return torch.repeat_interleave(
        torch.arange(structure.Nats, device=structure.RX.device),
        structure.n_shells_per_atom,
    )


def _sum_over_atoms(structure, q_sr: torch.Tensor) -> torch.Tensor:
    """Sum per-group charges into per-atom charges."""
    per_atom = torch.zeros(
        structure.Nats, dtype=q_sr.dtype, device=q_sr.device
    )
    per_atom.scatter_add_(0, _shell_to_atom(structure), q_sr)
    return per_atom


def _shell_index(structure, atom: int, shell: int) -> int:
    """Row of one orbital group, addressed the way ``Structure`` lays them out.

    ``H_INDEX_START_U[atom] + shell`` with shell 0/1/2/3 for s/p/d/f.  Never a
    hand-counted index: a hand-counted one would still "pass" if the entry
    landed on the neighbouring atom's rows.
    """
    return int(structure.H_INDEX_START_U[atom]) + shell


# ---------------------------------------------------------------------------
# Task 1: the loop tracks and feeds back charge one orbital group at a time
# ---------------------------------------------------------------------------


def test_eu_n_settles_with_per_group_charge(tmp_path):
    """Eu-N at 2.655 A reaches tolerance while tracking charge per group.

    A count strictly between 0 and the cap is the whole claim: 0 would mean the
    loop never ran, and the cap would mean it stopped because it was told to,
    not because the charges stopped moving.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=True)

        count = structure.scf_iter_count
        assert isinstance(count, int), (
            f"scf_iter_count should be a plain int, got {type(count).__name__} "
            f"({count!r})"
        )
        assert count != DID_NOT_CONVERGE, (
            "Eu-N at {} A did not settle while tracking charge per orbital "
            "group: scf_iter_count == {} (the gave-up sentinel), cap {}".format(
                EU_N_SEPARATION, count, SCF_ITERATION_CAP
            )
        )
        assert 0 < count < SCF_ITERATION_CAP, (
            "Eu-N at {} A reported scf_iter_count = {}, outside the open "
            "interval (0, {})".format(EU_N_SEPARATION, count, SCF_ITERATION_CAP)
        )

    run_with_float64(check)


def test_per_group_charges_have_one_entry_per_group(tmp_path):
    """``q_sr`` carries one finite number per orbital group, not per atom.

    The expected length is derived from ``n_shells_per_atom`` (Eu contributes
    s, p, d and f; N contributes s and p), never typed in as a literal.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=True)

        expected = int(structure.n_shells_per_atom.sum())
        assert structure.q_sr is not None, (
            "structure.q_sr is None with MAGNETIC_HUBBARD_LDEP set; the loop "
            "did not report per-group charges at all"
        )
        assert structure.q_sr.shape == (expected,), (
            "q_sr has shape {} but n_shells_per_atom = {} sums to {}".format(
                tuple(structure.q_sr.shape),
                structure.n_shells_per_atom.tolist(),
                expected,
            )
        )
        assert torch.isfinite(structure.q_sr).all(), (
            f"q_sr carries a non-finite entry: {structure.q_sr.tolist()}"
        )

    run_with_float64(check)


def test_per_group_charges_sum_to_the_per_atom_charges(tmp_path):
    """The two resolutions are two views of one answer.

    Summing the per-group charges over each atom must reproduce the per-atom
    charges every existing caller reads.  If this drifts, ``structure.q`` has
    quietly stopped describing the state the loop actually converged to.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=True)

        summed = _sum_over_atoms(structure, structure.q_sr)
        diff = (summed - structure.q).abs().max().item()
        assert diff < RESOLUTION_AGREEMENT_TOL, (
            "per-group charges summed over atoms disagree with the per-atom "
            "charges by {:.3e} (tolerance {:.0e}).\n"
            "  q_sr    = {}\n"
            "  summed  = {}\n"
            "  q       = {}".format(
                diff,
                RESOLUTION_AGREEMENT_TOL,
                structure.q_sr.tolist(),
                summed.tolist(),
                structure.q.tolist(),
            )
        )

    run_with_float64(check)


def test_total_charge_is_still_conserved(tmp_path):
    """A neutral molecule stays neutral at the finer resolution.

    The tolerance is the Fermi-level bisection floor already used throughout
    the code (``eps=1e-9`` per call), not a value pin.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=True)

        total = structure.q_sr.sum().item()
        assert abs(total) < CHARGE_CONSERVATION_TOL, (
            "per-group charges sum to {:.3e} for a neutral molecule "
            "(tolerance {:.0e}); q_sr = {}".format(
                total, CHARGE_CONSERVATION_TOL, structure.q_sr.tolist()
            )
        )

    run_with_float64(check)


def test_per_group_charge_actually_changes_the_answer(tmp_path):
    """The finer description reaches the answer instead of being ignored.

    This is the point of requirement SCC-03.  Phase 4 decision D-14 built the
    shell-resolved data and never consumed it, and a test that only checked the
    data exists would pass in exactly that broken state.  So the assertion is a
    *difference* between the two runs, never either value.
    """

    def check():
        _, off = _run_eu_n(tmp_path, shell_resolved=False)
        _, on = _run_eu_n(tmp_path, shell_resolved=True)

        assert off.scf_iter_count != DID_NOT_CONVERGE, (
            "the per-atom control did not settle (scf_iter_count == "
            f"{DID_NOT_CONVERGE}); the comparison below would be meaningless"
        )
        assert on.scf_iter_count != DID_NOT_CONVERGE, (
            "the shell-resolved run did not settle (scf_iter_count == "
            f"{DID_NOT_CONVERGE}); the comparison below would be meaningless"
        )

        change = (on.q - off.q).abs().max().item()
        assert change > ANSWER_CHANGED_TOL, (
            "the converged per-atom charges are the same with the switch on "
            "and off (largest change {:.3e}, floor {:.0e}), so the per-group "
            "charges are being built and then ignored - the exact state Phase "
            "4 decision D-14 left behind.\n"
            "  switch off = {}\n"
            "  switch on  = {}".format(
                change,
                ANSWER_CHANGED_TOL,
                off.q.tolist(),
                on.q.tolist(),
            )
        )

    run_with_float64(check)


def test_switch_off_leaves_no_per_group_charges(tmp_path):
    """With the switch unset the old path runs and reports no per-group charge.

    ``q_sr`` still has to *exist* as an attribute, so a caller can ask without
    an AttributeError; it is ``None`` because nothing per-group was computed.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=False)

        assert structure.q_sr is None, (
            "MAGNETIC_HUBBARD_LDEP is unset but structure.q_sr is {!r}; the "
            "per-atom path must not produce per-group charges".format(
                structure.q_sr
            )
        )
        assert structure.C_sr is None, (
            "MAGNETIC_HUBBARD_LDEP is unset but structure.C_sr was built"
        )
        assert structure.scf_iter_count != DID_NOT_CONVERGE, (
            "the untouched per-atom path stopped settling: scf_iter_count == "
            f"{DID_NOT_CONVERGE}"
        )

    run_with_float64(check)


def test_f_free_molecule_runs_at_the_finer_resolution():
    """Methane settles at the finer resolution too - the branch is not f-only.

    The control that the new code path is a general shell-resolved path and not
    something wired only for europium.
    """

    def check():
        _, structure = _run_ch4(shell_resolved=True)

        assert structure.scf_iter_count != DID_NOT_CONVERGE, (
            "methane did not settle at the finer resolution: scf_iter_count "
            f"== {DID_NOT_CONVERGE}, cap {SCF_ITERATION_CAP}"
        )
        assert structure.q_sr is not None, (
            "methane produced no per-group charges with MAGNETIC_HUBBARD_LDEP "
            "set"
        )
        expected = int(structure.n_shells_per_atom.sum())
        assert structure.q_sr.shape == (expected,), (
            "methane q_sr has shape {} but n_shells_per_atom = {} sums to "
            "{}".format(
                tuple(structure.q_sr.shape),
                structure.n_shells_per_atom.tolist(),
                expected,
            )
        )
        summed = _sum_over_atoms(structure, structure.q_sr)
        diff = (summed - structure.q).abs().max().item()
        assert diff < RESOLUTION_AGREEMENT_TOL, (
            "methane's two resolutions disagree by {:.3e} (tolerance "
            "{:.0e}).\n  q_sr   = {}\n  summed = {}\n  q      = {}".format(
                diff,
                RESOLUTION_AGREEMENT_TOL,
                structure.q_sr.tolist(),
                summed.tolist(),
                structure.q.tolist(),
            )
        )

    run_with_float64(check)


def test_per_group_request_refuses_when_no_matrix_can_be_built(tmp_path):
    """Asking for the finer resolution under PME fails out loud.

    ``ESDriver.forward`` builds the shell-resolved matrix only on the
    real-space branch; under ``COUL_METHOD="PME"`` there is no shell-resolved
    reciprocal-space counterpart, so ``structure.C_sr`` would stay ``None``.
    Serving that request from the per-atom matrix would hand back a coarser
    answer wearing the finer answer's name, which is the codebase's "fail
    loudly, never silently zero" pattern in reverse.

    The message must name both configuration keys involved and must carry no
    filesystem path (Phase 4 policy T-04-07).
    """

    def check():
        params = _eu_n_params(
            tmp_path,
            shell_resolved=True,
            COUL_METHOD="PME",
            CELL=[25.0, 25.0, 25.0],
        )

        with pytest.raises(NotImplementedError) as excinfo:
            _drive(params)

        message = str(excinfo.value)
        assert "COUL_METHOD" in message, (
            f"the refusal does not name COUL_METHOD: {message!r}"
        )
        assert "MAGNETIC_HUBBARD_LDEP" in message, (
            f"the refusal does not name MAGNETIC_HUBBARD_LDEP: {message!r}"
        )
        for leak in (".skf", ".xyz", "SKFPATH", "FILENAME"):
            assert leak not in message, (
                f"the refusal leaks {leak!r} into its message: {message!r}"
            )

    run_with_float64(check)


def test_reference_occupations_and_nuclear_charges_agree(tmp_path):
    """The two tallies subtract the same total number of electrons.

    The per-atom tally starts from minus the nuclear charge; the per-group
    tally starts from minus the reference occupation of each group.  If those
    two totals disagreed, the two resolutions could never sum to each other,
    and the failure would look like a physics problem rather than the
    bookkeeping problem it is.
    """

    def check():
        _, structure = _run_eu_n(tmp_path, shell_resolved=True)

        el = structure.el_per_shell.sum().item()
        z = structure.Znuc.sum().item()
        assert abs(el - z) < 1e-9, (
            "el_per_shell sums to {:.12f} but Znuc sums to {:.12f} (difference "
            "{:.3e}).\n  el_per_shell = {}\n  Znuc         = {}".format(
                el, z, el - z,
                structure.el_per_shell.tolist(),
                structure.Znuc.tolist(),
            )
        )

    run_with_float64(check)


# ---------------------------------------------------------------------------
# Task 2: europium's f electrons are charged at the f rate
# ---------------------------------------------------------------------------


def test_europium_f_group_is_charged_at_the_f_rate(tmp_path):
    """Europium's f group is charged at the f table's strength, not the s one.

    In plain words: the cost of putting charge into europium's f group is about
    2.4 times the cost the per-atom path uses, and seven of europium's nine
    outer electrons live in that f group.  So this is the difference between
    charging most of the atom's electrons correctly and charging them at about
    42 percent of the right value.

    Asserted as a table equality and an inequality.  Neither strength is typed
    in as a number - they are read from ``Constants``, which reads them from
    the SKF parameter files.
    """

    def check():
        const, structure = _run_eu_n(tmp_path, shell_resolved=True)

        eu = const.symbol_to_number["Eu"]
        eu_atom = int((structure.TYPE == eu).nonzero()[0])
        f_row = _shell_index(structure, eu_atom, 3)

        label = int(structure.shell_types[f_row])
        assert label == 4, (
            "row {} of the shell tables is labelled {} but the f group is "
            "labelled 4 (s/p/d/f -> 1/2/3/4); shell_types = {}, atom {} is "
            "element type {}".format(
                f_row,
                label,
                structure.shell_types.tolist(),
                eu_atom,
                eu,
            )
        )

        f_strength = structure.Hubbard_U_sr[f_row].item()
        table_f = const.Uf[eu].item()
        assert abs(f_strength - table_f) < 1e-12, (
            "the f group of atom {} (element type {}) is charged at {:.9f} eV "
            "but const.Uf for that element is {:.9f} eV".format(
                eu_atom, eu, f_strength, table_f
            )
        )

        per_atom = structure.Hubbard_U[eu_atom].item()
        assert f_strength > 2.0 * per_atom, (
            "the f strength {:.9f} eV is not more than twice the per-atom "
            "strength {:.9f} eV for atom {} (element type {}); the ratio is "
            "{:.4f} and the defect this phase routes around depends on it "
            "being large".format(
                f_strength,
                per_atom,
                eu_atom,
                eu,
                f_strength / per_atom if per_atom else float("inf"),
            )
        )

    run_with_float64(check)


def test_the_per_atom_strength_still_comes_from_the_s_group(tmp_path):
    """The per-atom defect is pinned in place, deliberately, not repaired.

    ``Constants.py:232`` reads ``self.U = torch.nn.Parameter(US, ...)``: the
    per-atom repulsion strength is the s group's value for every element.
    Changing that line would move numbers in every existing calculation
    containing an f element.  It is not required by this phase's bar, and the
    human ruling of 2026-08-04 named the shell-resolved path as the remedy
    instead.  So this test pins the defect rather than fixing it: if someone
    later changes that line, this failure tells them they have made a decision,
    not a tidy-up.  Plan 06-04 carries the written record.
    """

    def check():
        const, structure = _run_eu_n(tmp_path, shell_resolved=True)

        eu = const.symbol_to_number["Eu"]
        eu_atom = int((structure.TYPE == eu).nonzero()[0])

        per_atom = structure.Hubbard_U[eu_atom].item()
        table_s = const.U[eu].item()
        assert abs(per_atom - table_s) < 1e-12, (
            "the per-atom strength for atom {} (element type {}) is {:.9f} eV "
            "but const.U for that element is {:.9f} eV; if Constants.py:232 "
            "was changed on purpose, record it as a decision".format(
                eu_atom, eu, per_atom, table_s
            )
        )

        table_f = const.Uf[eu].item()
        assert abs(table_s - table_f) > 1e-9, (
            "const.U and const.Uf coincide for element type {} ({:.9f} eV vs "
            "{:.9f} eV), so this fixture no longer demonstrates the defect and "
            "the shell-resolved path cannot be told from the per-atom one on "
            "it".format(eu, table_s, table_f)
        )

    run_with_float64(check)


@pytest.mark.parametrize("case", ("eu_n", "ch4"))
def test_the_two_resolutions_charge_the_same_total_electrons(tmp_path, case):
    """Shapes and totals agree between the per-atom and per-group tables.

    One entry per orbital group in the finer table, one per atom in the coarser
    one, and the same total electron count described by both.  Fails loudly if
    a future element's parameter file disagrees with itself.
    """

    def check():
        if case == "eu_n":
            _, structure = _run_eu_n(tmp_path, shell_resolved=True)
        else:
            _, structure = _run_ch4(shell_resolved=True)

        n_groups = int(structure.n_shells_per_atom.sum())
        assert structure.Hubbard_U_sr.shape == (n_groups,), (
            "{}: Hubbard_U_sr has shape {} but n_shells_per_atom = {} sums to "
            "{}".format(
                case,
                tuple(structure.Hubbard_U_sr.shape),
                structure.n_shells_per_atom.tolist(),
                n_groups,
            )
        )
        assert structure.Hubbard_U.shape == (structure.Nats,), (
            "{}: Hubbard_U has shape {} but the molecule has {} atoms".format(
                case, tuple(structure.Hubbard_U.shape), structure.Nats
            )
        )

        el = structure.el_per_shell.sum().item()
        z = structure.Znuc.sum().item()
        assert abs(el - z) < 1e-9, (
            "{}: el_per_shell sums to {:.12f} but Znuc sums to {:.12f} "
            "(difference {:.3e}); el_per_shell = {}, Znuc = {}".format(
                case, el, z, el - z,
                structure.el_per_shell.tolist(),
                structure.Znuc.tolist(),
            )
        )

    run_with_float64(check)
