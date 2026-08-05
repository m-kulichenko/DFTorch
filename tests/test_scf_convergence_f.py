"""The self-consistent charge loop settles for Eu-N, and says so in a number.

Phase 6 plan 06-01, requirement SCC-01.

Plain-language orientation
--------------------------
Part of a DFTB calculation is a repeat-until-settled loop: guess how the
electrons are spread across the atoms, recompute, compare, repeat.  "Converged"
means the change from one pass to the next fell below a tolerance.  Before this
plan the only way a caller could learn whether that loop settled or simply ran
out of passes was to read printed text on stdout.  This module pins the
replacement: the loop returns the pass count on success and the literal ``-1``
when it gave up, and the driver stores it as ``structure.scf_iter_count``.

It also pins the interim disabling of the Krylov accelerator
(``kernel_update_lr``) for f-containing systems.  A human ruling recorded in
``06-RESEARCH.md`` on 2026-08-04 defers repairing that accelerator to a later
phase; Phase 6 sets ``KRYLOV_START`` above the iteration cap for f systems only
and proceeds.  ``test_krylov_accelerator_is_untouched_for_an_f_free_system`` is
the control proving the switch cannot move a single f-free number.

What is deliberately NOT here
-----------------------------
No converged energy and no converged charge is compared against a recorded
number.  Decision D-6.08 forbids freezing a value produced by a loop whose
behaviour this phase is still changing.  The one magnitude bound present (a
charge of at most 2.0 e) is a runaway detector with a factor of three of
headroom, not a reference value.

Decision D-6.05 sets the bar at the neighbourhood of 2.655 A, not at all 21
separations of the 1.60-3.60 A scan.  A separation elsewhere in that range that
still fails to settle is reported through the ``-1`` result and does not turn
this suite red -- no test here asserts convergence away from the neighbourhood.

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

import torch


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.

    Copied from ``tests/test_single_shot_energy.py`` -- the established harness
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


# --- Eu-N reference case ----------------------------------------------------
#
# Isolated Eu-N diatomic at the D-24 target separation of 2.655 A, using the
# nine-file f-orbital SKF fixture set in tests/f_orbital_data/.  The parameter
# dict is copied from tests/test_single_shot_energy.py so that the two modules
# describe the same physical system; the only addition is the opt-in silence
# flag, which suppresses per-iteration chatter and changes no numeric path
# (see dftorch._tools.library_output_enabled).
EU_N_SEPARATION = 2.655

# D-6.01's actual bar: the neighbourhood of the target, not one lucky point.
EU_N_NEIGHBOURHOOD = (2.60, 2.655, 2.70)

# The library default in _scf.py: `dftorch_params.get("SCF_MAX_ITER", 100)`.
# Named here so a failure message can quote the cap it was measured against.
SCF_ITERATION_CAP = 100

# The sentinel a loop returns when it exhausted its cap without reaching
# tolerance.  Chosen after scipy's iterative solvers, which return a positive
# count on success and a sentinel on failure.
DID_NOT_CONVERGE = -1

# A runaway detector, NOT a reference value.  The two observed divergence modes
# of this system moved a charge to -2.9 e and to +5.0 e, while a healthy answer
# sits near 0.6 e.  The bound deliberately leaves a factor of three of headroom
# so a legitimate shift in the answer does not turn this test red.
CHARGE_RUNAWAY_BOUND = 2.0

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}

# The f-free control: methane on mio-1-1, copied from tests/test_scf.py.
# KRYLOV_START is deliberately absent.  test_scf.py sets it to 5; leaving it out
# here means the f-free branch of the interim helper -- not its
# caller-already-asked branch -- is what has to return False.
CH4_PARAMS = {
    "CELL": [25.0, 25.0, 25.0],
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 8.0,
    "RCUT_REPULSIVE": 4.0,
    "COUL_METHOD": "PME",
    "SCF_MAX_ITER": 25,
    "VERBOSE_LIBRARY_OUTPUT": False,
}


def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"


def _write_eu_n_xyz(path: Path, separation: float = EU_N_SEPARATION) -> None:
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 6 self-consistent charge case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _run_scf(tmp_path: Path, **overrides):
    """Build the Eu-N system and drive it through ``forward(do_scf=True)``.

    Returns the structure so callers can inspect what the self-consistent
    branch is contracted to populate.
    """
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    separation = overrides.pop("separation", EU_N_SEPARATION)
    xyz_path = tmp_path / f"eu_n_{separation:.3f}.xyz"
    _write_eu_n_xyz(xyz_path, separation)

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_skf_dir()) + os.sep
    params.update(overrides)

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    driver(structure, const, do_scf=True)
    return structure


def _run_scf_capturing_loop_params(tmp_path: Path, **overrides):
    """Drive Eu-N and also capture the parameter dict handed to ``SCFx``.

    Asserting on the driver's own dict would prove nothing about what the loop
    received; the interim helper returns a *copy*, so the two can legitimately
    differ.  This wraps the module-global ``SCFx`` that ``forward`` resolves at
    call time, which is the connection itself rather than either end of it.
    """
    # import_module, not ``from dftorch import ESDriver``: the package exports a
    # *class* of that name, so the plain import shadows the module it lives in.
    from importlib import import_module

    esdriver_module = import_module("dftorch.ESDriver")

    captured = {}
    real_scfx = esdriver_module.SCFx

    def recording_scfx(dftorch_params, *args, **kwargs):
        captured["params"] = dict(dftorch_params)
        return real_scfx(dftorch_params, *args, **kwargs)

    esdriver_module.SCFx = recording_scfx
    try:
        structure = _run_scf(tmp_path, **overrides)
    finally:
        esdriver_module.SCFx = real_scfx
    return structure, captured["params"]


def _run_ch4_scf():
    """Drive the f-free methane control through ``forward(do_scf=True)``."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    root = Path(__file__).resolve().parents[1]
    params = dict(CH4_PARAMS)
    params["FILENAME"] = str(root / "tests" / "ch4.xyz")
    params["SKFPATH"] = str(root / "tests" / "data_skf_mio-1-1") + os.sep

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    driver(structure, const, do_scf=True)
    return structure


# --- Task 1: the loop settles, and reports how it went ----------------------


def test_eu_n_scf_converges_at_target_separation(tmp_path):
    """Eu-N at 2.655 A reaches tolerance rather than running out of passes.

    A count strictly between 0 and the cap is the whole claim: 0 would mean the
    loop never ran, and the cap would mean it stopped because it was told to,
    not because the charges stopped moving.
    """

    def check():
        structure = _run_scf(tmp_path)

        count = getattr(structure, "scf_iter_count", None)
        assert count is not None, (
            "structure.scf_iter_count was never assigned; the driver did not "
            "unpack the loop's convergence result"
        )
        assert isinstance(count, int) and not isinstance(count, bool), (
            "scf_iter_count must be a plain int (a pass count), not "
            f"{type(count).__name__} -- a boolean would throw the count away"
        )
        assert 0 < count < SCF_ITERATION_CAP, (
            f"Eu-N at {EU_N_SEPARATION} A did not settle: "
            f"scf_iter_count = {count}, iteration cap = {SCF_ITERATION_CAP} "
            f"({DID_NOT_CONVERGE} means the loop gave up)"
        )

    run_with_float64(check)


def test_eu_n_scf_converges_across_the_target_neighbourhood(tmp_path):
    """The bar is the neighbourhood of 2.655 A, not one lucky separation.

    Decision D-6.01 sets the requirement here.  A single point could settle by
    accident of where the initial guess happened to land.
    """

    def check():
        counts = {}
        for separation in EU_N_NEIGHBOURHOOD:
            structure = _run_scf(tmp_path, separation=separation)
            counts[separation] = getattr(structure, "scf_iter_count", None)

        failed = [
            separation
            for separation, count in counts.items()
            if not (isinstance(count, int) and 0 < count < SCF_ITERATION_CAP)
        ]
        assert not failed, (
            "Eu-N failed to settle at "
            + ", ".join(f"{sep} A" for sep in failed)
            + ". Counts across the neighbourhood (cap "
            + f"{SCF_ITERATION_CAP}, {DID_NOT_CONVERGE} means gave up): "
            + ", ".join(f"{sep} A -> {counts[sep]}" for sep in EU_N_NEIGHBOURHOOD)
        )

    run_with_float64(check)


def test_eu_n_converged_charges_are_not_a_runaway(tmp_path):
    """The settled charges are physical in scale, not piled onto one atom.

    Two assertions with different jobs.  The sum is charge conservation on a
    neutral molecule -- an exact property, not a measured value, so a tight
    1e-6 is right.  The magnitude bound is a runaway detector: it separates a
    healthy answer near 0.6 e from the observed failure modes at -2.9 e and
    +5.0 e, and its factor-of-three headroom means it is not a reference value
    in disguise (decision D-6.08).
    """

    def check():
        structure = _run_scf(tmp_path)
        q = structure.q

        assert q.shape == (2,), f"expected two per-atom charges, got {tuple(q.shape)}"
        total = q.sum().item()
        assert abs(total) < 1e-6, (
            f"charge is not conserved: the two per-atom charges sum to {total}, "
            f"which should be zero for a neutral Eu-N molecule (q = {q.tolist()})"
        )
        for index, value in enumerate(q.tolist()):
            assert abs(value) <= CHARGE_RUNAWAY_BOUND, (
                f"charge runaway on atom {index}: q = {value} e exceeds the "
                f"runaway bound of {CHARGE_RUNAWAY_BOUND} e "
                f"(all charges: {q.tolist()})"
            )

    run_with_float64(check)


def test_eu_n_charge_transfer_runs_the_chemical_direction(tmp_path):
    """Electrons move from the metal to the non-metal, Eu -> N.

    Sign only, never magnitude.  This codebase builds charges as
    ``q = population - Znuc`` (``_scf.py``, the final charge assembly), so a
    positive q means surplus electrons.  Europium is the electropositive metal
    and nitrogen the electronegative non-metal, so Eu must come out negative
    and N positive.  A run that settled to the opposite signs would be settled
    and wrong, which no convergence count can detect.
    """

    def check():
        structure = _run_scf(tmp_path)
        q_eu, q_n = structure.q.tolist()

        assert q_eu < 0.0, (
            f"europium should lose electrons to nitrogen, but q_Eu = {q_eu} e "
            f"(convention: q = population - Znuc, so positive means surplus "
            f"electrons); both charges: {structure.q.tolist()}"
        )
        assert q_n > 0.0, (
            f"nitrogen should gain electrons from europium, but q_N = {q_n} e; "
            f"both charges: {structure.q.tolist()}"
        )

    run_with_float64(check)


def test_krylov_accelerator_is_disabled_for_the_f_system(tmp_path):
    """An f system takes the interim path, and says so on the structure.

    The accelerator diverges on this system; repairing it is deferred by the
    human ruling of 2026-08-04 recorded in 06-RESEARCH.md.  This attribute is
    how a caller -- or a later phase removing the interim -- can tell which
    path a given result came from.
    """

    def check():
        structure = _run_scf(tmp_path)

        assert getattr(structure, "krylov_disabled_for_f", None) is True, (
            "Eu-N contains a 16-orbital atom, so the interim Krylov switch-off "
            "should have engaged, but structure.krylov_disabled_for_f = "
            f"{getattr(structure, 'krylov_disabled_for_f', '<unset>')}"
        )

    run_with_float64(check)


def test_krylov_accelerator_is_untouched_for_an_f_free_system():
    """The control: methane never enters the interim branch at all.

    This is the test that makes "no existing f-free calculation changes by a
    single digit" checkable rather than asserted.  CH4 on mio-1-1 has no
    16-orbital atom, so the helper must return the caller's dict unchanged and
    the accelerator must keep its normal default.
    """

    def check():
        structure = _run_ch4_scf()

        assert getattr(structure, "krylov_disabled_for_f", None) is False, (
            "CH4 contains no f-shell element, so the interim Krylov switch-off "
            "must not engage, but structure.krylov_disabled_for_f = "
            f"{getattr(structure, 'krylov_disabled_for_f', '<unset>')}"
        )
        count = getattr(structure, "scf_iter_count", None)
        assert isinstance(count, int) and not isinstance(count, bool), (
            "CH4 must report a convergence result too, but scf_iter_count = "
            f"{count!r}"
        )
        assert count > 0, (
            f"CH4 self-consistency did not settle: scf_iter_count = {count} "
            f"({DID_NOT_CONVERGE} means the loop gave up)"
        )

    run_with_float64(check)


def test_caller_supplied_krylov_start_is_never_overridden(tmp_path):
    """A caller who states a preference keeps it, f system or not.

    The interim switch-off exists so that an ordinary caller gets a converging
    f calculation without knowing a magic key.  It must not become a ceiling on
    a caller who does know the key and has chosen a value on purpose -- for
    instance to reproduce a pre-Phase-6 result, or to study the accelerator
    while the repair is being written.
    """

    def check():
        structure, loop_params = _run_scf_capturing_loop_params(
            tmp_path, KRYLOV_START=3
        )

        assert getattr(structure, "krylov_disabled_for_f", None) is False, (
            "the caller set KRYLOV_START explicitly, so the interim switch-off "
            "must stand down, but structure.krylov_disabled_for_f = "
            f"{getattr(structure, 'krylov_disabled_for_f', '<unset>')}"
        )
        assert loop_params.get("KRYLOV_START") == 3, (
            "the caller asked for KRYLOV_START = 3, but the loop was handed "
            f"KRYLOV_START = {loop_params.get('KRYLOV_START')!r}"
        )

    run_with_float64(check)
