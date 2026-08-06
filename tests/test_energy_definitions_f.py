"""The one-pass and settled energies are two different quantities, and stay so.

Phase 6 plan 06-04, requirement SCC-01.  The written verdict these tests hold
is ``docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md``; read that first, then read this
module as the machine half of it.

Plain-language orientation
--------------------------
There are two ways to get an energy out of this code for the same molecule at
the same geometry.

* **One pass** (``do_scf=False``) puts the electrons in their free-atom
  arrangement, solves once, and reports the answer.  It reports an
  electron-repulsion energy of exactly zero, deliberately, by Phase 4 decision
  D-11 -- ``ESDriver.forward`` hands ``energy()`` a ``None`` where the Coulomb
  matrix would go, which selects that function's zero arm.
* **Settled** (``do_scf=True``) solves, looks at where the electrons moved to,
  rebuilds, and repeats until the answer stops changing.  It includes the
  electron-repulsion energy.

"Electron-repulsion energy" here means the electrostatic price of having moved
charge off one atom and onto another -- ``structure.e_coul``.  It is a
different thing from ``structure.e_repulsion``, which is the short-range
pairwise term keeping nuclei apart and is identical on both paths.

At Eu-N 2.655 A the two totals differ by about 10.6 eV, which is two to five
times a chemical bond.  The verdict document explains why: it is a difference
of definition.  These four tests hold that verdict's three load-bearing claims
plus one tripwire.

What is deliberately NOT here
-----------------------------
**No energy and no charge is compared against a typed-in value.**  Decision
D-6.08 forbids freezing a number produced by behaviour this phase is still
changing, and this module is the most tempting place in the phase to break that
rule, because the verdict it holds is full of measured numbers.  Every
assertion below is an exact zero, a sign, an inequality or a ratio.  The
measured numbers live in the verdict document as prose evidence, and in the
docstrings here as scale, never in an assertion.

The one number imported rather than retyped is ``EU_N_REFERENCE_E_TOT``, the
Phase 4 one-pass pin, so that the suite holds exactly one copy of it.

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

# The one number this module does not own.  Imported, never retyped, so the
# suite carries exactly one copy of the Phase 4 pin.  The cross-module import
# follows tests/test_dtype_contract.py, which already imports a helper from a
# sibling test module this way.
from test_single_shot_energy import EU_N_REFERENCE_E_TOT


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
# The same molecule, geometry and parameter dict as
# tests/test_single_shot_energy.py and tests/test_scf_convergence_f.py, so all
# three modules describe one physical system.  The only addition is the opt-in
# silence flag, which suppresses per-pass chatter and changes no numeric path.
EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}

# How much more charge the one-pass path must move than the settled path.
#
# This is a mechanism check, not a value pin.  Measured 2026-08-06 the two
# paths move 2.703 e and 0.600 e, a ratio of 4.51, so a threshold of three
# leaves better than a third of headroom.  What it is really asserting is that
# switching the electrostatic penalty on changes where the electrons go by a
# lot -- which is the whole mechanism behind the 10.6 eV difference.  A ratio
# that fell towards one would mean the penalty had stopped biting.
CHARGE_TRANSFER_RATIO_FLOOR = 3.0

# How far the settled total must sit from the Phase 4 one-pass pin, in eV.
#
# Also not a value pin, and pointing the opposite way from a normal tolerance:
# it is a floor on a difference, not a ceiling.  The observed difference is
# 10.59 eV, an order of magnitude of headroom.  It exists so that a future
# change making the two paths agree fails here loudly, because agreement would
# mean Phase 4 decision D-11 had been silently undone.
MINIMUM_GAP_TO_THE_PHASE_FOUR_PIN = 1.0

# Which of the four on-site energy tables each entry of ``shell_present``
# switches on, in the order Constants stores them.
SHELL_LABELS = ("s", "p", "d", "f")
SHELL_TABLES = ("Es", "Ep", "Ed", "Ef")

# Eu is written first in the geometry below and N second.  Europium carries 16
# orbitals (s + p + d + f) and nitrogen 4 (s + p); the tests assert that rather
# than trusting the ordering, so a reordered fixture fails loudly instead of
# silently inverting a comparison.
EU_ORBITAL_COUNT = 16
N_ORBITAL_COUNT = 4


def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"


def _write_eu_n_xyz(path: Path, separation: float = EU_N_SEPARATION) -> None:
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 6 energy-definition case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _build(tmp_path: Path, **overrides):
    """Assemble the Eu-N system without running anything through the driver.

    Returns ``(params, const, structure)``.  Split out from ``_run`` because
    the energy-level ordering test needs the parameter tables and nothing else:
    it is a statement about the parameter files, and reading it off a
    calculation would make it a statement about a calculation instead.
    """
    from dftorch.Constants import Constants
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
    return params, const, structure


def _run(tmp_path: Path, do_scf: bool, **overrides):
    """Drive the Eu-N system through ``forward`` on the requested path."""
    from dftorch.ESDriver import ESDriver

    params, const, structure = _build(tmp_path, **overrides)
    ESDriver(params, device="cpu")(structure, const, do_scf=do_scf)
    return structure


def _as_float(value) -> float:
    """Unwrap a tensor-or-number energy to a plain float.

    The one-pass path reports ``e_coul`` as the Python int ``0`` straight out of
    ``_energy.py``'s zero arm, while the settled path reports a tensor.  That
    difference is itself part of the story, so it is unwrapped here rather than
    papered over in the assertions.
    """
    if torch.is_tensor(value):
        return value.item()
    return float(value)


def _present_levels(const, atomic_number: int):
    """Return ``{label: on-site energy}`` for the shells this element has.

    Read straight from the four ``Constants`` tables, restricted by
    ``shell_present``.  An absent shell has a zero sitting in its table -- a
    placeholder, not an energy -- so including it would compare against a
    number that means nothing.
    """
    present = const.shell_present[atomic_number].tolist()
    return {
        label: getattr(const, table)[atomic_number].item()
        for label, table, is_present in zip(SHELL_LABELS, SHELL_TABLES, present)
        if is_present
    }


# --- Claim 1: the definitional difference itself ----------------------------


def test_the_one_pass_path_carries_no_electron_repulsion_term(tmp_path):
    """One pass reports exactly zero electron repulsion; settled reports some.

    Both halves in one test, because the pair *is* the definitional difference
    the verdict document is about.  Splitting them would let one half go green
    while the claim they jointly make had stopped being true.

    Zero-ness and non-zero-ness only.  The settled magnitude was +1.699 eV when
    the verdict was written; asserting that number would freeze a value this
    phase is still changing (decision D-6.08), and would also miss the point --
    the claim is that the term is *absent* on one path and *present* on the
    other, not that it has any particular size.
    """

    def check():
        one_pass = _run(tmp_path, do_scf=False)
        e_coul_one_pass = _as_float(one_pass.e_coul)
        assert e_coul_one_pass == 0.0, (
            "the one-pass path must carry no electron-repulsion term at all, "
            f"but structure.e_coul = {e_coul_one_pass!r}. This is Phase 4 "
            "decision D-11: ESDriver.forward hands energy() a None where the "
            "Coulomb matrix goes (ESDriver.py:1030-1031), selecting the "
            "Ecoul = 0 arm (_energy.py:181-182). A non-zero value here means "
            "that None became a matrix, which redefines the one-pass energy "
            "and invalidates the Phase 4 pinned reference. See "
            "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md."
        )

        settled = _run(tmp_path, do_scf=True)
        e_coul_settled = _as_float(settled.e_coul)
        assert e_coul_settled == e_coul_settled, (
            "the settled path reported a NaN electron-repulsion energy "
            f"({e_coul_settled!r}); the loop settled on nothing usable"
        )
        assert abs(e_coul_settled) != float("inf"), (
            "the settled path reported an infinite electron-repulsion energy "
            f"({e_coul_settled!r})"
        )
        assert e_coul_settled != 0.0, (
            "the settled path must carry a real electron-repulsion term, but "
            f"structure.e_coul = {e_coul_settled!r}. An exact zero here means "
            "the settled path took the same arm as the one-pass path, so the "
            "two paths are now computing the same quantity and the verdict in "
            "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md no longer describes this "
            f"code. Settled run: e_tot = {_as_float(settled.e_tot)}, "
            f"scf_iter_count = {getattr(settled, 'scf_iter_count', '<unset>')}"
        )

    run_with_float64(check)


# --- Claim 2: the mechanism, measured ---------------------------------------


def test_the_two_paths_move_very_different_amounts_of_charge(tmp_path):
    """Switching the electrostatic penalty on changes where the electrons go.

    This is the mechanism behind the 10.6 eV difference, and it is why an
    explanation that only talks about the electron-repulsion *term* is
    incomplete: 84 percent of the difference sits in the band-structure term,
    which moved because the electrons are somewhere else.

    Without a penalty the one-pass path pours charge onto nitrogen and banks
    the full band-structure reward while paying nothing for the resulting
    separation of positive and negative charge.  With the penalty on, the
    transfer collapses and most of that reward is handed back.

    A ratio, never a magnitude.  Measured 2026-08-06 the two paths move 2.703 e
    and 0.600 e; the floor of three leaves real headroom and asserts the
    mechanism rather than either number.
    """

    def check():
        one_pass = _run(tmp_path, do_scf=False)
        settled = _run(tmp_path, do_scf=True)

        q_one_pass = one_pass.q[0].item()
        q_settled = settled.q[0].item()

        assert q_settled != 0.0, (
            "the settled path moved no charge off europium at all "
            f"(q_Eu = {q_settled}), so there is no ratio to take. Both charge "
            f"vectors: one pass {one_pass.q.tolist()}, settled "
            f"{settled.q.tolist()}"
        )

        ratio = abs(q_one_pass) / abs(q_settled)
        assert ratio > CHARGE_TRANSFER_RATIO_FLOOR, (
            "the two paths should settle on very different charge states, but "
            f"they are close: |q_Eu| = {abs(q_one_pass)} e on the one-pass "
            f"path and {abs(q_settled)} e on the settled path, a ratio of "
            f"{ratio}, which does not clear the floor of "
            f"{CHARGE_TRANSFER_RATIO_FLOOR}. A ratio near one would mean the "
            "electrostatic penalty had stopped restraining the charge "
            "transfer, which is the mechanism the verdict in "
            "docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md rests on. Full charge "
            f"vectors: one pass {one_pass.q.tolist()}, settled "
            f"{settled.q.tolist()}"
        )

    run_with_float64(check)


# --- Claim 3: why the one-pass transfer runs away ---------------------------


def test_every_nitrogen_level_lies_below_every_europium_level(tmp_path):
    """The parameter set forces the one-pass runaway; no calculation needed.

    Each orbital has an on-site energy: roughly, how tightly that atom holds an
    electron in it.  Electrons fill the lowest levels first, across the whole
    molecule, against one shared filling level.  If every nitrogen level lies
    below every europium level, then with no charge penalty nitrogen fills
    completely at any separation -- which is exactly what the one-pass path
    does, pinning at -3.0000 electrons on europium out to 40 A.

    This is the classic charge-transfer failure of non-self-consistent tight
    binding and is the reason the self-consistent method exists.  It is not
    specific to f orbitals and not a defect in this implementation.

    Read from the ``Constants`` tables rather than from a run, because the
    claim is about the parameter files.  Measured 2026-08-06: nitrogen's
    highest is 2p at -6.8355 eV, europium's lowest is 6s at -2.634 eV, a margin
    of 4.20 eV.  The assertion is the ordering, never the margin.
    """

    def check():
        _, const, structure = _build(tmp_path)

        eu_z, n_z = (int(value) for value in structure.TYPE.tolist())
        assert const.n_orb[eu_z].item() == EU_ORBITAL_COUNT, (
            f"expected the first atom to be europium with {EU_ORBITAL_COUNT} "
            f"orbitals, but atomic number {eu_z} carries "
            f"{const.n_orb[eu_z].item()}; the geometry fixture was reordered "
            "and this test would otherwise compare the two elements the wrong "
            "way round"
        )
        assert const.n_orb[n_z].item() == N_ORBITAL_COUNT, (
            f"expected the second atom to be nitrogen with {N_ORBITAL_COUNT} "
            f"orbitals, but atomic number {n_z} carries "
            f"{const.n_orb[n_z].item()}"
        )

        eu_levels = _present_levels(const, eu_z)
        n_levels = _present_levels(const, n_z)

        assert eu_levels and n_levels, (
            "at least one element reported no present shells at all, so there "
            f"is nothing to order: Eu {eu_levels}, N {n_levels}"
        )

        highest_n = max(n_levels, key=n_levels.get)
        lowest_eu = min(eu_levels, key=eu_levels.get)
        margin = eu_levels[lowest_eu] - n_levels[highest_n]

        assert n_levels[highest_n] < eu_levels[lowest_eu], (
            "every nitrogen on-site level should lie below every europium one "
            "-- that ordering is what makes the one-pass path pour charge onto "
            "nitrogen without limit. It no longer holds: nitrogen's highest is "
            f"{highest_n} at {n_levels[highest_n]} eV and europium's lowest is "
            f"{lowest_eu} at {eu_levels[lowest_eu]} eV, a margin of {margin} eV "
            "(negative means they overlap). Nitrogen levels (eV): "
            + ", ".join(f"{label} {value}" for label, value in n_levels.items())
            + ". Europium levels (eV): "
            + ", ".join(f"{label} {value}" for label, value in eu_levels.items())
            + ". If this fails, the parameter files changed and the mechanism "
            "section of docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md needs rewriting."
        )

    run_with_float64(check)


# --- The tripwire -----------------------------------------------------------


def test_the_phase_four_pin_is_not_the_settled_answer(tmp_path):
    """The Phase 4 pinned reference belongs to the one-pass path only.

    A reader who compares ``EU_N_REFERENCE_E_TOT`` against a settled energy is
    comparing two different observables.  That is the single most likely way to
    draw a wrong conclusion from this codebase, and it is what
    docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md exists to prevent.

    The assertion points the opposite way from a normal tolerance: it is a
    floor on a difference, not a ceiling.  It fires if some future change makes
    the two paths agree, because agreement would mean Phase 4 decision D-11 --
    the one that keeps electron repulsion out of the one-pass energy -- had
    been silently undone.  The observed difference is 10.59 eV against a floor
    of 1.0, an order of magnitude of headroom.
    """

    def check():
        settled = _run(tmp_path, do_scf=True)
        settled_total = settled.e_tot.item()
        gap = abs(settled_total - EU_N_REFERENCE_E_TOT)

        assert gap > MINIMUM_GAP_TO_THE_PHASE_FOUR_PIN, (
            "the settled total energy has converged onto the Phase 4 one-pass "
            f"pin: settled e_tot = {settled_total} eV against "
            f"EU_N_REFERENCE_E_TOT = {EU_N_REFERENCE_E_TOT} eV, a difference "
            f"of only {gap} eV, below the floor of "
            f"{MINIMUM_GAP_TO_THE_PHASE_FOUR_PIN} eV. That is not good news -- "
            "the two paths compute different quantities by construction, so "
            "agreement means the one-pass path has started including the "
            "electron-repulsion term that Phase 4 decision D-11 excludes, or "
            "the settled path has stopped including it. Check the two None "
            "arguments at ESDriver.py:1030-1031. Settled run: e_coul = "
            f"{_as_float(settled.e_coul)}, scf_iter_count = "
            f"{getattr(settled, 'scf_iter_count', '<unset>')}, q = "
            f"{settled.q.tolist()}"
        )

    run_with_float64(check)
