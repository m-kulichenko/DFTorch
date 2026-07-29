"""Eu-N diatomic single-shot energy scan - the SIM-05 loose-band sanity gate.

Scans the interatomic separation of an isolated Eu-N diatomic from 1.60 A to
3.60 A and asserts that the single-shot (non-SCC) total energy has a well whose
minimum sits inside a loose band around 2.655 A.

**Target and band.** The target separation is 2.655 A, the mean of 16 Eu-N bond
lengths from a table of Eu-Bp coordination complexes across four ligand variants
(Bp, Bp^Me, Bp^Me2, Bp^CF3), range 2.606-2.716 A.  The band is +/-20%, giving
[2.124, 3.186] A.  The band is ALWAYS derived as a fraction of the target and
never as an absolute half-width - an earlier draft mis-stated it as
"2.4 +/- 0.2 A", which is +/-8% and would have silently tightened this gate far
past what the method can support.  ``test_tolerance_band_is_a_fraction_of_the_target``
exists solely so that such a substitution fails the suite instead of passing
quietly.

**Energy scan, never geometry optimization.**  f derivatives are unimplemented:
``calc_forces``, ``ESDriverBatch.calc_forces`` and analytical stress all raise
``FDerivativeUnsupportedError`` for any system containing a 16-orbital atom, so
an optimizer-based approach hits a hard wall immediately.  A scan needs only
energy, which the single-shot path provides.

**ASYMMETRIC FAILURE RULE - read this before concluding anything about the f
implementation.**  A failure of this gate does not mean the same thing in both
directions:

* **No interior minimum at all** (the minimum sits at either endpoint of the
  scanned range) is a **genuine red flag**.  The band assertion is meaningless
  without a well; inspect the curve for monotonic drift.
* **A minimum SHORT of the band** (below 2.124 A) is **plausibly correct
  isolated-diatomic physics and is NOT evidence that the f implementation is
  broken.**  The 2.655 A target is a mean over *dative* Eu-N bonds in a crowded
  8-9-coordinate sphere; an isolated gas-phase diatomic would plausibly be
  shorter.  This must be **escalated to human judgement**: compare against bulk
  rock-salt EuN (~2.45-2.5 A) and against the curve shape before touching the f
  code.
* **A minimum LONG of the band** (above 3.186 A) is a **genuine red flag**.  The
  diatomic-versus-coordination argument only runs short, so it does not excuse an
  over-long bond.

**There is no reference paper for this case and none is claimed.**  A green gate
is a smoke test: the +/-20% band absorbs an f-block error of 15%, single-shot is
not SCF, and a closed-shell treatment of an open-shell 4f7 ion is an
approximation.  See ``tests/f_orbital_data/README-EU-N-CASE.md`` for the full
case document - geometry, SKF parameter files, observable, units, tolerance,
provenance, and what a passing test does not prove.
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

    Copied from ``tests/test_f_orbital_skf.py`` - the established harness for
    every f-orbital test module in this project.  Deliberately copied rather
    than imported across test modules, matching this suite's convention.
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


# --- Tolerance band (D-20, D-22, D-24) --------------------------------------
#
# DERIVATION ORDER MATTERS.  The two band edges below are arithmetic
# expressions over the target and the fraction.  Do NOT replace either with a
# numeric literal, and do NOT re-express the band as an absolute half-width:
# both are exactly the regressions test_tolerance_band_is_a_fraction_of_the_target
# is here to catch.  Widening or narrowing the band requires a recorded decision,
# not a constant edit.
EU_N_TARGET_ANGSTROM = 2.655
EU_N_BAND_FRACTION = 0.20
EU_N_BAND_MIN_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 - EU_N_BAND_FRACTION)
EU_N_BAND_MAX_ANGSTROM = EU_N_TARGET_ANGSTROM * (1.0 + EU_N_BAND_FRACTION)

# --- Scan grid ---------------------------------------------------------------
#
# The range must BRACKET both band edges.  If it stopped at a band edge, a
# minimum that genuinely wanted to sit outside the band would be clipped to the
# edge and this gate would report a false pass.  1.60 A also reaches low enough
# to *observe* a short minimum rather than hiding it, which the asymmetric
# failure rule requires.
SCAN_MIN_ANGSTROM = 1.60
SCAN_MAX_ANGSTROM = 3.60
SCAN_STEP_ANGSTROM = 0.10

# Driver parameters, pinned to match tests/test_single_shot_energy.py exactly so
# that the reference energies recorded there transfer to this curve.
EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}


def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"


def scan_separations() -> list[float]:
    """Return the uniform scan grid, inclusive of both endpoints.

    Each value is rounded to two decimals so the grid is exact in printed
    diagnostics and reproducible across platforms, rather than carrying the
    accumulated error of repeated float addition.
    """
    count = int(round((SCAN_MAX_ANGSTROM - SCAN_MIN_ANGSTROM) / SCAN_STEP_ANGSTROM)) + 1
    return [round(SCAN_MIN_ANGSTROM + SCAN_STEP_ANGSTROM * i, 2) for i in range(count)]


def _write_eu_n_xyz(path: Path, separation: float) -> None:
    """Eu at the origin, N displaced along +x by ``separation``."""
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 4 separation scan)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _eu_n_params(separation: float, tmp_dir: Path) -> dict:
    xyz_path = Path(tmp_dir) / f"eu_n_{separation:.2f}.xyz"
    _write_eu_n_xyz(xyz_path, separation)

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_skf_dir()) + os.sep
    return params


def single_shot_energy(separation, tmp_dir, *, const=None, driver=None):
    """Return ``structure.e_tot`` for the Eu-N diatomic at ``separation``.

    Writes the geometry into ``tmp_dir``, builds ``Constants`` and ``Structure``
    with the module's pinned parameters, and drives the system through
    ``forward(do_scf=False)``.

    This helper does NOT call ``calc_forces``, ``calc_stress``, or any optimizer
    routine, and must never be changed to: those raise
    ``FDerivativeUnsupportedError`` for f systems, so an optimizer-based approach
    to locating the minimum is a hard wall (D-19).  Only the energy is needed.

    ``const`` and ``driver`` are optional caches.  Called with neither, the
    helper is fully standalone and builds both from scratch.  ``Constants`` reads
    the geometry file only to recover the species list, which is identical
    (Eu, N) at every point of this scan, so the module-scoped fixture builds it
    once and threads it in here.  That is a pure speed measure and was verified
    to be bit-identical: constructing ``Constants`` per point versus reusing one
    instance gives the same 21 energies to the last digit, while the scan drops
    from 46 s to 9 s - which is what keeps this module inside the 30-second
    feedback budget.
    """
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    params = _eu_n_params(separation, tmp_dir)

    if const is None:
        const = Constants(params).to("cpu")
    if driver is None:
        driver = ESDriver(params, device="cpu")

    structure = Structure(params, const, device="cpu")
    driver(structure, const, do_scf=False)

    # do_scf=False leaves SCF bookkeeping attributes (H, Hcoul, KK, Q, dq_p1,
    # ...) unset by design.  e_tot is the only attribute this scan reads.
    return structure.e_tot


def locate_minimum(separations, energies):
    """Return ``(index, separation)`` of the lowest energy on the grid.

    Minimises the tuple ``(energy, separation)``, so on an exact float64 tie
    between two grid points the SMALLER separation wins.

    That tie-break is deterministic, and it is also the conservative choice given
    the asymmetric failure rule in this module's docstring: a short reading is
    escalated for human judgement (it is plausibly correct diatomic physics),
    whereas a long reading is a genuine red flag.  Breaking ties toward the
    shorter separation therefore biases an ambiguous curve toward the outcome
    that gets a human involved rather than the one that would be read as a
    definite defect.  ``test_eu_n_minimum_is_unique_on_the_grid`` records that
    the real curve never actually exercises this branch.
    """
    index = min(range(len(energies)), key=lambda i: (energies[i], separations[i]))
    return index, separations[index]


def _format_curve(separations, energies) -> str:
    return "\n".join(
        f"    {r:.2f} A   {e: .10f} eV" for r, e in zip(separations, energies)
    )


@pytest.fixture(scope="module")
def eu_n_scan(tmp_path_factory):
    """Compute the whole 21-point scan once for the entire module.

    Computing the curve once is what keeps this module inside the 30-second
    feedback budget; recomputing it inside each of the six tests would multiply
    the cost by six.  Returns plain Python lists (no tensors and no dftorch
    objects), because ``run_with_float64`` unloads the dftorch modules and
    restores the default dtype on the way out.
    """

    def compute():
        from dftorch.Constants import Constants
        from dftorch.ESDriver import ESDriver

        tmp_dir = tmp_path_factory.mktemp("eu_n_scan")
        separations = scan_separations()

        # Built once from the first grid point and reused; see single_shot_energy.
        params = _eu_n_params(separations[0], tmp_dir)
        const = Constants(params).to("cpu")
        driver = ESDriver(params, device="cpu")

        energies = []
        dtypes = []
        for separation in separations:
            e_tot = single_shot_energy(
                separation, tmp_dir, const=const, driver=driver
            )
            energies.append(float(e_tot.item()))
            dtypes.append(str(e_tot.dtype))

        return {
            "separations": separations,
            "energies": energies,
            "dtypes": dtypes,
        }

    return run_with_float64(compute)


def test_tolerance_band_is_a_fraction_of_the_target():
    """The band is derived from the target, not hardcoded as a half-width.

    No physics runs here.  This is the regression guard for D-24: an edit that
    substitutes an absolute half-width (for example the mis-stated
    "2.4 +/- 0.2 A", which is +/-8% rather than +/-20%) silently tightens this
    gate past what the method can support, and would make the suite fail for
    reasons unrelated to the f implementation.  Such an edit must fail HERE,
    loudly, instead of passing quietly.
    """
    assert EU_N_TARGET_ANGSTROM == 2.655
    assert EU_N_BAND_FRACTION == 0.20

    assert EU_N_BAND_MIN_ANGSTROM == EU_N_TARGET_ANGSTROM * (1.0 - EU_N_BAND_FRACTION)
    assert EU_N_BAND_MAX_ANGSTROM == EU_N_TARGET_ANGSTROM * (1.0 + EU_N_BAND_FRACTION)

    assert abs(EU_N_BAND_MIN_ANGSTROM - 2.124) < 1e-12
    assert abs(EU_N_BAND_MAX_ANGSTROM - 3.186) < 1e-12

    # The band must stay loose (D-20/D-22: roughly 10-20%).  A fraction below
    # 0.10 is a tightening that the method cannot support.
    assert EU_N_BAND_FRACTION >= 0.10


def test_scan_grid_brackets_the_tolerance_band():
    """The grid straddles both band edges, so a near-edge minimum is resolvable.

    No physics runs here.  If the scan stopped at a band edge, a minimum that
    genuinely wanted to sit outside the band would be clipped to the edge and
    this gate would report a false pass.  Both endpoints must therefore lie
    outside the band, and grid points must exist strictly on both sides of each
    edge.
    """
    separations = scan_separations()

    assert SCAN_MIN_ANGSTROM < EU_N_BAND_MIN_ANGSTROM
    assert SCAN_MAX_ANGSTROM > EU_N_BAND_MAX_ANGSTROM

    # Both scan endpoints lie outside the band.
    assert separations[0] < EU_N_BAND_MIN_ANGSTROM
    assert separations[-1] > EU_N_BAND_MAX_ANGSTROM

    for edge_name, edge in (
        ("band minimum", EU_N_BAND_MIN_ANGSTROM),
        ("band maximum", EU_N_BAND_MAX_ANGSTROM),
    ):
        below = [r for r in separations if r < edge]
        above = [r for r in separations if r > edge]
        assert below, f"no grid point strictly below the {edge_name} ({edge:.3f} A)"
        assert above, f"no grid point strictly above the {edge_name} ({edge:.3f} A)"

    assert len(separations) == 21
    assert separations[0] == SCAN_MIN_ANGSTROM
    assert separations[-1] == SCAN_MAX_ANGSTROM


def test_eu_n_energies_are_finite_and_float64(eu_n_scan):
    """Every scanned energy is a finite float64 scalar.

    A NaN or an inf anywhere in the curve makes ``locate_minimum`` meaningless,
    and a float32 energy would mean the float64 harness stopped taking effect -
    either way the SIM-05 gate below would be measuring nothing.
    """
    separations = eu_n_scan["separations"]
    energies = eu_n_scan["energies"]
    dtypes = eu_n_scan["dtypes"]

    assert len(energies) == len(separations) == 21

    for separation, energy in zip(separations, energies):
        assert isinstance(energy, float)
        assert energy == energy, f"NaN energy at {separation:.2f} A"
        assert abs(energy) != float("inf"), f"infinite energy at {separation:.2f} A"

    assert set(dtypes) == {"torch.float64"}, f"non-float64 energies: {set(dtypes)}"


def test_eu_n_energy_scan_locates_interior_minimum(eu_n_scan):
    """SIM-05: the energy well's minimum sits inside 2.655 A +/- 20%.

    The three failure branches below implement D-24's asymmetric failure rule.
    Read the module docstring before acting on a failure: a minimum SHORT of the
    band is plausibly correct isolated-diatomic physics and is NOT evidence that
    the f implementation is broken, while a minimum LONG of the band, or no
    interior minimum at all, is a genuine red flag.
    """
    separations = eu_n_scan["separations"]
    energies = eu_n_scan["energies"]

    index, separation = locate_minimum(separations, energies)
    last = len(separations) - 1

    context = (
        f"\n  located minimum: {separation:.2f} A at grid index {index} of {last}"
        f"\n  tolerance band:  [{EU_N_BAND_MIN_ANGSTROM:.3f}, "
        f"{EU_N_BAND_MAX_ANGSTROM:.3f}] A"
        f"  (target {EU_N_TARGET_ANGSTROM} A, "
        f"+/-{EU_N_BAND_FRACTION * 100:.0f}%)"
        f"\n  scanned range:   [{SCAN_MIN_ANGSTROM:.2f}, {SCAN_MAX_ANGSTROM:.2f}] A "
        f"step {SCAN_STEP_ANGSTROM:.2f} A"
        f"\n  full curve:\n{_format_curve(separations, energies)}"
    )

    if index == 0 or index == last:
        no_interior_minimum = (
            "GENUINE RED FLAG: no interior minimum in the Eu-N energy scan.\n"
            "The lowest energy sits at an endpoint of the scanned range, so the "
            "curve has no well and the tolerance-band check below is meaningless "
            "regardless of whether it would pass.\n"
            "Inspect the scanned curve for monotonic drift: a single clean well "
            "is expected, with the energy rising on both sides of the minimum. "
            "Check first that no charge-fluctuation Coulomb term has been added "
            "to the single-shot energy - feeding first-iterate Mulliken charges "
            "(q_Eu around -2.7, drifting to -2.99 at 4 A) into the "
            "electrostatics was measured to destroy the well entirely.\n"
            "See tests/f_orbital_data/README-EU-N-CASE.md."
            + context
        )
        raise AssertionError(no_interior_minimum)

    if separation < EU_N_BAND_MIN_ANGSTROM:
        short_of_band = (
            "MINIMUM SHORT OF THE BAND - ESCALATE TO HUMAN JUDGEMENT.\n"
            "This is NOT evidence that the f implementation is broken.\n"
            "A short minimum is plausibly correct isolated-diatomic physics: the "
            f"{EU_N_TARGET_ANGSTROM} A target is the mean of 16 *dative* Eu-N "
            "bonds in crowded 8-9-coordinate Eu-Bp complexes, not a diatomic "
            "measurement, and an isolated gas-phase Eu-N diatomic would "
            "plausibly be SHORTER than a coordination-sphere mean.\n"
            "A human must adjudicate before anything is concluded about the f "
            "code. Compare the located minimum against bulk rock-salt EuN "
            "(~2.45-2.5 A) and against the shape of the curve below, then decide "
            "whether to widen the band with a recorded decision or to "
            "investigate. Do NOT 'fix' the f implementation on this evidence "
            "alone, and do NOT silently edit the band constants.\n"
            "See tests/f_orbital_data/README-EU-N-CASE.md."
            + context
        )
        raise AssertionError(short_of_band)

    if separation > EU_N_BAND_MAX_ANGSTROM:
        long_of_band = (
            "GENUINE RED FLAG: minimum long of the band.\n"
            "The located minimum is longer than the upper band edge. Unlike a "
            "short minimum, this is NOT explained by the difference between an "
            "isolated diatomic and a dative coordination bond - that argument "
            "only runs short. An over-long Eu-N bond points at the f "
            "implementation, the Slater-Koster tables, or the repulsive spline.\n"
            "Investigate. See tests/f_orbital_data/README-EU-N-CASE.md."
            + context
        )
        raise AssertionError(long_of_band)

    assert EU_N_BAND_MIN_ANGSTROM <= separation <= EU_N_BAND_MAX_ANGSTROM


def test_eu_n_minimum_is_unique_on_the_grid(eu_n_scan):
    """Exactly one grid point attains the minimum energy.

    ``locate_minimum`` breaks an exact float64 tie toward the smaller
    separation.  This test records that the real curve never exercises that
    branch, so the located minimum is unambiguous rather than a consequence of
    the tie-break policy.
    """
    separations = eu_n_scan["separations"]
    energies = eu_n_scan["energies"]

    index, separation = locate_minimum(separations, energies)
    lowest = energies[index]
    winners = [r for r, e in zip(separations, energies) if e == lowest]

    assert winners == [separation], (
        "more than one grid point attains the minimum energy "
        f"{lowest:.10f} eV: {winners}. locate_minimum broke the tie toward the "
        f"smaller separation ({separation:.2f} A), so the located minimum "
        "reflects the tie-break policy rather than the curve.\n"
        f"  full curve:\n{_format_curve(separations, energies)}"
    )


def test_eu_n_curve_is_a_single_well(eu_n_scan):
    """The curve decreases strictly into the minimum and rises strictly after.

    This is the automated half of the manual "single well, no monotonic drift"
    sanity check the phase's validation strategy records.  A curve with a second
    local minimum, a plateau, or monotonic drift would still let the band
    assertion pass while meaning something quite different physically.
    """
    separations = eu_n_scan["separations"]
    energies = eu_n_scan["energies"]

    index, _ = locate_minimum(separations, energies)
    curve = _format_curve(separations, energies)

    for i in range(index):
        assert energies[i] > energies[i + 1], (
            f"energy does not decrease strictly from {separations[i]:.2f} A to "
            f"{separations[i + 1]:.2f} A on the approach to the minimum at "
            f"{separations[index]:.2f} A\n  full curve:\n{curve}"
        )

    for i in range(index, len(energies) - 1):
        assert energies[i] < energies[i + 1], (
            f"energy does not increase strictly from {separations[i]:.2f} A to "
            f"{separations[i + 1]:.2f} A beyond the minimum at "
            f"{separations[index]:.2f} A\n  full curve:\n{curve}"
        )
