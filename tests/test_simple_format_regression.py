"""REG-02 regression baseline for the tutorial notebook's simple-format calculations.

REG-02 asks that "existing simple-format calculations in
``experiments/1_tutorial.ipynb`` remain scientifically unchanged within documented
tolerances".  The notebook itself cannot be executed in this environment, for three
separately verified reasons: its first computational cell reads a ``COORD.pdb`` that was
never committed, one cell hardcodes a CUDA device on a machine with no CUDA driver, and
no notebook-execution tooling (nbformat / nbclient / papermill / nbval / ipykernel /
matplotlib) is installed.

At the plan 05-02 checkpoint the developer selected ``option-a``: reproduce the
notebook's runnable simple-format calculations here, as ordinary pytest cases, and pin
their numbers.  **REG-02 is therefore satisfied in substance, not in form.**  This module
never executes the notebook, so a change that breaks only the notebook's own plumbing --
a renamed keyword argument, a changed constructor signature -- will not be caught here.
Cell 5's PBC + PME + MD path is not covered at all, because its input file is missing
from the repository.

``docs/REG-02-NOTEBOOK-BASELINE.md`` carries the full decision record, the evidence for
each obstacle, and the complete list of what this approach does not cover.  Read it
before reporting REG-02 as satisfied.

**Why this is stronger than what already existed.**  ``tests/test_scf.py`` runs a
similar calculation but asserts only finiteness and shape, so it would not notice a
numeric drift.  This module pins total energies, full force arrays and per-atom Mulliken
charges, which is what Phase 5 needs while D-01, D-02 and D-04 edit roughly twenty files.

**All reference values were measured on commit db63487, before any 05-0* commit landed.**
A baseline captured after the code it is meant to protect has already changed is not a
baseline.  None of these numbers came from the notebook's stored cell outputs, which were
produced on unknown hardware running unknown code.
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

    Copied from ``tests/test_f_orbital_skf.py`` - the established harness for every
    numeric test module in this project.  Deliberately copied rather than imported
    across test modules, matching this suite's convention.
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


def _experiments_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments"


# --- Tolerances -------------------------------------------------------------
#
# These rest on a measurement, not on a guess.  Two things were established on
# commit db63487 before the numbers below were chosen:
#
#   1. Repeating any case in the same process reproduces energy, forces and charges
#      bit-identically (difference exactly 0.0).  So no tolerance here is absorbing
#      run-to-run noise; the bands exist to survive a different LAPACK on a different
#      machine, which is what CI provides.
#
#   2. Sensitivity to SCF convergence was measured directly, by tightening SCF_TOL from
#      its default 1e-6 to 1e-10 and diffing:
#
#          total energy      shifted by 0.0 exactly
#          force component   shifted by 3.9e-08 eV/Angstrom
#          Mulliken charge   shifted by 5.0e-09 electrons
#
#      The energy does not move at all because it is variational in the converged
#      density: a density error enters the energy at second order.  Forces and charges
#      are first order in that same error, which is why they get looser bands than the
#      energy even though they are numerically smaller quantities.
#
# Each band therefore sits roughly 20-25x above its own measured floor: tight enough to
# catch a real change, loose enough not to be a detector for the SCF iteration path.
ENERGY_TOL_EV = 1e-8
FORCE_TOL_EV_PER_ANGSTROM = 1e-6
CHARGE_TOL_ELECTRONS = 1e-7

# Neutrality is exact by construction (Mulliken population minus nuclear charge, summed
# over a neutral system), so this one is a hard identity check rather than a physical
# tolerance.  Measured residuals are ~1e-14 and ~1e-16.
CHARGE_SUM_TOL_ELECTRONS = 1e-9

# The largest relative energy tolerance this module will permit, enforced by
# test_tolerances_are_documented_not_ad_hoc.  Phase 4's D-24 established the principle:
# a band that can be quietly widened is not a gate.
MAX_RELATIVE_ENERGY_TOL = 1e-6


# --- Case table -------------------------------------------------------------
#
# Every numeric literal below was measured on commit db63487 (2026-07-30) on CPU under
# torch.set_default_dtype(torch.float64), by running the case's own parameter dictionary
# through Constants -> Structure -> ESDriver(do_scf=True) -> calc_forces.  The notebook
# cell each case derives from is named in its "notebook_cell" field and repeated in the
# provenance comment above each reference block.

SIMPLE_FORMAT_CASES = {
    "o2_mio_unrestricted": {
        "notebook_cell": 17,
        "docstring": (
            "Notebook cell 17: open-shell O2 with the mio-1-1 parameter set. "
            "Reproduced faithfully; cell 17 already guards its device selection "
            "with torch.cuda.is_available(), so pinning it to CPU changes nothing "
            "about what the cell asks for."
        ),
        "geometry": "O2.xyz",
        "skf": ("sk_orig", "mio-1-1", "mio-1-1"),
        "params": {
            "T_ELECTRONIC": 500.0,
            "RCUT_ELECTRONIC": 8.0,
            "RCUT_REPULSIVE": 4.0,
            "UNRESTRICTED": True,
            "SPIN_POL": 2,
            # "!PME" is copied verbatim from the notebook cell rather than normalised
            # to "FULL", so that this case runs whatever the notebook actually runs.
            "COUL_METHOD": "!PME",
        },
        "n_atoms": 2,
        "system_charge": 0.0,
        # Measured: cell 17 case, commit db63487.
        "e_tot": -9.031346598527405,
        # Measured: cell 17 case, commit db63487.  The y and z rows are zero by the
        # symmetry of a diatomic lying along x; they were measured at |f| < 4e-15 and
        # are recorded as exact zeros so that this assertion is a symmetry check rather
        # than a pin on meaningless last-digit noise.  The two x components sum to
        # 7e-15, i.e. Newton's third law holds to machine precision.
        "f_tot": [
            [-1.2964871718334248, 1.296487171833432],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # Measured: cell 17 case, commit db63487.  Both atoms carry exactly zero net
        # charge because O2 is homonuclear; measured at |q| < 2e-16.
        "charges": [0.0, 0.0],
        "tolerance_rationale": (
            "Energy 1e-8 eV: this case's energy did not move at all (0.0) when SCF_TOL "
            "was tightened from 1e-6 to 1e-10, because the total energy is variational "
            "in the converged density. 1e-8 eV is 1e-9 relative here, far inside the "
            "1e-6 relative ceiling, and leaves headroom for a different LAPACK. "
            "Forces 1e-6 eV/Angstrom: forces are first order in the density error, so "
            "they carry the SCF tolerance rather than its square; this case's forces "
            "also happened to be bit-stable under the SCF tightening, but the band is "
            "set from the general first-order argument, not from this one lucky case. "
            "Charges 1e-7 electrons: same first-order argument, against a measured "
            "5e-9 sensitivity on the water case."
        ),
    },
    "o2_3ob_dftb3": {
        "notebook_cell": 25,
        "docstring": (
            "Notebook cell 25: the same open-shell O2 geometry with the 3ob-3-1 "
            "parameter set and DFTB3 third-order corrections enabled. This is the "
            "only case in the module exercising the DFTB3 branch."
        ),
        "geometry": "O2.xyz",
        "skf": ("sk_orig", "3ob-3-1"),
        "params": {
            "T_ELECTRONIC": 500.0,
            "RCUT_ELECTRONIC": 8.0,
            "RCUT_REPULSIVE": 4.0,
            "UNRESTRICTED": True,
            "SPIN_POL": 2,
            "COUL_METHOD": "!PME",
            "DFTB3": True,
        },
        "n_atoms": 2,
        "system_charge": 0.0,
        # Measured: cell 25 case, commit db63487.
        "e_tot": -8.270081470047817,
        # Measured: cell 25 case, commit db63487.  Transverse rows zero by symmetry as
        # in the mio case above (measured |f| < 4e-15).
        "f_tot": [
            [-0.6831752639601807, 0.6831752639601487],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        # Measured: cell 25 case, commit db63487.  Homonuclear, so exactly zero;
        # measured at |q| < 4e-15.
        "charges": [0.0, 0.0],
        "tolerance_rationale": (
            "Identical basis to o2_mio_unrestricted -- the tolerances are properties "
            "of the quantities (an extensive energy in eV, a force component in "
            "eV/Angstrom, a Mulliken charge in electrons) and of the measured SCF "
            "sensitivity, not of the parameter set. DFTB3 adds a third-order term to "
            "the energy expression but does not change the numerical character of any "
            "of the three observables, so it does not warrant a different band."
        ),
    },
    "water8_mio_full": {
        "notebook_cell": 14,
        "docstring": (
            "Notebook cell 14, adapted: a single 8-water box on CPU rather than the "
            "cell's four-member batch on CUDA. Two deliberate departures, both "
            "recorded in docs/REG-02-NOTEBOOK-BASELINE.md. (1) The batch of four "
            "structures is reduced to one, taking CELL [21, 21, 21] -- the first "
            "member of the cell's four-cell tensor. StructureBatch, ESDriverBatch and "
            "the batched force path are therefore NOT exercised by this module. "
            "(2) The cell's unconditional device = 'cuda' is replaced by CPU, because "
            "this cell is the one place in the notebook with no is_available() guard "
            "and there is no CUDA driver here. This case is the module's only PBC and "
            "only COUL_METHOD='FULL' coverage, and its only many-atom force and charge "
            "array."
        ),
        "geometry": "COORD_8WATER.xyz",
        "skf": ("sk_orig", "mio-1-1", "mio-1-1"),
        "params": {
            "CELL": [21.0, 21.0, 21.0],
            "T_ELECTRONIC": 1000.0,
            "RCUT_ELECTRONIC": 8.0,
            "RCUT_REPULSIVE": 4.0,
            "COUL_METHOD": "FULL",
        },
        "n_atoms": 24,
        "system_charge": 0.0,
        # Measured: cell 14 case (single-structure CPU variant), commit db63487.
        "e_tot": -110.65745489032126,
        # Measured: cell 14 case (single-structure CPU variant), commit db63487.
        # Shape (3, 24): rows are x, y, z; columns are atoms in file order.
        "f_tot": [
            [
                0.33829149310132745, -0.06858800507676532, -0.293449491556139,
                1.6332070635930434, -1.6140239959802507, 0.007005327179393839,
                -0.6740990448669351, 0.26028858105516406, 0.4602621549636241,
                1.1874252327318349, -0.6309022474032178, -0.5857998522010225,
                -3.0849050352406984, 2.823361659768203, 0.29646729159041385,
                0.1151662020430641, -0.06765174104795046, -0.017140543821765775,
                0.21948818815513782, -0.9184190384331892, 0.6664804411993221,
                2.6301212261295515, -2.503497170777642, -0.17908869510459358,
            ],
            [
                -1.507580263364825, -0.7803542804431345, 2.21605490085365,
                0.8892952822413981, -0.9752326963414926, 0.25244879177260193,
                0.434759034689165, -0.3665594207777054, -0.06002277288896085,
                -2.6340548589681436, -2.3707854908346087, 5.0400712664194,
                1.3823909441471773, -1.9935200471542966, 0.5470012212213371,
                -0.8467907671503898, 0.49431010591771773, 0.3086380955861241,
                -3.706795851531089, 0.890298933899401, 2.8177318973845686,
                -7.194141800302871, 8.570064510579156, -1.4072267349540875,
            ],
            [
                2.8723984799029987, 0.03760642441320683, -2.906949509032102,
                -0.011345036584545554, -0.3161763655022367, 0.4389522472021046,
                -0.06291465285018738, 0.2856754412762933, -0.27524415557350324,
                -8.961616351117362, 2.3646173434332205, 6.539321273515879,
                -0.6110524654959502, 1.6207651220915196, -0.9500907060261858,
                0.1484690088610563, 0.28996030281871565, -0.4223237091648864,
                0.21358029581270443, -0.7232901922822457, 0.51733453613769,
                -3.419540655792117, 3.131042231633078, 0.2008210923229159,
            ],
        ],
        # Measured: cell 14 case (single-structure CPU variant), commit db63487.
        # Per-atom Mulliken charges in file order: each water contributes O, H, H, so
        # the repeating (positive, negative, negative) pattern is the expected
        # oxygen-then-two-hydrogens ordering with charge flowing from H to O.
        "charges": [
            0.6097396035982232, -0.31617296823830465, -0.2982078941432126,
            0.6103271517150612, -0.3060071159070604, -0.29872916791038184,
            0.5953588856959675, -0.2938296325431986, -0.3007502348523923,
            0.5997567589195569, -0.29199019823717287, -0.3155930714594106,
            0.5986306638356247, -0.3020426976008316, -0.2888883958594852,
            0.594580490343501, -0.2950777876886501, -0.29863195212918714,
            0.5790330284034444, -0.30212993727574455, -0.2766980578854067,
            0.6029506246488221, -0.30965458493453113, -0.2959735104952512,
        ],
        "tolerance_rationale": (
            "This is the case the tolerances were actually measured against, because "
            "it is the only one with a non-trivial charge distribution. Tightening "
            "SCF_TOL from 1e-6 to 1e-10 moved the energy by exactly 0.0, the largest "
            "force component by 3.9e-8 eV/Angstrom, and the largest charge by 5.0e-9 "
            "electrons. The chosen bands (1e-8 eV, 1e-6 eV/Angstrom, 1e-7 electrons) "
            "sit 25x above the force floor and 20x above the charge floor. The energy "
            "band of 1e-8 eV is 9e-11 relative on this -110.66 eV total, which is the "
            "tightest relative pin in the module and still far above the ~1e-12 "
            "absolute drift a different LAPACK could plausibly introduce at this "
            "magnitude."
        ),
    },
}


# Results are cached because each case is a full SCF plus a force evaluation, and three
# separate tests interrogate each one.  Recomputing per test would triple the module's
# runtime for no additional coverage -- the runs were verified bit-identical.
_RESULT_CACHE = {}


def run_case(case_name):
    """Build and run one case, returning (e_tot, f_tot, charges) as plain tensors.

    Mirrors the construction sequence in ``tests/test_scf.py`` and in notebook cells
    14, 17 and 25: Constants -> Structure -> ESDriver(do_scf=True) -> calc_forces.
    """
    if case_name in _RESULT_CACHE:
        return _RESULT_CACHE[case_name]

    case = SIMPLE_FORMAT_CASES[case_name]
    experiments = _experiments_dir()
    geometry = experiments / case["geometry"]
    skf_dir = experiments.joinpath(*case["skf"])

    assert geometry.is_file(), f"Missing notebook geometry: {geometry}"
    assert skf_dir.is_dir(), f"Missing notebook SKF directory: {skf_dir}"

    def compute():
        from dftorch.Constants import Constants
        from dftorch.ESDriver import ESDriver
        from dftorch.Structure import Structure

        params = dict(case["params"])
        params["FILENAME"] = str(geometry)
        params["SKFPATH"] = str(skf_dir) + os.sep

        const = Constants(params).to("cpu")
        structure = Structure(params, const, device="cpu")
        driver = ESDriver(params, device="cpu")
        driver(structure, const, do_scf=True)
        driver.calc_forces(structure, const)

        return (
            structure.e_tot.detach().clone(),
            structure.f_tot.detach().clone(),
            structure.q.detach().clone(),
        )

    result = run_with_float64(compute)
    _RESULT_CACHE[case_name] = result
    return result


CASE_NAMES = sorted(SIMPLE_FORMAT_CASES)


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_total_energy_is_pinned(case_name):
    """The total energy matches the value measured on commit db63487.

    A failure here does not mean the test went stale.  It means a simple-format total
    energy moved, which is exactly what REG-02 exists to detect while Phase 5 edits the
    radial-grid lookup (D-01), gates the library print calls (D-02) and audits the
    orbital-count sites (D-04).  Investigate the move; do not widen the band.
    """
    case = SIMPLE_FORMAT_CASES[case_name]
    e_tot, _, _ = run_case(case_name)

    assert torch.isfinite(e_tot).all()
    observed = e_tot.item()
    expected = case["e_tot"]
    assert abs(observed - expected) < ENERGY_TOL_EV, (
        f"{case_name} (notebook cell {case['notebook_cell']}) total energy moved: "
        f"expected {expected!r} eV, got {observed!r} eV, "
        f"difference {abs(observed - expected):.3e} eV exceeds {ENERGY_TOL_EV:.0e} eV"
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_forces_are_pinned(case_name):
    """Forces keep their shape, stay finite, and match the pinned array.

    Forces are the more sensitive probe of the two: they are first order in the
    converged density where the total energy is second order, so a change that the
    energy absorbs variationally can still show up here.
    """
    case = SIMPLE_FORMAT_CASES[case_name]
    _, f_tot, _ = run_case(case_name)

    assert f_tot.shape == (3, case["n_atoms"]), (
        f"{case_name} force array shape changed: expected (3, {case['n_atoms']}), "
        f"got {tuple(f_tot.shape)}"
    )
    assert torch.isfinite(f_tot).all()

    expected = torch.tensor(case["f_tot"], dtype=f_tot.dtype)
    deviation = (f_tot - expected).abs().max().item()
    assert deviation < FORCE_TOL_EV_PER_ANGSTROM, (
        f"{case_name} (notebook cell {case['notebook_cell']}) forces moved: "
        f"maximum per-component deviation {deviation:.3e} eV/Angstrom exceeds "
        f"{FORCE_TOL_EV_PER_ANGSTROM:.0e} eV/Angstrom"
    )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_mulliken_charges_are_pinned(case_name):
    """Per-atom Mulliken charges match the pinned array and still sum to the system charge.

    The sum check is a separate and much tighter assertion than the per-atom one: charge
    neutrality is an exact identity of the construction (Mulliken population minus
    nuclear charge over a neutral system), not a converged quantity, so it holds to
    ~1e-14 regardless of how well the SCF converged.  A per-atom drift with the sum
    still exact means charge moved between atoms; a broken sum means something more
    fundamental changed.
    """
    case = SIMPLE_FORMAT_CASES[case_name]
    _, _, charges = run_case(case_name)

    assert charges.shape == (case["n_atoms"],), (
        f"{case_name} charge array shape changed: expected ({case['n_atoms']},), "
        f"got {tuple(charges.shape)}"
    )
    assert torch.isfinite(charges).all()

    expected = torch.tensor(case["charges"], dtype=charges.dtype)
    deviation = (charges - expected).abs().max().item()
    assert deviation < CHARGE_TOL_ELECTRONS, (
        f"{case_name} (notebook cell {case['notebook_cell']}) Mulliken charges moved: "
        f"maximum per-atom deviation {deviation:.3e} e exceeds "
        f"{CHARGE_TOL_ELECTRONS:.0e} e"
    )

    charge_sum = charges.sum().item()
    assert abs(charge_sum - case["system_charge"]) < CHARGE_SUM_TOL_ELECTRONS, (
        f"{case_name} charges no longer sum to the system charge: "
        f"sum {charge_sum!r} e, expected {case['system_charge']!r} e"
    )


def test_tolerances_are_documented_not_ad_hoc():
    """Every case must justify its tolerance, and no energy band may exceed 1e-6 relative.

    Phase 4's D-24 established the principle this test enforces: a band that can be
    quietly widened is not a gate.  If a future change makes one of the pinned numbers
    fail, the correct response is to investigate the move and, if it is a deliberate and
    understood change, re-measure the reference with a recorded reason -- not to loosen
    the tolerance until the comparison passes.

    The relative cap is checked against each case's own pinned energy, so it binds
    hardest on the largest system, which is where an absolute band is weakest.
    """
    assert SIMPLE_FORMAT_CASES, "SIMPLE_FORMAT_CASES must not be empty"

    for case_name, case in SIMPLE_FORMAT_CASES.items():
        rationale = case.get("tolerance_rationale")
        assert isinstance(rationale, str), (
            f"{case_name} has no tolerance_rationale string"
        )
        assert rationale.strip(), f"{case_name} has an empty tolerance_rationale"

        relative_energy_tol = ENERGY_TOL_EV / abs(case["e_tot"])
        assert relative_energy_tol <= MAX_RELATIVE_ENERGY_TOL, (
            f"{case_name} energy tolerance is too loose to be a gate: "
            f"{ENERGY_TOL_EV:.0e} eV is {relative_energy_tol:.3e} relative against a "
            f"pinned energy of {case['e_tot']!r} eV, exceeding the "
            f"{MAX_RELATIVE_ENERGY_TOL:.0e} relative ceiling"
        )


def test_uncovered_notebook_paths_are_recorded():
    """The limitations of this substitute must stay written down next to it.

    REG-02's literal wording is not satisfied by this module, and the risk the plan
    named explicitly (T-05-06, "REG-02 reported satisfied by an artifact that does not
    test it") is a documentation risk rather than a code one.  This test is the only
    thing standing between that document being deleted or hollowed out and REG-02
    silently reading green.  It checks the decision record still names the selected
    option, the missing input file, and the section that lists what is not covered.
    """
    doc = Path(__file__).resolve().parents[1] / "docs" / "REG-02-NOTEBOOK-BASELINE.md"
    assert doc.is_file(), f"REG-02 decision record is missing: {doc}"

    text = doc.read_text(encoding="utf-8")
    for needle in (
        "option-a",
        "COORD.pdb",
        "What this does NOT cover",
    ):
        assert needle in text, (
            f"docs/REG-02-NOTEBOOK-BASELINE.md no longer contains {needle!r}. "
            "REG-02 is satisfied in substance and not in form; the record of that "
            "distinction is not optional."
        )
