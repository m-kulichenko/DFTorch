"""Characterization tests for the Slater-Koster radial grid and knot lookup.

WHY THIS MODULE EXISTS
----------------------
`_bond_integral.get_skf_tensors` already stores every element pair's own radial
grid in `R_tensor`, but it exports a single global `R_orb` chosen as the LONGEST
grid it saw (`_bond_integral.py:1064-1065`).  `_h0ands.H0_and_S_vectorized`
derives the spline knot from that one array for every pair type
(`_h0ands.py:243-245`), so a pair whose real grid differs is interpolated
against another pair's ruler.  Decision D-01 closes that (requirement REG-06).

Every parameter directory currently checked into this repository uses a single
radial STEP throughout, so the defect is inert today.  There is therefore no
observed failure to work backwards from, only a model of one.  That is exactly
why these tests are written BEFORE the lookup is rewritten: they record what the
code does now, so that a rewrite which quietly moves a knot fails a named test
instead of silently shifting every interpolated integral (requirement REG-01).

PROVENANCE OF THE LITERALS
--------------------------
Every numeric literal below was MEASURED on this machine on 2026-07-30/31 by
loading the real fixtures, not copied from prose.  Two chains of provenance:

* Grid step and length come from the SKF grid line parsed at
  `_bond_integral.py:597` (`step`, `npts_read`), padded to `npts_read + 50` at
  `:599`, and turned into an Angstrom grid at `:713-718` by
  `R = arange(1, npts_pad + 1) * step * BOHR_TO_ANGSTROM`.
* `idx` and `dx` come from re-running the production expression at
  `_h0ands.py:243-245` on the same inputs.

UNITS ARE SETTLED, AND THIS MODULE DOES NOT REOPEN THEM
-------------------------------------------------------
The grid is in ANGSTROM.  It starts one step in (`R[0] == step_A`), not at zero.
The docstring at `_ml_sk.py:437-450` claims `R_orb` is Angstrom while `dR_mskd`
is Bohr and that `searchsorted` therefore compares mixed-unit arrays.  That
claim is wrong, and the same file contradicts it eleven lines later at
`_ml_sk.py:522`, which annotates `dR_mskd` as Angstrom.
`test_grid_is_angstrom_uniform_progression` and
`test_ch4_knot_indices_are_unchanged` together pin the Angstrom reading:
a CH4 C-H separation of 1.0566812742799978 A over a 0.0105835442 A step lands on
`idx = 98` and `R_orb[98] = 1.0477708758` A, which is the correct physical knot.
Under the docstring's Bohr theory the knot would sit at roughly twice the true
separation.  Correcting that docstring belongs to plan 05-04, which owns
`_ml_sk.py`; this module only records the measurement.

ASCII ONLY.  Phase 4 recorded that em dashes render as replacement characters on
a cp1252 console and destroy the diagnosability of pytest output.
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

import pytest
import torch


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.

    Copied from ``tests/test_f_orbital_skf.py`` and ``tests/test_eu_n_scan.py``,
    the established harness for every numeric test module in this project.
    Deliberately copied rather than imported across test modules, matching this
    suite's convention.
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


# --- Fixture locations -------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mio_dir() -> Path:
    return _repo_root() / "tests" / "data_skf_mio-1-1"


def _f_dir() -> Path:
    return _repo_root() / "tests" / "f_orbital_data"


def _ch4_xyz() -> Path:
    return _repo_root() / "tests" / "ch4.xyz"


# --- Pinned grid geometry ----------------------------------------------------
#
# mio-1-1: the C-C / C-H / H-C / H-H files all carry the grid line "0.02, 500".
# 500 tabulated points padded by 50 (_bond_integral.py:599) gives 550, and the
# step in Angstrom is 0.02 Bohr * BOHR_TO_ANGSTROM.
MIO_GRID_LENGTH = 550
MIO_STEP_ANGSTROM = 0.0105835442
MIO_LAST_ANGSTROM = 5.82094931

# f_orbital_data: all nine files carry the grid line "0.04, 433".  433 + 50 gives
# 483, and 0.04 Bohr in Angstrom is 0.0211670884.
F_GRID_LENGTH = 483
F_STEP_ANGSTROM = 0.0211670884
F_LAST_ANGSTROM = 10.2237036972

# `_bond_integral.py:976` hardcodes npts = 1300, so R_tensor is (n_pairs, 1301)
# regardless of how long the real grids are.  Every index past a pair's own
# grid length is zero today; that is what Task 2 of plan 05-01 repairs.
R_TENSOR_WIDTH = 1301


# --- Pinned knot arithmetic --------------------------------------------------
#
# CH4 + mio-1-1, built exactly as tests/test_scf.py builds it (25 A cubic cell,
# RCUT_ELECTRONIC 8.0).  The neighbour list yields 20 ordered pairs (5 atoms,
# no self pairs).  idx/dx come from re-running _h0ands.py:243-245.
#
# NOTE ON idx.max(): the 05-01 plan text predicted 178.  The measured value on
# this machine is 177, and the plan explicitly instructs recording the measured
# value rather than forcing the predicted one.  177 is what
# `searchsorted(R_orb, 1.8875517915355329, right=True) - 1` returns:
# 1.8875517915355329 / 0.0105835442 = 178.34..., floor 178, minus 1 = 177.
CH4_PAIR_COUNT = 20
CH4_DR_MIN = 1.0566812742799978
CH4_DR_MAX = 1.8875517915355329
CH4_IDX_MIN = 98
CH4_IDX_MAX = 177
CH4_DX_MIN = 0.002540771531197583
CH4_DX_MAX = 0.009085704018195973
CH4_R_ORB_AT_98 = 1.0477708758

# Eu-N diatomic at the Phase 4 target separation of 2.655 A (see
# tests/f_orbital_data/README-EU-N-CASE.md) against tests/f_orbital_data.
# Two ordered pairs, Eu->N and N->Eu, both at the same separation.
EU_N_SEPARATION = 2.655
EU_N_PAIR_COUNT = 2
EU_N_IDX = 124
EU_N_DX = 0.009113950000000148

# --- Pinned effective cutoffs ------------------------------------------------
#
# `_ml_sk.build_pair_type_rcut(coeffs_tensor, R_orb)` walks each pair's spline
# coefficients back from the tail to the last nonzero interval and returns the
# radius at which the tabulated data stops.  The value is in the SAME ANGSTROM
# units as the grid.  mio-1-1 stops at 500 * 0.0105835442 = 5.2917721 A
# (10.0 Bohr); f_orbital_data stops at 433 * 0.0211670884 = 9.1653493 A
# (17.32 Bohr).  This test exists so that a later change to the lookup cannot
# move an effective cutoff unnoticed.
MIO_RCUT_ANGSTROM = 5.291772099999999
F_RCUT_ANGSTROM = 9.165349277199999

# --- Pinned CH4 H0/S checksums ----------------------------------------------
#
# Originally recorded on the PRE-D-01 code path (global `R_orb` lookup) so that Task 2
# had a real oracle for its bit-identity claim rather than only comparing the new path
# against itself.  That purpose is preserved; the values were RE-RECORDED 2026-08-02.
#
# Why they moved: the original numbers were reductions over 20x20 matrices, because the
# SKF parser gave hydrogen a phantom p shell (4 orbitals instead of 1) from a rounding-
# noise placeholder `Ep = 0.000039` in mio-1-1's H-H.skf.  CH4 is one carbon with 4
# orbitals plus four hydrogens with 1 each, so the correct matrix is 8x8 -- exactly what
# the comment beside the old `shape == (20, 20)` assertion already said.  The phantom
# orbitals also made the overlap matrix indefinite, which is why CH4 could never reach
# SCF convergence.  See tests/test_shell_count_parsing.py and the shell-presence block in
# `_bond_integral.read_skf_table`.
#
# These are exact float64 reprs of reductions over the 8x8 H0/S matrices and their
# 3x8x8 Cartesian derivatives, assembled by `H0_and_S_vectorized` for CH4 + mio-1-1.
# Scalar reductions are compared with zero relative tolerance and a four-ULP absolute
# bound: equivalent reduction orders can move the last few bits even when every tensor
# byte is identical.  The global-versus-per-pair tensors are still checked with
# `torch.equal` below, so elementwise bit identity remains exact and any larger checksum
# movement is still a regression under REG-01.
CH4_CHECKSUM_MAX_ULPS = 4
CH4_H0_SUM = -148.51616944363101
CH4_S_SUM = 12.604087520050435
CH4_H0_ABS_SUM = 251.90634572197465
CH4_S_ABS_SUM = 18.651448940203256
CH4_DH0_ABS_SUM = 500.92420893056124
CH4_DS_ABS_SUM = 27.86760866378902


def _assert_ch4_checksum(actual: float, expected: float, label: str) -> None:
    """Keep a scalar reduction within four ULPs of its historical value.

    Floating-point addition is non-associative, so CPU/vector reduction order
    can change a sum's final bits without moving any input tensor element.  A
    zero-relative four-ULP bound admits only that measured reduction noise; it
    is far narrower than a physics tolerance and still rejects meaningful
    Hamiltonian drift.
    """
    absolute_tolerance = CH4_CHECKSUM_MAX_ULPS * math.ulp(expected)
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=absolute_tolerance,
    ), (
        f"{label} checksum moved: actual={actual!r}, expected={expected!r}, "
        f"abs_diff={abs(actual - expected)!r}, allowed={absolute_tolerance!r} "
        f"({CH4_CHECKSUM_MAX_ULPS} ULP)"
    )


# --- Shared builders ---------------------------------------------------------

def _mio_params(xyz_path: Path) -> dict:
    """Driver parameters pinned to match tests/test_scf.py exactly."""
    return {
        "FILENAME": str(xyz_path),
        "CELL": [25.0, 25.0, 25.0],
        "SKFPATH": str(_mio_dir()) + os.sep,
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 8.0,
        "RCUT_REPULSIVE": 4.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
    }


def _f_params(xyz_path: Path) -> dict:
    """Driver parameters pinned to match tests/test_eu_n_scan.py exactly."""
    return {
        "FILENAME": str(xyz_path),
        "SKFPATH": str(_f_dir()) + os.sep,
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 10.0,
        "RCUT_REPULSIVE": 6.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
    }


def _write_eu_n_xyz(path: Path, separation: float) -> None:
    """Eu at the origin, N displaced along +x.  Mirrors tests/test_eu_n_scan.py."""
    path.write_text(
        "2\n"
        "Eu-N diatomic (radial grid characterization)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _write_eu_ga_n_xyz(path: Path) -> None:
    """All three f-fixture species, so all nine ordered pairs get loaded."""
    path.write_text(
        "3\n"
        "Eu-Ga-N (loads all nine f_orbital_data pairs)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        "Ga 3.00000000 0.00000000 0.00000000\n"
        "N 6.00000000 0.00000000 0.00000000\n"
    )


def _neighbour_distances(struct, const, rcut):
    """Return ``(dR_mskd, IJ_pair_type)`` exactly as _h0ands computes them.

    Reproduces `_h0ands.py:120-210`: the pairwise displacement, its norm, and
    the `nnType != -1` mask that drops neighbour-list zero padding.  Kept in
    lockstep with production on purpose; a characterization test records what
    the code does, so it mirrors the expression rather than deriving its own.
    """
    from dftorch._nearestneighborlist import vectorized_nearestneighborlist

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
        IJ_pair_type,
        JI_pair_type,
    ) = vectorized_nearestneighborlist(
        struct.TYPE,
        struct.RX,
        struct.RY,
        struct.RZ,
        struct.cell,
        rcut,
        struct.Nats,
        const,
        upper_tri_only=False,
    )

    Rab_X = nnRx - struct.RX.reshape(-1, 1)
    Rab_Y = nnRy - struct.RY.reshape(-1, 1)
    Rab_Z = nnRz - struct.RZ.reshape(-1, 1)
    dR = torch.sqrt(Rab_X**2 + Rab_Y**2 + Rab_Z**2)
    nn_mask = nnType != -1

    return dR[nn_mask], IJ_pair_type


def _global_knot_lookup(R_orb, dR_mskd):
    """The pre-D-01 production expression, verbatim from _h0ands.py:243-245."""
    idx = torch.searchsorted(R_orb, dR_mskd, right=True) - 1
    idx = torch.clamp(idx, 0, len(R_orb))
    dx = dR_mskd - R_orb[idx]
    return idx, dx


def _ch4_h0_and_s(tmp_path, *, per_pair: bool = False):
    """Assemble CH4 + mio-1-1 H0/S through the live single-system path.

    Returns ``(const, H0, dH0, S, dS)``.  Deliberately calls
    ``H0_and_S_vectorized`` directly rather than driving ``ESDriver``, so that
    the checksums pin the H0/S assembly alone and are not perturbed by SCF,
    Coulomb, or repulsion.

    ``per_pair=False`` passes neither ``R_tensor`` nor ``n_grid``, which selects
    the pre-D-01 global ``R_orb`` fallback.  ``per_pair=True`` passes both,
    which selects ``_pair_knot_lookup``.  Both are exercised, because the whole
    REG-01 claim is that they agree bit for bit on a single-grid directory.
    """
    from dftorch.Constants import Constants
    from dftorch.Structure import Structure
    from dftorch._h0ands import H0_and_S_vectorized

    params = _mio_params(_ch4_xyz())
    const = Constants(params).to("cpu")
    struct = Structure(params, const, device="cpu")

    from dftorch._nearestneighborlist import vectorized_nearestneighborlist

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
        IJ_pair_type,
        JI_pair_type,
    ) = vectorized_nearestneighborlist(
        struct.TYPE,
        struct.RX,
        struct.RY,
        struct.RZ,
        struct.cell,
        params["RCUT_ELECTRONIC"],
        struct.Nats,
        const,
        upper_tri_only=False,
    )

    extra = {}
    if per_pair:
        extra["R_tensor"] = const.R_tensor
        extra["n_grid"] = const.n_grid

    H0, dH0, S, dS = H0_and_S_vectorized(
        struct.TYPE,
        struct.RX,
        struct.RY,
        struct.RZ,
        struct.diagonal,
        struct.H_INDEX_START,
        nnRx,
        nnRy,
        nnRz,
        nnType,
        const,
        neighbor_I,
        neighbor_J,
        IJ_pair_type,
        JI_pair_type,
        const.R_orb,
        const.coeffs_tensor,
        **extra,
    )
    return const, H0, dH0, S, dS


# --- Synthetic mixed-grid fixture generator ----------------------------------

def write_mixed_grid_skf_pair(
    path,
    elem_a: str,
    elem_b: str,
    *,
    step_bohr: float,
    npts: int,
) -> None:
    """Write one minimal simple-format SKF file with a caller-chosen grid line.

    No mixed-STEP parameter directory is checked into this repository, so the
    hazard decision D-01 closes cannot be reproduced from real fixtures.  This
    helper synthesises the case at test time.  See
    ``tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md`` for the full rationale
    and for the honest statement of what a synthetic fixture does and does not
    prove.

    The file layout follows the comment at ``_bond_integral.py:607-611``:

        homonuclear:   grid line, atomic header, mass/poly line, electronic table
        heteronuclear: grid line, mass/poly line, electronic table

    followed by a repulsive ``Spline`` block.  Modelled on
    ``tests/data_skf_mio-1-1/C-C.skf`` and on ``write_simple_skf`` in
    ``tests/test_f_orbital_skf.py``.

    Both elements are given an s shell only (``Es`` and ``fs`` nonzero, every
    other on-site value zero), which ``_shell_metadata_from_presence`` maps to
    ``n_orb == 1``.  That keeps the synthetic system small and keeps this helper
    focused on the radial grid rather than on angular routing.

    Parameters
    ----------
    path : Path or str
        Destination file.  The caller is responsible for naming it
        ``{elem_a}-{elem_b}.skf`` inside the intended SKFPATH directory.
    elem_a, elem_b : str
        Element symbols.  ``elem_a == elem_b`` selects the homonuclear layout,
        which is the one carrying the atomic header line.
    step_bohr : float
        Radial grid spacing in BOHR, written verbatim as the first field of the
        grid line.  ``_bond_integral.py:713-718`` scales it by
        ``BOHR_TO_ANGSTROM``, so the loaded grid is in Angstrom.
    npts : int
        Number of tabulated points, written as the second field of the grid
        line.  The parser reads ``npts - 1`` electronic rows
        (``_bond_integral.py:703``), appends one zero knot, and pads the tail to
        ``npts + 50`` (``:599``, ``:707-711``).  The loaded grid length is
        therefore ``npts + 50``, not ``npts``.

    Returns
    -------
    None
        Writes the file and makes no assertions.  Plan 05-01 Task 2 and plan
        05-04 both consume it.
    """
    if npts < 2:
        raise ValueError(f"npts must be at least 2 to emit a table, got {npts}")

    homonuclear = elem_a == elem_b

    lines = [f"{step_bohr:.8f} {npts}"]
    if homonuclear:
        # Simple-format homonuclear header, 10 values in the order the parser
        # unpacks them at _bond_integral.py:640-651:
        #   Ed Ep Es SPE Ud Up Us fd fp fs
        lines.append(
            "0.00000000 0.00000000 -0.10000000 0.00000000 "
            "0.00000000 0.00000000 0.01000000 "
            "0.00000000 0.00000000 1.00000000"
        )
    # Mass / polynomial line: eight values, unused by this test path.
    lines.append("0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0")

    # Electronic table: npts - 1 rows of 20 simple-format channel values.  The
    # values decay smoothly with radius so the cubic spline through them is well
    # behaved; their magnitudes are arbitrary and carry no physical meaning.
    for row in range(npts - 1):
        r = (row + 1) * step_bohr
        base = 0.05 * (2.718281828459045 ** (-r))
        values = [base * (1.0 + 0.01 * col) for col in range(20)]
        lines.append(" ".join(f"{value:.8f}" for value in values))

    # Minimal repulsive spline block: two intervals, all coefficients zero.
    lines.extend(
        [
            "Spline",
            "2 2.0",
            "0.0 0.0 0.0",
            "0.0 1.0 0.0 0.0 0.0 0.0",
            "1.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")


# --- Tests -------------------------------------------------------------------

def test_grid_is_angstrom_uniform_progression():
    """The grid is a uniform arithmetic progression starting one step in.

    `R = arange(1, npts_pad + 1) * step * BOHR_TO_ANGSTROM`
    (`_bond_integral.py:713-718`) means `R[0] == step`, NOT `R[0] == 0`, and
    every successive difference equals that same step.  Both properties are
    load-bearing:

    * `R[0] == step` is why a distance shorter than one step floors to
      `idx = -1` and gets clamped to 0 rather than landing on a real knot.
    * uniformity is why `searchsorted` and `floor(dR / step) - 1` agree, and it
      is the invariant that makes extending a row past its tabulated end (plan
      05-01 Task 2) a well-defined operation rather than an extrapolation.

    Checked for BOTH real parameter directories, because a claim about "the
    grid" that holds for only one of them is not an invariant.
    """

    def check():
        from dftorch.Constants import Constants

        results = []
        for params_builder, xyz_writer in (
            (_mio_params, None),
            (_f_params, _write_eu_n_xyz),
        ):
            if xyz_writer is None:
                xyz = _ch4_xyz()
            else:
                xyz = Path(_tmp_dir()) / "grid_progression.xyz"
                xyz_writer(xyz, EU_N_SEPARATION)
            const = Constants(params_builder(xyz)).to("cpu")
            R = const.R_orb.detach()
            step = (R[1] - R[0]).item()
            diffs = R[1:] - R[:-1]
            results.append(
                (
                    R[0].item(),
                    step,
                    (diffs - step).abs().max().item(),
                )
            )
        return results

    for first, step, max_diff_error in run_with_float64(check):
        # The grid starts one step in, not at zero.
        assert abs(first - step) < 1e-12, (
            f"grid does not start one step in: R[0]={first!r}, step={step!r}"
        )
        # Successive differences are constant.
        assert max_diff_error < 1e-12, (
            f"grid is not a uniform progression: max deviation {max_diff_error!r}"
        )


def test_mio_grid_pins_step_and_length():
    """mio-1-1 loads a 550-point Angstrom grid at 0.02 Bohr spacing.

    Provenance: the C-C / C-H / H-C / H-H grid lines all read "0.02, 500".
    500 tabulated points are padded to 550 at `_bond_integral.py:599`, and the
    step is converted to Angstrom at `:713-718`.  The last knot is
    550 * 0.0105835442 = 5.82094931 A.
    """

    def check():
        from dftorch.Constants import Constants

        const = Constants(_mio_params(_ch4_xyz())).to("cpu")
        R = const.R_orb.detach()
        return (
            len(R),
            (R[1] - R[0]).item(),
            R[-1].item(),
            tuple(const.R_tensor.shape),
        )

    length, step, last, r_tensor_shape = run_with_float64(check)

    assert length == MIO_GRID_LENGTH
    assert abs(step - MIO_STEP_ANGSTROM) < 1e-10
    assert abs(last - MIO_LAST_ANGSTROM) < 1e-8
    # 4 ordered pairs for a C/H system, 1301 columns from the hardcoded
    # npts = 1300 at _bond_integral.py:976.
    assert r_tensor_shape == (4, R_TENSOR_WIDTH)


def test_f_fixture_grids_are_all_identical():
    """All nine f_orbital_data pairs share ONE 483-point 0.04 Bohr grid.

    THIS IS WHY A SYNTHETIC FIXTURE IS NEEDED.  Every one of the nine files
    carries the grid line "0.04, 433", so `R_tensor` has nine identical rows and
    the shared-ruler defect D-01 closes cannot be reproduced from these fixtures
    at all.  `write_mixed_grid_skf_pair` in this module synthesises the mixed
    case instead.

    The Phase 3 write-up at
    `.planning/phases/03-h0-s-routing-and-f-angular-blocks/deferred-items.md`
    section 3 claims these grids "genuinely differ".  That claim is
    CONTRADICTED HERE and must not be relied on.  Read that document for the
    mechanism of the defect, never for its fixture claim.
    """

    def check():
        from dftorch.Constants import Constants

        xyz = Path(_tmp_dir()) / "eu_ga_n.xyz"
        _write_eu_ga_n_xyz(xyz)
        const = Constants(_f_params(xyz)).to("cpu")
        R_tensor = const.R_tensor.detach()
        rows_equal = all(
            torch.equal(R_tensor[i], R_tensor[0]) for i in range(R_tensor.shape[0])
        )
        # Read the tabulated length from const.n_grid, not from a nonzero count.
        # Since plan 05-01 Task 2 the row tail continues each pair's arithmetic
        # progression instead of being zero, so counting nonzero entries would
        # now return the full 1301-column width for every pair.
        lengths = [int(v) for v in const.n_grid.detach().tolist()]
        R = const.R_orb.detach()
        return (
            R_tensor.shape[0],
            rows_equal,
            lengths,
            len(R),
            (R[1] - R[0]).item(),
            R[-1].item(),
        )

    n_pairs, rows_equal, lengths, length, step, last = run_with_float64(check)

    # Eu, Ga, N give 3 x 3 = 9 ordered pairs.
    assert n_pairs == 9
    assert rows_equal, "f_orbital_data R_tensor rows are NOT identical"
    assert lengths == [F_GRID_LENGTH] * 9
    assert length == F_GRID_LENGTH
    assert abs(step - F_STEP_ANGSTROM) < 1e-10
    assert abs(last - F_LAST_ANGSTROM) < 1e-8


def test_ch4_knot_indices_are_unchanged():
    """CH4 + mio-1-1 knot indices and residuals, recorded before D-01.

    Built exactly as `tests/test_scf.py` builds it: `tests/ch4.xyz`, a 25 A
    cubic cell, `RCUT_ELECTRONIC = 8.0`.  Five atoms give 20 ordered neighbour
    pairs.  `idx` and `dx` are recomputed with the production expression from
    `_h0ands.py:243-245` and pinned as literals.

    On idx.max(): the 05-01 plan predicted 178.  The MEASURED value on this
    machine is 177, and the plan instructs recording the measurement rather than
    forcing the prediction.  177 is arithmetically right:
    1.8875517915355329 / 0.0105835442 = 178.34, floor 178, minus 1 gives 177.

    dR is asserted to 1e-4 (the plan's stated tolerance for the distances) while
    idx is asserted exactly, because idx is an integer knot selection and a
    one-off there is exactly the silent shift REG-01 forbids.
    """

    def check():
        from dftorch.Constants import Constants
        from dftorch.Structure import Structure

        params = _mio_params(_ch4_xyz())
        const = Constants(params).to("cpu")
        struct = Structure(params, const, device="cpu")
        dR_mskd, _ = _neighbour_distances(struct, const, params["RCUT_ELECTRONIC"])
        idx, dx = _global_knot_lookup(const.R_orb.detach(), dR_mskd)
        return (
            dR_mskd.numel(),
            dR_mskd.min().item(),
            dR_mskd.max().item(),
            int(idx.min()),
            int(idx.max()),
            dx.min().item(),
            dx.max().item(),
            const.R_orb[98].item(),
        )

    (
        n_pairs,
        dr_min,
        dr_max,
        idx_min,
        idx_max,
        dx_min,
        dx_max,
        r_at_98,
    ) = run_with_float64(check)

    assert n_pairs == CH4_PAIR_COUNT
    assert abs(dr_min - CH4_DR_MIN) < 1e-4
    assert abs(dr_max - CH4_DR_MAX) < 1e-4
    assert idx_min == CH4_IDX_MIN
    assert idx_max == CH4_IDX_MAX
    assert abs(dx_min - CH4_DX_MIN) < 1e-10
    assert abs(dx_max - CH4_DX_MAX) < 1e-10
    # The Angstrom sanity check: the shortest C-H distance lands on a knot at
    # 1.0478 A, just under the 1.0567 A separation.  A Bohr reading would put
    # the knot near twice the true separation.
    assert abs(r_at_98 - CH4_R_ORB_AT_98) < 1e-8
    assert r_at_98 < CH4_DR_MIN


def test_eu_n_knot_indices_are_unchanged():
    """Eu-N at 2.655 A against f_orbital_data, recorded before D-01.

    Same shape as the CH4 pin, on the other real parameter directory and on the
    only f-containing case this project validates end to end (see
    `tests/f_orbital_data/README-EU-N-CASE.md`).  Both ordered pairs (Eu->N and
    N->Eu) sit at the same separation, so both must select the same knot.
    """

    def check():
        from dftorch.Constants import Constants
        from dftorch.Structure import Structure

        xyz = Path(_tmp_dir()) / "eu_n_knot.xyz"
        _write_eu_n_xyz(xyz, EU_N_SEPARATION)
        params = _f_params(xyz)
        const = Constants(params).to("cpu")
        struct = Structure(params, const, device="cpu")
        dR_mskd, _ = _neighbour_distances(struct, const, params["RCUT_ELECTRONIC"])
        idx, dx = _global_knot_lookup(const.R_orb.detach(), dR_mskd)
        return (
            dR_mskd.numel(),
            idx.tolist(),
            dx.tolist(),
        )

    n_pairs, idx_list, dx_list = run_with_float64(check)

    assert n_pairs == EU_N_PAIR_COUNT
    assert idx_list == [EU_N_IDX] * EU_N_PAIR_COUNT
    for value in dx_list:
        assert abs(value - EU_N_DX) < 1e-10
    # 2.655 A over a 0.0211670884 A step: floor(125.4) - 1 = 124.
    assert 0.0 <= EU_N_DX < F_STEP_ANGSTROM


def test_effective_cutoffs_are_unchanged():
    """Effective per-pair cutoffs, in the same Angstrom units as the grid.

    `_ml_sk.build_pair_type_rcut` reports where each pair's tabulated data
    stops, derived from the last nonzero spline interval in `coeffs_tensor`.
    mio-1-1 stops at 5.2917721 A (10.0 Bohr); f_orbital_data stops at
    9.1653493 A (17.32 Bohr).

    This test exists so that a later change to the knot lookup cannot move an
    effective cutoff unnoticed.  A shifted cutoff means bonds silently switch on
    or off at a different radius, which is a physics change wearing a refactor's
    clothes.
    """

    def check():
        from dftorch.Constants import Constants
        from dftorch._ml_sk import build_pair_type_rcut

        mio_const = Constants(_mio_params(_ch4_xyz())).to("cpu")
        mio_rcut = build_pair_type_rcut(
            mio_const.coeffs_tensor.detach(), mio_const.R_orb.detach()
        ).tolist()

        xyz = Path(_tmp_dir()) / "eu_n_rcut.xyz"
        _write_eu_n_xyz(xyz, EU_N_SEPARATION)
        f_const = Constants(_f_params(xyz)).to("cpu")
        f_rcut = build_pair_type_rcut(
            f_const.coeffs_tensor.detach(), f_const.R_orb.detach()
        ).tolist()
        return mio_rcut, f_rcut

    mio_rcut, f_rcut = run_with_float64(check)

    assert len(mio_rcut) == 4
    for value in mio_rcut:
        assert abs(value - MIO_RCUT_ANGSTROM) < 1e-6
    assert len(f_rcut) == 4
    for value in f_rcut:
        assert abs(value - F_RCUT_ANGSTROM) < 1e-6
    # The cutoff is a radius on the same grid, so it must lie inside it.
    assert MIO_RCUT_ANGSTROM < MIO_LAST_ANGSTROM
    assert F_RCUT_ANGSTROM < F_LAST_ANGSTROM


def test_ch4_h0_s_checksums_are_unchanged():
    """CH4 + mio-1-1 H0/S/dH0/dS reductions, recorded on the pre-D-01 path.

    These six numbers are the oracle for Task 2's bit-identity claim.  Without a
    value recorded BEFORE the per-pair lookup lands, a post-change test can only
    compare the new path against itself, which proves nothing about REG-01.

    Compared with zero relative tolerance and a four-ULP absolute bound.  That
    admits last-bit changes caused solely by reduction order while still
    rejecting any meaningful movement under REG-01.
    """

    def check():
        _, H0, dH0, S, dS = _ch4_h0_and_s(_tmp_dir())
        return (
            tuple(H0.shape),
            H0.sum().item(),
            S.sum().item(),
            H0.abs().sum().item(),
            S.abs().sum().item(),
            dH0.abs().sum().item(),
            dS.abs().sum().item(),
        )

    (
        shape,
        h0_sum,
        s_sum,
        h0_abs,
        s_abs,
        dh0_abs,
        ds_abs,
    ) = run_with_float64(check)

    # CH4: one C with 4 orbitals plus four H with 1 each. This assertion previously read
    # (20, 20) directly beneath that comment -- the prose was right and the number was
    # not, because hydrogen carried a phantom p shell.
    assert shape == (8, 8)
    _assert_ch4_checksum(h0_sum, CH4_H0_SUM, "H0 sum")
    _assert_ch4_checksum(s_sum, CH4_S_SUM, "S sum")
    _assert_ch4_checksum(h0_abs, CH4_H0_ABS_SUM, "H0 absolute sum")
    _assert_ch4_checksum(s_abs, CH4_S_ABS_SUM, "S absolute sum")
    _assert_ch4_checksum(dh0_abs, CH4_DH0_ABS_SUM, "dH0 absolute sum")
    _assert_ch4_checksum(ds_abs, CH4_DS_ABS_SUM, "dS absolute sum")


def test_r_tensor_rows_are_strictly_increasing():
    """Every R_tensor row must be strictly increasing across its FULL width.

    This was the single concrete blocker for the per-pair lookup, and this test
    was written as a strict xfail in plan 05-01 Task 1 so that Task 2 would flip
    it rather than invent it.  `R_tensor` is allocated as
    `zeros((n_pairs, 1301))` at `_bond_integral.py:985` and only
    `R_tensor[i, :len(R_orb_i)]` was written, so row 550 onward was 0.0 for
    mio-1-1 while row 549 was 5.82095.  `torch.searchsorted` requires a sorted
    boundary tensor; against a row that rises then drops to zero its result is
    arbitrary, which is why the per-pair lookup could not simply index
    `R_tensor` as it stood.

    Task 2 fills the tail by continuing each pair's own arithmetic progression
    (`R[k] == (k + 1) * step`, and the grid starts one step in, so `R_orb_i[0]`
    IS the step).  Masking around the zero tail at the call site was considered
    and rejected: it makes the lookup correct only by caller discipline, while
    filling the tail makes it correct by construction.
    """

    def check():
        from dftorch.Constants import Constants

        report = []
        mio_const = Constants(_mio_params(_ch4_xyz())).to("cpu")
        report.append(("mio-1-1", mio_const.R_tensor.detach().clone()))

        xyz = Path(_tmp_dir()) / "eu_n_monotonic.xyz"
        _write_eu_n_xyz(xyz, EU_N_SEPARATION)
        f_const = Constants(_f_params(xyz)).to("cpu")
        report.append(("f_orbital_data", f_const.R_tensor.detach().clone()))
        return report

    for name, R_tensor in run_with_float64(check):
        assert R_tensor.shape[1] == R_TENSOR_WIDTH
        for i in range(R_tensor.shape[0]):
            row = R_tensor[i]
            assert bool((row[1:] > row[:-1]).all()), (
                f"{name} R_tensor row {i} is not strictly increasing across its "
                f"full {R_TENSOR_WIDTH}-entry width"
            )


# --- Per-pair lookup (plan 05-01 Task 2, decision D-01, requirement REG-06) --

def test_per_pair_lookup_matches_global_for_single_grid():
    """The per-pair lookup reproduces the global one exactly on one grid.

    This is the REG-01 gate on the D-01 rewrite.  Both real parameter
    directories use a single radial step throughout, so every pair's row is a
    prefix of `R_orb` and `_pair_knot_lookup` must return exactly what the old
    global expression returned.  `idx` is compared with `torch.equal` (integer
    equality) and `dx` with `atol=0.0`, so a last-bit difference fails.

    Also asserts that `const.R_orb` is still a registered attribute with an
    unchanged shape for both directories.  That is not incidental: `_stress`,
    `_ml_sk`, the SEDACS interface and the batched H0/S path all still read it,
    and the per-pair path was added ALONGSIDE it rather than in place of it
    precisely so that those out-of-scope callers keep working (REG-03).
    """

    def check():
        from dftorch.Constants import Constants
        from dftorch.Structure import Structure
        from dftorch._h0ands import _pair_knot_lookup

        results = []

        mio = _mio_params(_ch4_xyz())
        f_xyz = Path(_tmp_dir()) / "eu_n_perpair.xyz"
        _write_eu_n_xyz(f_xyz, EU_N_SEPARATION)
        eu = _f_params(f_xyz)

        for label, params, expected_r_orb_len in (
            ("mio-1-1", mio, MIO_GRID_LENGTH),
            ("f_orbital_data", eu, F_GRID_LENGTH),
        ):
            const = Constants(params).to("cpu")
            struct = Structure(params, const, device="cpu")
            dR_mskd, IJ_pair_type = _neighbour_distances(
                struct, const, params["RCUT_ELECTRONIC"]
            )

            idx_global, dx_global = _global_knot_lookup(
                const.R_orb.detach(), dR_mskd
            )
            idx_pair, dx_pair = _pair_knot_lookup(
                const.R_tensor.detach(),
                const.n_grid.detach(),
                IJ_pair_type,
                dR_mskd,
            )
            results.append(
                (
                    label,
                    bool(torch.equal(idx_global, idx_pair)),
                    (dx_global - dx_pair).abs().max().item(),
                    hasattr(const, "R_orb"),
                    tuple(const.R_orb.shape),
                    expected_r_orb_len,
                    tuple(const.n_grid.shape),
                    const.R_tensor.shape[0],
                )
            )
        return results

    for (
        label,
        idx_equal,
        dx_max_diff,
        has_r_orb,
        r_orb_shape,
        expected_len,
        n_grid_shape,
        n_pairs,
    ) in run_with_float64(check):
        assert idx_equal, f"{label}: per-pair idx differs from the global idx"
        assert dx_max_diff == 0.0, (
            f"{label}: per-pair dx differs from the global dx by {dx_max_diff!r}"
        )
        assert has_r_orb, f"{label}: const.R_orb was removed"
        assert r_orb_shape == (expected_len,), (
            f"{label}: const.R_orb shape changed to {r_orb_shape}"
        )
        # n_grid carries one entry per ordered element pair.
        assert n_grid_shape == (n_pairs,)


def test_ch4_h0_s_bit_identical_after_per_pair_lookup():
    """CH4 + mio-1-1 H0/S are bit-identical through the per-pair path.

    Two independent checks, and both are needed:

    1. The per-pair matrices equal the global-fallback matrices under
       `torch.equal`, NOT `torch.allclose`.  Element for element, zero
       tolerance.
    2. The per-pair reductions stay within four ULPs of the checksum literals
       recorded in Task 1 on the PRE-D-01 code path.  Check 1 alone would only
       compare the new path against itself in the same process; check 2 is what
       ties it back to behaviour that existed before the rewrite (REG-01,
       CLN-05) without mistaking reduction-order noise for tensor drift.

    A refactor that moves a checksum beyond the four-ULP reduction bound is
    still a regression here.
    """

    def check():
        _, H0_g, dH0_g, S_g, dS_g = _ch4_h0_and_s(_tmp_dir(), per_pair=False)
        _, H0_p, dH0_p, S_p, dS_p = _ch4_h0_and_s(_tmp_dir(), per_pair=True)
        return (
            bool(torch.equal(H0_g, H0_p)),
            bool(torch.equal(S_g, S_p)),
            bool(torch.equal(dH0_g, dH0_p)),
            bool(torch.equal(dS_g, dS_p)),
            H0_p.sum().item(),
            S_p.sum().item(),
            H0_p.abs().sum().item(),
            S_p.abs().sum().item(),
            dH0_p.abs().sum().item(),
            dS_p.abs().sum().item(),
        )

    (
        h0_same,
        s_same,
        dh0_same,
        ds_same,
        h0_sum,
        s_sum,
        h0_abs,
        s_abs,
        dh0_abs,
        ds_abs,
    ) = run_with_float64(check)

    assert h0_same, "H0 moved when the per-pair lookup was used"
    assert s_same, "S moved when the per-pair lookup was used"
    assert dh0_same, "dH0 moved when the per-pair lookup was used"
    assert ds_same, "dS moved when the per-pair lookup was used"

    _assert_ch4_checksum(h0_sum, CH4_H0_SUM, "per-pair H0 sum")
    _assert_ch4_checksum(s_sum, CH4_S_SUM, "per-pair S sum")
    _assert_ch4_checksum(h0_abs, CH4_H0_ABS_SUM, "per-pair H0 absolute sum")
    _assert_ch4_checksum(s_abs, CH4_S_ABS_SUM, "per-pair S absolute sum")
    _assert_ch4_checksum(dh0_abs, CH4_DH0_ABS_SUM, "per-pair dH0 absolute sum")
    _assert_ch4_checksum(ds_abs, CH4_DS_ABS_SUM, "per-pair dS absolute sum")


# Synthetic mixed-grid parameters.  0.1 Bohr is coarse for real physics but the
# electronic table written by write_mixed_grid_skf_pair carries no physical
# meaning; only the grid line matters here.
MIXED_STEP_BOHR = 0.1
MIXED_SHORT_NPTS = 40
MIXED_LONG_NPTS = 80
# The parser pads every table by 50 (_bond_integral.py:599), so the LOADED grid
# lengths are npts + 50, not npts.
MIXED_SHORT_LENGTH = MIXED_SHORT_NPTS + 50
MIXED_LONG_LENGTH = MIXED_LONG_NPTS + 50
BOHR_TO_ANGSTROM_LITERAL = 0.52917721


def _write_mixed_skf_dir(directory: Path, pair_npts: dict, pair_step: dict) -> None:
    """Write a four-file SKFPATH for the two-element systems below.

    ``pair_npts`` and ``pair_step`` are keyed by the ``"H-H"`` style pair name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("H-H", "H-N", "N-H", "N-N"):
        elem_a, elem_b = name.split("-")
        write_mixed_grid_skf_pair(
            directory / f"{name}.skf",
            elem_a,
            elem_b,
            step_bohr=pair_step[name],
            npts=pair_npts[name],
        )


def _write_h_n_xyz(path: Path) -> None:
    path.write_text(
        "2\n"
        "synthetic mixed-grid pair\n"
        "H 0.00000000 0.00000000 0.00000000\n"
        "N 1.00000000 0.00000000 0.00000000\n"
    )


def _mixed_const(skf_dir: Path, xyz: Path):
    from dftorch.Constants import Constants

    return Constants(
        {
            "FILENAME": str(xyz),
            "SKFPATH": str(skf_dir) + os.sep,
        }
    ).to("cpu")


def test_mixed_length_grids_use_their_own_rows():
    """A short pair clamps to its OWN grid length; a long pair does not.

    Built from `write_mixed_grid_skf_pair`: H-H, H-N and N-H are given 40
    tabulated points (loaded length 90) while N-N is given 80 (loaded length
    130), all at the same 0.1 Bohr step.  This is the BENIGN mixed case from
    decision D-01, where the knots coincide and only the tabulated extent
    differs, so it must load rather than being rejected.

    The probe distance sits past the short pairs' own tabulated end but inside
    the long pair's.  With the per-pair lookup the short pair's index clamps to
    its own `n_grid` while the long pair's index does not, and the clamped index
    selects an all-zero spline interval, so the short pair contributes nothing
    instead of borrowing a coefficient from a longer grid.  The old global
    expression gave BOTH pairs the same unclamped index, which is exactly the
    shared-ruler behaviour REG-06 closes.

    See `tests/f_orbital_data/README-MIXED-GRID-FIXTURE.md` for why this fixture
    is synthesised and for the honest limitation of a synthetic case.
    """

    def check():
        from dftorch._h0ands import _pair_knot_lookup

        skf_dir = Path(_tmp_dir()) / "mixed_length_skf"
        xyz = Path(_tmp_dir()) / "mixed_length.xyz"
        _write_h_n_xyz(xyz)
        _write_mixed_skf_dir(
            skf_dir,
            {
                "H-H": MIXED_SHORT_NPTS,
                "H-N": MIXED_SHORT_NPTS,
                "N-H": MIXED_SHORT_NPTS,
                "N-N": MIXED_LONG_NPTS,
            },
            dict.fromkeys(("H-H", "H-N", "N-H", "N-N"), MIXED_STEP_BOHR),
        )
        const = _mixed_const(skf_dir, xyz)

        # TYPE holds atomic numbers, so pair_lookup is indexed by them.
        pt_short = int(const.pair_lookup[1, 1])
        pt_long = int(const.pair_lookup[7, 7])

        step_a = const.R_tensor[pt_short, 0].item()
        # Past the short pair's own tabulated end, inside the long pair's.
        probe = step_a * (MIXED_SHORT_LENGTH + 13.5)

        pair_type = torch.tensor([pt_short, pt_long])
        dR = torch.tensor([probe, probe])
        idx, dx = _pair_knot_lookup(
            const.R_tensor.detach(), const.n_grid.detach(), pair_type, dR
        )
        idx_global, _ = _global_knot_lookup(const.R_orb.detach(), dR)

        return (
            int(const.n_grid[pt_short]),
            int(const.n_grid[pt_long]),
            len(const.R_orb),
            step_a,
            probe,
            [int(v) for v in idx.tolist()],
            [int(v) for v in idx_global.tolist()],
            const.coeffs_tensor[pt_short, int(idx[0])].abs().sum().item(),
        )

    (
        n_short,
        n_long,
        r_orb_len,
        step_a,
        probe,
        idx_pair,
        idx_global,
        short_coeff_magnitude,
    ) = run_with_float64(check)

    # The directory loaded, and the two pairs really do carry different lengths.
    assert n_short == MIXED_SHORT_LENGTH
    assert n_long == MIXED_LONG_LENGTH
    # R_orb still holds the LONGEST grid seen.
    assert r_orb_len == MIXED_LONG_LENGTH
    assert abs(step_a - MIXED_STEP_BOHR * BOHR_TO_ANGSTROM_LITERAL) < 1e-10

    # The short pair clamps to its own n_grid; the long pair does not.
    assert idx_pair[0] == n_short, (
        f"short pair index {idx_pair[0]} did not clamp to its own n_grid {n_short}"
    )
    assert idx_pair[1] < n_long, (
        f"long pair index {idx_pair[1]} clamped although {probe!r} is inside its grid"
    )
    assert idx_pair[0] != idx_pair[1], (
        "both pairs selected the same knot, so the rows are still shared"
    )

    # The old global expression gave BOTH pairs the long grid's index.  That is
    # the shared-ruler defect, recorded here so the difference is visible rather
    # than asserted in the abstract.
    assert idx_global[0] == idx_global[1] == idx_pair[1]

    # The backstop truth: past its own cutoff the short pair lands on the zero
    # spline interval rather than on a borrowed coefficient.
    assert short_coeff_magnitude == 0.0


def test_mixed_step_grids_read_their_own_radius():
    """The hazardous case: differing STEP, where a shared ruler reads wrong.

    Not required by the 05-01 acceptance criteria, and added deliberately.
    `test_mixed_length_grids_use_their_own_rows` covers the BENIGN case where
    the knots coincide, so on its own it never demonstrates that the defect
    D-01 closes changes any radius at all.  Decision D-01 is explicit that the
    hazard is a differing grid STEP: the f dataset uses 0.04 Bohr while
    `mio-1-1`, `3ob-3-1`, `pbc-0-3` and `trans3d-0-1` all use 0.02, so a single
    SKFPATH mixing them is wrong by a factor of two.

    Here H-H (and the heteronuclear pairs) get 0.1 Bohr with 60 points, while
    N-N gets 0.2 Bohr with 40.  H-H's grid is the longer one, so it becomes the
    global `R_orb`.  The global expression therefore reads N-N's spline at
    H-H's radius; the per-pair lookup reads it at N-N's own.  The two indices
    differ by roughly the ratio of the steps, which is the factor-of-two error
    D-01 describes.

    UPDATED BY PLAN 05-04.  When this test was written such a directory loaded
    without complaint, and the docstring said so.  It no longer does: plan 05-04
    added `_bond_integral._require_uniform_grid_step`, which refuses a
    mixed-step SKFPATH outright (requirement REG-05).  The refusal is asserted
    by `test_mixed_step_skfpath_is_refused` below.

    This test therefore now bypasses that guard deliberately, by replacing it
    with a no-op for the single `Constants` construction.  The point of the test
    is to keep showing WHAT THE GUARD PREVENTS as a measured radius rather than
    as prose: with the guard removed the numbers below are what a user would
    silently have received.  Deleting this test in favour of the refusal test
    would leave the refusal justified only by assertion.
    """

    def check():
        import dftorch._bond_integral as bond_integral
        from dftorch._h0ands import _pair_knot_lookup

        skf_dir = Path(_tmp_dir()) / "mixed_step_skf"
        xyz = Path(_tmp_dir()) / "mixed_step.xyz"
        _write_h_n_xyz(xyz)
        _write_mixed_skf_dir(
            skf_dir,
            {"H-H": 60, "H-N": 60, "N-H": 60, "N-N": 40},
            {"H-H": 0.1, "H-N": 0.1, "N-H": 0.1, "N-N": 0.2},
        )
        # Bypass the REG-05 guard for this one load.  Restored in `finally` so a
        # failure here cannot disarm the guard for any later test.
        original_guard = bond_integral._require_uniform_grid_step
        bond_integral._require_uniform_grid_step = lambda steps_by_file: None
        try:
            const = _mixed_const(skf_dir, xyz)
        finally:
            bond_integral._require_uniform_grid_step = original_guard

        pt_fine = int(const.pair_lookup[1, 1])
        pt_coarse = int(const.pair_lookup[7, 7])

        probe = 2.0  # Angstrom, comfortably inside both grids.
        pair_type = torch.tensor([pt_fine, pt_coarse])
        dR = torch.tensor([probe, probe])
        idx, _ = _pair_knot_lookup(
            const.R_tensor.detach(), const.n_grid.detach(), pair_type, dR
        )
        idx_global, _ = _global_knot_lookup(const.R_orb.detach(), dR)

        return (
            const.R_tensor[pt_fine, 0].item(),
            const.R_tensor[pt_coarse, 0].item(),
            len(const.R_orb),
            [int(v) for v in idx.tolist()],
            [int(v) for v in idx_global.tolist()],
            const.R_tensor[pt_coarse, int(idx[1])].item(),
            const.R_tensor[pt_coarse, int(idx_global[1])].item(),
        )

    (
        step_fine,
        step_coarse,
        r_orb_len,
        idx_pair,
        idx_global,
        radius_own,
        radius_borrowed,
    ) = run_with_float64(check)

    # The two rows really do carry different steps, at a ratio of two.
    assert abs(step_fine - 0.1 * BOHR_TO_ANGSTROM_LITERAL) < 1e-10
    assert abs(step_coarse - 0.2 * BOHR_TO_ANGSTROM_LITERAL) < 1e-10
    # R_orb is the LONGEST grid, which is H-H's 60 + 50 = 110 points.
    assert r_orb_len == 110

    # The fine pair is unaffected: its own row IS the global R_orb.
    assert idx_pair[0] == idx_global[0]
    # The coarse pair is not.  The shared ruler put its knot roughly twice as
    # far along its own grid as it belongs.
    assert idx_pair[1] != idx_global[1]
    assert idx_global[1] > idx_pair[1]

    # Stated as a radius rather than an index, which is what actually matters:
    # the per-pair knot sits just below the 2.0 A probe, the borrowed one does
    # not sit anywhere near it.
    assert radius_own <= 2.0
    assert radius_borrowed > 2.0
    assert radius_borrowed > 1.9 * radius_own


def test_mio_c_h_p_has_genuinely_mixed_grid_lengths():
    """A REAL mixed-length case already lives in tests/data_skf_mio-1-1.

    Not required by the 05-01 acceptance criteria, and added deliberately.  The
    plan text states that no mixed-grid fixture exists in this repository.  That
    is true of a mixed STEP but false of a mixed LENGTH: measured 2026-07-31,
    the `mio-1-1` grid lines are `0.02, 500` for most pairs, `0.02 600` for
    every Zn pair, and `0.02, 619` for every P pair.

    A C/H/P system therefore loads nine ordered pairs whose rows carry two
    genuinely different real lengths, 550 and 669, at one common step.  Pinning
    that here means the per-pair machinery is exercised against real data and
    not only against a generator, which is the weakest part of the synthetic
    tests above.  It is also the benign case D-01 says must keep loading, so
    this test doubles as a guard that the REG-05 work in plan 05-04 does not
    start rejecting `mio-1-1`.
    """

    def check():
        from dftorch.Constants import Constants

        xyz = Path(_tmp_dir()) / "c_h_p.xyz"
        xyz.write_text(
            "3\n"
            "C/H/P from mio-1-1: two real grid lengths at one step\n"
            "C 0.00000000 0.00000000 0.00000000\n"
            "H 1.10000000 0.00000000 0.00000000\n"
            "P 2.90000000 0.00000000 0.00000000\n"
        )
        const = Constants(_mio_params(xyz)).to("cpu")
        R_tensor = const.R_tensor.detach()
        monotonic = all(
            bool((R_tensor[i, 1:] > R_tensor[i, :-1]).all())
            for i in range(R_tensor.shape[0])
        )
        return (
            sorted(set(int(v) for v in const.n_grid.detach().tolist())),
            len(const.R_orb),
            monotonic,
            R_tensor.shape[0],
        )

    lengths, r_orb_len, monotonic, n_pairs = run_with_float64(check)

    # C, H, P give 3 x 3 = 9 ordered pairs.
    assert n_pairs == 9
    # 500 + 50 for the C/H pairs, 619 + 50 for every P pair.
    assert lengths == [550, 669]
    # R_orb is still the longest grid seen.
    assert r_orb_len == 669
    assert monotonic, "a real mixed-length load produced a non-monotonic row"


# --- REG-05: the mixed-step refusal (plan 05-04) -----------------------------
#
# The two steps below are the REAL hazard pair named by decision D-01, not
# arbitrary numbers: tests/f_orbital_data is tabulated at 0.04 Bohr while
# mio-1-1, 3ob-3-1, pbc-0-3 and trans3d-0-1 are all at 0.02 Bohr, so any single
# SKFPATH mixing the f dataset with a mainstream parameter set is wrong by a
# factor of two.
GUARD_FINE_STEP_BOHR = 0.02
GUARD_COARSE_STEP_BOHR = 0.04
GUARD_NPTS = 40


def _write_step_mismatch_dir(directory: Path) -> None:
    """Write a four-file SKFPATH in which N-N alone declares the coarse step."""
    _write_mixed_skf_dir(
        directory,
        dict.fromkeys(("H-H", "H-N", "N-H", "N-N"), GUARD_NPTS),
        {
            "H-H": GUARD_FINE_STEP_BOHR,
            "H-N": GUARD_FINE_STEP_BOHR,
            "N-H": GUARD_FINE_STEP_BOHR,
            "N-N": GUARD_COARSE_STEP_BOHR,
        },
    )


def _load_expecting_guard(skf_dir: Path, xyz: Path):
    """Construct ``Constants`` and report whether the REG-05 guard fired.

    Returns ``(outcome, message)`` where ``outcome`` is ``"refused"``,
    ``"loaded"`` or ``"other-error"``.  Primitives are returned rather than the
    exception object because ``run_with_float64`` evicts and restores the
    ``dftorch`` modules afterwards, so a class captured inside the callable is
    not the same object as one imported at assert time.
    """
    from dftorch._bond_integral import SKFRadialGridStepMismatchError

    try:
        _mixed_const(skf_dir, xyz)
    except SKFRadialGridStepMismatchError as exc:
        return "refused", str(exc)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return "other-error", f"{type(exc).__name__}: {exc}"
    return "loaded", ""


def test_mixed_step_skfpath_is_refused():
    """A SKFPATH whose files disagree on grid STEP refuses to load (REG-05).

    Decision D-01 keeps this guard even though plan 05-01 made the per-pair
    lookup correct, because a per-pair lookup cannot rescue a mixed-STEP
    directory.  Two tabulations sampled at different radial resolutions carry
    different information at different radii; combining them is a parameter-set
    error, not an indexing one.

    Also pins the exception's base class.  `ValueError` is deliberate: this is a
    rejected input, unlike the four `NotImplementedError` subclasses in
    `_slater_koster_pair` which mark capability DFTorch has yet to implement.
    """

    def check():
        from dftorch._bond_integral import SKFRadialGridStepMismatchError

        skf_dir = Path(_tmp_dir()) / "guard_mixed_step"
        xyz = Path(_tmp_dir()) / "guard_mixed_step.xyz"
        _write_h_n_xyz(xyz)
        _write_step_mismatch_dir(skf_dir)
        outcome, message = _load_expecting_guard(skf_dir, xyz)
        return (
            outcome,
            message,
            issubclass(SKFRadialGridStepMismatchError, ValueError),
        )

    outcome, message, is_value_error = run_with_float64(check)

    assert outcome == "refused", (
        "a mixed-step SKFPATH must raise SKFRadialGridStepMismatchError; "
        f"got outcome {outcome!r} with {message!r}"
    )
    assert is_value_error, (
        "SKFRadialGridStepMismatchError must subclass ValueError: it reports a "
        "rejected input, not an unimplemented capability"
    )


def test_mixed_step_error_names_both_files_and_steps():
    """The refusal names the offending files AND both steps, not just 'mismatch'.

    A message saying only that grids disagree leaves the user to find which of
    possibly hundreds of SKF files is the odd one out.  The assertions below are
    on the exact grouped lines the guard emits, and NOT merely on the substrings
    `0.02` and `0.04`: the explanatory paragraph
    `SKF_GRID_STEP_MISMATCH_MESSAGE` mentions both of those numbers in prose, so
    a bare substring check would pass even if the guard reported no measured
    value at all.
    """

    def check():
        skf_dir = Path(_tmp_dir()) / "guard_message"
        xyz = Path(_tmp_dir()) / "guard_message.xyz"
        _write_h_n_xyz(xyz)
        _write_step_mismatch_dir(skf_dir)
        return _load_expecting_guard(skf_dir, xyz)

    outcome, message = run_with_float64(check)

    assert outcome == "refused", f"expected a refusal, got {outcome!r}: {message!r}"

    fine_line = "step 0.02 Bohr: H-H.skf, H-N.skf, N-H.skf"
    coarse_line = "step 0.04 Bohr: N-N.skf"
    assert fine_line in message, (
        f"the message does not group the fine-step files: expected {fine_line!r} "
        f"in:\n{message}"
    )
    assert coarse_line in message, (
        f"the message does not name the odd file out: expected {coarse_line!r} "
        f"in:\n{message}"
    )
    # The message must also tell the reader NOT to respond by padding files,
    # which is the plausible wrong repair once someone sees "grid" and "500".
    assert "NUMBERS OF POINTS at the same step are FINE" in message


def test_same_step_different_length_is_accepted():
    """Same step, different point counts: BENIGN, and must keep loading.

    This is the prohibition half of REG-05 and it matters more than the refusal.
    The obvious implementation of the guard -- compare grid LENGTHS -- passes
    the mixed-step case it exists to catch while rejecting real shipped
    parameter sets:

    * `3ob-3-1` ships C-C at 650 points and Br-Br at 850 points, both at the
      same 0.02 Bohr step.
    * `mio-1-1` mixes 500, 600 (Zn) and 619 (P) points, all at 0.02 Bohr.

    Rejecting either would break a working parameter set for no benefit, since
    plan 05-01 gave every pair its own grid row.  The two lengths are asserted
    to be genuinely distinct in `const.n_grid`, so the test cannot pass on a
    directory that quietly collapsed to one grid.
    """

    def check():
        skf_dir = Path(_tmp_dir()) / "guard_same_step"
        xyz = Path(_tmp_dir()) / "guard_same_step.xyz"
        _write_h_n_xyz(xyz)
        _write_mixed_skf_dir(
            skf_dir,
            {
                "H-H": MIXED_SHORT_NPTS,
                "H-N": MIXED_SHORT_NPTS,
                "N-H": MIXED_SHORT_NPTS,
                "N-N": MIXED_LONG_NPTS,
            },
            dict.fromkeys(("H-H", "H-N", "N-H", "N-N"), MIXED_STEP_BOHR),
        )
        outcome, message = _load_expecting_guard(skf_dir, xyz)
        if outcome != "loaded":
            return outcome, message, []
        const = _mixed_const(skf_dir, xyz)
        return (
            outcome,
            "",
            sorted(set(int(v) for v in const.n_grid.detach().tolist())),
        )

    outcome, message, lengths = run_with_float64(check)

    assert outcome == "loaded", (
        "the guard rejected a same-step / different-length SKFPATH, which is "
        "the benign configuration 3ob-3-1 and mio-1-1 both ship; "
        f"got {outcome!r}: {message!r}"
    )
    assert lengths == [MIXED_SHORT_LENGTH, MIXED_LONG_LENGTH], (
        "the accepted directory must really carry two different grid lengths, "
        f"got {lengths}"
    )


def test_real_parameter_sets_still_load():
    """Both parameter sets in this repository still construct Constants.

    The blunt regression check that the guard did not become over-eager.
    `tests/data_skf_mio-1-1` is uniform at 0.02 Bohr across a set that mixes
    500 / 600 / 619 point counts, and `tests/f_orbital_data` is uniform at 0.04
    Bohr; each is internally consistent in step and must load unchanged.  Every
    other test in the suite that runs a real calculation depends on this.
    """

    def check():
        from dftorch.Constants import Constants

        outcomes = []

        ch4 = _ch4_xyz()
        try:
            mio_const = Constants(_mio_params(ch4)).to("cpu")
            outcomes.append(("mio-1-1", "loaded", len(mio_const.R_orb)))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            outcomes.append(("mio-1-1", f"{type(exc).__name__}: {exc}", 0))

        eu_n = Path(_tmp_dir()) / "guard_real_eu_n.xyz"
        _write_eu_n_xyz(eu_n, EU_N_SEPARATION)
        try:
            f_const = Constants(_f_params(eu_n)).to("cpu")
            outcomes.append(("f_orbital_data", "loaded", len(f_const.R_orb)))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            outcomes.append(("f_orbital_data", f"{type(exc).__name__}: {exc}", 0))

        return outcomes

    outcomes = run_with_float64(check)

    for name, status, _length in outcomes:
        assert status == "loaded", f"{name} no longer loads: {status}"

    lengths = {name: length for name, _status, length in outcomes}
    assert lengths["mio-1-1"] == MIO_GRID_LENGTH
    assert lengths["f_orbital_data"] == F_GRID_LENGTH


def test_real_f_and_mio_files_mixed_are_refused(tmp_path):
    """The backstop: the guard fires on REAL files from the two real datasets.

    Every other refusal test above builds its SKFPATH with
    `write_mixed_grid_skf_pair`, so on its own the guard is only ever shown
    firing on a construction of this suite's own making.  Here the directory is
    assembled by COPYING shipped files: Eu-Eu, Eu-N and N-Eu come from
    `tests/f_orbital_data` at 0.04 Bohr, while N-N is taken from
    `tests/data_skf_mio-1-1` at 0.02 Bohr.

    That is precisely the case decision D-01 describes -- an Eu system whose
    light-element ligand parameters are pulled from a mainstream set -- and it
    is the combination this guard exists for.  The measured grid lines are
    `0.04, 433` in the f files and `0.02, 500,2` in mio's N-N.skf.
    """
    import shutil

    skf_dir = tmp_path / "eu_n_mixed_real"
    skf_dir.mkdir()
    for name in ("Eu-Eu.skf", "Eu-N.skf", "N-Eu.skf"):
        shutil.copy(_f_dir() / name, skf_dir / name)
    shutil.copy(_mio_dir() / "N-N.skf", skf_dir / "N-N.skf")

    xyz = tmp_path / "eu_n_mixed_real.xyz"
    _write_eu_n_xyz(xyz, EU_N_SEPARATION)

    def check():
        return _load_expecting_guard(skf_dir, xyz)

    outcome, message = run_with_float64(check)

    assert outcome == "refused", (
        "mixing the real 0.04 Bohr f dataset with mio-1-1's real 0.02 Bohr "
        f"N-N.skf must be refused; got {outcome!r}: {message!r}"
    )
    assert "step 0.02 Bohr: N-N.skf" in message, (
        f"the borrowed mio file is not named as the odd one out:\n{message}"
    )
    assert "step 0.04 Bohr: Eu-Eu.skf, Eu-N.skf, N-Eu.skf" in message, (
        f"the f-dataset files are not grouped at their own step:\n{message}"
    )


def test_guard_message_leaks_no_path(tmp_path):
    """The refusal names basenames and numbers only, never a path.

    Phase 4 threat T-04-07 set the policy that an exception message names
    orbital counts and modes but never interpolates a path; the same applies to
    this guard, whose message is the one place in plan 05-04 where
    caller-supplied text could reach a log or a shared traceback.

    The assertion is on the specific `tmp_path` string rather than on a general
    path-shaped regex.  A regex for "looks like a path" could be satisfied by
    accident on a short temporary directory name, or could fail on a message
    that legitimately contains a slash; asserting that THIS directory's own path
    is absent is exact.  Both the native form and the forward-slash form are
    checked, because `os.sep` is a backslash here and code that normalises a
    path before formatting it would otherwise slip through.
    """
    skf_dir = tmp_path / "guard_leak_check_dir"
    xyz = tmp_path / "guard_leak_check.xyz"
    _write_h_n_xyz(xyz)
    _write_step_mismatch_dir(skf_dir)

    def check():
        return _load_expecting_guard(skf_dir, xyz)

    outcome, message = run_with_float64(check)

    assert outcome == "refused", f"expected a refusal, got {outcome!r}: {message!r}"

    forbidden = {
        "skf directory (native)": str(skf_dir),
        "skf directory (posix)": skf_dir.as_posix(),
        "tmp_path (native)": str(tmp_path),
        "tmp_path (posix)": tmp_path.as_posix(),
        "geometry file (native)": str(xyz),
        "geometry file (posix)": xyz.as_posix(),
    }
    leaked = sorted(
        label for label, value in forbidden.items() if value in message
    )
    assert not leaked, (
        f"the guard's message leaked {leaked} into text a user may paste into a "
        f"bug report:\n{message}"
    )
    # The message is still useful: it names the files, just not where they live.
    assert "N-N.skf" in message


# --- tmp dir helper ----------------------------------------------------------
#
# Several tests need a scratch directory but run their real work inside
# `run_with_float64`, which is a plain callable rather than a pytest fixture, so
# a `tmp_path` argument cannot be threaded through it.  A module-scoped
# directory under pytest's own base temp keeps the artefacts out of the source
# tree while staying inspectable after a failure.
_TMP_DIR = None


def _tmp_dir():
    global _TMP_DIR
    if _TMP_DIR is None:
        import tempfile

        _TMP_DIR = tempfile.mkdtemp(prefix="dftorch_radial_grid_")
    return _TMP_DIR
