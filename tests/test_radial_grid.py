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
# Recorded on the PRE-D-01 code path (global `R_orb` lookup) so that Task 2 has
# a real oracle for its bit-identity claim rather than only comparing the new
# path against itself.  These are exact float64 reprs of reductions over the
# 20x20 H0/S matrices and their 3x20x20 Cartesian derivatives, assembled by
# `H0_and_S_vectorized` for CH4 + mio-1-1.  Compared with `==` on the float,
# not `pytest.approx`: any movement at all is a regression under REG-01.
CH4_H0_SUM = -150.13565671807473
CH4_S_SUM = 24.694869162805045
CH4_H0_ABS_SUM = 353.67703472587453
CH4_S_ABS_SUM = 36.60802871760147
CH4_DH0_ABS_SUM = 777.8475256952822
CH4_DS_ABS_SUM = 43.245812432176216


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


def _ch4_h0_and_s(tmp_path):
    """Assemble CH4 + mio-1-1 H0/S through the live single-system path.

    Returns ``(const, H0, dH0, S, dS)``.  Deliberately calls
    ``H0_and_S_vectorized`` directly rather than driving ``ESDriver``, so that
    the checksums pin the H0/S assembly alone and are not perturbed by SCF,
    Coulomb, or repulsion.
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
        nonzero_lengths = [
            int((R_tensor[i] != 0).sum()) for i in range(R_tensor.shape[0])
        ]
        R = const.R_orb.detach()
        return (
            R_tensor.shape[0],
            rows_equal,
            nonzero_lengths,
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

    Compared with `==` on the float, not `pytest.approx`.  Under REG-01 a
    refactor that moves a digit is a regression even if it moves it in a
    direction someone considers an improvement.
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

    # CH4: one C with 4 orbitals plus four H with 1 each.
    assert shape == (20, 20)
    assert h0_sum == CH4_H0_SUM
    assert s_sum == CH4_S_SUM
    assert h0_abs == CH4_H0_ABS_SUM
    assert s_abs == CH4_S_ABS_SUM
    assert dh0_abs == CH4_DH0_ABS_SUM
    assert ds_abs == CH4_DS_ABS_SUM


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R_tensor rows are zero-padded past each pair's own grid length "
        "(_bond_integral.py:985 allocates zeros, :1061 writes only the head), "
        "so they are NOT monotonic and torch.searchsorted against a row is "
        "undefined behaviour.  Plan 05-01 Task 2 fills the tail by continuing "
        "the same arithmetic progression; this xfail flips to a pass there."
    ),
)
def test_r_tensor_rows_are_strictly_increasing():
    """Every R_tensor row must be strictly increasing across its FULL width.

    This is the single concrete blocker for the per-pair lookup.  `R_tensor` is
    allocated as `zeros((n_pairs, 1301))` at `_bond_integral.py:985` and only
    `R_tensor[i, :len(R_orb_i)]` is written at `:1061`, so today row 550 onward
    is 0.0 for mio-1-1 while row 549 is 5.82095.  `torch.searchsorted` requires
    a sorted boundary tensor; against a row that rises then drops to zero its
    result is arbitrary, which is why Task 2 cannot simply index `R_tensor` as
    it stands.

    Masking around the zero tail at the call site was considered and rejected:
    it makes the lookup correct only by caller discipline.  Filling the tail
    makes it correct by construction.
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
