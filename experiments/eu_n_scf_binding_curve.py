"""Europium-nitrogen binding curve, recomputed live: one pass versus settled.

Phase 6 plan 06-05, requirements SCC-01, SCC-02, SCC-03.  This script exists for
one purpose: to put a freshly computed picture in front of a person so they can
say whether it looks right.  Decision D-6.02 makes that human judgement the
phase's completion bar, in place of a numeric one.

Plain-language orientation
--------------------------
A *binding curve* is the molecule's energy plotted against how far apart its two
atoms are.  It should fall as the atoms approach and find each other, reach a
lowest point where the bond is most comfortable, then rise steeply as they are
pushed too close.  A single clean well.  Anything else - two wells, a straight
slope, a flat line - says something is wrong.

Three curves are computed, at every one of the 21 separations:

1. **one pass** - the charge distribution is worked out once and never revisited.
   This is the curve a human already approved in Phase 4.  It carries no
   electron-repulsion term at all, by design (Phase 4 decision D-11).
2. **settled, per-atom charge** - the repeat-until-settled loop, tracking one
   charge number per atom.  This is the default self-consistent path and the
   comparison decision D-6.09 mandates.
3. **settled, per-orbital-group charge** - the same loop tracking one charge per
   orbital group (s, p, d, f) instead of one lump per atom, which is what plans
   06-02 and 06-03 built.  Selected by the existing ``MAGNETIC_HUBBARD_LDEP``
   parameter key, which is off by default.

The three answer different questions and are expected to sit at different
heights.  ``docs/SINGLE-SHOT-VS-SELF-CONSISTENT.md`` is the written verdict on
why, and concludes the difference is one of definition, not a defect.

Everything here is recomputed from the code as it stands at the moment of the
run.  Nothing is replayed from a recorded table, a cached file, or a number
copied out of a research document: a replay would make the sign-off describe a
tree that no longer exists (threat T-06-28).

No number this script produces is written into a test as a reference value, and
nothing here compares the settled curve's lowest point against Phase 4's
tolerance band.  Decision D-6.01 explicitly declined that as this phase's bar and
decision D-6.08 froze no numbers this phase.

Running it
----------
::

    python experiments/eu_n_scf_binding_curve.py

It prints a 21-row table and writes ``figures/eu_n_scf_binding_curve.png``.

**It needs matplotlib in the interpreter that runs it.**  matplotlib is a
*development-time* dependency of this project, carried in the ``dev`` extra of
``pyproject.toml`` beside pytest and ruff, and deliberately NOT a runtime
dependency: figures like this one are how the physics gets verified during
development, but nobody installing dftorch to compute with it needs a plotting
library.  ``uv run`` supplies it.  A bare interpreter may not, so this script
checks for matplotlib before spending any time on physics and says what to do
if it is missing.

ASCII only, in the source and in every string emitted, per the Phase 4 rule that
a failure be diagnosable from a cp1252 console.  Write "A" for Angstrom and "->"
for an arrow.
"""

import os

# Disable TorchDynamo/Inductor compilation (keeps the run deterministic and
# avoids requiring a C++ toolchain).  Mirrors tests/test_eu_n_scan.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import datetime
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGURE_DIR = REPO / "figures"
FIGURE_PATH = FIGURE_DIR / "eu_n_scf_binding_curve.png"
SKF_DIR = REPO / "tests" / "f_orbital_data"

# --- Scan grid ---------------------------------------------------------------
#
# Exactly the grid tests/test_eu_n_scan.py uses, kept by decision D-6.07.  The
# two curves cannot be compared point for point on different grids, and
# overlaying them is the whole reason this figure exists.  Extending past
# 3.60 A was offered to the user and declined.
SCAN_MIN_ANGSTROM = 1.60
SCAN_MAX_ANGSTROM = 3.60
SCAN_STEP_ANGSTROM = 0.10

# A separation reported for context only, never as a bar this curve must clear.
# It is the mean of 16 Eu-N bond lengths in crowded Eu-Bp coordination
# complexes, not a diatomic measurement.  Decision D-6.01 explicitly declined to
# make it this phase's gate.
COORDINATION_MEAN_ANGSTROM = 2.655

# Driver parameters, pinned to match tests/test_eu_n_scan.py so the Phase 4
# one-pass curve this reproduces is the same curve that was approved.
#
# VERBOSE_LIBRARY_OUTPUT lives HERE, in this script's own parameter dictionary,
# and nowhere else.  The library's noisy default is deliberate (Phase 5 decision
# D-02) and must not be changed globally; without the override the per-iteration
# chatter from roughly 2,500 passes buries the table this script prints.
EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}

# The sentinel plan 06-01 returns when a charge loop exhausted its cap without
# the charges settling.  Read from structure.scf_iter_count, never scraped out
# of printed text.  This is the only machine-readable signal of which
# separations the solver gave up on, and marking them is what decision D-6.09
# requires (threat T-06-27).
DID_NOT_CONVERGE = -1

# structure.q is built as (population - Znuc), so a POSITIVE value means surplus
# electrons.  Nitrogen is atom 1 in the geometry written below, so q[1] is
# literally "how many electrons moved onto nitrogen".
NITROGEN_INDEX = 1

FIGURE_DPI = 200

# The three series, in the order they are computed and drawn.  ``shell_resolved``
# selects the per-orbital-group charge description through MAGNETIC_HUBBARD_LDEP;
# ``do_scf`` selects whether the charge loop runs at all.
SERIES_SPECS = (
    {
        "key": "one_pass",
        "label": "One pass, charge never revisited",
        "do_scf": False,
        "shell_resolved": False,
        "marker": "s",
    },
    {
        "key": "settled_per_atom",
        "label": "Settled, one charge per atom",
        "do_scf": True,
        "shell_resolved": False,
        "marker": "o",
    },
    {
        "key": "settled_per_group",
        "label": "Settled, one charge per orbital group",
        "do_scf": True,
        "shell_resolved": True,
        "marker": "^",
    },
)


# --- Geometry and parameters -------------------------------------------------


def scan_separations():
    """Return the uniform scan grid, inclusive of both endpoints.

    Each value is rounded to two decimals so the grid is exact in printed
    output rather than carrying the accumulated error of repeated float
    addition.  Copied from tests/test_eu_n_scan.py::scan_separations.
    """
    count = int(round((SCAN_MAX_ANGSTROM - SCAN_MIN_ANGSTROM) / SCAN_STEP_ANGSTROM)) + 1
    return [round(SCAN_MIN_ANGSTROM + SCAN_STEP_ANGSTROM * i, 2) for i in range(count)]


def _write_eu_n_xyz(path, separation):
    """Eu at the origin, N displaced along +x by ``separation``."""
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 6 settled binding curve)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _eu_n_params(separation, tmp_dir, shell_resolved):
    tag = "group" if shell_resolved else "atom"
    xyz_path = Path(tmp_dir) / f"eu_n_{tag}_{separation:.2f}.xyz"
    _write_eu_n_xyz(xyz_path, separation)

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(SKF_DIR) + os.sep
    params["MAGNETIC_HUBBARD_LDEP"] = shell_resolved
    return params


# --- Computation -------------------------------------------------------------


def compute_curves(tmp_dir, echo=print):
    """Recompute all three curves from live code and return them as plain data.

    Returns ``{"separations": [...], "series": [...], "started": ..., ...}`` with
    every value a plain Python float, int or None - no tensors - so the caller
    can hand the result straight to the renderer.

    ``Constants`` is built ONCE per charge resolution and reused across all 21
    separations.  It reads the geometry file only to recover the species list,
    which is (Eu, N) at every point of this scan, so reuse is a pure speed
    measure: tests/test_eu_n_scan.py verified it bit-identical to per-point
    construction while dropping that scan from 46 s to under 16 s.  Two
    instances are needed rather than one because ``MAGNETIC_HUBBARD_LDEP`` is
    read by ``Constants.__init__`` and stored on the object, so the per-atom and
    per-orbital-group paths cannot share one.

    A run that raises is recorded and the scan continues.  A single bad point
    must not cost a human their review - but it is reported in the tally, never
    swallowed.
    """
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    separations = scan_separations()

    constants = {}
    for shell_resolved in (False, True):
        params0 = _eu_n_params(separations[0], tmp_dir, shell_resolved)
        constants[shell_resolved] = Constants(params0).to("cpu")

    series = []
    for spec in SERIES_SPECS:
        # A driver per series.  ESDriver stores the dictionary it is given and
        # writes defaults back into it, so each gets its own copy.
        driver_params = _eu_n_params(separations[0], tmp_dir, spec["shell_resolved"])
        driver = ESDriver(driver_params, device="cpu")
        const = constants[spec["shell_resolved"]]

        energies = []
        charges_on_n = []
        counts = []
        errors = []

        echo(f"  computing: {spec['label']} ...")
        for separation in separations:
            try:
                params = _eu_n_params(separation, tmp_dir, spec["shell_resolved"])
                structure = Structure(params, const, device="cpu")
                driver(structure, const, do_scf=spec["do_scf"])

                energies.append(float(structure.e_tot.item()))
                q = structure.q.detach().cpu().tolist()
                charges_on_n.append(float(q[NITROGEN_INDEX]))
                # Absent on the one-pass path, which runs no loop at all.
                counts.append(getattr(structure, "scf_iter_count", None))
            except Exception as exc:  # recorded, never hidden
                energies.append(None)
                charges_on_n.append(None)
                counts.append(None)
                errors.append((separation, type(exc).__name__, str(exc)))
                echo(
                    f"    FAILED at {separation:.2f} A: "
                    f"{type(exc).__name__}: {exc}"
                )

        series.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "marker": spec["marker"],
                "runs_a_loop": spec["do_scf"],
                "energies": energies,
                "charges_on_n": charges_on_n,
                "counts": counts,
                "errors": errors,
            }
        )

    return {"separations": separations, "series": series}


def locate_minimum(separations, energies):
    """Return ``(index, separation)`` of the lowest computed energy, or None.

    Points that raised (energy ``None``) are skipped.  Ties break toward the
    smaller separation, matching tests/test_eu_n_scan.py::locate_minimum.
    """
    usable = [i for i, e in enumerate(energies) if e is not None]
    if not usable:
        return None
    index = min(usable, key=lambda i: (energies[i], separations[i]))
    return index, separations[index]


def converged_count(series):
    """How many separations this series settled at.

    A positive pass count means the charges stopped moving.  ``-1`` means the
    loop exhausted its cap and gave up.  ``None`` means either the point raised
    or the series runs no loop at all.
    """
    return sum(
        1
        for count in series["counts"]
        if isinstance(count, int) and count > 0
    )


def gave_up_separations(result, series):
    """The separations where this series' loop came back with the -1 sentinel."""
    return [
        separation
        for separation, count in zip(result["separations"], series["counts"])
        if count == DID_NOT_CONVERGE
    ]


# --- Printed table -----------------------------------------------------------


def _cell(value, width, decimals):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{decimals}f}"


def _int_cell(value, width):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}d}"


def print_table(result, echo=print):
    """Print one row per separation, then a tally line per settled series."""
    separations = result["separations"]
    by_key = {s["key"]: s for s in result["series"]}
    one_pass = by_key["one_pass"]
    per_atom = by_key["settled_per_atom"]
    per_group = by_key["settled_per_group"]

    echo("")
    echo(
        "  r (A) |     E one pass |    E per atom |   E per group "
        "|  qN one pass |  qN per atom | qN per group | it per atom | it per group"
    )
    echo("  " + "-" * 129)

    for i, separation in enumerate(separations):
        echo(
            f"  {separation:5.2f} | "
            f"{_cell(one_pass['energies'][i], 14, 6)} | "
            f"{_cell(per_atom['energies'][i], 13, 6)} | "
            f"{_cell(per_group['energies'][i], 13, 6)} | "
            f"{_cell(one_pass['charges_on_n'][i], 12, 6)} | "
            f"{_cell(per_atom['charges_on_n'][i], 12, 6)} | "
            f"{_cell(per_group['charges_on_n'][i], 12, 6)} | "
            f"{_int_cell(per_atom['counts'][i], 11)} | "
            f"{_int_cell(per_group['counts'][i], 12)}"
        )

    echo("")
    echo("  Energies in eV. Charges in electrons, positive meaning electrons")
    echo("  moved ONTO nitrogen (q = population - Znuc). Iteration counts are")
    echo(f"  passes taken; {DID_NOT_CONVERGE} means the loop gave up.")

    echo("")
    total = len(separations)
    for series in result["series"]:
        if not series["runs_a_loop"]:
            continue
        settled = converged_count(series)
        echo(f"  {series['label']}: converged at {settled} of {total} separations")
        gave_up = gave_up_separations(result, series)
        if gave_up:
            echo(
                "    gave up at: "
                + ", ".join(f"{r:.2f} A" for r in gave_up)
            )

    for series in result["series"]:
        if series["errors"]:
            echo("")
            echo(f"  {series['label']}: {len(series['errors'])} separation(s) raised")
            for separation, kind, message in series["errors"]:
                echo(f"    {separation:.2f} A -> {kind}: {message}")

    echo("")
    for series in result["series"]:
        found = locate_minimum(separations, series["energies"])
        if found is None:
            echo(f"  {series['label']}: no computed point, no lowest point")
            continue
        index, separation = found
        echo(
            f"  {series['label']}: lowest point at {separation:.2f} A, "
            f"{series['energies'][index]:.6f} eV"
        )


# --- Figure ------------------------------------------------------------------

# Presentation values copied from experiments/diatomic_scans/style.py, which is
# the established figure style in this repository.  They are inlined as a
# fallback because that directory is not tracked by git, so a fresh checkout
# would otherwise be unable to render this figure at all.  The shared module is
# preferred whenever it is present, so the two cannot drift apart in practice.
FALLBACK_NAVY = "#1f3864"
FALLBACK_SLATE = "#6b7280"
FALLBACK_CRIMSON = "#b3282d"
FALLBACK_GREEN = "#2e8b57"
FALLBACK_GOLD = "#b8860b"

FALLBACK_RC = {
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 10.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}


def _shared_style():
    """Return experiments/diatomic_scans/style.py if it is present, else None."""
    style_dir = HERE / "diatomic_scans"
    if not (style_dir / "style.py").is_file():
        return None
    if str(style_dir) not in sys.path:
        sys.path.insert(0, str(style_dir))
    try:
        import style
    except ImportError:
        return None
    return style


def _apply_style():
    """Apply the house figure style and return its named colours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = _shared_style()
    if style is not None:
        style.apply()
        return (
            getattr(style, "NAVY", FALLBACK_NAVY),
            getattr(style, "SLATE", FALLBACK_SLATE),
            getattr(style, "CRIMSON", FALLBACK_CRIMSON),
            getattr(style, "GREEN", FALLBACK_GREEN),
            getattr(style, "GOLD", FALLBACK_GOLD),
            getattr(style, "finish", lambda ax: None),
        )

    plt.rcParams.update(FALLBACK_RC)

    def finish(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return (
        FALLBACK_NAVY,
        FALLBACK_SLATE,
        FALLBACK_CRIMSON,
        FALLBACK_GREEN,
        FALLBACK_GOLD,
        finish,
    )


def _series_legend_label(result, series):
    """The in-plot label: what the series is, where it bottoms out, how it went.

    The converged-of-21 count rides in the legend deliberately, so a reader can
    answer "how many settled?" without counting markers (threat T-06-27).
    """
    parts = [series["label"]]
    found = locate_minimum(result["separations"], series["energies"])
    if found is not None:
        parts.append(f"lowest at {found[1]:.2f} A")
    if series["runs_a_loop"]:
        total = len(result["separations"])
        parts.append(f"converged at {converged_count(series)} of {total}")
    return "  -  ".join(parts)


def _judgeable_energy_limits(result):
    """Vertical limits for the energy panel, set by the points that settled.

    A charge loop that gave up can hand back an energy hundreds of eV above the
    true scale.  Auto-scaling the panel to contain those would compress every
    well into a band a few pixels tall - and the shape of that well is precisely
    what a human is being asked to judge here, so an auto-scaled panel would
    defeat the gate this whole script exists to serve.

    The limits are therefore taken from the points where the loop settled, plus
    the one-pass path, which runs no loop at all.  The runaway points visibly
    leave the panel rather than being deleted, and every separation where a loop
    gave up is shaded on BOTH panels, so a reader cannot mistake an off-scale
    point for missing data (threat T-06-27).  The lower panel is left on full
    autoscale, which is where the runaway magnitudes stay legible.

    Returns ``None`` when nothing settled, in which case the caller leaves the
    panel on autoscale rather than inventing a range.
    """
    usable = []
    for series in result["series"]:
        for energy, count in zip(series["energies"], series["counts"]):
            if energy is None:
                continue
            if series["runs_a_loop"] and not (isinstance(count, int) and count > 0):
                continue
            usable.append(energy)

    if not usable:
        return None

    low, high = min(usable), max(usable)
    padding = 0.08 * (high - low) if high > low else 1.0
    return low - padding, high + padding


def _shade_gave_up(ax, result, gold, with_label):
    """Shade every separation where any charge loop gave up.

    A vertical band marks the separation itself, which stays visible whatever
    the panel's vertical range does - unlike a marker on the point, which
    disappears the moment the point is off-scale.  Decision D-6.09 requires the
    separations to be visibly marked; this is what makes that true on both
    panels at once.
    """
    half_step = 0.5 * SCAN_STEP_ANGSTROM
    separations = sorted(
        {
            r
            for series in result["series"]
            for r in gave_up_separations(result, series)
        }
    )

    label = None
    if with_label and separations:
        label = (
            f"Loop gave up here (scf_iter_count = {DID_NOT_CONVERGE}): "
            f"{len(separations)} separations"
        )

    for separation in separations:
        ax.axvspan(
            separation - half_step,
            separation + half_step,
            color=gold,
            alpha=0.16,
            lw=0,
            zorder=0,
            label=label,
        )
        label = None

    return separations


def _plot_panel(ax, result, colours, values_key, gold):
    """Draw the three series onto one panel, marking every gave-up point."""
    separations = result["separations"]

    for series, colour in zip(result["series"], colours):
        xs = [
            r
            for r, v in zip(separations, series[values_key])
            if v is not None
        ]
        ys = [v for v in series[values_key] if v is not None]
        ax.plot(
            xs,
            ys,
            linestyle="-",
            marker=series["marker"],
            color=colour,
            ms=6.0,
            lw=2.0,
            label=_series_legend_label(result, series),
        )

        found = locate_minimum(separations, series["energies"])
        if found is not None and values_key == "energies":
            index, separation = found
            ax.plot(
                [separation],
                [series["energies"][index]],
                marker="*",
                linestyle="none",
                color=colour,
                ms=18,
                zorder=6,
            )

        gave_up = [
            (r, v)
            for r, v, count in zip(separations, series[values_key], series["counts"])
            if count == DID_NOT_CONVERGE and v is not None
        ]
        if gave_up:
            # No legend entry here: the shaded band carries it, and the band is
            # what stays visible when a runaway point sits off the panel.
            ax.plot(
                [r for r, _ in gave_up],
                [v for _, v in gave_up],
                linestyle="none",
                marker="X",
                mfc="none",
                mec=gold,
                mew=2.6,
                ms=16,
                zorder=7,
            )


def render(result, echo=print):
    """Write the two-panel figure and return the path it was written to.

    Upper panel: total energy against separation - the shape a human is being
    asked to judge.  Lower panel: how much charge moved onto nitrogen, which is
    where a runaway is legible even when the energy scale hides it.  Phase 4's
    own sign-off found that a single linear energy axis could not support the
    shape judgement being asked for and was resolved with a second panel; this
    repeats that lesson rather than rediscovering it.

    No prose explanation box, caption block or footer text is drawn.  The
    narrative belongs in the message to the human, not baked into the image.
    """
    navy, slate, crimson, green, gold, finish = _apply_style()
    import matplotlib.pyplot as plt

    colours = (slate, navy, crimson)

    fig, (ax_energy, ax_charge) = plt.subplots(
        2, 1, figsize=(12.0, 11.0), sharex=True
    )
    fig.suptitle(
        "Europium-Nitrogen Binding Curve: One-Pass Charge versus "
        "Self-Consistent Charge\nat Per-Atom and Per-Orbital-Group Resolution",
        fontsize=16.0,
        fontweight="bold",
        y=0.975,
    )

    gave_up = _shade_gave_up(ax_energy, result, gold, with_label=True)
    _plot_panel(ax_energy, result, colours, "energies", gold)
    ax_energy.axvline(
        COORDINATION_MEAN_ANGSTROM,
        color=green,
        ls="--",
        lw=1.5,
        alpha=0.9,
        label=(
            f"Eu-N coordination mean, {COORDINATION_MEAN_ANGSTROM} A "
            "(context, not a pass mark)"
        ),
    )
    limits = _judgeable_energy_limits(result)
    if limits is not None:
        ax_energy.set_ylim(*limits)
    energy_title = "Total energy: stars mark each curve's lowest point"
    if gave_up:
        energy_title += (
            "\n(vertical range set by the separations that settled; "
            "runaway points leave the panel)"
        )
    ax_energy.set_ylabel("Total energy  $E_\\mathrm{tot}$  (eV)")
    ax_energy.set_title(energy_title)
    # Pinned rather than "best": on the measured curves "best" puts the box over
    # the top-left corner, which is exactly where the two settled curves climb
    # out of their wells at short separation.
    ax_energy.legend(loc="center right")
    finish(ax_energy)

    _shade_gave_up(ax_charge, result, gold, with_label=False)
    _plot_panel(ax_charge, result, colours, "charges_on_n", gold)
    ax_charge.axhline(0.0, color="#444444", lw=1.1)
    ax_charge.axvline(
        COORDINATION_MEAN_ANGSTROM, color=green, ls="--", lw=1.5, alpha=0.9
    )
    ax_charge.set_xlabel("Eu-N separation  $r$  (A)")
    ax_charge.set_ylabel("Electrons moved onto nitrogen  $q_\\mathrm{N}$  (e)")
    ax_charge.set_title(
        "Charge transfer: above zero means nitrogen gained electrons\n"
        "(full vertical range, so a runaway stays legible)"
    )
    ax_charge.legend(loc="best")
    finish(ax_charge)

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI)
    plt.close(fig)

    echo(f"\n  wrote {FIGURE_PATH}")
    return FIGURE_PATH


# --- Entry point -------------------------------------------------------------


def _matplotlib_is_available():
    return importlib.util.find_spec("matplotlib") is not None


def _timestamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def main():
    if not _matplotlib_is_available():
        print(
            "ERROR: matplotlib is not importable in this interpreter, so the\n"
            "figure cannot be drawn. Nothing was computed - the check runs\n"
            "first so a missing plotting library does not cost you two minutes\n"
            "of physics.\n"
            "\n"
            f"  interpreter: {sys.executable}\n"
            "\n"
            "matplotlib is a development-time dependency of this project, in\n"
            "the 'dev' extra of pyproject.toml. It is deliberately not a runtime\n"
            "dependency: the library computes the numbers, it does not draw.\n"
            "\n"
            "Run this through the project environment, which supplies it:\n"
            "\n"
            "  uv run python experiments/eu_n_scf_binding_curve.py\n"
            "\n"
            "If that is what you just did, the dev extra is not installed. Run\n"
            "'uv sync --extra dev' and try again.",
            file=sys.stderr,
        )
        return 2

    started = _timestamp()
    print("Eu-N binding curve, recomputed live from the current tree.")
    print(f"  run started (UTC): {started}")
    print(f"  interpreter:       {sys.executable}")
    print(f"  torch:             {torch.__version__}")
    print(
        f"  grid:              {SCAN_MIN_ANGSTROM:.2f} to "
        f"{SCAN_MAX_ANGSTROM:.2f} A in steps of {SCAN_STEP_ANGSTROM:.2f} "
        f"({len(scan_separations())} points)"
    )
    print("")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with tempfile.TemporaryDirectory(prefix="eu_n_curve_") as tmp_dir:
            result = compute_curves(tmp_dir)
    finally:
        torch.set_default_dtype(previous_dtype)

    print_table(result)
    render(result)
    print(f"  run finished (UTC): {_timestamp()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
