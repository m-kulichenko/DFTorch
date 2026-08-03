"""The D-02 verbose flag: default stays noisy, and the print inventory stays complete.

Two contracts, both claimed elsewhere and neither previously enforced.

1. ``docs/LIBRARY-OUTPUT-INVENTORY.md`` states that
   ``tests/test_verbose_flag.py::test_inventory_covers_every_print`` walks the package and
   asserts the row count equals the shipped ``print(`` count, "so a print added by a later
   phase without a row here fails the suite rather than accumulating quietly." That test did
   not exist -- the document described a gate nobody had written, which is worse than no
   gate, because it reads as covered. Recorded as `.planning/WINDOWS.md` entry 2.

2. ``05-VALIDATION.md`` requires a capsys assertion that the default output is *unchanged*
   and that the flag suppresses status chatter but not warnings.

On the default: D-02 rejected both a quiet default and a migration to ``logging``, because
REG-02 and REG-03 require existing simple-format callers and the tutorial notebook to see
byte-identical output. This phase makes the noise suppressible, not absent. A test that
asserted a quiet default would be asserting the option the user declined.
"""

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import contextlib
import io
import pathlib
import re

import pytest
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "src" / "dftorch"
INVENTORY = REPO_ROOT / "docs" / "LIBRARY-OUTPUT-INVENTORY.md"


def _shipped_print_sites() -> list[tuple[str, int, str]]:
    """Every ``print(`` occurrence in the shipped package.

    Deliberately the same mechanical walk the inventory documents, so the two cannot
    drift by construction.
    """
    sites: list[tuple[str, int, str]] = []
    for p in sorted(PKG_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if "print(" in line:
                sites.append((p.as_posix(), i + 1, line.strip()))
    return sites


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ------------------------------------------------------------------ inventory gate


def test_inventory_covers_every_print():
    """The gate the inventory claims: row count must equal shipped print count."""
    assert INVENTORY.is_file(), f"missing {INVENTORY}"

    sites = _shipped_print_sites()
    text = INVENTORY.read_text(encoding="utf-8")

    # A data row is one whose FIRST cell is a backticked source path. Matching anywhere
    # in the line instead would also catch the glob patterns in this document's prose
    # (`src/dftorch/**/*.py`) and the summary rows above the table.
    rows = [
        ln
        for ln in text.splitlines()
        if re.match(r"^\|\s*`src/dftorch/[\w/]+\.py`\s*\|", ln)
    ]

    assert len(rows) == len(sites), (
        f"docs/LIBRARY-OUTPUT-INVENTORY.md has {len(rows)} rows but src/dftorch ships "
        f"{len(sites)} print( occurrences. A print added without a row here would "
        f"otherwise accumulate silently, which is exactly what this gate exists to stop. "
        f"Re-run the walk documented in that file and add the missing rows."
    )


def test_every_inventoried_file_still_exists():
    """A row naming a deleted file means the inventory is stale, not that it is complete."""
    text = INVENTORY.read_text(encoding="utf-8")
    # Backticked paths only, so the glob patterns in the prose are not treated as files.
    named = {m for m in re.findall(r"`(src/dftorch/[\w/]+\.py)`", text)}
    missing = sorted(n for n in named if not (REPO_ROOT / n).is_file())
    assert not missing, (
        f"inventory names files that no longer exist: {missing}. "
        f"src/dftorch/script.py was removed in plan 05-03; rows for deleted files must go."
    )


# ------------------------------------------------------------------ D-02 behaviour

CH4_PARAMS = {
    "CELL": [25.0, 25.0, 25.0],
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 8.0,
    "RCUT_REPULSIVE": 4.0,
    "COUL_METHOD": "FULL",
    "SCF_MAX_ITER": 25,
    "KRYLOV_START": 5,
}


def _run_ch4(extra: dict) -> tuple[str, float]:
    """Run a full CH4 SCF, capturing stdout. Returns (stdout, e_tot)."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    params = dict(CH4_PARAMS)
    params["FILENAME"] = str(REPO_ROOT / "tests" / "ch4.xyz")
    params["SKFPATH"] = str(REPO_ROOT / "tests" / "data_skf_mio-1-1") + os.sep
    params.update(extra)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        const = Constants(params)
        structure = Structure(params, const, device="cpu")
        ESDriver(params, device="cpu")(structure, const, do_scf=True)
    return buf.getvalue(), float(structure.e_tot)


@pytest.fixture(scope="module")
def ch4_runs():
    torch.set_default_dtype(torch.float64)
    default_out, default_e = _run_ch4({})
    quiet_out, quiet_e = _run_ch4({"VERBOSE_LIBRARY_OUTPUT": False})
    return default_out, default_e, quiet_out, quiet_e


def test_default_is_noisy(ch4_runs):
    """D-02: omitting the key must reproduce the output callers saw before it existed."""
    default_out, _, _, _ = ch4_runs
    assert default_out.strip(), (
        "default output is empty. D-02 requires the flag to default to TODAY'S NOISY "
        "behaviour so REG-02/REG-03 callers and the tutorial notebook are unaffected. "
        "A quiet default was explicitly rejected."
    )
    assert "### Do _scf ###" in default_out


def test_flag_suppresses_status_chatter(ch4_runs):
    """Opting in must actually quieten the run."""
    default_out, _, quiet_out, _ = ch4_runs
    assert len(quiet_out) < len(default_out) / 2, (
        f"VERBOSE_LIBRARY_OUTPUT=False produced {len(quiet_out)} chars against a default "
        f"of {len(default_out)}; the flag is not suppressing status chatter."
    )
    assert "### Do _scf ###" not in quiet_out
    assert "coulomb_matrix_vectorized" not in quiet_out


def test_flag_does_not_suppress_warnings(ch4_runs):
    """Gating a failure warning trades noise for silent wrongness.

    CH4 emits `zero norm_dr` from _xl_tools. It is a genuine warning about the run, not
    status chatter, so it must survive the flag. If a future change makes CH4 stop
    emitting it, this test should be re-pointed at another unconditional warning rather
    than deleted.
    """
    _, _, quiet_out, _ = ch4_runs
    assert "zero norm_dr" in quiet_out, (
        "the failure warning was suppressed along with the status chatter. Warnings "
        "(cnt == MaxIt, zero norm_dr, Not converged, the spinw.txt load failure) are "
        "unconditional by design."
    )


def test_flag_does_not_change_the_physics(ch4_runs):
    """Output verbosity must not perturb a single bit of the result."""
    _, default_e, _, quiet_e = ch4_runs
    assert default_e == quiet_e, (
        f"e_tot differs between verbosity settings: {default_e!r} vs {quiet_e!r}"
    )
