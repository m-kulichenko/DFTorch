"""Every hardcoded orbital-count site is dispositioned, and every guard is reached.

The machine half of decision **D-04** (requirements REG-04, CLN-03).
``tests/orbital_count_sweep.py`` enumerates the sites;
``docs/ORBITAL-COUNT-INVENTORY.md`` dispositions them; this module holds the two
together and proves each refusal actually fires.

Three properties are gated here, and the first is the one that keeps the audit
from rotting:

1.  **Completeness, in both directions.** A site present in the code but absent
    from the inventory fails, and a row in the inventory with no matching site
    fails too. Without the second direction an inventory could drift by keeping
    rows for code that has since moved.
2.  **No unresolved rows.** ``needs-action`` is a state Task 1 of plan 05-06 was
    allowed to record and Task 2 had to clear; it may never reappear silently.
3.  **Reachability, not existence.** A test asserting that a guard function
    exists proves nothing -- a guard that is never called looks identical from
    the outside. Each ``guarded`` row is exercised by driving a real f system
    (Eu-N on ``tests/f_orbital_data``) into the site and observing the named
    exception, and each guard is separately shown inert for CH4 + mio-1-1.

Why that third property is worth the setup: a silently unhandled orbital-count
site returns a well-formed matrix with zeros where the f values belong. The
worked example is commit ``4dbffaa``, where a placeholder ``Ep = 0.000039`` in
mio-1-1's ``H-H.skf`` gave hydrogen four orbitals instead of one, made the
overlap matrix indefinite, and stopped CH4 converging -- with every shape,
finiteness and symmetry assertion in the suite still green. See
``tests/test_shell_count_parsing.py``.

ASCII only, per the Phase 4 rule: a failure must be diagnosable from pytest
output on a cp1252 console.
"""

import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic
# and avoids requiring a C++ toolchain).  Mirrors tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from orbital_count_sweep import (
    FAMILIES,
    count_prose_mentions,
    flatten_source,
    pattern_for,
    sweep_orbital_count_sites,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "dftorch"
INVENTORY_PATH = REPO_ROOT / "docs" / "ORBITAL-COUNT-INVENTORY.md"
TESTS_DIR = Path(__file__).resolve().parent

DISPOSITIONS = ("extended", "guarded", "unreachable", "needs-action")


def run_with_float64(fn):
    """Run ``fn`` under float64 defaults, restoring dftorch module state after.

    Copied from ``tests/test_f_orbital_skf.py`` -- the established harness for
    every f-orbital test module in this project.
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


# --------------------------------------------------------------------------
# Reading the inventory back
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^###\s+`src/(?P<path>[^`]+)`")
_ROW_START = re.compile(r"^\|\s*\d+\s*\|")


@dataclass(frozen=True)
class InventoryRow:
    path: str
    line: int
    snippet: str
    family: str
    owner: str
    decides: str
    disposition: str
    evidence: str
    test: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.path, self.line, self.snippet)


def _unfence(cell: str) -> str:
    cell = cell.strip()
    if cell.startswith("`") and cell.endswith("`") and len(cell) >= 2:
        return cell[1:-1]
    return cell


def parse_inventory_rows() -> list[InventoryRow]:
    """Read ``docs/ORBITAL-COUNT-INVENTORY.md`` back into records.

    Splitting is done on unescaped pipes only: one evidence cell quotes a
    ``grep`` alternation containing a literal ``\\|``, and a naive
    ``str.split("|")`` shears that row into the wrong number of columns.
    """
    rows: list[InventoryRow] = []
    current_path = None
    for raw in INVENTORY_PATH.read_text(encoding="utf-8").splitlines():
        heading = _HEADING.match(raw)
        if heading:
            current_path = heading.group("path")
            continue
        if not _ROW_START.match(raw):
            continue
        assert current_path is not None, (
            f"inventory row appears before any '### `src/...`' heading: {raw[:80]}"
        )
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", raw)[1:-1]]
        assert len(cells) == 8, (
            f"inventory row in {current_path} has {len(cells)} columns, expected 8. "
            f"Row: {raw[:120]}"
        )
        rows.append(
            InventoryRow(
                path=current_path,
                line=int(cells[0]),
                snippet=_unfence(cells[1]),
                family=_unfence(cells[2]),
                owner=_unfence(cells[3]),
                decides=cells[4],
                disposition=_unfence(cells[5]),
                evidence=cells[6],
                test=_unfence(cells[7]),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------

EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
    "VERBOSE_LIBRARY_OUTPUT": False,
}


def _build_eu_n(tmp_path: Path, **overrides):
    """Eu-N diatomic on the f fixture set: driver, structure, const, params."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    xyz_path = tmp_path / "eu_n_orbital_count.xyz"
    xyz_path.write_text(
        "2\n"
        "Eu-N diatomic (orbital-count guard case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {EU_N_SEPARATION:.8f} 0.00000000 0.00000000\n"
    )
    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(TESTS_DIR / "f_orbital_data") + os.sep
    params.update(overrides)

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    return driver, structure, const, params


def _build_ch4(**overrides):
    """f-free CH4 on mio-1-1: driver, structure, const, params."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    params = {
        "FILENAME": str(TESTS_DIR / "ch4.xyz"),
        "SKFPATH": str(TESTS_DIR / "data_skf_mio-1-1") + os.sep,
        "CELL": [25.0, 25.0, 25.0],
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 8.0,
        "RCUT_REPULSIVE": 4.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
        "VERBOSE_LIBRARY_OUTPUT": False,
    }
    params.update(overrides)
    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    return driver, structure, const, params


def _synthetic_batch_call(const, type_row):
    """Drive ``H0_and_S_vectorized_batch`` far enough to reach its f refusal.

    The refusal at ``_h0ands.py:636`` is not the function's first statement --
    it sits after the direction-cosine arithmetic -- so reaching it needs real
    tensors rather than sentinels.  Everything here is the smallest shape that
    gets there: one batch, two atoms, one neighbour slot, one pair.
    """
    from dftorch._h0ands import H0_and_S_vectorized_batch

    n_batch, n_atoms, max_nn, hdim = 1, 2, 1, 20
    TYPE = torch.tensor([type_row], dtype=torch.long)
    RX = torch.zeros(n_batch, n_atoms)
    RY = torch.zeros(n_batch, n_atoms)
    RZ = torch.zeros(n_batch, n_atoms)
    RX[0, 1] = EU_N_SEPARATION
    nnRx = torch.full((n_batch, n_atoms, max_nn), EU_N_SEPARATION)
    nnRy = torch.zeros(n_batch, n_atoms, max_nn)
    nnRz = torch.zeros(n_batch, n_atoms, max_nn)
    nnType = torch.zeros(n_batch, n_atoms, max_nn, dtype=torch.long)
    return H0_and_S_vectorized_batch(
        TYPE,
        RX,
        RY,
        RZ,
        torch.zeros(n_batch, hdim),
        torch.zeros(n_batch, n_atoms, dtype=torch.long),
        nnRx,
        nnRy,
        nnRz,
        nnType,
        const,
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([[0]]),
        torch.tensor([[0]]),
        const.R_orb,
        const.coeffs_tensor,
        False,
    )


def _synthetic_pair_grad_call(const, n_orb_I, n_orb_J):
    """Drive ``_stress._pair_grad_from_sk`` far enough to reach its f refusal.

    Its refusal also sits after arithmetic (the direction cosines and the knot
    offset ``dR - const.R_orb[idx]``), so the metadata dict has to be real
    enough to survive that much.
    """
    from dftorch._stress import _pair_grad_from_sk

    metadata = {
        "Rab_mskd": torch.tensor([[1.0, 0.0, 0.0]]),
        "idx": torch.tensor([0]),
        "IJ_pair_type": torch.tensor([0]),
        "JI_pair_type": torch.tensor([0]),
        "i0": torch.tensor([0]),
        "j0": torch.tensor([16]),
        "n_orb_I": torch.tensor([n_orb_I], dtype=torch.uint8),
        "n_orb_J": torch.tensor([n_orb_J], dtype=torch.uint8),
    }
    return _pair_grad_from_sk(metadata, const, torch.ones(20), torch.eye(3) * 25.0)


# --------------------------------------------------------------------------
# The guard cases, keyed by the id the inventory's Test column names
# --------------------------------------------------------------------------


def _case_esdriver_calc_forces(tmp_path):
    from dftorch._slater_koster_pair import FDerivativeUnsupportedError

    driver, structure, const, _ = _build_eu_n(tmp_path)
    return FDerivativeUnsupportedError, lambda: driver.calc_forces(structure, const)


def _case_esdriver_forward_unrestricted(tmp_path):
    from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError

    driver, structure, const, _ = _build_eu_n(tmp_path, UNRESTRICTED=True)
    return FSpinPolarizationUnsupportedError, lambda: driver(
        structure, const, do_scf=False
    )


def _case_h0_and_s_batch(tmp_path):
    from dftorch._slater_koster_pair import FAngularFormulaSourceError

    _, _, const, _ = _build_eu_n(tmp_path)
    # Eu (63) is the 16-orbital atom; N (7) is sp.
    return FAngularFormulaSourceError, lambda: _synthetic_batch_call(const, [63, 7])


def _case_pair_grad_from_sk(tmp_path):
    from dftorch._slater_koster_pair import FDerivativeUnsupportedError

    _, _, const, _ = _build_eu_n(tmp_path)
    return FDerivativeUnsupportedError, lambda: _synthetic_pair_grad_call(const, 16, 4)


def _case_forces_spin(tmp_path):
    from dftorch._forces import forces_spin
    from dftorch._slater_koster_pair import FDerivativeUnsupportedError

    _, structure, const, _ = _build_eu_n(tmp_path)
    return FDerivativeUnsupportedError, lambda: forces_spin(
        None, None, None, 2, const, structure.TYPE
    )


def _case_get_h_spin(tmp_path):
    from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError
    from dftorch._spin import get_h_spin_eager

    shell_types = torch.tensor([1, 2, 3, 4, 1, 2])  # Eu s/p/d/f then N s/p
    return FSpinPolarizationUnsupportedError, lambda: get_h_spin_eager(
        None, None, None, None, shell_types
    )


def _case_get_h_spin_diag(tmp_path):
    from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError
    from dftorch._spin import get_h_spin_diag_eager

    shell_types = torch.tensor([1, 2, 3, 4, 1, 2])
    return FSpinPolarizationUnsupportedError, lambda: get_h_spin_diag_eager(
        None, None, None, None, shell_types
    )


# ``ewald_real_space_vectorized_sr`` was a guard case until Phase 6.  Its
# refusal (``FShellResolvedCoulombUnsupportedError``) is retired by requirement
# SCC-02, which built the seven missing f shell-pair blocks, and the function
# has no orbital-count or shell-count site left for the sweep to find.  The
# exception class itself is deliberately kept -- the taxonomy stays at four
# names (``test_no_new_f_exception_class_was_defined`` below) -- and
# ``tests/test_shell_resolved_coulomb_f.py`` now owns that function's coverage.
GUARD_CASES = {
    "ESDriver.calc_forces": _case_esdriver_calc_forces,
    "ESDriver.forward-unrestricted": _case_esdriver_forward_unrestricted,
    "H0_and_S_vectorized_batch": _case_h0_and_s_batch,
    "_pair_grad_from_sk": _case_pair_grad_from_sk,
    "forces_spin": _case_forces_spin,
    "get_h_spin": _case_get_h_spin,
    "get_h_spin_diag": _case_get_h_spin_diag,
}


def _f_error_classes():
    from dftorch import _slater_koster_pair as skp

    return (
        skp.FAngularFormulaSourceError,
        skp.FDerivativeUnsupportedError,
        skp.FSpinPolarizationUnsupportedError,
        skp.FShellResolvedCoulombUnsupportedError,
    )


def _assert_no_f_refusal(call, case_id: str):
    """Assert ``call`` does not raise any f-unsupported error.

    Some inert checks drive a real entry point with f-free data that is not a
    complete calculation, so the call may still fail downstream on its own
    arguments.  Only the refusal is under test here; a downstream failure means
    the guard let the f-free system through, which is the property being
    asserted.  The end-to-end counterweight is
    ``test_ch4_runs_end_to_end_after_the_guards``.
    """
    try:
        call()
    except _f_error_classes() as exc:
        raise AssertionError(
            f"guard case {case_id!r} refused an f-FREE system: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except Exception:
        pass


def _inert_case(case_id: str):
    """Call the guard behind ``case_id`` with f-free (CH4 + mio-1-1) inputs."""
    if case_id == "ESDriver.calc_forces":
        from dftorch.ESDriver import _require_f_derivatives

        _, structure, const, _ = _build_ch4()
        assert _require_f_derivatives(structure, const, "inert-check") is None
    elif case_id == "ESDriver.forward-unrestricted":
        from dftorch.ESDriver import _require_closed_shell_f_system

        _, structure, const, params = _build_ch4(UNRESTRICTED=True)
        assert params["UNRESTRICTED"] is True
        assert (
            _require_closed_shell_f_system(structure, const, params, "inert-check")
            is None
        )
    elif case_id == "H0_and_S_vectorized_batch":
        _, _, const, _ = _build_ch4()
        # C (6) and H (1): both f-free, so the batched refusal must not fire.
        _assert_no_f_refusal(lambda: _synthetic_batch_call(const, [6, 1]), case_id)
    elif case_id == "_pair_grad_from_sk":
        _, _, const, _ = _build_ch4()
        _assert_no_f_refusal(lambda: _synthetic_pair_grad_call(const, 4, 1), case_id)
    elif case_id == "forces_spin":
        from dftorch._forces import _require_no_f_spin_forces

        _, structure, const, _ = _build_ch4()
        assert (
            _require_no_f_spin_forces(const, structure.TYPE, "inert-check") is None
        )
    elif case_id in ("get_h_spin", "get_h_spin_diag"):
        from dftorch._spin import _require_no_f_spin_shells

        # CH4 shells: C is s+p (types 1, 2), each H is s (type 1).
        f_free_shell_types = torch.tensor([1, 2, 1, 1, 1, 1])
        assert _require_no_f_spin_shells(f_free_shell_types, "inert-check") is None
        assert _require_no_f_spin_shells(None, "inert-check") is None
    else:  # pragma: no cover - guarded by test_guard_cases_are_exhaustive
        raise AssertionError(f"no inert check written for {case_id!r}")


# --------------------------------------------------------------------------
# 1. Completeness
# --------------------------------------------------------------------------


def test_inventory_covers_every_site():
    """Swept sites and inventory rows must match as multisets, both ways.

    Reporting only one direction is what lets an inventory rot: "the inventory
    is stale" is useless without saying which way it is stale, so both lists are
    printed in full on failure.
    """
    swept = Counter(site.key for site in sweep_orbital_count_sites(PACKAGE_ROOT))
    listed = Counter(row.key for row in parse_inventory_rows())

    missing = swept - listed
    extra = listed - swept

    def _fmt(counter):
        return "\n".join(
            f"    {path}:{line}  {snippet}" + (f"  (x{n})" if n > 1 else "")
            for (path, line, snippet), n in sorted(counter.items())
        )

    assert not missing and not extra, (
        "docs/ORBITAL-COUNT-INVENTORY.md is out of sync with the sweep.\n"
        f"In the code but NOT in the inventory ({sum(missing.values())}):\n"
        f"{_fmt(missing) or '    (none)'}\n"
        f"In the inventory but NOT in the code ({sum(extra.values())}):\n"
        f"{_fmt(extra) or '    (none)'}\n"
        "Regenerate with: uv run python tests/orbital_count_sweep.py"
    )


def test_inventory_row_count_matches_record_count():
    """One row per record, stated as its own assertion so the number is visible."""
    swept = sweep_orbital_count_sites(PACKAGE_ROOT)
    rows = parse_inventory_rows()
    assert len(rows) == len(swept), (
        f"inventory has {len(rows)} rows but the sweep returns {len(swept)} records"
    )
    assert len(swept) >= 70, (
        f"sweep returned only {len(swept)} records; the audit covered 157 at plan "
        "close, so a collapse to double digits means the patterns stopped matching"
    )


def test_sweep_finds_the_wrapped_pair_masks_in_h0ands():
    """At least 20 sites in _h0ands.py.

    A result near 12 is the signature of a line-oriented sweep that lost the
    flattening, which is how the plan-time research undercounted this file by
    roughly half.
    """
    count = sum(
        1
        for site in sweep_orbital_count_sites(PACKAGE_ROOT)
        if site.path == "dftorch/_h0ands.py"
    )
    assert count >= 20, (
        f"only {count} sites found in _h0ands.py; a result near 12 means the "
        "newline flattening is no longer being applied"
    )


def test_flattening_is_load_bearing():
    """A per-line sweep undercounts _h0ands.py; flattening restores the count.

    Measured on live code rather than on a constructed example, and stated
    exactly rather than approximately.  Using the pair-level form
    ``(a == x) & (b == y)`` -- the shape the plan-time research swept for --
    against the real ``_h0ands.py``:

    * matched **per physical line**, the way a ``grep`` pipeline works: 9 hits;
    * matched after ``flatten_source``: 25 hits.

    That is the undercount D-04's research fell into, and it lands in exactly
    the region the decision is most concerned with, because the single-system
    16-orbital masks are among the wrapped ones.

    An honest caveat, recorded here rather than left implied: this module's own
    patterns separate their parts with ``\\s*``, which already crosses newlines,
    so for *these* patterns matching the whole file is equivalent to flattening
    it (52 either way at comparison level).  The flattening is kept because it
    makes the guarantee structural instead of dependent on every future pattern
    author remembering to use ``\\s*`` rather than a literal space -- and this
    test measures the property that actually distinguishes them.
    """
    pattern = pattern_for("orbital-count")
    pair_pattern = re.compile(
        pattern.pattern + r"\s*\)\s*[&*]\s*\(\s*" + pattern.pattern
    )
    source = (PACKAGE_ROOT / "_h0ands.py").read_text(encoding="utf-8")

    def per_line(text):
        return sum(len(pair_pattern.findall(line)) for line in text.splitlines())

    raw = per_line(source)
    flattened = per_line(flatten_source(source))

    assert flattened >= 20, (
        f"the flattened pair-level sweep found {flattened} masks in _h0ands.py; "
        "a result near 12 or below means the flattening stopped working"
    )
    assert raw < flattened, (
        "a per-line sweep was supposed to undercount the wrapped pair masks in "
        f"_h0ands.py, but it found {raw} against the flattened {flattened}. If "
        "the file has been reformatted so nothing wraps any more, this test no "
        "longer proves anything and should be re-pointed at code that does."
    )


def test_prose_mentions_are_excluded_and_counted():
    """Comments and message text decide nothing and must not become rows."""
    assert count_prose_mentions(PACKAGE_ROOT) > 0, (
        "no prose mentions found at all, which means either the blanking is "
        "matching everything or the patterns stopped matching; both are bugs"
    )
    swept = {site.key for site in sweep_orbital_count_sites(PACKAGE_ROOT)}
    # ESDriver's guards write "n_orb == 16" into their own f-string messages.
    # Those lines must NOT be rows; the executable "counts == 16" lines must be.
    #
    # 61 and 96 are the two Phase 5 guards.  165 is Phase 6's
    # _krylov_params_for_f_interim, whose docstring also spells out "n_orb == 16"
    # and "16 orbitals" in prose -- so it is a third chance for the blanking to
    # go wrong, and its executable comparison being the only line listed is the
    # same property this test has always held.
    esdriver = {line for path, line, _ in swept if path == "dftorch/ESDriver.py"}
    assert esdriver == {61, 96, 165}, (
        f"ESDriver.py sites are {sorted(esdriver)}; expected the three executable "
        "comparisons at 61, 96 and 165. Lines 64 and 104 are f-string message "
        "text, and the prose inside _krylov_params_for_f_interim's docstring is "
        "comment text; all of them appear when FSTRING_*/COMMENT tokens are not "
        "blanked."
    )


# --------------------------------------------------------------------------
# 2. Every row is dispositioned
# --------------------------------------------------------------------------


def test_inventory_has_no_unresolved_sites():
    open_rows = [r for r in parse_inventory_rows() if r.disposition == "needs-action"]
    assert not open_rows, (
        "rows still marked needs-action:\n"
        + "\n".join(f"    {r.path}:{r.line}  {r.snippet}" for r in open_rows)
    )


def test_every_row_has_a_known_disposition_and_evidence():
    for row in parse_inventory_rows():
        assert row.disposition in DISPOSITIONS, (
            f"{row.path}:{row.line} carries disposition {row.disposition!r}, "
            f"which is not one of {DISPOSITIONS}"
        )
        assert row.family in FAMILIES, (
            f"{row.path}:{row.line} carries family {row.family!r}"
        )
        assert len(row.evidence) >= 40, (
            f"{row.path}:{row.line} has evidence of {len(row.evidence)} chars, "
            "which is too short to say anything: " + row.evidence
        )
        assert row.decides, f"{row.path}:{row.line} does not say what it decides"


def test_every_unreachable_row_states_how_it_was_established():
    """`unreachable` is never accepted on the strength of reading the code.

    The evidence must name the upstream guard (and where it fires) or the import
    search that found no importer.
    """
    markers = ("grep", "first statement", "refusal at", "guard at", "importer")
    for row in parse_inventory_rows():
        if row.disposition != "unreachable":
            continue
        assert any(m in row.evidence.lower() for m in markers), (
            f"{row.path}:{row.line} is marked unreachable but its evidence names "
            f"neither an upstream guard nor an import search: {row.evidence}"
        )


def test_every_guarded_row_names_an_exercised_case():
    """Each guarded row must point at a case this module actually drives."""
    expected = {
        f"test_every_guarded_site_raises_for_f[{case}]" for case in GUARD_CASES
    }
    for row in parse_inventory_rows():
        if row.disposition != "guarded":
            continue
        assert row.test in expected, (
            f"{row.path}:{row.line} is marked guarded but its Test column reads "
            f"{row.test!r}, which is not one of the exercised cases: "
            f"{sorted(expected)}"
        )


def test_every_guard_case_is_claimed_by_at_least_one_row():
    """The reverse direction: no guard case exists that no row cites.

    Without this, a case could be deleted from the inventory and this module
    would still pass, which is the same rot the completeness test prevents for
    sites.
    """
    cited = {row.test for row in parse_inventory_rows() if row.disposition == "guarded"}
    for case in GUARD_CASES:
        name = f"test_every_guarded_site_raises_for_f[{case}]"
        assert name in cited, (
            f"guard case {case!r} is exercised here but no guarded inventory row "
            "cites it"
        )


# --------------------------------------------------------------------------
# 3. Reachability, and inertness for f-free systems
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case_id", sorted(GUARD_CASES))
def test_every_guarded_site_raises_for_f(case_id, tmp_path):
    """Drive a real f system into each guarded site and observe the refusal."""

    def check():
        expected, call = GUARD_CASES[case_id](tmp_path)
        with pytest.raises(expected) as excinfo:
            call()
        message = str(excinfo.value)
        assert "16" in message or "f shell" in message or "f-shell" in message, (
            f"{case_id}: refusal message names neither the orbital count nor the "
            f"f shell: {message}"
        )

    run_with_float64(check)


@pytest.mark.parametrize("case_id", sorted(GUARD_CASES))
def test_guards_ignore_f_free_systems(case_id):
    """No guard may fire for CH4 + mio-1-1.

    This is the counterweight to the audit: adding refusals is only acceptable
    if no calculation that works today acquires a new failure mode.
    """
    run_with_float64(lambda: _inert_case(case_id))


def test_ch4_runs_end_to_end_after_the_guards(tmp_path):
    """The whole f-free path still runs, not just the guard functions.

    ``test_guards_ignore_f_free_systems`` isolates each guard; this proves the
    system they sit in front of still completes.
    """

    def check():
        driver, structure, const, _ = _build_ch4()
        driver(structure, const, do_scf=False)
        assert bool(torch.isfinite(structure.H0).all())
        assert bool(torch.isfinite(structure.S).all())
        overlap = structure.S.detach()
        overlap = 0.5 * (overlap + overlap.T)
        assert float(torch.linalg.eigvalsh(overlap).min()) > 0.0

    run_with_float64(check)


def test_refusal_messages_leak_no_path(tmp_path):
    """No refusal may interpolate a filesystem path (Phase 4 threat T-04-07).

    A bare "no forward slash" assertion would be wrong here: the shared message
    constants legitimately contain ``dH0/dS`` and ``H0/S``.  What must not
    appear is the SKF directory, the geometry file, the temporary directory, or
    any file extension or drive-letter form.
    """

    def check():
        forbidden_fragments = [
            str(TESTS_DIR / "f_orbital_data"),
            str(TESTS_DIR / "data_skf_mio-1-1"),
            str(tmp_path),
            ".skf",
            ".xyz",
            "\\",
            "://",
        ]
        for case_id in sorted(GUARD_CASES):
            expected, call = GUARD_CASES[case_id](tmp_path)
            with pytest.raises(expected) as excinfo:
                call()
            message = str(excinfo.value)
            for fragment in forbidden_fragments:
                assert fragment not in message, (
                    f"{case_id}: refusal message contains {fragment!r}, which is a "
                    f"path form. Message: {message}"
                )
            assert not re.search(r"[A-Za-z]:[\\/]", message), (
                f"{case_id}: refusal message contains a drive-letter path. "
                f"Message: {message}"
            )

    run_with_float64(check)


def test_no_new_f_exception_class_was_defined():
    """D-04 requires reusing the established taxonomy, not extending it.

    The four classes are the whole f support policy and are meant to be
    reviewable as one list.  A fifth added quietly would split it.
    """

    def check():
        import inspect

        from dftorch import _slater_koster_pair as skp

        defined = sorted(
            name
            for name, obj in vars(skp).items()
            if inspect.isclass(obj)
            and name.startswith("F")
            and name.endswith("Error")
            and obj.__module__ == skp.__name__
        )
        assert defined == [
            "FAngularFormulaSourceError",
            "FDerivativeUnsupportedError",
            "FShellResolvedCoulombUnsupportedError",
            "FSpinPolarizationUnsupportedError",
        ], f"the f exception taxonomy changed: {defined}"

    run_with_float64(check)


# --------------------------------------------------------------------------
# The `extended` and `unreachable` rows' own gates
# --------------------------------------------------------------------------


def test_f_pair_reaches_single_system_sk_assembly(tmp_path):
    """The `extended` rows in _h0ands.py: an f pair really is assembled.

    Before the 16-orbital pair masks existed, an f-containing neighbour pair
    matched none of the nine 1/4/9 masks and was omitted from H0/S with no
    signal.  The off-diagonal block coupling Eu's f AOs (local 9-15) to N's AOs
    must therefore be non-zero -- a zero block is exactly what the silent drop
    produced, and it is indistinguishable from a correct result by shape,
    finiteness or symmetry.
    """

    def check():
        driver, structure, const, _ = _build_eu_n(tmp_path)
        driver(structure, const, do_scf=False)
        H0 = structure.H0.detach()
        S = structure.S.detach()
        assert H0.shape == (20, 20), f"unexpected AO dimension {tuple(H0.shape)}"
        f_block_H0 = H0[9:16, 16:20].abs().max()
        f_block_S = S[9:16, 16:20].abs().max()
        assert float(f_block_H0) > 1e-6, (
            "the Eu f / N block of H0 is zero, which is the signature of f pairs "
            "being dropped by the 1/4/9 masks rather than routed"
        )
        assert float(f_block_S) > 1e-8, (
            "the Eu f / N block of S is zero; f overlap did not reach assembly"
        )

    run_with_float64(check)


def test_structure_layout_tables_span_f():
    """The `extended` rows in Structure.py: the layout tables include f."""

    def check():
        from dftorch.Structure import (
            AO_LABEL_TEMPLATE,
            SHELL_DIMS,
            SHELL_LOCAL_STARTS,
            SHELL_TYPE_IDS,
        )

        assert SHELL_DIMS == (1, 3, 5, 7)
        assert SHELL_LOCAL_STARTS == (0, 1, 4, 9)
        assert SHELL_TYPE_IDS == (1, 2, 3, 4)
        assert sum(SHELL_DIMS) == 16
        assert len(AO_LABEL_TEMPLATE) == 16
        assert SHELL_LOCAL_STARTS[-1] + SHELL_DIMS[-1] == 16

    run_with_float64(check)


def test_canonical_shell_dim_table_spans_f():
    """The `extended` row in Constants.py, and the reason the copies are wrong.

    ``Constants.shell_dim`` is the correct five-entry table.  ``_spin`` and
    ``_forces`` each declare a truncated four-entry copy of it; those are the
    rows this plan had to guard, and the contrast is the finding.
    """

    def check():
        _, _, const, _ = _build_ch4()
        assert const.shell_dim.tolist() == [0, 1, 3, 5, 7]

    run_with_float64(check)


def test_legacy_h0ands_has_no_importer():
    """The `unreachable` rows in _legacy/H0andS.py, established not asserted.

    Only ``src/`` is scanned, and only import statements are matched: this
    module and ``orbital_count_sweep`` both *name* ``_legacy`` in prose, and a
    bare substring search would flag the audit that established the property.
    """
    import_pattern = re.compile(
        r"^\s*(?:from\s+[\w.]*_legacy[\w.]*\s+import|import\s+[\w.]*_legacy[\w.]*|"
        r"from\s+[\w.]*\bH0andS\b\s+import|import\s+[\w.]*\bH0andS\b)",
        re.MULTILINE,
    )
    offenders = []
    for path in sorted(REPO_ROOT.joinpath("src").rglob("*.py")):
        if "_legacy" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if import_pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)).replace(os.sep, "/"))
    assert not offenders, (
        "src/dftorch/_legacy/ is dispositioned unreachable because nothing "
        f"imports it, but these files now reference it: {offenders}"
    )
    assert not (PACKAGE_ROOT / "_legacy" / "__init__.py").exists(), (
        "src/dftorch/_legacy/ gained an __init__.py, so it is now an importable "
        "package and its 54 unreachable rows need re-dispositioning"
    )


def test_atomic_density_matrix_has_no_production_importer():
    """The `unreachable` row in _atomic_density_matrix.py.

    ``Structure._atomic_density_matrix_from_shells`` is a different symbol and
    does not match the word-boundary pattern used here.
    """
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "_atomic_density_matrix.py" or "__pycache__" in path.parts:
            continue
        if "_legacy" in path.parts:
            continue  # itself unreachable; see test_legacy_h0ands_has_no_importer
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*(from\s+\.?_atomic_density_matrix\s+import|"
                     r"import\s+\S*\b_atomic_density_matrix\b)", text, re.MULTILINE):
            offenders.append(path.name)
    assert not offenders, (
        "_atomic_density_matrix is dispositioned unreachable because only "
        f"_legacy imports it, but these production modules now do: {offenders}"
    )


# --------------------------------------------------------------------------
# The site handed over by plan 05-05, classified out of D-04's row set
# --------------------------------------------------------------------------


def test_batched_kspace_site_has_a_disposition():
    """`do_vec=True` must refuse instead of printing an apology and returning.

    The old branch printed "vectorized k-space is not implemented for batched
    data" and then returned ``None`` from a function annotated
    ``-> tuple[torch.Tensor, torch.Tensor]``, so its only caller failed with
    ``cannot unpack non-iterable NoneType object`` one frame later, the
    explanation already gone.
    """

    def check():
        from dftorch._coulomb_matrix_batch import ewald_k_space_vectorized_eager

        lattice = torch.eye(3).unsqueeze(0) * 20.0
        with pytest.raises(NotImplementedError) as excinfo:
            ewald_k_space_vectorized_eager(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                lattice,
                torch.zeros(2),
                2,
                1.0e-6,
                0.5,
                False,
                do_vec=True,
            )
        message = str(excinfo.value)
        assert "do_vec" in message, message
        assert "PHY-04" in message, message

    run_with_float64(check)


def test_inventory_records_the_batched_kspace_determination():
    """The determination, not just the fix, has to be written down.

    The plan required deciding whether the site was in D-04's scope and
    recording the reasoning; a passing behaviour test alone would leave the
    next reader to re-derive it.
    """
    text = INVENTORY_PATH.read_text(encoding="utf-8")
    assert "Out of D-04's row set, resolved anyway" in text
    assert "not an orbital-count site" in text
    assert "PHY-04" in text
    assert "do_vec" in text


def test_sedacs_result_is_stated():
    """`sedacs/` must be reported as checked, whether or not it has sites."""
    text = INVENTORY_PATH.read_text(encoding="utf-8")
    assert "sedacs" in text.lower()
    sedacs_sites = [
        site
        for site in sweep_orbital_count_sites(PACKAGE_ROOT)
        if site.path.startswith("dftorch/sedacs/")
    ]
    if sedacs_sites:
        rows = {row.key for row in parse_inventory_rows()}
        for site in sedacs_sites:
            assert site.key in rows, f"unlisted sedacs site {site.key}"
    else:
        assert "checked, no sites" in text, (
            "the sweep finds no sites under sedacs/, and the inventory must say "
            "so explicitly rather than leaving the absence unstated"
        )
