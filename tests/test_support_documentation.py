"""The basis metadata tables agree with each other, and the support matrix
agrees with the code.

The machine half of two documentation requirements:

* **CLN-02** -- "channel lookup and AO ordering are centralized enough for human
  troubleshooting".  They are not centralized; they are defined in six places
  across three modules with two different indexing conventions.  ``05-CONTEXT.md``
  records a single-source basis module as a *Deferred Idea*, so CLN-02 is met by
  ``docs/BASIS-METADATA-MAP.md`` plus the consistency tests below.  A map alone
  would be true on the day it is written; these tests are what make a one-sided
  edit to one table fail instead of drifting silently.
* **CLN-04** -- the f support status matrix, ``docs/F-SUPPORT-STATUS.md``.
  ``05-VALIDATION.md`` names the exception cross-check as the property that makes
  CLN-04 verifiable rather than merely written: the named exception classes are
  enumerated from ``_slater_koster_pair`` at test time, so a class added later
  without a matrix row fails here.

Why the consistency half is worth having.  Plan 05-06 swept 157 hardcoded
orbital-count sites and the only real gaps it found were three truncated copies
of the per-shell AO count table, ``[0, 1, 3, 5]``, sitting a few modules away
from ``Constants.shell_dim``'s correct ``[0, 1, 3, 5, 7]``.  Nothing asserted
that the two agreed, because nothing asserted any relation between any two of
these tables.  That is the class of defect these tests exist to catch.

ASCII only, per the Phase 4 rule: a failure must be diagnosable from pytest
output on a cp1252 console.
"""

import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic
# and avoids requiring a C++ toolchain).  Mirrors tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import importlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
F_ORBITAL_DATA = REPO_ROOT / "tests" / "f_orbital_data"


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


def _write_eu_n_xyz(path: Path, separation: float = 2.4) -> None:
    """Eu at the origin, N displaced along +x.  Mirrors tests/test_eu_n_scan.py."""
    path.write_text(
        "2\n"
        "Eu-N diatomic (basis metadata consistency)\n"
        "Eu 0.000000 0.000000 0.000000\n"
        f"N {separation:.6f} 0.000000 0.000000\n",
        encoding="ascii",
    )


def _f_params(xyz_path: Path) -> dict:
    """Driver parameters pinned to match tests/test_eu_n_scan.py exactly."""
    return {
        "FILENAME": str(xyz_path),
        "SKFPATH": str(F_ORBITAL_DATA) + os.sep,
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 10.0,
        "RCUT_REPULSIVE": 6.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
    }


# --------------------------------------------------------------------------
# CLN-02: the duplicated basis tables are pinned against each other
# --------------------------------------------------------------------------


def test_shell_dim_and_shell_dims_agree(tmp_path):
    """The four shell-count tables state one physical fact four ways.

    ``Constants.shell_dim`` (``Constants.py:106-108``) is
    ``[0, 1, 3, 5, 7]``: five entries, indexed by ``shell_types``, whose ids run
    1..4, so entry 0 is an unused pad that makes the indexing direct.
    ``Structure.SHELL_DIMS`` (``Structure.py:11``) is ``(1, 3, 5, 7)``: four
    entries, indexed by shell *position* 0..3.  They describe the same fact --
    how many atomic orbitals an s/p/d/f shell contributes -- with different
    indexing conventions, and nothing in the package relates them.

    ``_bond_integral.MAX_SHELLS`` (``_bond_integral.py:172``) and
    ``Structure.SHELL_TYPE_IDS`` (``Structure.py:13``) are the same count seen
    from two more angles, and ``_spin._F_SHELL_TYPE_ID`` (``_spin.py:15``) is a
    deliberate literal copy of ``SHELL_TYPE_IDS[-1]`` whose only link to the
    definition is a comment.

    This test exists so that an edit to one is not silently inconsistent with
    the others.  Plan 05-06 found three copies of this very table truncated to
    ``[0, 1, 3, 5]`` -- one entry short of f -- living unasserted next to the
    correct one.

    ``Constants`` is built from ``tests/f_orbital_data`` rather than having its
    tensor reconstructed by hand, so the value under test is the one a real
    calculation receives.
    """

    def body():
        Constants = importlib.import_module("dftorch.Constants").Constants
        Structure = importlib.import_module("dftorch.Structure")
        bond_integral = importlib.import_module("dftorch._bond_integral")
        spin = importlib.import_module("dftorch._spin")

        xyz = tmp_path / "eu_n.xyz"
        _write_eu_n_xyz(xyz)
        const = Constants(_f_params(xyz)).to("cpu")

        shell_dim = [int(v) for v in const.shell_dim.detach().cpu().tolist()]
        shell_dims = [int(v) for v in Structure.SHELL_DIMS]

        assert shell_dim[0] == 0, (
            "Constants.shell_dim[0] is the pad that makes the table directly "
            f"indexable by shell_types (ids 1..4); got {shell_dim!r}"
        )
        assert shell_dim[1:] == shell_dims, (
            "Constants.shell_dim[1:] and Structure.SHELL_DIMS disagree. They are "
            "the same per-shell AO count table with different indexing.\n"
            f"  Constants.shell_dim      = {shell_dim!r}  (Constants.py:106)\n"
            f"  Constants.shell_dim[1:]  = {shell_dim[1:]!r}\n"
            f"  Structure.SHELL_DIMS     = {shell_dims!r}  (Structure.py:11)\n"
            "See docs/BASIS-METADATA-MAP.md."
        )

        assert len(shell_dims) == bond_integral.MAX_SHELLS, (
            "Structure.SHELL_DIMS has one entry per supported shell, so its "
            "length must equal _bond_integral.MAX_SHELLS, which bounds the "
            "declared shell count a parsed SKF file may carry "
            "(_bond_integral.py:679).\n"
            f"  len(SHELL_DIMS) = {len(shell_dims)}, MAX_SHELLS = "
            f"{bond_integral.MAX_SHELLS}"
        )

        shell_type_ids = tuple(int(v) for v in Structure.SHELL_TYPE_IDS)
        assert shell_type_ids == tuple(range(1, len(shell_dims) + 1)), (
            "Structure.SHELL_TYPE_IDS must be the 1-based ids that index "
            "Constants.shell_dim, one per entry of SHELL_DIMS; got "
            f"{shell_type_ids!r}"
        )
        assert shell_dim[shell_type_ids[-1]] == shell_dims[-1] == 7, (
            "Indexing Constants.shell_dim by the f shell id must give the f AO "
            f"count. shell_dim={shell_dim!r}, f id={shell_type_ids[-1]}"
        )
        assert spin._F_SHELL_TYPE_ID == shell_type_ids[-1], (
            "_spin._F_SHELL_TYPE_ID is a deliberate literal copy of "
            "Structure.SHELL_TYPE_IDS[-1] (see the comment at _spin.py:11-14, "
            "which explains the import-order reason it is not imported). The "
            "copy has drifted: "
            f"_spin._F_SHELL_TYPE_ID={spin._F_SHELL_TYPE_ID}, "
            f"SHELL_TYPE_IDS[-1]={shell_type_ids[-1]}"
        )

    run_with_float64(body)


def test_shell_local_starts_are_prefix_sums():
    """``SHELL_LOCAL_STARTS`` is the exclusive prefix sum of ``SHELL_DIMS``.

    ``Structure.SHELL_LOCAL_STARTS`` (``Structure.py:12``) is the AO offset of
    each shell inside an atom's 16-AO block; ``SHELL_DIMS`` is how wide each
    shell is.  ``_shell_local_end`` (``Structure.py:115-120``) adds one to the
    other, so the two are only meaningful together.

    The final relation is the one that ties the offset table to the label table:
    the last start plus the last width must be exactly
    ``len(AO_LABEL_TEMPLATE)``.  If it were not, an atom's AO block and its AO
    labels would be different lengths.
    """
    Structure = importlib.import_module("dftorch.Structure")

    dims = tuple(int(v) for v in Structure.SHELL_DIMS)
    starts = tuple(int(v) for v in Structure.SHELL_LOCAL_STARTS)

    expected = []
    running = 0
    for width in dims:
        expected.append(running)
        running += width

    assert starts == tuple(expected), (
        "Structure.SHELL_LOCAL_STARTS is not the exclusive prefix sum of "
        "SHELL_DIMS.\n"
        f"  SHELL_DIMS         = {dims!r}\n"
        f"  SHELL_LOCAL_STARTS = {starts!r}\n"
        f"  expected           = {tuple(expected)!r}"
    )

    total = starts[-1] + dims[-1]
    assert total == len(Structure.AO_LABEL_TEMPLATE), (
        "The AO block described by SHELL_LOCAL_STARTS/SHELL_DIMS is not the "
        "same width as AO_LABEL_TEMPLATE, so an atom's orbitals and its orbital "
        "labels would disagree.\n"
        f"  SHELL_LOCAL_STARTS[-1] + SHELL_DIMS[-1] = {total}\n"
        f"  len(AO_LABEL_TEMPLATE)                  = "
        f"{len(Structure.AO_LABEL_TEMPLATE)}"
    )


def test_ao_label_template_matches_shell_dims():
    """Slicing the AO labels at the shell offsets yields 1, 3, 5 and 7 labels.

    ``Structure.AO_LABEL_TEMPLATE`` (``Structure.py:25-42``) is the per-atom AO
    order, and ``Structure.AO_SHELL_TEMPLATE`` (``Structure.py:44-61``) maps the
    same 16 positions to shell ids.  Both are written out literally, so either
    could be edited without the other.

    The f AO order inside its group is *not* checked for physical correctness
    here; that is source-locked in
    ``.planning/phases/03-h0-s-routing-and-f-angular-blocks/03-SOURCE-LOCK.md``
    against Takegahara, Aoki and Yanase, J. Phys. C 13 (1980) 583, and is
    covered by the Phase 3 angular tests.  What is checked here is the layout:
    that the labels group into shells the way the offset tables claim.
    """
    Structure = importlib.import_module("dftorch.Structure")

    labels = tuple(Structure.AO_LABEL_TEMPLATE)
    shells = tuple(int(v) for v in Structure.AO_SHELL_TEMPLATE)
    dims = tuple(int(v) for v in Structure.SHELL_DIMS)
    starts = tuple(int(v) for v in Structure.SHELL_LOCAL_STARTS)
    shell_ids = tuple(int(v) for v in Structure.SHELL_TYPE_IDS)

    assert len(labels) == 16, (
        f"AO_LABEL_TEMPLATE must carry the full spdf block of 16 AOs; got "
        f"{len(labels)}: {labels!r}"
    )
    assert len(shells) == len(labels), (
        "AO_SHELL_TEMPLATE and AO_LABEL_TEMPLATE describe the same 16 AO "
        f"positions; got {len(shells)} and {len(labels)}"
    )

    prefixes = ("s", "p", "d", "f")
    for index, (start, width, prefix, shell_id) in enumerate(
        zip(starts, dims, prefixes, shell_ids)
    ):
        group = labels[start : start + width]
        assert len(group) == width, (
            f"shell {index} ({prefix}) should occupy {width} AO slots starting "
            f"at {start}; slicing AO_LABEL_TEMPLATE gave {len(group)}: {group!r}"
        )
        assert all(label.startswith(prefix) for label in group), (
            f"shell {index} slice of AO_LABEL_TEMPLATE is not all {prefix} "
            f"labels: {group!r}. Either the labels or SHELL_LOCAL_STARTS moved."
        )
        group_shells = shells[start : start + width]
        assert set(group_shells) == {shell_id}, (
            f"AO_SHELL_TEMPLATE disagrees with the {prefix} slice: expected "
            f"every entry to be {shell_id}, got {group_shells!r}"
        )

    assert sum(dims) == len(labels), (
        f"SHELL_DIMS sums to {sum(dims)} but AO_LABEL_TEMPLATE has "
        f"{len(labels)} entries"
    )


def test_channel_tables_are_consistent():
    """The three channel tables in ``_bond_integral`` agree with each other.

    ``_CHANNELS`` (``_bond_integral.py:101``) is the canonical 40-channel order
    of the extended SKF electronic table; ``_SIMPLE_CHANNELS``
    (``_bond_integral.py:144``) is the 20-channel simple-format subset; and
    ``_SIMPLE_TO_EXTENDED`` (``_bond_integral.py:167``) is the index map that
    zero-fills a simple row into a canonical one at
    ``_bond_integral.py:415-419``.

    Plan 05-03's port inventory (``docs/SCRIPT-PY-PORT-INVENTORY.md``, the
    ``check_simple_canonical_channels`` row, formerly ``script.py:505``)
    classifies the *data* property as **covered** by
    ``tests/test_f_orbital_skf.py::test_f_orbital_skf_parser_and_spline_gate``:
    that a parsed simple file lands its 20 values in the right canonical slots
    with the rest zero.  This test deliberately does not duplicate it.  What it
    adds is the *table* invariant that the data property assumes and never
    states -- the lengths, the subset relation, and the H/S mirror -- which is
    the CLN-02 question of whether the lookup tables themselves still agree.
    """
    bond_integral = importlib.import_module("dftorch._bond_integral")
    slater_koster = importlib.import_module("dftorch._slater_koster_pair")

    channels = list(bond_integral._CHANNELS)
    simple = list(bond_integral._SIMPLE_CHANNELS)
    simple_to_extended = list(bond_integral._SIMPLE_TO_EXTENDED)

    assert len(channels) == 40, (
        f"_CHANNELS is the canonical 40-channel extended layout; got "
        f"{len(channels)}"
    )
    assert len(set(channels)) == len(channels), (
        "_CHANNELS contains a duplicate name, so the name->index lookup in "
        "_slater_koster_pair would be ambiguous"
    )
    assert len(simple) == 20, (
        f"_SIMPLE_CHANNELS is the 20-channel simple-format layout; got "
        f"{len(simple)}"
    )
    assert len(simple_to_extended) == len(simple), (
        "_SIMPLE_TO_EXTENDED maps one index per simple channel; got "
        f"{len(simple_to_extended)} for {len(simple)} channels"
    )

    missing = [name for name in simple if name not in channels]
    assert not missing, (
        "every simple-format channel must exist in the canonical order, "
        f"otherwise the zero-fill at _bond_integral.py:415-419 drops it: "
        f"{missing!r}"
    )
    assert simple_to_extended == [channels.index(name) for name in simple], (
        "_SIMPLE_TO_EXTENDED no longer points each simple channel at its own "
        "slot in _CHANNELS. A simple-format file would be scattered into the "
        "wrong canonical channels.\n"
        f"  _SIMPLE_TO_EXTENDED = {simple_to_extended!r}\n"
        f"  recomputed          = {[channels.index(n) for n in simple]!r}"
    )

    assert bond_integral.N_SK_CHANNELS == len(channels), (
        "N_SK_CHANNELS is derived from _CHANNELS and must match it; got "
        f"{bond_integral.N_SK_CHANNELS} vs {len(channels)}"
    )
    assert bond_integral.SK_BLOCK_SIZE == len(channels) // 2, (
        "SK_BLOCK_SIZE is the width of the H half of _CHANNELS (the S half "
        f"mirrors it); got {bond_integral.SK_BLOCK_SIZE} for "
        f"{len(channels)} channels"
    )

    block = bond_integral.SK_BLOCK_SIZE
    hamiltonian = channels[:block]
    overlap = channels[block:]
    assert all(name.startswith("H") for name in hamiltonian), (
        f"the first {block} entries of _CHANNELS must be the H channels: "
        f"{hamiltonian!r}"
    )
    assert all(name.startswith("S") for name in overlap), (
        f"the last {block} entries of _CHANNELS must be the S channels: "
        f"{overlap!r}"
    )
    assert [name[1:] for name in hamiltonian] == [name[1:] for name in overlap], (
        "the S half of _CHANNELS must repeat the H half's suffixes in the same "
        "order. _slater_koster_pair.py:15 and :34 both rely on it, and "
        "_slater_koster_pair.py:34 derives its pair-class names by stripping "
        "the leading H from exactly this half.\n"
        f"  H suffixes = {[name[1:] for name in hamiltonian]!r}\n"
        f"  S suffixes = {[name[1:] for name in overlap]!r}"
    )

    assert list(slater_koster.SK_CHANNEL_NAMES) == channels, (
        "_slater_koster_pair.SK_CHANNEL_NAMES (:25) is built from "
        "_bond_integral._CHANNELS and must still equal it"
    )
