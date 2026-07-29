"""Shell-resolved Hubbard / charge data for the f-containing Eu-N diatomic.

Phase 4 decision **D-14** calls this deliberate de-risking: extend *and validate*
the f dimension in the charge / Hubbard / Coulomb data structures now, even
though D-11's single-shot energy path will not consume them this phase, so the
later self-consistent SCF work does not discover the gaps from scratch.  These
are validated-but-unconsumed structures by design, not dead code.

Decision **D-16 as amended by D-23** fixes the config gate at the *existing*
``MAGNETIC_HUBBARD_LDEP`` key rather than a new ``SHELL_RESOLVED`` flag: its
docstring at ``Constants.py:39`` already reads "Use shell-dependent
(l-dependent) Hubbard U parameters", which is exactly D-16's intent.  Every test
in this module that inspects construction therefore runs under **both** flag
settings.

Covers SIM-03.
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

    Copied from ``tests/test_f_orbital_skf.py`` — the established harness for
    every f-orbital test module in this project.  Copied rather than imported
    across test modules, matching the existing convention.
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


def write_xyz(
    path: Path, elements: list[str], spacing: float = 1.5, axis: str = "x"
) -> None:
    """Write atoms evenly spaced along ``axis``.

    Copied verbatim from ``tests/test_f_orbital_skf.py``.  With the default
    ``axis="x"`` every ordered pair (I, J) with J after I has direction cosines
    (L, M, N) = (1, 0, 0), which is what the metadata regressions rely on.
    """
    offsets = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
    lines = [str(len(elements)), "metadata regression"]
    for idx, sym in enumerate(elements):
        step = spacing * idx
        lines.append(
            f"{sym} {step * offsets[0]:.8f} {step * offsets[1]:.8f} "
            f"{step * offsets[2]:.8f}"
        )
    path.write_text("\n".join(lines) + "\n")


# --- Eu-N reference case (D-19, D-24) ---------------------------------------
#
# Isolated Eu-N diatomic at the D-24 target separation of 2.655 A, using the
# nine-file f-orbital SKF fixture set in tests/f_orbital_data/.  Same parameter
# dict as tests/test_single_shot_energy.py so the two modules describe the same
# physical system.
EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}

#: ``tests/f_orbital_data/Eu-Eu.skf`` line 3 carries ``Uf = 0.50`` Hartree in
#: the extended-format on-site block (field order Ef Ed Ep Es SPE Uf Ud Up Us
#: ff fd fp fs).  ``_bond_integral.py`` multiplies it by
#: ``EV_PER_HARTREE = 27.21138625`` on the way into ``const.Uf``.  D-15 requires
#: the f Hubbard U to be SKF-sourced rather than defaulted, and this literal is
#: what proves it.
EU_F_HUBBARD_U_HARTREE = 0.50
EV_PER_HARTREE = 27.21138625
EU_F_HUBBARD_U_EV = 13.605693125  # == 0.50 * 27.21138625

#: Both flag settings.  D-16/D-23 require both paths to be exercised.
LDEP_SETTINGS = (False, True)


def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"


def _build_eu_n(tmp_path: Path, magnetic_hubbard_ldep: bool):
    """Build ``(const, structure, eu, n)`` for the Eu-N diatomic.

    ``eu`` and ``n`` are element indices resolved through
    ``const.symbol_to_number`` rather than hardcoded 63 and 7, so the assertions
    stay readable and survive an element-table change.

    Constants and Structure are constructed directly (as
    ``tests/test_single_shot_energy.py`` does) rather than through
    ``test_f_orbital_skf.build_structure``, because that helper pins its own
    parameter dict and cannot carry ``MAGNETIC_HUBBARD_LDEP``.
    """
    from dftorch.Constants import Constants
    from dftorch.Structure import Structure

    xyz_path = tmp_path / "eu_n.xyz"
    write_xyz(xyz_path, ["Eu", "N"], spacing=EU_N_SEPARATION, axis="x")

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_skf_dir()) + os.sep
    params["MAGNETIC_HUBBARD_LDEP"] = magnetic_hubbard_ldep

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    return const, structure, const.symbol_to_number["Eu"], const.symbol_to_number["N"]


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_shell_counts(tmp_path, ldep):
    """Eu carries all four shells; N carries s and p only.

    ``const.n_orb[N]`` is 4, so ``HDIM`` is 20 rather than 25.  N has **two**
    shells, not three.
    """

    def check():
        const, structure, eu, n = _build_eu_n(tmp_path, ldep)

        # Structure.py:438 — n_shells_per_atom is shell_present.sum(dim=1).
        assert structure.n_shells_per_atom.tolist() == [4, 2]
        assert const.shell_present[eu].tolist() == [True, True, True, True]
        assert const.shell_present[n].tolist() == [True, True, False, False]

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_hubbard_u_sr_length_matches_active_shells(tmp_path, ldep):
    """One shell-resolved Hubbard entry per *present* shell, f included."""

    def check():
        _, structure, _, _ = _build_eu_n(tmp_path, ldep)

        # Structure.py:435 — Hubbard_U_sr = template_U[shell_present].
        assert structure.Hubbard_U_sr.shape[0] == 6
        assert (
            structure.Hubbard_U_sr.shape[0]
            == int(structure.n_shells_per_atom.sum())
        )

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_hubbard_u_sr_values(tmp_path, ldep):
    """Entry-by-entry: Eu s/p/d/f then N s/p, with the f entry real.

    The f entry (index 3) is the one this phase is about: it is
    ``const.Uf[Eu]``, i.e. 13.605693125 eV, not a zero fill.
    """

    def check():
        const, structure, eu, n = _build_eu_n(tmp_path, ldep)

        # Structure.py:413-422 stacks (Us, Up, Ud, Uf) per atom and then masks
        # by shell_present, so the flat order is Eu s,p,d,f then N s,p.
        expected = torch.tensor(
            [
                const.U[eu],
                const.Up[eu],
                const.Ud[eu],
                const.Uf[eu],
                const.U[n],
                const.Up[n],
            ],
            dtype=structure.Hubbard_U_sr.dtype,
        )
        assert torch.allclose(structure.Hubbard_U_sr, expected, atol=1e-9)
        assert abs(structure.Hubbard_U_sr[3].item() - EU_F_HUBBARD_U_EV) < 1e-6

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_f_hubbard_u_traces_to_skf(tmp_path, ldep):
    """D-15: the f Hubbard U comes from the SKF header, not from a default.

    ``tests/f_orbital_data/Eu-Eu.skf`` line 3 carries ``Uf = 0.50`` Hartree.
    A failure here means the extended-format on-site parse stopped reaching the
    f field — D-15 requires that to surface as a blocker, never as a zero fill.
    """

    def check():
        const, structure, eu, _ = _build_eu_n(tmp_path, ldep)

        assert (
            abs(
                structure.Hubbard_U_sr[3].item()
                - EU_F_HUBBARD_U_HARTREE * EV_PER_HARTREE
            )
            < 1e-6
        )
        assert abs(const.Uf[eu].item() - EU_F_HUBBARD_U_EV) < 1e-6
        assert const.Uf[eu].item() != 0.0

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_shell_types_label_the_f_shell(tmp_path, ldep):
    """``shell_types`` labels s/p/d/f as 1/2/3/4; the Eu f shell is index 3."""

    def check():
        _, structure, _, _ = _build_eu_n(tmp_path, ldep)

        # Structure.py:423-431 builds (1, 2, 3, 4) per atom, then masks.
        assert structure.shell_types.tolist() == [1, 2, 3, 4, 1, 2]
        assert int(structure.shell_types[3]) == 4  # the Eu f shell

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_electrons_per_shell(tmp_path, ldep):
    """Eu's f shell carries 7 reference electrons — Eu 4f7.

    The values come from the SKF on-site occupations: ``Eu-Eu.skf`` line 3 ends
    ``7.0 0.0 0.0 2.0`` (ff fd fp fs), and ``N-N.skf`` gives fp = 3.0, fs = 2.0.
    """

    def check():
        _, structure, _, _ = _build_eu_n(tmp_path, ldep)

        # Structure.py:432/437 — stack (n_s, n_p, n_d, n_f), then mask.
        assert structure.el_per_shell.tolist() == [2.0, 0.0, 0.0, 7.0, 2.0, 3.0]
        assert float(structure.el_per_shell[3]) == 7.0

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_shell_index_ranges(tmp_path, ldep):
    """The per-atom shell index ranges span exactly ``n_shells_per_atom``."""

    def check():
        _, structure, _, _ = _build_eu_n(tmp_path, ldep)

        # Structure.py:439-441 — cumulative start, inclusive end.
        assert structure.H_INDEX_START_U.tolist() == [0, 4]
        assert structure.H_INDEX_END_U.tolist() == [3, 5]
        spans = structure.H_INDEX_END_U - structure.H_INDEX_START_U + 1
        assert torch.equal(spans, structure.n_shells_per_atom)

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_eu_n_reference_density_includes_f_shell(tmp_path, ldep):
    """SIM-03: the reference charge data carries the seven Eu f AOs.

    ``D0`` is the flat diagonal of the atomic reference density.  Indices 9-15
    are the Eu f AOs, each carrying 0.5 — Eu's 7 reference f electrons, halved
    by ``Structure.py:451``.  This is the concrete evidence that reference
    charge data already includes the f shell.
    """

    def check():
        _, structure, _, _ = _build_eu_n(tmp_path, ldep)

        assert structure.D0.ndim == 1
        assert structure.D0.shape == (20,)  # Eu 16 AOs + N 4 AOs
        assert float(structure.D0[0]) == 1.0  # Eu s: 2 electrons, halved
        assert torch.all(structure.D0[1:9] == 0.0)  # Eu p and d are empty
        assert torch.all(structure.D0[9:16] == 0.5)  # the seven Eu f AOs
        assert float(structure.D0[16]) == 1.0  # N s
        assert torch.all(structure.D0[17:20] == 0.5)  # N p
        assert float(structure.D0.sum()) == 7.0

        # Structure.py:410 — the AO-level shell labels agree: 4 == f.
        assert structure.ao_shell_types[9:16].tolist() == [4] * 7

    run_with_float64(check)


@pytest.mark.parametrize("ldep", LDEP_SETTINGS)
def test_n_has_no_f_hubbard_entry(tmp_path, ldep):
    """N contributes exactly two shell entries and has no f Hubbard U.

    ``const.Uf[N]`` is 0.0 because N genuinely has no f shell — and because
    ``shell_present[N][3]`` is False, that zero never enters ``Hubbard_U_sr``.
    A zero *inside* ``Hubbard_U_sr`` would be the D-15 failure mode; a zero in
    ``const.Uf`` for a shell-less element is not.
    """

    def check():
        const, structure, _, n = _build_eu_n(tmp_path, ldep)

        assert float(const.Uf[n]) == 0.0
        assert structure.Hubbard_U_sr[4:].shape[0] == 2
        assert not bool((structure.Hubbard_U_sr == 0.0).any())

    run_with_float64(check)


def test_ldep_flag_does_not_change_construction(tmp_path):
    """``MAGNETIC_HUBBARD_LDEP`` selects *consumption*, never construction.

    ``Hubbard_U_sr``, ``shell_types``, ``el_per_shell`` and
    ``n_shells_per_atom`` are all built unconditionally at
    ``Structure.py:413-441`` — nothing there reads the flag.  This is the
    non-obvious invariant the Coulomb-path gating depends on: the flag can be
    read at the point of *use* without any risk that the two branches were
    handed differently-shaped data.  Deliberately not parametrised, because it
    is the cross-flag comparison itself.
    """

    def check():
        _, off, _, _ = _build_eu_n(tmp_path / "off", False)
        _, on, _, _ = _build_eu_n(tmp_path / "on", True)

        assert torch.equal(off.Hubbard_U_sr, on.Hubbard_U_sr)
        assert torch.equal(off.shell_types, on.shell_types)
        assert torch.equal(off.el_per_shell, on.el_per_shell)
        assert torch.equal(off.n_shells_per_atom, on.n_shells_per_atom)

    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    run_with_float64(check)
