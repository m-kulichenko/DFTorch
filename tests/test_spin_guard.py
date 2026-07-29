"""Spin-polarized f systems must be refused, not approximated (D-12).

Eu 4f7 is genuinely open-shell, so returning a closed-shell number for a
spin-polarized request would be quietly wrong rather than merely approximate.
Phase 4 therefore raises a named exception instead.  This is the unsupported
branch of SIM-02; the supported branch lives in
``tests/test_single_shot_energy.py``.

The guard must fire for BOTH entry points.  Before it existed, Eu-N with
``UNRESTRICTED=True`` failed with a bare ``IndexError: index 4 is out of bounds
for dimension 0 with size 4`` raised deep inside ``scf_x_os`` when
``do_scf=True``, and failed with nothing at all — no error, no energy — when
``do_scf=False``.

f-free open-shell calculations are an existing supported capability and must be
completely unaffected.
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


EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}


def _tests_dir() -> Path:
    return Path(__file__).resolve().parent


def _build_eu_n(tmp_path: Path, **overrides):
    """Build the Eu-N system and its driver without running anything yet."""
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    xyz_path = tmp_path / "eu_n_spin.xyz"
    xyz_path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 4 spin guard case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {EU_N_SEPARATION:.8f} 0.00000000 0.00000000\n"
    )

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_tests_dir() / "f_orbital_data") + os.sep
    params.update(overrides)

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    return driver, structure, const


def _build_ch4(**overrides):
    """Build an f-free CH4 system from the mio-1-1 fixture set."""
    from dftorch.Constants import Constants
    from dftorch.Structure import Structure

    xyz_path = _tests_dir() / "ch4.xyz"
    skf_dir = _tests_dir() / "data_skf_mio-1-1"
    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

    params = {
        "FILENAME": str(xyz_path),
        "SKFPATH": str(skf_dir) + os.sep,
        "CELL": [25.0, 25.0, 25.0],
        "T_ELECTRONIC": 1000.0,
        "RCUT_ELECTRONIC": 8.0,
        "RCUT_REPULSIVE": 4.0,
        "COUL_METHOD": "FULL",
        "CHARGE": 0,
    }
    params.update(overrides)

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    return structure, const, params


def test_unrestricted_f_system_raises_on_scf(tmp_path):
    """do_scf=True previously died with a bare IndexError inside scf_x_os."""

    def check():
        from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError

        driver, structure, const = _build_eu_n(tmp_path, UNRESTRICTED=True)
        with pytest.raises(FSpinPolarizationUnsupportedError):
            driver(structure, const, do_scf=True)

    run_with_float64(check)


def test_unrestricted_f_system_raises_on_single_shot(tmp_path):
    """do_scf=False previously did nothing at all — silence, not an error."""

    def check():
        from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError

        driver, structure, const = _build_eu_n(tmp_path, UNRESTRICTED=True)
        with pytest.raises(FSpinPolarizationUnsupportedError):
            driver(structure, const, do_scf=False)

    run_with_float64(check)


def test_error_message_names_the_mode_and_the_atom(tmp_path):
    """The message must be self-explaining: where, what mode, which atoms.

    It must NOT leak filesystem paths (threat T-04-02): only the requested mode
    and the orbital count belong in the text.
    """

    def check():
        from dftorch._slater_koster_pair import FSpinPolarizationUnsupportedError

        driver, structure, const = _build_eu_n(tmp_path, UNRESTRICTED=True)
        with pytest.raises(FSpinPolarizationUnsupportedError) as excinfo:
            driver(structure, const, do_scf=False)

        message = str(excinfo.value)
        assert "ESDriver.forward" in message
        assert "UNRESTRICTED" in message
        assert "n_orb == 16" in message
        assert str(tmp_path) not in message
        assert "f_orbital_data" not in message

    run_with_float64(check)


def test_restricted_f_system_does_not_raise(tmp_path):
    """Closed-shell f calculations are unchanged by the guard."""

    def check():
        driver, structure, const = _build_eu_n(tmp_path)
        driver(structure, const, do_scf=False)

        assert hasattr(structure, "e_tot")
        assert torch.isfinite(structure.e_tot)

    run_with_float64(check)


def test_guard_ignores_f_free_systems():
    """f-free open-shell calculations must be completely unaffected.

    The guard is called directly rather than through the driver so that this
    assertion isolates the guard from the unrelated open-shell machinery: a
    driver-level assertion could pass or fail for reasons that have nothing to
    do with whether the guard correctly ignored an f-free system.
    """

    def check():
        from dftorch.ESDriver import _require_closed_shell_f_system

        structure, const, params = _build_ch4(UNRESTRICTED=True)
        assert params["UNRESTRICTED"] is True

        assert (
            _require_closed_shell_f_system(
                structure, const, params, "ESDriver.forward"
            )
            is None
        )

    run_with_float64(check)
