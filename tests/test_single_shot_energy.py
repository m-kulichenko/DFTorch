"""Single-shot (non-SCC) energy path for the f-containing Eu-N diatomic.

Phase 4 D-11 fixes the completion bar for f systems at *single-shot* energy:
build H0/S, diagonalize once at the reference charge state, and report band
plus repulsion.  There is no charge self-consistency and, deliberately, no
charge-fluctuation Coulomb term — see
``test_eu_n_single_shot_has_no_coulomb_term`` for why that exclusion is
load-bearing rather than an omission.

Covers SIM-01 (H0/S reach assembly through the driver for an f system) and the
supported branch of SIM-02 (closed-shell energy).  The unsupported branch of
SIM-02 lives in ``tests/test_spin_guard.py``.
"""

import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic
# and avoids requiring a C++ toolchain).  Mirrors tests/test_scf.py.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import sys
from pathlib import Path

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


# --- Eu-N reference case (D-19, D-24) ---------------------------------------
#
# Isolated Eu-N diatomic at the D-24 target separation of 2.655 A, using the
# nine-file f-orbital SKF fixture set in tests/f_orbital_data/.
EU_N_SEPARATION = 2.655

EU_N_PARAMS = {
    "T_ELECTRONIC": 1000.0,
    "RCUT_ELECTRONIC": 10.0,
    "RCUT_REPULSIVE": 6.0,
    "COUL_METHOD": "FULL",
    "CHARGE": 0,
}

# Reference values recorded at plan time by direct computation of the algorithm
# specified in 04-01-PLAN.md Task 1.  See
# test_eu_n_single_shot_reference_energy for the interpretation of a mismatch.
EU_N_REFERENCE_E_TOT = -17.510444238744924
EU_N_REFERENCE_E_BAND0 = -16.948081509141794


def _skf_dir() -> Path:
    return Path(__file__).resolve().parent / "f_orbital_data"


def _write_eu_n_xyz(path: Path, separation: float = EU_N_SEPARATION) -> None:
    path.write_text(
        "2\n"
        "Eu-N diatomic (Phase 4 single-shot reference case)\n"
        "Eu 0.00000000 0.00000000 0.00000000\n"
        f"N {separation:.8f} 0.00000000 0.00000000\n"
    )


def _run_single_shot(tmp_path: Path, **overrides):
    """Build the Eu-N system and drive it through ``forward(do_scf=False)``.

    Returns the structure so callers can inspect the attributes the single-shot
    branch is contracted to populate.
    """
    from dftorch.Constants import Constants
    from dftorch.ESDriver import ESDriver
    from dftorch.Structure import Structure

    xyz_path = tmp_path / "eu_n.xyz"
    _write_eu_n_xyz(xyz_path, overrides.pop("separation", EU_N_SEPARATION))

    params = dict(EU_N_PARAMS)
    params["FILENAME"] = str(xyz_path)
    params["SKFPATH"] = str(_skf_dir()) + os.sep
    params.update(overrides)

    const = Constants(params).to("cpu")
    structure = Structure(params, const, device="cpu")
    driver = ESDriver(params, device="cpu")
    driver(structure, const, do_scf=False)
    return structure


def test_eu_n_h0_s_shape_and_symmetry(tmp_path):
    """SIM-01: the f system reaches H0/S assembly through the driver intact.

    Eu contributes 16 orbitals and N contributes 4 (N has s and p only, not
    s/p/d), so the expected dimension is 20, not 25.  The second run pins the
    related contract that the single-shot branch must not fold the external
    field term into the stored ``structure.H0``: H0 is field-independent, so a
    run with a non-zero field must produce a byte-identical H0.
    """

    def check():
        structure = _run_single_shot(tmp_path)

        assert structure.H0.shape == (20, 20)
        assert structure.S.shape == (20, 20)
        assert torch.isfinite(structure.H0).all()
        assert torch.isfinite(structure.S).all()
        assert torch.allclose(structure.H0, structure.H0.T, atol=1e-10)
        assert torch.allclose(structure.S, structure.S.T, atol=1e-10)

        field_structure = _run_single_shot(
            tmp_path, ELECTRIC_FIELD=[0.01, 0.0, 0.0]
        )
        assert torch.allclose(field_structure.H0, structure.H0, atol=1e-12)

    run_with_float64(check)


def test_eu_n_single_shot_populates_e_tot(tmp_path):
    """SIM-02 supported branch: do_scf=False yields a finite float64 energy.

    Before this path existed, ``forward(do_scf=False)`` returned None and left
    ``structure.e_tot`` unset entirely.
    """

    def check():
        structure = _run_single_shot(tmp_path)

        assert hasattr(structure, "e_tot")
        assert torch.is_tensor(structure.e_tot)
        assert structure.e_tot.dim() == 0
        assert torch.isfinite(structure.e_tot)
        assert structure.e_tot.dtype is torch.float64

    run_with_float64(check)


def test_eu_n_single_shot_has_no_coulomb_term(tmp_path):
    """The single-shot energy is band + repulsion, with Ecoul exactly zero.

    D-11 forbids charge self-consistency.  Building a Coulomb term out of the
    first-iterate Mulliken charges would not be a single-shot energy, it would
    be one broken SCF step: those charges are around q_Eu = -2.7 here and drift
    to -2.99 at 4 A, i.e. the atoms are not even neutral at separation.  Feeding
    them into the electrostatic energy destroys the binding curve entirely.
    ``energy()`` is therefore called with C=None and dq_p1=None, selecting its
    ``Ecoul = 0`` arm.
    """

    def check():
        structure = _run_single_shot(tmp_path)

        assert structure.e_coul == 0.0
        assert torch.allclose(
            structure.e_tot, structure.e_elec_tot + structure.e_repulsion
        )

    run_with_float64(check)


def test_eu_n_single_shot_reference_energy(tmp_path):
    """Pin the non-SCC energy definition to its plan-time value.

    ``-17.510444238744924`` eV total and ``-16.948081509141794`` eV band energy
    are reference values recorded at plan time by direct computation with the
    parameter dict pinned in this module.  A mismatch here does NOT mean the
    test went stale — it means the definition of the non-SCC energy drifted.
    Downstream Phase 4 work (the Eu-N separation scan and the SIM-05 loose-band
    minimum gate) records numbers against exactly this definition, so a drift
    invalidates them.
    """

    def check():
        structure = _run_single_shot(tmp_path)

        assert abs(structure.e_tot.item() - EU_N_REFERENCE_E_TOT) < 1e-6
        assert abs(structure.e_band0.item() - EU_N_REFERENCE_E_BAND0) < 1e-6

    run_with_float64(check)


def test_eu_n_single_shot_charges_are_diagnostic_only(tmp_path):
    """Mulliken charges are recorded for inspection, never fed into the energy.

    ``q_Eu`` comes out large and negative (about -2.7 e) because these are
    first-iterate Mulliken charges with no self-consistency behind them.  That
    is precisely why D-11 keeps them out of the energy: they are a diagnostic,
    not a converged charge state.  Overall neutrality still holds because the
    charges are constructed as (Mulliken population - nuclear charge) summed
    over a neutral system.
    """

    def check():
        structure = _run_single_shot(tmp_path)

        assert structure.q.shape == (2,)
        assert abs(structure.q.sum().item()) < 1e-9
        assert structure.q[0].item() < -1.0

    run_with_float64(check)
