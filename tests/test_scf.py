import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic and avoids C++ toolchain)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import pathlib


@pytest.mark.parametrize("device", ["cpu"])
def test_energy_smoke_import_and_call(device):
    """
    CI smoke test:
    - Imports core modules.
    - Runs a *small* _scf + forces calculation on CPU.
    - Skips automatically if required SKF test data is not present.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = root / "tests" / "ch4.xyz"
    skf_dir = root / "tests" / "data_skf_mio-1-1"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

    import torch

    torch.set_default_dtype(torch.float64)

    from dftorch.Constants import Constants
    from dftorch.Structure import Structure
    from dftorch.ESDriver import ESDriver

    dftorch_params = {
        "FILENAME": str(xyz_path),
        "CELL": [25.0, 25.0, 25.0],  # Simulation box vectors in Angstroms
        "SKFPATH": str(skf_dir) + "/",  # Path to SKF files
        "T_ELECTRONIC": 1000.0,  # Electronic temperature in Kelvin for Fermi smearing
        "RCUT_ELECTRONIC": 8.0,  # Cutoff for electronic interactions in Angstroms. Should be >= largest cutoff in SKF files for the element pairs present in the system.
        "RCUT_REPULSIVE": 4.0,  # Cutoff for repulsive interactions in Angstroms. Should be >= largest cutoff in SKF files for the element pairs present in the system.
        "COUL_METHOD": "PME",  # 'FULL' for full coulomb matrix, 'PME' for Particle Mesh Ewald method
        "SCF_MAX_ITER": 25,  # Maximum number of _scf iterations
        "KRYLOV_START": 5,  # Number of initial _scf iterations before starting Krylov acceleration
    }

    # Create constants container. Set path to SKF files.
    const = Constants(
        dftorch_params,
    ).to(device)

    # Create structure object. Define total charge and electronic temperature.
    structure1 = Structure(dftorch_params, const, device=device)

    # Create ESDriver object and run _scf calculation
    # electronic_rcut and repulsive_rcut are in Angstroms.
    # They should be >= cutoffs defined in SKF files for the element pair with largest cutoff present in the system.
    es_driver = ESDriver(dftorch_params, device=device)
    es_driver(structure1, const, do_scf=True)
    es_driver.calc_forces(structure1, const)  # Calculate forces after _scf

    assert hasattr(structure1, "e_tot")
    assert torch.isfinite(structure1.e_tot).all()
    assert hasattr(structure1, "f_tot")
    assert structure1.f_tot.shape == (3, structure1.Nats)
    assert torch.isfinite(structure1.f_tot).all()
