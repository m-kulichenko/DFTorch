import os

# Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic and avoids C++ toolchain)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import pathlib


def test_import_dftorch():
    import dftorch  # noqa: F401


def test_import_key_modules():
    """
    Import modules.
    """
    from dftorch import (  # noqa: F401
        AtomicDensityMatrix,
        _bond_integral,
        CoulombMatrix,
        DM_Fermi,
        Energy,
        Forces,
        H0andS,
        SCF,
        SlaterKosterPair,
        Tools,
        nearestneighborlist,
    )


@pytest.mark.parametrize("device", ["cpu"])
def test_nearestneighborlist_small_xyz(device):
    """
    Test neighbor list construction for a small molecule.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = root / "tests" / "ch4.xyz"
    skf_dir = root / "tests" / "data_skf_mio-1-1"

    assert xyz_path.is_file(), f"Missing required test geometry: {xyz_path}"
    assert skf_dir.is_dir(), f"Missing required SKF directory: {skf_dir}"

    import torch

    torch.set_default_dtype(torch.float64)

    from dftorch.Constants import Constants
    from dftorch.nearestneighborlist import vectorized_nearestneighborlist

    # Minimal coordinates for 3 atoms in a triangle
    Rx = torch.tensor([0.0, 1.0, 0.0])
    Ry = torch.tensor([0.0, 0.0, 1.0])
    Rz = torch.tensor([0.0, 0.0, 0.0])
    TYPE = torch.tensor([6, 6, 6], dtype=torch.int32)  # Carbon atoms
    LBox = torch.tensor([10.0, 10.0, 10.0])
    Rcut = 4.0
    N = 3
    const = Constants(
        str(xyz_path), str(skf_dir) + "/", magnetic_hubbard_ldep=False
    ).to(device)

    result = vectorized_nearestneighborlist(
        TYPE,
        Rx,
        Ry,
        Rz,
        LBox,
        Rcut,
        N,
        const,
        upper_tri_only=True,
        remove_self_neigh=True,
        min_image_only=True,
        verbose=False,
    )

    # Check that result is a tuple and contains expected tensors
    assert isinstance(result, tuple)
    # Example: check that at least one neighbor pair is found
    neighbor_pairs = result[0]
    assert torch.is_tensor(neighbor_pairs)
    assert neighbor_pairs.shape[0] > 0


@pytest.mark.parametrize("device", ["cpu"])
def test_energy_smoke_import_and_call(device):
    """
    CI smoke test:
    - Imports core modules.
    - Runs a *small* SCF + forces calculation on CPU.
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
        "coul_method": "FULL",  # 'FULL' for full coulomb matrix, 'PME' for PME method
        "Coulomb_acc": 5e-5,  # Coulomb accuracy for full coulomb calcs or t_err for PME
        "cutoff": 10.0,  # Coulomb cutoff
        "PME_order": 4,  # Ignored for FULL coulomb method
        "SCF_MAX_ITER": 25,  # Maximum number of SCF iterations
        "SCF_TOL": 1e-6,  # SCF convergence tolerance on density matrix
        "SCF_ALPHA": 0.2,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
        "KRYLOV_MAXRANK": 10,  # Maximum Krylov subspace rank
        "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in SCF
        "KRYLOV_TOL_MD": 1e-4,  # Krylov subspace convergence tolerance in MD SCF
        "KRYLOV_START": 5,  # Number of initial SCF iterations before starting Krylov acceleration
    }

    LBox = torch.tensor(
        [25.0, 25.0, 25.0], device=device
    )  # Simulation box size in Angstroms. Only cubic boxes supported for now.
    # Create constants container. Set path to SKF files.
    const = Constants(
        str(xyz_path), str(skf_dir) + "/", magnetic_hubbard_ldep=False
    ).to(device)

    # Create structure object. Define total charge and electronic temperature.
    structure1 = Structure(
        str(xyz_path), LBox, const, charge=0, Te=1000.0, device=device
    )

    # Create ESDriver object and run SCF calculation
    # electronic_rcut and repulsive_rcut are in Angstroms.
    # They should be >= cutoffs defined in SKF files for the element pair with largest cutoff present in the system.
    es_driver = ESDriver(
        dftorch_params, electronic_rcut=8.0, repulsive_rcut=6.0, device=device
    )
    es_driver(structure1, const, do_scf=True)
    es_driver.calc_forces(structure1, const)  # Calculate forces after SCF

    assert hasattr(structure1, "e_tot")
    assert torch.isfinite(structure1.e_tot).all()
    assert hasattr(structure1, "f_tot")
    assert structure1.f_tot.shape == (3, structure1.Nats)
    assert torch.isfinite(structure1.f_tot).all()
