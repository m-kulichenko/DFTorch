# import os
# # Disable TorchDynamo/Inductor compilation in tests (keeps tests deterministic and avoids C++ toolchain)
# os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import pathlib

def test_import_dftorch():
    import dftorch 

def test_import_key_modules():
    '''
    Import modules.
    '''
    from dftorch import (  # noqa: F401
        AtomicDensityMatrix,
        BondIntegral,
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
def test_energy_smoke_import_and_call(device):
    """
    Smoke test: verifies that core modules import and that calling a SCF routine does not crash.
    """

    root = pathlib.Path(__file__).resolve().parents[1]  # DFTorch/

    xyz_path = str(root / "tests" / "COORD.xyz")
    skf_dir = str(root / "experiments" / "sk_orig" / "mio-1-1" / "mio-1-1") + '/'


    import torch
    torch.set_default_dtype(torch.float64)

    from dftorch.Constants import Constants
    from dftorch.Structure import Structure
    from dftorch.ESDriver import ESDriver


    dftorch_params = {
        'coul_method': 'FULL', # 'FULL' for full coulomb matrix, 'PME' for PME method
        'Coulomb_acc': 5e-5,   # Coulomb accuracy for full coulomb calcs or t_err for PME
        'cutoff': 10.0,        # Coulomb cutoff
        'PME_order': 4,        # Ignored for FULL coulomb method

        'SCF_MAX_ITER': 100,    # Maximum number of SCF iterations
        'SCF_TOL': 1e-6,       # SCF convergence tolerance on density matrix
        'SCF_ALPHA': 0.2,      # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.

        'KRYLOV_MAXRANK': 20,  # Maximum Krylov subspace rank
        'KRYLOV_TOL': 1e-6,    # Krylov subspace convergence tolerance in SCF
        'KRYLOV_TOL_MD': 1e-4, # Krylov subspace convergence tolerance in MD SCF
        'KRYLOV_START': 5,     # Number of initial SCF iterations before starting Krylov acceleration
                    }
                    
    LBox = torch.tensor([25.0, 25.0, 25.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.
    # Create constants container. Set path to SKF files.
    const = Constants(xyz_path,
                      skf_dir,
                      magnetic_hubbard_ldep=False
                     ).to(device)

    # Create structure object. Define total charge and electronic temperature.
    structure1 = Structure(xyz_path, LBox, const, charge=0, Te=1000.0, device=device)

    # Create ESDriver object and run SCF calculation
    # electronic_rcut and repulsive_rcut are in Angstroms.
    # They should be >= cutoffs defined in SKF files for the element pair with largest cutoff present in the system.
    es_driver = ESDriver(dftorch_params, electronic_rcut=8.0, repulsive_rcut=6.0, device=device)
    es_driver(structure1, const, do_scf=True)
    es_driver.calc_forces(structure1, const) # Calculate forces after SCF

    assert hasattr(structure1, "e_tot")
    assert torch.isfinite(structure1.e_tot).all()
    assert hasattr(structure1, "f_tot")
    assert structure1.f_tot.shape == (3, structure1.Nats)
    assert torch.isfinite(structure1.f_tot).all()
