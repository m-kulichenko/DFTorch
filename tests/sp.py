import os

os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")


import torch
from dftorch.Constants import Constants
from dftorch.Structure import Structure
from dftorch.ESDriver import ESDriver

torch.set_default_dtype(torch.float64)

device = "cpu"

dftorch_params = {
    "coul_method": "FULL",  # 'FULL' for full coulomb matrix, 'PME' for PME method
    "Coulomb_acc": 5e-5,  # Coulomb accuracy for full coulomb calcs or t_err for PME
    "cutoff": 10.0,  # Coulomb cutoff
    "PME_order": 4,  # Ignored for FULL coulomb method
    "SCF_MAX_ITER": 100,  # Maximum number of _scf iterations
    "SCF_TOL": 1e-6,  # _scf convergence tolerance on density matrix
    "SCF_ALPHA": 0.2,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
    "KRYLOV_MAXRANK": 20,  # Maximum Krylov subspace rank
    "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in _scf
    "KRYLOV_TOL_MD": 1e-4,  # Krylov subspace convergence tolerance in MD _scf
    "KRYLOV_START": 5,  # Number of initial _scf iterations before starting Krylov acceleration
}

xyz_path = "COORD.xyz"  # Solvated acetylacetone and glycine molecules in H20, Na, Cl
skf_dir = "/home/maxim/Projects/DFTB/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/"


LBox = torch.tensor(
    [25.0, 25.0, 25.0], device=device
)  # Simulation box size in Angstroms. Only cubic boxes supported for now.
# Create constants container. Set path to SKF files.
const = Constants(xyz_path, skf_dir, magnetic_hubbard_ldep=False).to(device)

# Create structure object. Define total charge and electronic temperature.
structure1 = Structure(xyz_path, LBox, const, charge=0, Te=1000.0, device=device)

# Create ESDriver object and run _scf calculation
# electronic_rcut and repulsive_rcut are in Angstroms.
# They should be >= cutoffs defined in SKF files for the element pair with largest cutoff present in the system.
es_driver = ESDriver(
    dftorch_params, electronic_rcut=8.0, repulsive_rcut=6.0, device=device
)
es_driver(structure1, const, do_scf=True)
es_driver.calc_forces(structure1, const)  # Calculate forces after _scf
