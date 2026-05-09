# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %                   SCF-TB - PROXY APPLICATION                      %
# %                   A.M.N. Niklasson, M. Kulichenko. T1, LANL       %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Total Energy Function:                                            %
# % E = 2Tr[H0(D-D0)] + (1/2)*sum_i U_i q_i^2 +                       %
# %      + (1/2)sum_{i,j (i!=j)} q_i C_{ij} q_j - Efield*dipole       %
# % dipole = sum_i R_{i} q_i                                          %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import logging
import os
import warnings

import torch

# to disable torchdynamo completely. Faster for smaller systems and single-point calculations.
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # hard-disable capture

import matplotlib.pyplot as plt
import numpy as np

from dftorch.Constants import Constants
from dftorch.ESDriver import ESDriver
from dftorch.MD import MDXLOS
from dftorch.Structure import Structure

### Configure torch and torch.compile ###
# Silence warnings and module logs
warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = ""  # disable PT2 logging
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
logging.getLogger("torch.fx").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)
# Enable dynamic shape capture for dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# default data type
torch.set_default_dtype(torch.float64)

torch.cuda.empty_cache()

dftorch_params = {
    "UNRESTRICTED": True,
    "SHARED_MU": False,  # if True, use shared chemical potential for both spin channels in unrestricted calculations. Otherwise, use separate chemical potentials for each spin channel.
    "BROKEN_SYM": True,  # if True, mix 2 % of lumo in homo at initialization
    "DELTA_SCF": True,  # if True, perform delta SCF for targeted, non-aufbau excited state. Performs GS SCF, then ES SCF.
    "DELTA_SCF_TARGET": "SINGLET",  # options: '"SINGLET" or "TRIPLET"'. desired lowest excited state
    "DELTA_SCF_SMEARING": False,  # if True, occupations for GS orbital and target ES orbital will be set to 0.5
    "COUL_METHOD": "!PME",  # 'FULL' for full coulomb matrix, 'PME' for PME method
    "COULOMB_ACC": 1e-6,  # Coulomb accuracy for full coulomb calcs or t_err for PME
    "COULOMB_CUTOFF": 12.0,  # Coulomb cutoff
    "PME_ORDER": 4,  # Ignored for FULL coulomb method
    "SCF_MAX_ITER": 90,  # Maximum number of SCF iterations
    "SCF_TOL": 1e-6,  # SCF convergence tolerance on density matrix
    "SCF_ALPHA": 0.1,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
    "KRYLOV_MAXRANK": 15,  # Maximum Krylov subspace rank
    "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in SCF
    "KRYLOV_TOL_MD": 1e-6,  # Krylov subspace convergence tolerance in MD SCF
    "KRYLOV_START": 10,  # Number of initial SCF iterations before starting Krylov acceleration
}

# Initial data, load atoms and coordinates, etc in COORD.dat
device = "cuda" if torch.cuda.is_available() else "cpu"
filename = "COORD_ACETONE.xyz"
# filename = "O2.xyz"
# LBox = None
LBox = torch.tensor(
    [12.0, 12.0, 12.0], device=device
)  # Simulation box size in Angstroms. Only cubic boxes supported for now.

# Create constants container. Set path to SKF files.
const = Constants(
    filename,
    "/Users/anthonybaldo/Documents/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/",
    magnetic_hubbard_ldep=False,
).to(device)

# Create structure object. Define total charge and electronic temperature.
# charge=0 and spin_pol need to be specified for open-shell systems.
# spin_pol is the number of unpaired electrons, not multiplicity.
# Shared chemical potential is used for both spin channels, so spin_pol has no effect but needs to be set anyways.
structure1 = Structure(
    filename,
    LBox,
    const,
    charge=0,
    spin_pol=0,
    os=dftorch_params["UNRESTRICTED"],
    Te=300.0,
    device=device,
)

# Create ESDriver object and run SCF calculation
# electronic_rcut and repulsive_rcut are in Angstroms.
# They should be >= cutoffs defined in SKF files for the element pair with largest cutoff present in the system.
es_driver = ESDriver(
    dftorch_params, electronic_rcut=8.0, repulsive_rcut=6.0, device=device
)
es_driver(structure1, const, do_scf=True)
es_driver.calc_forces(structure1, const)  # Calculate forces after SCF


dt = 0.125
steps = 2000


torch.manual_seed(0)
temperature_K = torch.tensor(300.0, device=structure1.device)
mdDriver = MDXLOS(es_driver, const, temperature_K=temperature_K)
# Set number of steps, time step (fs), dump interval and trajectory filename
mdDriver.run(
    structure1,
    dftorch_params,
    num_steps=steps,
    dt=dt,
    dump_interval=1,
    traj_filename="md_trj.xyz",
)

# print(vars(mdDriver))
Energy = mdDriver.E_array

timesteps = np.arange(0, len(Energy), 1)
timesteps = timesteps * dt

deltaE = (Energy - Energy[0]) * 1000
plt.plot(timesteps, deltaE, label=f"dt = {dt}")
plt.xlabel("Time (fs)")
plt.ylabel(r"$\Delta$E (meV)")


dt_array = [0.25, steps / 2], [0.5, steps / 4], [1.0, steps / 8]

for dt, step in dt_array:
    step = int(step)
    es_driver(structure1, const, do_scf=True)
    es_driver.calc_forces(structure1, const)  # Calculate forces after SCF
    torch.manual_seed(0)
    temperature_K = torch.tensor(300.0, device=structure1.device)
    mdDriver = MDXLOS(es_driver, const, temperature_K=temperature_K)
    mdDriver.run(
        structure1,
        dftorch_params,
        num_steps=step,
        dt=dt,
        dump_interval=10,
        traj_filename="md_trj.xyz",
    )
    Energy = mdDriver.E_array
    deltaE = (Energy - Energy[0]) * 1000
    print(f"number of steps and len of Energy= {step} {len(Energy)}")
    timesteps = np.arange(0, len(Energy), 1)
    timesteps = timesteps * dt
    print(timesteps)
    plt.plot(timesteps, deltaE, label=f"dt = {dt}")


plt.legend()
plt.show()
