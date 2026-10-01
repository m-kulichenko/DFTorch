'''
module load miniconda3
conda create -p $HOME/conda/metis -c conda-forge metis -y
cd /vast/projects/ml4chem/Maksim/2025/DFTorch-main/DFTorch
uv pip install --python .venv/bin/python metis
export LD_LIBRARY_PATH=$HOME/conda/metis/lib:$LD_LIBRARY_PATH
export METIS_DLL=$HOME/conda/metis/lib/libmetis.so
'''
import os
import argparse
import time

# import warnings
import logging

import sys

# Toggle this flag, then restart the kernel and rerun from Cell 1.
ENABLE_TORCH_COMPILE = False
os.environ["DFTORCH_ENABLE_COMPILE"] = "1" if ENABLE_TORCH_COMPILE else "0"
if ENABLE_TORCH_COMPILE:
    os.environ.pop("TORCHDYNAMO_DISABLE", None)
else:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.distributed as dist

#proxya_path = "/home/maxim/Projects/SEDACS_github/sedacs/src/"
#sys.path.insert(1, proxya_path)



import dftorch._nearestneighborlist as _nnl
_nnl.USE_ALCHEMI = True   # all subsequent calls use alchemi by default

from dftorch.sedacs import (
    scf,
    MDXL_Graph,
    prepare_structure,
    prepare_initial_graph_data,
)


### Configure torch and torch.compile ###
# Silence warnings and module logs
# warnings.filterwarnings("ignore")
# os.environ["TORCH_LOGS"] = ""               # disable PT2 logging
# os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
# os.environ["TORCHDYNAMO_VERBOSE"] = "0"
logging.getLogger("torch.fx").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.CRITICAL)
logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)
# Enable dynamic shape capture for dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# default data type
torch.set_default_dtype(torch.float64)

# # Environment variables set by torch.distributed.launch
# LOCAL_RANK = int(os.environ["LOCAL_RANK"])
# WORLD_SIZE = int(os.environ["WORLD_SIZE"])
# WORLD_RANK = int(os.environ["RANK"])

MAX_DEG = 500
GTHRESH = 0.001
NJUMPS = 1


def init_processes(backend):
    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    WORLD_RANK = int(os.environ["RANK"])

    INT_DTYPE = torch.int32
    nparts = 128

    dftorch_params = {
    
    "FILENAME": "C1.pdb",
    "SKFPATH": '/global/homes/m/maxim/programs/dftorch/DFTorch/experiments/sk_orig/mio-1-1/mio-1-1/',  # Path to SKF files
    "T_ELECTRONIC": 4000.0,  # Electronic temperature in Kelvin for Fermi smearing
    "RCUT_ELECTRONIC": 7.0,  # Cutoff for electronic interactions in Angstroms. Should be >= largest cutoff in SKF files for the element pairs present in the system.
    "RCUT_REPULSIVE": 4.5,  # Cutoff for repulsive interactions in Angstroms. Should be >= largest cutoff in SKF files for the element pairs present in the system.
    "COUL_METHOD": "PME",  # 'FULL' for full coulomb matrix, 'PME' for Particle Mesh Ewald method
    'COULOMB_CUTOFF': 10.0,        # Coulomb cutoff
    
    'graph_cutoff': 4.5,        # Graph cutoff

    'SCF_MAX_ITER': 100,    # Maximum number of SCF iterations
    'SCF_TOL': 1e-6,       # SCF convergence tolerance on density matrix
    'SCF_ALPHA': 0.15,      # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
    'KRYLOV_MAXRANK': 10,  # Maximum Krylov subspace rank
    'KRYLOV_TOL': 1e-7,    # Krylov subspace convergence tolerance in SCF
    'KRYLOV_TOL_MD': 1e-2, # Krylov subspace convergence tolerance in MD SCF
    'KRYLOV_START': 10,     # Number of initial SCF iterations before starting Krylov acceleration
    #"h_damp_exp": 3.7,  # Exponential damping factor for hydrogen bonds. Default is None.
                }

    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    # choose device for collectives
    if backend == "nccl":
        device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    structure, dftorch_params = prepare_structure(dftorch_params, device)

    if dftorch_params["COULOMB_CUTOFF"] < dftorch_params["graph_cutoff"]:
        raise ValueError(
            "Coulomb cutoff must be greater than or equal to graph cutoff for this implementation."
        )

    (
        nbr_state,
        disps,
        dists,
        nl,
        fullGraph,
        partsCoreHalo,
        numCores,
    ) = prepare_initial_graph_data(
        structure,
        dftorch_params,
        nparts,
        NJUMPS,
        MAX_DEG,
        INT_DTYPE,
        device,
    )

    works_per_rank = nparts // WORLD_SIZE
    if nparts % WORLD_SIZE != 0:
        raise ValueError("nparts must be divisible by WORLD_SIZE.")
    if WORLD_SIZE > nparts:
        raise ValueError("WORLD_SIZE must be less than or equal to nparts.")

    cur_rank = dist.get_rank()
    start = cur_rank * works_per_rank
    end = start + works_per_rank
    mu0, fullGraph = scf(
        structure,
        dftorch_params,
        fullGraph,
        partsCoreHalo[start:end],
        numCores[start:end],
        nbr_state,
        disps,
        dists,
        nl,
        works_per_rank,
        NJUMPS,
        GTHRESH,
        MAX_DEG,
        device,
    )
    

    torch.manual_seed(0)
    temperature_K = torch.tensor(330.0, device=structure.device)
    mdDriver = MDXL_Graph(
        structure.const, temperature_K, NJUMPS, GTHRESH, MAX_DEG, INT_DTYPE
    )
    # Set number of steps, time step (fs), dump interval and trajectory filename
    mdDriver.run(
        structure,
        dftorch_params,
        num_steps=24000,
        dt=0.5,
        mu0=mu0,
        fullGraph=fullGraph,
        nbr_state=nbr_state,
        ch=partsCoreHalo[start:end],
        core_size=numCores[start:end],
        works_per_rank=works_per_rank,
        device=device,
        dump_interval=100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    start_time1 = time.perf_counter()
    try:
        init_processes(backend=args.backend)
    finally:
        # avoid NCCL resource leak warning on exit
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    print("  t TOTAL {:.1f} s\n".format(time.perf_counter() - start_time1))