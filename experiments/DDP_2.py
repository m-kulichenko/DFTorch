import os
import argparse
import time
import torch
import torch.distributed as dist

# import warnings
import logging

# to disable torchdynamo completely. Faster for smaller systems and single-point calculations.
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # hard-disable capture
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # disables torch.compile / Inductor globally


from dftorch.Constants import Constants
from dftorch.Structure import Structure

from sedacs.graph import (
    get_initial_graph,
)
from sedacs.graph_partition import get_coreHaloIndices, graph_partition

from dftorch._tools import (
    calculate_dist_dips,
)

# from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy

from dftorch.ewald_pme.neighbor_list import NeighborState


from dftorch.sedacs import pack_lol_int, unpack_lol_int, bcast_1d_int, scf, MDXL_Graph


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

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

MAX_DEG = 600
GTHRESH = 0.0002
NJUMPS = 1


def prepare_structure(device):
    ### Prepare DFTB+DFTorch parameters, structure, neighbor list, and graph partitioning ###
    dftorch_params = {
        "coul_method": "PME",  # 'FULL' for full coulomb matrix, 'PME' for PME method
        "Coulomb_acc": 5e-5,  # Coulomb accuracy for full coulomb calcs or t_err for PME
        "cutoff": 10.0,  # Coulomb cutoff
        "h0_cutoff": 8.0,  # Coulomb cutoff
        "graph_cutoff": 5.0,  # Graph cutoff
        "PME_order": 4,  # Ignored for FULL coulomb method
        "SCF_MAX_ITER": 100,  # Maximum number of _scf iterations
        "SCF_TOL": 1e-6,  # _scf convergence tolerance on density matrix
        "SCF_ALPHA": 0.5,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
        "KRYLOV_MAXRANK": 10,  # Maximum Krylov subspace rank
        "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in _scf
        "KRYLOV_TOL_MD": 1e-4,  # Krylov subspace convergence tolerance in MD _scf
        "KRYLOV_START": 3,  # Number of initial _scf iterations before starting Krylov acceleration
    }

    # filename = 'COORD_far.xyz'            # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    # LBox = torch.tensor([25.0, 25.0, 25.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.

    # filename = 'COORD_8WATER.xyz'            # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    # LBox = torch.tensor([30.0, 30.0, 30.0], device=device) # Simulation box size in Angstroms. Only cubic boxes supported for now.

    filename = (
        "water_30.xyz"  # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    )
    LBox = torch.tensor(
        [35.0, 35.0, 35.0], device=device
    )  # Simulation box size in Angstroms. Only cubic boxes supported for now.

    const = Constants(
        filename,
        #'/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/ptbp/complete_set',
        #'C:\\000_MyFiles\\Programs\\DFTorch\\tests\\sk_orig\\ptbp\\complete_set\\',
        "/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/mio-1-1/mio-1-1/",
        magnetic_hubbard_ldep=False,
    ).to(device)
    structure = Structure(filename, LBox, const, charge=0, Te=5000.0, device=device)
    structure.SpecClustNN = 20
    structure.interface = "dftorch"
    return structure, dftorch_params


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    # choose device for collectives
    if backend == "nccl":
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")

    nparts = 8
    structure, dftorch_params = prepare_structure(device)
    if dftorch_params["cutoff"] < dftorch_params["graph_cutoff"]:
        raise ValueError(
            "Coulomb cutoff must be greater than or equal to graph cutoff for this implementation."
        )

    INT_DTYPE = torch.int32

    if WORLD_RANK == 0:
        positions = torch.stack(
            (structure.RX, structure.RY, structure.RZ),
        )

        nbr_state = NeighborState(
            positions,
            structure.lattice_vecs,
            None,
            dftorch_params["cutoff"],
            is_dense=True,
            buffer=1.0,
            use_triton=False,
        )
        disps, dists, nl = calculate_dist_dips(
            positions, nbr_state, dftorch_params["cutoff"]
        )
        print("nl.dtype", nl.dtype)

        if dftorch_params["graph_cutoff"] < dftorch_params["cutoff"]:
            nl_init = torch.where(
                (dists > dftorch_params["graph_cutoff"]) | (dists == 0.0), -1, nl
            )
            nl_init = nl_init.sort(dim=1, descending=True)[0]
            nl_init = nl_init[:, : torch.max(torch.sum(nl_init != -1, dim=1))]

        elif dftorch_params["graph_cutoff"] > dftorch_params["cutoff"]:
            raise ValueError(
                "Coulomb cutoff must be greater than or equal to graph cutoff for this implementation."
            )

        # num_neighbors = torch.sum(nl != -1, dim=1)
        # nl = torch.cat((num_neighbors.unsqueeze(1), nl), dim=1)

        fullGraph, eweights = get_initial_graph(
            positions.T.cpu().numpy(),
            nl_init.cpu().numpy(),
            dftorch_params["graph_cutoff"],
            MAX_DEG,
            structure.LBox.cpu().numpy(),
        )
        parts = graph_partition(
            structure,
            structure,
            fullGraph,
            "SpectralClustering",
            nparts,
            positions.T.cpu().numpy(),
            verb=False,
        )

        partsCoreHalo = []
        numCores = []
        print("\nCore and halos indices for every part:")
        for i in range(nparts):
            coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, NJUMPS)
            partsCoreHalo.append(coreHalo)
            numCores.append(nc)

        numCores = torch.tensor(numCores, dtype=INT_DTYPE, device=device)
        fullGraph = torch.tensor(fullGraph, dtype=INT_DTYPE, device=device)
        flat, offsets = pack_lol_int(partsCoreHalo, INT_DTYPE)
        nl_shape = torch.tensor(list(nl.shape), dtype=INT_DTYPE, device=device)
        fullGraph_shape = torch.tensor(
            list(fullGraph.shape), dtype=INT_DTYPE, device=device
        )
    else:
        partsCoreHalo = None
        numCores = torch.empty((nparts,), dtype=INT_DTYPE, device=device)
        nbr_state = None
        disps, dists = None, None
        flat, offsets = None, None
        nl_shape = torch.empty((2,), dtype=INT_DTYPE, device=device)
        fullGraph_shape = torch.empty((2,), dtype=INT_DTYPE, device=device)
    dist.broadcast(numCores, 0)
    dist.broadcast(nl_shape, 0)
    dist.broadcast(fullGraph_shape, 0)

    if WORLD_RANK != 0:
        nl = torch.empty(
            tuple(int(x) for x in nl_shape.tolist()), dtype=INT_DTYPE, device=device
        )
        fullGraph = torch.empty(
            tuple(int(x) for x in fullGraph_shape.tolist()),
            dtype=INT_DTYPE,
            device=device,
        )

    dist.broadcast(nl, 0)
    dist.broadcast(fullGraph, 0)

    flat = bcast_1d_int(flat, INT_DTYPE, device=device, src=0)
    offsets = bcast_1d_int(offsets, INT_DTYPE, device=device, src=0)
    partsCoreHalo = unpack_lol_int(flat.cpu(), offsets.cpu())

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
    temperature_K = torch.tensor(500.0, device=structure.device)
    mdDriver = MDXL_Graph(
        structure.const, temperature_K, NJUMPS, GTHRESH, MAX_DEG, INT_DTYPE
    )
    # Set number of steps, time step (fs), dump interval and trajectory filename
    mdDriver.run(
        structure,
        dftorch_params,
        num_steps=100,
        dt=0.3,
        mu0=mu0,
        fullGraph=fullGraph,
        nbr_state=nbr_state,
        ch=partsCoreHalo[start:end],
        core_size=numCores[start:end],
        works_per_rank=works_per_rank,
        device=device,
        dump_interval=5,
        traj_filename="md_trj.xyz",
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
