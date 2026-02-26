"""run.py:"""
# ruff: noqa

#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import warnings
import logging

# to disable torchdynamo completely. Faster for smaller systems and single-point calculations.
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # hard-disable capture

import numpy as np

from dftorch._scf import SCFx, SCFx_batch
from dftorch.Constants import Constants
from dftorch.Structure import Structure, StructureBatch
from dftorch.MD import MDXL, MDXLBatch, MDXLOS
from dftorch.ESDriver import ESDriver, ESDriverBatch

from sedacs.graph import get_initial_graph
from dftorch._tools import calculate_dist_dips

from sedacs.graph_partition import (
    get_coreHaloIndices,
    graph_partition,
)

# from sedacs.ewald import calculate_PME_ewald, init_PME_data, calculate_alpha_and_num_grids, ewald_energy
from dftorch.ewald_pme import (
    calculate_PME_ewald,
    init_PME_data,
    calculate_alpha_and_num_grids,
    ewald_energy,
)
from dftorch.ewald_pme.neighbor_list import NeighborState
from dftorch._nearestneighborlist import vectorized_nearestneighborlist

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

"""Blocking point-to-point communication."""


def run(rank, partsCoreHalo, structure1):
    tensor = torch.zeros(1)
    if rank == 0:
        structure1.TYPE[0] = 10
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print("Rank ", rank, " has data ", structure1.TYPE)

    cur_ch = partsCoreHalo[rank]

    (
        _,
        _,
        nnRx,
        nnRy,
        nnRz,
        nnType,
        _,
        _,
        neighbor_I,
        neighbor_J,
        IJ_pair_type,
        JI_pair_type,
    ) = vectorized_nearestneighborlist(
        structure1.TYPE[cur_ch],
        structure1.RX[cur_ch],
        structure1.RY[cur_ch],
        structure1.RZ[cur_ch],
        structure1.LBox,
        8.0,
        len(cur_ch),
        structure1.const,
        upper_tri_only=False,
        remove_self_neigh=False,
        min_image_only=False,
        verbose=False,
    )

    # print('Rank ', rank, ' has data ', structure1.TYPE)
    print("Rank ", rank, structure1.TYPE.untyped_storage().data_ptr())
    print("child TYPE is_shared:", structure1.TYPE.is_shared())


def init_process(rank, size, partsCoreHalo, structure1, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, partsCoreHalo, structure1)


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")

    dftorch_params = {
        "coul_method": "FULL",  # 'FULL' for full coulomb matrix, 'PME' for PME method
        "Coulomb_acc": 5e-5,  # Coulomb accuracy for full coulomb calcs or t_err for PME
        "cutoff": 12.0,  # Coulomb cutoff
        "PME_order": 4,  # Ignored for FULL coulomb method
        "SCF_MAX_ITER": 100,  # Maximum number of _scf iterations
        "SCF_TOL": 1e-6,  # _scf convergence tolerance on density matrix
        "SCF_ALPHA": 0.1,  # Scaled delta function coefficient. Acts as linear mixing coefficient used before Krylov acceleration starts.
        "KRYLOV_MAXRANK": 20,  # Maximum Krylov subspace rank
        "KRYLOV_TOL": 1e-6,  # Krylov subspace convergence tolerance in _scf
        "KRYLOV_TOL_MD": 1e-4,  # Krylov subspace convergence tolerance in MD _scf
        "KRYLOV_START": 10,  # Number of initial _scf iterations before starting Krylov acceleration
    }
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    filename = (
        "COORD_far.xyz"  # Solvated acetylacetone and glycine molecules in H20, Na, Cl
    )
    LBox = torch.tensor(
        [25.0, 25.0, 25.0], device=device
    )  # Simulation box size in Angstroms. Only cubic boxes supported for now.
    const = Constants(
        filename,
        #'/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/ptbp/complete_set',
        #'C:\\000_MyFiles\\Programs\\DFTorch\\tests\\sk_orig\\ptbp\\complete_set\\',
        "/home/maxim/Projects/DFTB/DFTorch/tests/sk_orig/mio-1-1/mio-1-1/",
        magnetic_hubbard_ldep=False,
    ).to(device)
    structure1 = Structure(filename, LBox, const, charge=0, Te=500.0, device=device)
    es_driver = ESDriver(
        dftorch_params, electronic_rcut=8.0, repulsive_rcut=6.0, device=device
    )

    positions = torch.stack(
        (structure1.RX, structure1.RY, structure1.RZ),
    )
    CALPHA, grid_dimensions = calculate_alpha_and_num_grids(
        structure1.lattice_vecs.cpu().numpy(),
        dftorch_params["cutoff"],
        dftorch_params["Coulomb_acc"],
    )
    PME_data = init_PME_data(
        grid_dimensions, structure1.lattice_vecs, CALPHA, dftorch_params["PME_order"]
    )
    nbr_state = NeighborState(
        positions,
        structure1.lattice_vecs,
        None,
        dftorch_params["cutoff"],
        is_dense=True,
        buffer=0.0,
        use_triton=False,
    )
    disps, dists, nl = calculate_dist_dips(
        positions, nbr_state, dftorch_params["cutoff"]
    )
    num_neighbors = torch.sum(nl != -1, dim=1)
    nl = torch.cat(
        (num_neighbors.unsqueeze(1), nl.sort(dim=1, descending=True)[0]), dim=1
    )
    nl = nl.cpu().numpy()
    graphNL = get_initial_graph(positions.T.cpu().numpy(), nl, 5.0, 20, structure1.LBox)
    structure1.SpecClustNN = 8
    structure1.interface = "dftorch"
    nparts = 4
    parts = graph_partition(
        structure1,
        structure1,
        graphNL,
        "SpectralClustering",
        nparts,
        positions.T.cpu().numpy(),
        verb=False,
    )
    njumps = 1
    partsCoreHalo = []
    numCores = []
    print("\nCore and halos indices for every part:")
    fullGraph = graphNL
    for i in range(nparts):
        coreHalo, nc, nh = get_coreHaloIndices(parts[i], fullGraph, njumps)
        partsCoreHalo.append(coreHalo)
        numCores.append(nc)
        print("coreHalo for part", i, "=", coreHalo)

    for rank in range(world_size):
        p = mp.Process(
            target=init_process, args=(rank, world_size, partsCoreHalo, structure1, run)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
