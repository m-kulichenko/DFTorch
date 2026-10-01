#!/bin/bash
#SBATCH --account=m4988_g
#SBATCH --job-name=sed1
#SBATCH --output=out.out
#SBATCH --nodes=8
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=13:00:00
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding
#SBATCH --qos=regular

module purge
module load conda
module load python
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit

PROJ=/global/homes/m/maxim/programs/dftorch/DFTorch
NPROC_PER_NODE=4
export UV_PYTHON=$PROJ/.venv/bin/python

export PYTHONWARNINGS="ignore:Unverified HTTPS request"
if [ "$SLURM_CPUS_PER_TASK" -lt "$NPROC_PER_NODE" ]; then
    echo "ERROR: need at least one CPU per local torchrun worker" >&2
    exit 1
fi
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# On Perlmutter the first torch.compile/Triton pass can consume the whole debug
# allocation before SCF makes progress. Run DFTorch in eager mode here.
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID:-$$}
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-$$}

# Triton compiles a small C extension that includes Python.h. The DFTorch venv
# reports /usr/include/python3.11, which does not exist on Perlmutter compute
# nodes, so point gcc at a real NERSC Python 3.11 header tree.
PYVER=$($UV_PYTHON - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
PYINC=$(ls -d /global/common/software/nersc/pe/conda-envs/*/python-${PYVER}/nersc-python/include/python${PYVER} 2>/dev/null | sort -V | tail -1)
if [ -z "$PYINC" ] || [ ! -f "$PYINC/Python.h" ]; then
    echo "ERROR: could not find Python.h for python${PYVER}" >&2
    exit 1
fi
export CPATH="$PYINC${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
echo "Using Python.h from $PYINC"

export LD_LIBRARY_PATH=$HOME/conda/metis/lib:$LD_LIBRARY_PATH
# mpi4py 4.x needs the ABI-compat dir where libmpi.so.12 lives
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib-abi-mpich:$LD_LIBRARY_PATH
export METIS_DLL=$HOME/conda/metis/lib/libmetis.so

# KEY FIX: disable ALL MPI/PMI init — torchrun manages its own rendezvous
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pmix=^slurm
unset PMI_FD PMI_RANK PMI_SIZE PMIX_RANK

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Perlmutter has Slingshot hsn0-hsn3, not InfiniBand ib0.
# Pin PyTorch sockets to the high-speed network instead of the management NIC.
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN

# print interfaces on each node for debugging
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c \
    "echo \$SLURMD_NODENAME interfaces: \$(ip -o link show | awk '{print \$2}' | tr '\n' ' ')"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

cd /global/homes/m/maxim/calcs/dftorch/C1

# srun launches one task per node; each task runs torchrun for its 4 local GPUs
srun --mpi=none env UV_CACHE_DIR=/tmp/uv_${SLURM_JOB_ID:-$$} UV_PYTHON=$PROJ/.venv/bin/python uv run torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    main.py --backend nccl
