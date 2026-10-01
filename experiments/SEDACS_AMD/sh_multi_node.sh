#!/bin/bash
#SBATCH --job-name=sed1
#SBATCH --output=out.out
#SBATCH --partition=shared-gpu-amd-mi250
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding
#SBATCH -C gpu_count:8
#SBATCH --exclude=cn4043          # explicitly exclude the problematic node

module purge
module load miniconda3
module load rocm
module load openmpi

PROJ=/vast/projects/ml4chem/Maksim/2025/DFTorch-main/DFTorch
WORKDIR=/vast/projects/ml4chem/Maksim/2025/tests/dftorch/sedacs_water
PYTHON_BIN=${UV_PYTHON:-$PROJ/.venv-rocm/bin/python}

export PYTHONWARNINGS="ignore:Unverified HTTPS request"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export LD_LIBRARY_PATH=$HOME/conda/metis/lib:$LD_LIBRARY_PATH
export METIS_DLL=$HOME/conda/metis/lib/libmetis.so
# KEY FIX: disable ALL MPI/PMI init — torchrun manages its own rendezvous
export MPI4PY_RC_INITIALIZE=0
export OMPI_MCA_pmix=^slurm
unset PMI_FD PMI_RANK PMI_SIZE PMIX_RANK

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# no InfiniBand on these nodes — use eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1         # ← disable IB since it doesn't exist
export NCCL_DEBUG=WARN            # back to WARN now that interface is known
export NCCL_NET_GDR_LEVEL=0      # disable GPUDirect RDMA over eth

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python environment not found: $PYTHON_BIN"
	echo "Set UV_PYTHON to the ROCm environment or create $PROJ/.venv-rocm first."
	exit 1
fi

cd "$WORKDIR"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 "$PYTHON_BIN" - <<'PY'
import sys
import torch

runtime = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU-only"
print(f"torch={torch.__version__} runtime={runtime} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if getattr(torch.version, "hip", None) is None:
    raise SystemExit("This job targets MI250 GPUs. Use a ROCm PyTorch environment, not a CUDA-only wheel.")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("PyTorch cannot see AMD GPUs on this node.")
PY

# print interfaces on each node for debugging
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c \
    "echo \$SLURMD_NODENAME interfaces: \$(ip -o link show | awk '{print \$2}' | tr '\n' ' ')"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

# srun launches one task per node; each task runs torchrun for its 8 GPUs
srun --mpi=none "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=${SLURM_GPUS_ON_NODE:-8} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    main.py --backend nccl
