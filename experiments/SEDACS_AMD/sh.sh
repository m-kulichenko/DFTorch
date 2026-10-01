#!/bin/bash
#SBATCH --job-name=sed1
#SBATCH --output=out.out
##SBATCH --partition=shared-gpu-ampere
##SBATCH --partition=atdm-ml
#SBATCH --partition=shared-gpu-amd-mi250
##SBATCH --partition=volta-x86
##SBATCH --nodelist=cn4030,cn4031,cn4032,cn4035  # Specify the exact nodes to use
##SBATCH --exclude=cn4043
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH -C gpu_count:8

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

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python environment not found: $PYTHON_BIN"
	echo "Create a ROCm environment first, for example:"
	echo "  cd $PROJ"
	echo "  uv venv .venv-rocm --python 3.12"
	echo "  Install a ROCm-enabled PyTorch wheel compatible with the loaded rocm module"
	echo "  UV_PYTHON=$PROJ/.venv-rocm/bin/python uv pip install --python $PROJ/.venv-rocm/bin/python -e .[sedacs]"
	exit 1
fi

cd "$WORKDIR"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

runtime = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU-only"
print(f"torch={torch.__version__} runtime={runtime} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if getattr(torch.version, "hip", None) is None:
	raise SystemExit("This job targets MI250 GPUs. Use a ROCm PyTorch environment, not a CUDA-only wheel.")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
	raise SystemExit("PyTorch cannot see AMD GPUs on this node.")
PY

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 main.py --backend nccl
