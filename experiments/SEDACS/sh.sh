#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=m4988_g
#SBATCH --job-name=sed1
#SBATCH --output=out.out
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --qos=debug

module purge
module load conda
module load python
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit

#source activate sedacs
export UV_PYTHON=/global/homes/m/maxim/programs/dftorch/DFTorch/.venv/bin/python

export PYTHONWARNINGS="ignore:Unverified HTTPS request"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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

PROJ=/global/homes/m/maxim/programs/dftorch/DFTorch
UV_CACHE_DIR=/tmp/uv_$$ UV_PYTHON=$PROJ/.venv/bin/python uv run torchrun --standalone --nproc_per_node=4 main.py --backend nccl
