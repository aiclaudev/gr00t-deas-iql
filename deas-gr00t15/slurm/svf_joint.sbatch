#!/bin/bash
#SBATCH --account=sub
#SBATCH --qos=own
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=768G
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
# Time/log paths and all immutable training arguments come from submit_svf.py.
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && $# -ge 2 ]] || { echo 'Use scripts/submit_svf.py via sbatch.' >&2; exit 2; }
RUN_ROOT=$1
shift
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
TRAIN_ENV=/home/nas_main/dohyunlee/miniconda3/envs/groot-train
[[ $(realpath -m -- "$RUN_ROOT") == "$REPO_ROOT"/output/svf-joint/* ]] || exit 2
cd "$REPO_ROOT"
set +u
source /home/nas_main/dohyunlee/miniconda3/etc/profile.d/conda.sh
conda activate "$TRAIN_ENV"
set -u
export PYTHONPATH=$REPO_ROOT
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_SAVE_CODE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export TMPDIR=/home/nas_main/dohyunlee/.tmp/svf-${SLURM_JOB_ID}
export HF_MODULES_CACHE=$RUN_ROOT/.cache/hf-modules
export TRITON_CACHE_DIR=$RUN_ROOT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$RUN_ROOT/.cache/torchinductor
mkdir -p -- "$TMPDIR" "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
"$TRAIN_ENV/bin/python" - <<'CHECK'
import torch
if torch.cuda.device_count() != 4:
    raise RuntimeError(f'Expected four allocated GPUs; got {torch.cuda.device_count()}')
CHECK
exec "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=4 "$REPO_ROOT/scripts/train_svf.py" "$@"
