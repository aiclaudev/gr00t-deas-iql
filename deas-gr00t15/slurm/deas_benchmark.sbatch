#!/bin/bash
#SBATCH --account=sub
#SBATCH --qos=own
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && $# -ge 3 ]] || { echo 'Use scripts/submit_deas_benchmark.py.' >&2; exit 2; }
NUM_GPUS=$1
CASE_ROOT=$2
shift 2
case "$NUM_GPUS" in 1|4) ;; *) exit 2 ;; esac
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
[[ $(realpath -m -- "$CASE_ROOT") == "$REPO_ROOT"/output/deas-benchmark/* ]] || exit 2
cd "$REPO_ROOT"
TRAIN_ENV=/home/nas_main/dohyunlee/miniconda3/envs/groot-train
set +u
source /home/nas_main/dohyunlee/miniconda3/etc/profile.d/conda.sh
conda activate "$TRAIN_ENV"
set -u
export PYTHONPATH=$REPO_ROOT
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export TMPDIR=/home/nas_main/dohyunlee/.tmp/deas-bench-${SLURM_JOB_ID}
export HF_MODULES_CACHE=$CASE_ROOT/.cache/hf-modules
export TRITON_CACHE_DIR=$CASE_ROOT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$CASE_ROOT/.cache/torchinductor
mkdir -p -- "$TMPDIR" "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
"$TRAIN_ENV/bin/python" - "$NUM_GPUS" "$@" <<'CHECK'
import sys, torch
if "--video-backend" in sys.argv and sys.argv[sys.argv.index("--video-backend") + 1] == "torchcodec":
    from torchcodec.decoders import VideoDecoder
    print("Worker TorchCodec CPU library import: OK", flush=True)
if torch.cuda.device_count() != int(sys.argv[1]):
    raise RuntimeError(f'Expected {sys.argv[1]} GPUs, found {torch.cuda.device_count()}')
CHECK
exec "$TRAIN_ENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" "$REPO_ROOT/scripts/benchmark_deas_throughput.py" "$@"
