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

# Submit with bash_scripts/submit_deas_training.sh, which supplies the time,
# resources, log paths, and afterok dependencies separately for each stage.
set -euo pipefail

if [[ $# -ne 6 || -z ${SLURM_JOB_ID:-} ]]; then
    echo "Run this stage through bash_scripts/submit_deas_training.sh." >&2
    exit 2
fi
export DEAS_JOB_START_UNIX=$(date +%s)
STAGE=$1
RUN_ROOT=$2
NUM_GPUS=$3
GLOBAL_BATCH_SIZE=$4
MAX_STEPS=$5
SEED=$6
REPO_ROOT=/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
DATA_ROOT=/home/nas_main/dohyunlee/jh_ws/data
ORIGINAL_MODEL=/home/nas_main/dohyunlee/jh_ws/models/GR00T-N1.5-3B
TRAIN_ENV=/home/nas_main/dohyunlee/miniconda3/envs/groot-train
PYTHON_BIN=$TRAIN_ENV/bin/python

case "$NUM_GPUS" in 1|2|4) ;; *) echo "GPU count must be 1, 2, or 4." >&2; exit 2 ;; esac
[[ $SEED =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer." >&2; exit 2; }
if [[ $GLOBAL_BATCH_SIZE != 128 || ! $MAX_STEPS =~ ^[1-9][0-9]*$ ]]; then
    echo "This pipeline requires global batch 128 and a positive step count." >&2
    exit 2
fi
if [[ $RUN_ROOT == */../* || $RUN_ROOT == */.. || ! -d $RUN_ROOT ]]; then
    echo "Invalid pipeline output directory: $RUN_ROOT" >&2
    exit 2
fi
RUN_ROOT=$(realpath -e -- "$RUN_ROOT")
if [[ $RUN_ROOT != "$REPO_ROOT"/output/deas-training/* || $RUN_ROOT != /home/nas_main/dohyunlee/* ]]; then
    echo "Pipeline output must stay inside $REPO_ROOT/output/deas-training." >&2
    exit 2
fi
for child in 01-bc-demo 02-bc-rollout 03-critic logs .cache; do
    resolved_child=$(realpath -m -- "$RUN_ROOT/$child")
    if [[ $resolved_child != "$RUN_ROOT"/* ]]; then
        echo "Output/cache path escapes the run directory: $RUN_ROOT/$child" >&2
        exit 2
    fi
done
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
RUN_ID=${RUN_ROOT##*/}
cd "$REPO_ROOT"

set +u
source /home/nas_main/dohyunlee/miniconda3/etc/profile.d/conda.sh
conda activate "$TRAIN_ENV"
set -u

export PYTHONPATH=$REPO_ROOT
export USE_TF=0
export WANDB_PROJECT='gr00t1.5 finetune'
export WANDB_ENTITY=aiclaudev
export WANDB_MODE=online
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export WANDB_SAVE_CODE=false
export WANDB_RUN_GROUP="seed-${SEED}-${RUN_ID}"
export WANDB_JOB_TYPE=$STAGE
export WANDB_TAGS="seed-${SEED},${STAGE},steps-${MAX_STEPS}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
DEAS_JOB_CACHE=$RUN_ROOT/.cache/${STAGE}-${SLURM_JOB_ID}
export HF_MODULES_CACHE=$DEAS_JOB_CACHE/hf-modules
# Unix socket paths must remain short (Linux limit: 108 bytes).
export TMPDIR=/home/nas_main/dohyunlee/.tmp/deas-${SLURM_JOB_ID}
[[ $(realpath -m -- "$TMPDIR") == /home/nas_main/dohyunlee/* ]] || exit 2
export TRITON_CACHE_DIR=$DEAS_JOB_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$DEAS_JOB_CACHE/torchinductor
mkdir -p -- "$HF_MODULES_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

TASKS=(CoffeeSetupMug PnPMicrowaveToCounter TurnOffStove PnPCounterToMicrowave)
DATASETS=()
EXTRA_ARGS=()
case "$STAGE" in
    bc-demo)
        ENTRYPOINT=$REPO_ROOT/scripts/gr00t_finetune.py
        OUTPUT_DIR=$RUN_ROOT/01-bc-demo
        BASE_MODEL=$ORIGINAL_MODEL
        DATA_CONFIG=single_panda_gripper
        DATASETS=("$DATA_ROOT/robocasa_mg_gr00t_100")
        ;;
    bc-rollout)
        ENTRYPOINT=$REPO_ROOT/scripts/gr00t_finetune.py
        OUTPUT_DIR=$RUN_ROOT/02-bc-rollout
        BASE_MODEL=$RUN_ROOT/01-bc-demo
        DATA_CONFIG=single_panda_gripper
        for task in "${TASKS[@]}"; do
            DATASETS+=("$DATA_ROOT/deas_robocasa/demos/$task" "$DATA_ROOT/deas_robocasa/success_rollouts/$task")
        done
        ;;
    critic)
        ENTRYPOINT=$REPO_ROOT/scripts/gr00t_deas_critic_finetune.py
        OUTPUT_DIR=$RUN_ROOT/03-critic
        # The README critic uses the original pretrained model, independently
        # of the filtered BC actor. The job still follows BC in the pipeline.
        BASE_MODEL=$ORIGINAL_MODEL
        DATA_CONFIG=single_panda_gripper_rl
        for task in "${TASKS[@]}"; do
            DATASETS+=("$DATA_ROOT/deas_robocasa/demos/$task" "$DATA_ROOT/deas_robocasa/rollouts/$task")
        done
        EXTRA_ARGS=(--critic-action-horizon 16 --discount1 0.9 --discount2 0.99 --expectile 0.7)
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

export WANDB_DIR=$OUTPUT_DIR

# Fail before training if allocation or metadata is wrong. In particular,
# missing/invalid stats must not trigger whole-dataset preprocessing here.
"$PYTHON_BIN" - "$NUM_GPUS" "$BASE_MODEL" "${DATASETS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from gr00t.data.schema import DatasetStatisticalValues

expected_gpus = int(sys.argv[1])
actual_gpus = torch.cuda.device_count()
if actual_gpus != expected_gpus:
    raise RuntimeError(f"Requested {expected_gpus} GPUs, but this worker sees {actual_gpus}")
model = Path(sys.argv[2])
if not (model / "config.json").is_file():
    raise FileNotFoundError(f"Missing model config: {model / 'config.json'}")
for name in sys.argv[3:]:
    dataset = Path(name)
    for filename in ("info.json", "modality.json", "stats.json", "episodes.jsonl", "tasks.jsonl"):
        if not (dataset / "meta" / filename).is_file():
            raise FileNotFoundError(dataset / "meta" / filename)
    stats = json.loads((dataset / "meta/stats.json").read_text())
    for values in stats.values():
        DatasetStatisticalValues.model_validate(values)
print(f"Preflight passed: {actual_gpus} GPUs, {len(sys.argv) - 3} datasets", flush=True)
PY

TRAIN_ARGS=(
    "$ENTRYPOINT"
    --dataset-path "${DATASETS[@]}"
    --base-model-path "$BASE_MODEL"
    --output-dir "$OUTPUT_DIR"
    --data-config "$DATA_CONFIG"
    --num-gpus "$NUM_GPUS"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --seed "$SEED"
    --save-steps 1000
    --dataloader-num-workers 4
    --run-name "${RUN_ID}-${STAGE}-seed${SEED}"
    "${EXTRA_ARGS[@]}"
)
printf 'Stage=%s global_batch=%s per_gpu_batch=%s steps=%s output=%s\n' \
    "$STAGE" "$GLOBAL_BATCH_SIZE" "$BATCH_SIZE" "$MAX_STEPS" "$OUTPUT_DIR"
if [[ $NUM_GPUS -eq 1 ]]; then
    exec "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
else
    export IS_TORCHRUN=1
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" "${TRAIN_ARGS[@]}"
fi
