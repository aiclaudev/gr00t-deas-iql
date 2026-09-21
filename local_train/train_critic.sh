#!/usr/bin/env bash
# DEAS critic training run (GR00T N1.5).
#
# Trains the critic head on a frozen Eagle backbone: backbone_encoder, the twin Q
# and the value MLP, about 139M of 1.69B parameters. The actor is not touched
# here; evaluation takes actor and critic as separate checkpoints.
#
# Defaults: batch 128 on GPU 0 for 100k steps. Every knob is an environment
# variable, and any extra arguments are passed straight to the training script.
#
#   local_train/train_critic.sh
#   DATASET=~/data/real_robocasa BATCH_SIZE=64 local_train/train_critic.sh
#   nohup local_train/train_critic.sh > /dev/null 2>&1 &   # see LOG_FILE below
set -euo pipefail

LOCAL_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../local_eval/env.sh
source "${LOCAL_TRAIN_DIR}/../local_eval/env.sh"

DATASET="${DATASET:-/home/junhyeong/data/fake_robocasa_rl}"
RUN_NAME="${RUN_NAME:-deas_critic_bs128_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_TRAIN_DIR}/../local_outputs/critic_train/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
DATA_CONFIG="${DATA_CONFIG:-single_panda_gripper_rl}"

GPU_DEVICE="${GPU_DEVICE:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
WORKERS="${WORKERS:-8}"

CRITIC_ACTION_HORIZON="${CRITIC_ACTION_HORIZON:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EXPECTILE="${EXPECTILE:-0.9}"
DISCOUNT1="${DISCOUNT1:-0.995}"
DISCOUNT2="${DISCOUNT2:-0.995}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RESUME="${RESUME:-0}"

if [[ ! -d "${DATASET}" ]]; then
    echo "No dataset at ${DATASET}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"

# save_total_limit is 8 in the training script and each checkpoint is roughly
# 9 GB, so a long run can hold ~72 GB. Warn while there is still time to react.
AVAILABLE_GIB=$(df -BG --output=avail "${OUTPUT_DIR}" | tail -1 | tr -dc '0-9')
KEPT=$(( MAX_STEPS / SAVE_STEPS )); (( KEPT > 8 )) && KEPT=8
NEEDED_GIB=$(( KEPT * 9 ))
if (( AVAILABLE_GIB < NEEDED_GIB )); then
    echo "Warning: up to ${KEPT} checkpoints (~${NEEDED_GIB} GiB) but only ${AVAILABLE_GIB} GiB free." >&2
fi

EXTRA=()
[[ "${RESUME}" == "1" ]] && EXTRA+=(--resume)

{
    echo "Run        ${RUN_NAME}"
    echo "Base model ${BASE_MODEL}"
    echo "Dataset    ${DATASET}"
    echo "Output     ${OUTPUT_DIR}"
    echo "GPU ${GPU_DEVICE} x${NUM_GPUS} | batch ${BATCH_SIZE}/GPU | ${MAX_STEPS} steps | save every ${SAVE_STEPS} (keeping ${KEPT}, ~${NEEDED_GIB} GiB)"
    echo "Started    $(date -Is)"
} | tee "${LOG_FILE}"

cd "${DEAS_ROOT}"
conda run --no-capture-output -n "${DEAS_CONDA_ENV}" python scripts/gr00t_deas_critic_finetune.py \
    --dataset-path "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --base-model-path "${BASE_MODEL}" \
    --data-config "${DATA_CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --critic-action-horizon "${CRITIC_ACTION_HORIZON}" \
    --learning-rate "${LEARNING_RATE}" \
    --expectile "${EXPECTILE}" \
    --discount1 "${DISCOUNT1}" \
    --discount2 "${DISCOUNT2}" \
    --seed "${SEED}" \
    --dataloader-num-workers "${WORKERS}" \
    --report-to "${REPORT_TO}" \
    --run-name "${RUN_NAME}" \
    "${EXTRA[@]}" "$@" \
    2>&1 | tee -a "${LOG_FILE}"

echo "Finished $(date -Is)" | tee -a "${LOG_FILE}"
