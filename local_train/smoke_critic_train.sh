#!/usr/bin/env bash
# Short DEAS critic training run against a synthetic dataset.
#
# The actor comes from the public base model; the critic and value heads are
# initialised fresh by GR00T_N1_5_DEAS_Critic.from_pretrained(..., from_gr00t_n1_5=True).
# Nothing here is a real training run: the data is noise and the step count is
# small. It exists to prove the pipeline runs and saves a checkpoint.
set -euo pipefail

LOCAL_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../local_eval/env.sh
source "${LOCAL_TRAIN_DIR}/../local_eval/env.sh"

DATASET="${DATASET:-/home/junhyeong/data/fake_robocasa_rl}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_TRAIN_DIR}/../local_outputs/critic_smoke/$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
DATA_CONFIG="${DATA_CONFIG:-single_panda_gripper_rl}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MAX_STEPS="${MAX_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
CRITIC_ACTION_HORIZON="${CRITIC_ACTION_HORIZON:-4}"
WORKERS="${WORKERS:-0}"
REPORT_TO="${REPORT_TO:-tensorboard}"

if [[ ! -d "${DATASET}" ]]; then
    echo "No dataset at ${DATASET}; generate one with local_train/make_fake_dataset.py" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
mkdir -p "${OUTPUT_DIR}"

echo "Base model  ${BASE_MODEL}"
echo "Dataset     ${DATASET}"
echo "Output      ${OUTPUT_DIR}"
echo "${MAX_STEPS} steps, batch ${BATCH_SIZE}, critic action horizon ${CRITIC_ACTION_HORIZON}, GPU ${GPU_DEVICE}"

cd "${DEAS_ROOT}"
conda run --no-capture-output -n "${DEAS_CONDA_ENV}" python scripts/gr00t_deas_critic_finetune.py \
    --dataset-path "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --base-model-path "${BASE_MODEL}" \
    --data-config "${DATA_CONFIG}" \
    --num-gpus 1 \
    --batch-size "${BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --critic-action-horizon "${CRITIC_ACTION_HORIZON}" \
    --dataloader-num-workers "${WORKERS}" \
    --report-to "${REPORT_TO}" \
    --run-name critic-smoke \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
