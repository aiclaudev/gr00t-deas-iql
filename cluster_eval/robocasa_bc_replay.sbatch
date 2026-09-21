#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit this script with sbatch.' >&2; exit 2; }
ROOT=${1:?absolute repository root}
ACTOR=${2:?absolute BC2 checkpoint directory}
RUN=${3:?absolute new output directory}
source "$ROOT/cluster_eval/env.sh"
source "$ROOT/local_eval/env.sh"
set -u
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GR00T_EVAL_DIAGNOSTICS=1
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT TORCHELASTIC_USE_AGENT_STORE
mkdir -p "$RUN/scripts" "$RUN/actor_metadata"
cp "$DEAS_ROOT/scripts/robocasa/replay_bc_inputs.py" "$DEAS_ROOT/gr00t/eval/inference_trace.py" "$RUN/scripts/"
cp "$ACTOR/config.json" "$ACTOR/experiment_cfg/metadata.json" "$RUN/actor_metadata/"
for TASK in CoffeeSetupMug PnPCounterToMicrowave PnPMicrowaveToCounter TurnOffStove; do
  mkdir -p "$RUN/raw/$TASK"
  "$DEAS_PYTHON" -u "$DEAS_ROOT/scripts/eval_policy_robocasa.py" \
    --actor_model_path "$ACTOR" --model_type gr00tn15 --env_name "$TASK" \
    --num_episodes 2 --n_envs 1 --shared_memory --seed 42 \
    --action_horizon 16 --execute_horizon 16 --denoising_steps 4 --noise 0 \
    --output_path "$RUN/eval/$TASK" --save_video --save_inference_inputs \
    --collect_data --data_collection_path "$RUN/raw/$TASK" --report_to none
  "$DEAS_PYTHON" -u "$ROOT/local_collect/robocasa_replay_to_lerobot.py" \
    --hdf5 "$RUN/raw/$TASK/demo.hdf5" --output-root "$RUN/dataset" \
    --camera-size 256 --max-episodes 2
done
"$DEAS_PYTHON" -u "$RUN/scripts/replay_bc_inputs.py" \
  --dataset "$RUN" --actor "$ACTOR" --report "$RUN/reference_replay.json" --max-calls-per-task 3
"$DEAS_PYTHON" -u "$ROOT/local_collect/package_bc_replay.py" --root "$RUN"
