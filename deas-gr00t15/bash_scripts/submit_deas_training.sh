#!/bin/bash
# Prepare one independent Slurm job per training stage; no login GPU work.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash bash_scripts/submit_deas_training.sh --time HH:MM:SS [options]

Submits demo BC -> successful-rollout BC -> DEAS critic with afterok dependencies.
Defaults: --gpus 4 --steps 10000 --seed 42; global batch is always 128.
--gpus 1|2|4              Number of GPUs per stage.
--steps N                 Optimizer steps per stage; use a fresh run for short validation.
--seed N                  Nonnegative random seed shared across the three stages.
--initial-dependency afterany:JOB_ID  Delay only demo BC until a previous seed finishes.
--time HH:MM:SS            Default time limit for each stage.
--time-bc-demo HH:MM:SS    Override the demo BC time limit.
--time-bc-rollout HH:MM:SS Override the rollout BC time limit.
--time-critic HH:MM:SS     Override the critic time limit.
Each stage needs an explicit time, from --time or its override, before submission.
Choose times from expected runtime plus 20-30% headroom; none are assumed here.
--dry-run prints the three commands without submitting or creating any files.
USAGE
}
fail() { echo "Error: $*" >&2; exit 2; }

ORIGINAL_ARGS=("$@")
NUM_GPUS=4
MAX_STEPS=10000
SEED=42
INITIAL_DEPENDENCY=
TIME_LIMIT=
TIME_BC_DEMO=
TIME_BC_ROLLOUT=
TIME_CRITIC=
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -ge 2 ]] || fail "--gpus requires a value."
            NUM_GPUS=$2; shift 2 ;;
        --time)
            [[ $# -ge 2 ]] || fail "--time requires a value."
            TIME_LIMIT=$2; shift 2 ;;
        --steps)
            [[ $# -ge 2 ]] || fail "--steps requires a value."
            MAX_STEPS=$2; shift 2 ;;
        --seed)
            [[ $# -ge 2 ]] || fail "--seed requires a value."
            SEED=$2; shift 2 ;;
        --initial-dependency)
            [[ $# -ge 2 ]] || fail "--initial-dependency requires a value."
            INITIAL_DEPENDENCY=$2; shift 2 ;;
        --time-bc-demo)
            [[ $# -ge 2 ]] || fail "--time-bc-demo requires a value."
            TIME_BC_DEMO=$2; shift 2 ;;
        --time-bc-rollout)
            [[ $# -ge 2 ]] || fail "--time-bc-rollout requires a value."
            TIME_BC_ROLLOUT=$2; shift 2 ;;
        --time-critic)
            [[ $# -ge 2 ]] || fail "--time-critic requires a value."
            TIME_CRITIC=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
case "$NUM_GPUS" in 1|2|4) ;; *) fail "--gpus must be 1, 2, or 4." ;; esac
[[ $MAX_STEPS =~ ^[1-9][0-9]*$ ]] || fail "--steps must be a positive integer."
[[ $SEED =~ ^[0-9]+$ ]] || fail "--seed must be a nonnegative integer."
if [[ -n $INITIAL_DEPENDENCY && ! $INITIAL_DEPENDENCY =~ ^afterany:[0-9]+$ ]]; then
    placeholder_pattern='^afterany:<SEED_[0-9]+_CRITIC_JOB_ID>$'
    if [[ $DRY_RUN -ne 1 || ! $INITIAL_DEPENDENCY =~ $placeholder_pattern ]]; then
        fail "--initial-dependency must be afterany:JOB_ID with a numeric job ID."
    fi
fi
for value in "$TIME_LIMIT" "$TIME_BC_DEMO" "$TIME_BC_ROLLOUT" "$TIME_CRITIC"; do
    if [[ -n $value ]]; then
        [[ $value =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || fail "Time limits must use HH:MM:SS."
        [[ ! $value =~ ^0+:00:00$ ]] || fail "Time limits must be positive."
    fi
done
TIME_BC_DEMO=${TIME_BC_DEMO:-$TIME_LIMIT}
TIME_BC_ROLLOUT=${TIME_BC_ROLLOUT:-$TIME_LIMIT}
TIME_CRITIC=${TIME_CRITIC:-$TIME_LIMIT}
for stage_time in TIME_BC_DEMO TIME_BC_ROLLOUT TIME_CRITIC; do
    if [[ -z ${!stage_time} ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            printf -v "$stage_time" '%s' '<TIME_REQUIRED>'
        else
            fail "Supply --time HH:MM:SS or all stage-specific times; $stage_time has no limit."
        fi
    fi
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
WORKSPACE_ROOT=$(dirname -- "$REPO_ROOT")
DATA_ROOT=$WORKSPACE_ROOT/data
ORIGINAL_MODEL=$WORKSPACE_ROOT/models/GR00T-N1.5-3B
STAGE_SCRIPT=$REPO_ROOT/slurm/deas_train_stage.sbatch
TRAIN_ENV=/home/nas_main/dohyunlee/miniconda3/envs/groot-train
PYTHON_BIN=$TRAIN_ENV/bin/python
RUN_ID=$(date -u +%Y%m%dT%H%M%S.%NZ)
RUN_ROOT=$(realpath -m -- "$REPO_ROOT/output/deas-training/$RUN_ID")
[[ $RUN_ROOT == "$REPO_ROOT"/output/deas-training/* && $RUN_ROOT == /home/nas_main/dohyunlee/* ]] \
    || fail "Resolved output path escapes the repository or personal workspace: $RUN_ROOT"
GLOBAL_BATCH_SIZE=128
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
MEM_GIB=$((192 * NUM_GPUS))
TASKS=(CoffeeSetupMug PnPMicrowaveToCounter TurnOffStove PnPCounterToMicrowave)
STAGES=(bc-demo bc-rollout critic)

build_command() {
    local stage=$1 dependency=$2 stage_time
    case "$stage" in
        bc-demo) stage_time=$TIME_BC_DEMO ;;
        bc-rollout) stage_time=$TIME_BC_ROLLOUT ;;
        critic) stage_time=$TIME_CRITIC ;;
    esac
    COMMAND=(
        sbatch --parsable
        --account=sub --qos=own --partition=compute
        --nodes=1 --ntasks=1 --gres="gpu:$NUM_GPUS" --cpus-per-gpu=8 --mem="${MEM_GIB}G"
        --container=nvcr.io/nvidia/pytorch:25.04-py3 --export=NONE
        --time="$stage_time" --kill-on-invalid-dep=yes
        --job-name="deas-$stage" --chdir="$REPO_ROOT"
        --output="$RUN_ROOT/logs/$stage-%j.out" --error="$RUN_ROOT/logs/$stage-%j.err"
    )
    if [[ -n $dependency ]]; then
        COMMAND+=(--dependency="$dependency")
    fi
    COMMAND+=("$STAGE_SCRIPT" "$stage" "$RUN_ROOT" "$NUM_GPUS" "$GLOBAL_BATCH_SIZE" "$MAX_STEPS" "$SEED")
}
print_command() { printf '%q ' "${COMMAND[@]}"; printf '\n'; }

printf 'Run: %s\nGPUs: %s | batch/GPU: %s | global batch: %s | steps/stage: %s\n' \
    "$RUN_ROOT" "$NUM_GPUS" "$BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$MAX_STEPS"
printf 'Seed: %s\n' "$SEED"
printf 'Time limits: demo=%s | rollout=%s | critic=%s\n' "$TIME_BC_DEMO" "$TIME_BC_ROLLOUT" "$TIME_CRITIC"
if [[ $DRY_RUN -eq 1 ]]; then
    dependency=$INITIAL_DEPENDENCY
    for stage in "${STAGES[@]}"; do
        build_command "$stage" "$dependency"
        print_command
        case "$stage" in
            bc-demo) dependency='afterok:<BC_DEMO_JOB_ID>' ;;
            bc-rollout) dependency='afterok:<BC_ROLLOUT_JOB_ID>' ;;
        esac
    done
    exit 0
fi

command -v sbatch >/dev/null || fail "sbatch is not available."
command -v snode >/dev/null || fail "snode is required for the resource check."
command -v squeue >/dev/null || fail "squeue is required to inspect existing jobs."
[[ -x $PYTHON_BIN ]] || fail "Missing personal Python: $PYTHON_BIN"
[[ -r $STAGE_SCRIPT ]] || fail "Missing stage script: $STAGE_SCRIPT"
[[ -s $ORIGINAL_MODEL/config.json ]] || fail "Missing pretrained model config: $ORIGINAL_MODEL/config.json"
DATASETS=("$DATA_ROOT/robocasa_mg_gr00t_100")
for task in "${TASKS[@]}"; do
    for kind in demos success_rollouts rollouts; do
        DATASETS+=("$DATA_ROOT/deas_robocasa/$kind/$task")
    done
done
for dataset in "${DATASETS[@]}"; do
    for metadata in info.json modality.json stats.json episodes.jsonl tasks.jsonl; do
        [[ -s $dataset/meta/$metadata ]] || fail "Missing metadata: $dataset/meta/$metadata (prepare it before submission)."
    done
done

# Read resource availability before submitting. Existing own jobs may leave
# this chain pending; this helper does not cancel or change anyone's jobs.
CAPACITY_JSON=$(snode --json) || fail "snode resource check failed."
CURRENT_JOBS=$(squeue --user="$(id -un)" --format="%.18i %.25j %.12q %.12T %.40R") || fail "Existing-job check failed."
printf "Existing jobs for %s:\n%s\n" "$(id -un)" "$CURRENT_JOBS"
printf '%s\n' "$CAPACITY_JSON" | "$PYTHON_BIN" -c '
import getpass, json, sys
snapshot = json.load(sys.stdin)
account = snapshot["accounts"]["sub"]
cap = account["per_user_own_cap"]
gpus = int(sys.argv[1])
request = {"gpu": gpus, "cpu": gpus * 8, "mem_mib": gpus * 192 * 1024}
if any(request[key] > cap[key] for key in request):
    raise SystemExit(f"Requested resources exceed sub own quota: {request}; quota={cap}")
current = account["users"].get(getpass.getuser(), {}).get("own", {})
available = snapshot["cluster"]["avail"]
remaining = {key: max(0, cap[key] - current.get(key, 0)) for key in request}
print(f"Resource check: own cap={cap}; current own={current}; own remaining={remaining}")
account_available = account["avail"]
print(f"Requested={request}; account available={account_available}; cluster available={available}")
if any(request[key] > remaining[key] for key in request):
    print("QUEUE EXPECTED: existing own allocations must release quota before this request can run.")
elif account["avail"]["gpu"] < gpus or available["gpu"] < gpus:
    print("QUEUE EXPECTED: personal quota fits, but account or cluster GPU capacity is currently busy.")
else:
    print("Resource snapshot fits this request; Slurm still decides the actual start time.")
print("Existing jobs are left running. Slurm enforces allocation limits even if usage changes after this check.")
' "$NUM_GPUS"

mkdir -p -- "$REPO_ROOT/output/deas-training"
mkdir -- "$RUN_ROOT" || fail "Run directory already exists or cannot be created: $RUN_ROOT"
mkdir -- "$RUN_ROOT/01-bc-demo" "$RUN_ROOT/02-bc-rollout" "$RUN_ROOT/03-critic" "$RUN_ROOT/logs"
printf '%s\n' "$CAPACITY_JSON" > "$RUN_ROOT/capacity.json"
printf '%s\n' "$CURRENT_JOBS" > "$RUN_ROOT/existing-jobs.txt"
printf 'stage\tjob_id\tdependency\n' > "$RUN_ROOT/jobs.tsv"
{
    printf 'key\tvalue\n'
    printf 'run_id\t%s\nrepo\t%s\n' "$RUN_ID" "$REPO_ROOT"
    printf 'gpus\t%s\nglobal_batch\t%s\nbatch_per_gpu\t%s\nsteps_per_stage\t%s\n' \
        "$NUM_GPUS" "$GLOBAL_BATCH_SIZE" "$BATCH_SIZE" "$MAX_STEPS"
    printf 'time_default\t%s\ntime_bc_demo\t%s\ntime_bc_rollout\t%s\ntime_critic\t%s\n' \
        "$TIME_LIMIT" "$TIME_BC_DEMO" "$TIME_BC_ROLLOUT" "$TIME_CRITIC"
    printf 'cpus_per_gpu\t8\nmem_gib\t%s\n' "$MEM_GIB"
    printf 'seed\t%s\ninitial_dependency\t%s\n' "$SEED" "$INITIAL_DEPENDENCY"
    printf 'account\tsub\nqos\town\npartition\tcompute\n'
    printf 'container\tnvcr.io/nvidia/pytorch:25.04-py3\n'
    printf 'train_env\t%s\ndata_root\t%s\noriginal_model\t%s\n' "$TRAIN_ENV" "$DATA_ROOT" "$ORIGINAL_MODEL"
    printf 'critic_base\t%s\ncritic_horizon\t16\ndiscount1\t0.9\ndiscount2\t0.99\nexpectile\t0.7\n' "$ORIGINAL_MODEL"
    printf 'wandb_project\tgr00t1.5 finetune\nwandb_entity\taiclaudev\nwandb_group\t%s\nwandb_mode\tonline\nwandb_log_model\tfalse\nwandb_watch\tfalse\nwandb_save_code\tfalse\n' "seed-${SEED}-${RUN_ID}"
} > "$RUN_ROOT/arguments.tsv"
{ printf 'bash %q ' "${BASH_SOURCE[0]}"; printf '%q ' "${ORIGINAL_ARGS[@]}"; printf '\n'; } > "$RUN_ROOT/invocation.txt"

SUBMITTED_IDS=()
dependency=$INITIAL_DEPENDENCY
for stage in "${STAGES[@]}"; do
    build_command "$stage" "$dependency"
    print_command >> "$RUN_ROOT/submission-commands.txt"
    if result=$("${COMMAND[@]}" 2> "$RUN_ROOT/logs/submit-$stage.err"); then
        printf '%s\n' "$result" > "$RUN_ROOT/logs/submit-$stage.response"
        job_id=${result%%;*}
        if [[ ! $job_id =~ ^[0-9]+$ ]]; then
            printf 'Unexpected sbatch response for %s: %s\nAlready submitted IDs: %s\nManifest: %s/jobs.tsv\n' \
                "$stage" "$result" "${SUBMITTED_IDS[*]:-none}" "$RUN_ROOT" >&2
            echo "The scheduler may have accepted this job; inspect the response before resubmitting." >&2
            exit 1
        fi
    else
        status=$?
        cat "$RUN_ROOT/logs/submit-$stage.err" >&2
        printf 'Submission failed for %s (exit %s). Already submitted IDs: %s\nManifest: %s/jobs.tsv\n' \
            "$stage" "$status" "${SUBMITTED_IDS[*]:-none}" "$RUN_ROOT" >&2
        echo "No submitted jobs were cancelled." >&2
        exit "$status"
    fi
    SUBMITTED_IDS+=("$job_id")
    printf '%s\t%s\t%s\n' "$stage" "$job_id" "$dependency" >> "$RUN_ROOT/jobs.tsv"
    printf 'Submitted %s: %s\n' "$stage" "$job_id"
    dependency=afterok:$job_id
done
printf 'Pipeline submitted: %s\nJob IDs and arguments: %s/{jobs,arguments}.tsv\n' "${SUBMITTED_IDS[*]}" "$RUN_ROOT"

"$PYTHON_BIN" - "$SEED" "$RUN_ROOT" "${SUBMITTED_IDS[@]}" <<'PY_RESULT'
import json
import sys
from pathlib import Path

seed, root, demo, rollout, critic = sys.argv[1:]
result = {
    "seed": int(seed),
    "run_root": root,
    "job_ids": {"bc_demo": demo, "bc_rollout": rollout, "critic": critic},
    "output_dirs": {
        "bc_demo": str(Path(root) / "01-bc-demo"),
        "bc_rollout": str(Path(root) / "02-bc-rollout"),
        "critic": str(Path(root) / "03-critic"),
    },
}
print("DEAS_SUBMIT_RESULT=" + json.dumps(result, separators=(",", ":")))
PY_RESULT
