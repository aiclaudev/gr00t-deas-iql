#!/bin/bash
# Submit independent three-stage seed runs, releasing resources between jobs.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash bash_scripts/submit_deas_seed_sweep.sh --time HH:MM:SS [options]

Submit BC demo -> BC rollout -> critic for each seed (defaults: 42,43,44).
Each seed uses afterok dependencies. The next seed starts afterany the previous
seed's critic, so a failed seed does not prevent later seeds from running.

--seeds 42,43,44          Comma-separated, distinct nonnegative integer seeds.
--gpus 1|2|4              GPUs per stage (default: 4).
--steps N                 Optimizer steps per stage (default: 10000).
--time HH:MM:SS            Default time limit for each stage.
--time-bc-demo HH:MM:SS    Override demo BC time.
--time-bc-rollout HH:MM:SS Override rollout BC time.
--time-critic HH:MM:SS     Override critic time.
--dry-run                 Print the complete dependency chain without side effects.

Other options are forwarded to submit_deas_training.sh. Use --seeds rather than
--seed; --initial-dependency is managed by this wrapper. All jobs use own QOS.
A submission error stops the sweep and reports submitted IDs; it cancels no jobs.
USAGE
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

SEED_LIST=42,43,44
DRY_RUN=0
FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            [[ $# -ge 2 ]] || fail "--seeds requires a comma-separated list."
            SEED_LIST=$2; shift 2 ;;
        --seeds=*) SEED_LIST=${1#*=}; shift ;;
        --seed|--seed=*) fail "Use --seeds for this sweep; --seed is not accepted." ;;
        --initial-dependency|--initial-dependency=*)
            fail "This sweep manages --initial-dependency automatically." ;;
        --dry-run) DRY_RUN=1; FORWARDED_ARGS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) FORWARDED_ARGS+=("$1"); shift ;;
    esac
done

[[ $SEED_LIST =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] \
    || fail "--seeds must contain nonnegative integers without spaces or leading zeros."
IFS=, read -r -a SEEDS <<< "$SEED_LIST"
declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
    [[ ! ${SEEN_SEEDS[$seed]+present} ]] || fail "Duplicate seed: $seed"
    SEEN_SEEDS[$seed]=1
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
HELPER=$REPO_ROOT/bash_scripts/submit_deas_training.sh
[[ -r $HELPER ]] || fail "Missing single-seed submission helper: $HELPER"
if [[ $DRY_RUN -eq 0 ]]; then
    command -v python3 >/dev/null || fail "python3 is required to read submission results."
fi

RESULTS=()
report_submissions() {
    [[ ${#RESULTS[@]} -gt 0 ]] || return 0
    printf '%s\n' "${RESULTS[@]}" | python3 -c '
import json, sys
results = [json.loads(line) for line in sys.stdin if line.strip()]
print("Submitted seed runs:")
for result in results:
    jobs = result["job_ids"]
    print("  seed {}: BC demo={}, BC rollout={}, critic={}".format(
        result["seed"], jobs["bc_demo"], jobs["bc_rollout"], jobs["critic"]))
    print("    Run: " + result["run_root"])
    print("    Jobs: " + result["run_root"] + "/jobs.tsv")
    for stage, path in result["output_dirs"].items():
        print(f"    {stage}: {path}")
print("DEAS_SWEEP_RESULT=" + json.dumps({"runs": results}, separators=(",", ":")))
'
}

printf 'Seed sweep: %s | %s jobs | own QOS | seeds run in sequence\n' \
    "$SEED_LIST" "$((${#SEEDS[@]} * 3))"
previous_critic=
previous_seed=
for seed in "${SEEDS[@]}"; do
    command_args=(bash "$HELPER" "${FORWARDED_ARGS[@]}" --seed "$seed")
    if [[ -n $previous_critic ]]; then
        command_args+=(--initial-dependency "afterany:$previous_critic")
        printf '\nSeed %s waits for seed %s critic to finish (any outcome).\n' "$seed" "$previous_seed"
    else
        printf '\nSeed %s starts the sweep.\n' "$seed"
    fi
    if helper_output=$("${command_args[@]}"); then
        helper_status=0
    else
        helper_status=$?
    fi
    printf '%s\n' "$helper_output"
    if [[ $helper_status -ne 0 ]]; then
        printf 'Sweep stopped at seed %s (submission exit %s).\n' "$seed" "$helper_status" >&2
        report_submissions >&2
        printf 'Any partial submissions for this seed are listed in the helper output above. No jobs were cancelled.\n' >&2
        exit "$helper_status"
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        previous_critic="<SEED_${seed}_CRITIC_JOB_ID>"
        previous_seed=$seed
        continue
    fi

    result_json=
    result_count=0
    while IFS= read -r line; do
        case "$line" in
            DEAS_SUBMIT_RESULT=*)
                result_json=${line#DEAS_SUBMIT_RESULT=}
                result_count=$((result_count + 1)) ;;
        esac
    done <<< "$helper_output"
    if [[ $result_count -ne 1 ]]; then
        printf 'Missing or ambiguous submission result for seed %s; stopping before the next seed.\n' "$seed" >&2
        report_submissions >&2
        printf 'Inspect the helper output above for accepted job IDs before retrying. No jobs were cancelled.\n' >&2
        exit 1
    fi
    if ! previous_critic=$(printf '%s\n' "$result_json" | python3 -c '
import json, re, sys
try:
    result = json.load(sys.stdin)
    if int(result["seed"]) != int(sys.argv[1]):
        raise ValueError("seed does not match the requested run")
    for stage in ("bc_demo", "bc_rollout", "critic"):
        if not re.fullmatch(r"[0-9]+", str(result["job_ids"][stage])):
            raise ValueError("invalid job ID for " + stage)
        if not isinstance(result["output_dirs"][stage], str) or not result["output_dirs"][stage]:
            raise ValueError("missing output directory for " + stage)
    if not isinstance(result["run_root"], str) or not result["run_root"]:
        raise ValueError("missing run directory")
    print(result["job_ids"]["critic"])
except (KeyError, TypeError, ValueError) as exc:
    print("Invalid submission result: " + str(exc), file=sys.stderr)
    sys.exit(1)
' "$seed"); then
        report_submissions >&2
        printf 'Sweep stopped after seed %s; inspect its helper output before retrying. No jobs were cancelled.\n' "$seed" >&2
        exit 1
    fi
    RESULTS+=("$result_json")
    previous_seed=$seed
done

if [[ $DRY_RUN -eq 1 ]]; then
    printf '\nDry run complete: no jobs submitted and no files created.\n'
else
    printf '\n'
    report_submissions
fi
