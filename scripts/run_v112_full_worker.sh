#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run_v112_full_worker.sh SHARD_INDEX RUN_DIRECTORY" >&2
  exit 2
fi

: "${AGENT_DEFENSE_ROOT:?set AGENT_DEFENSE_ROOT}"
: "${AGENT_DEFENSE_PYTHON:?set AGENT_DEFENSE_PYTHON}"
: "${AGENT_DEFENSE_MODEL:?set AGENT_DEFENSE_MODEL}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one authorized device}"

shard_index="$1"
run_directory="$2"
case "$shard_index" in
  0|1|2|3) ;;
  *)
    echo "SHARD_INDEX must be 0, 1, 2, or 3" >&2
    exit 2
    ;;
esac

cd "$AGENT_DEFENSE_ROOT" || exit 2
mkdir -p "$run_directory/raw" "$run_directory/logs"

worker_failures=0
for suite_name in banking slack travel workspace; do
  manifest="runs/full-v112/manifests/${suite_name}-shard-${shard_index}.json"
  output="$run_directory/raw/${suite_name}-shard-${shard_index}.raw.jsonl"
  resume_args=()
  if [[ -s "$output" ]]; then
    resume_args=(--resume)
  fi
  echo "worker_event=start shard=${shard_index} suite=${suite_name}"
  PYTHONPATH=src "$AGENT_DEFENSE_PYTHON" -m agent_defense matrix-run \
    "$manifest" \
    "$output" \
    --model "$AGENT_DEFENSE_MODEL" \
    --continue-on-error \
    "${resume_args[@]}"
  status=$?
  echo "worker_event=finish shard=${shard_index} suite=${suite_name} status=${status}"
  if [[ $status -ne 0 ]]; then
    worker_failures=$((worker_failures + 1))
  fi
done

echo "worker_event=complete shard=${shard_index} failed_manifests=${worker_failures}"
exit "$worker_failures"
