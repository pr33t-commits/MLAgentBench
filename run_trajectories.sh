#!/usr/bin/env bash
# Run independent trajectories sequentially against an already running model server.
# Usage: MODEL=gpt-qwen-local bash run_trajectories.sh [trajectory_count]
set -uo pipefail

N="${1:-5}"
MODEL="${MODEL:-gpt-qwen-local}"
PYTHON="$(command -v python)" || exit 1

[[ "$N" =~ ^[1-9][0-9]*$ ]] || {
  echo "Trajectory count must be a positive integer." >&2
  exit 1
}

[[ -f MLAgentBench/runner.py ]] || {
  echo "Run this script from the MLAgentBench repository root." >&2
  exit 1
}

SUFFIX="$("$PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:8])')" || exit 1
BATCH_ID="demand-forecasting-$(date -u +%Y%m%dT%H%M%SZ)-$SUFFIX"
LOG_ROOT="$PWD/runs/$BATCH_ID"
WORK_ROOT="$PWD/workspaces/$BATCH_ID"
mkdir -p "$PWD/runs" "$PWD/workspaces" || exit 1
mkdir "$LOG_ROOT" "$WORK_ROOT" || exit 1

# Identical settings for every trajectory; only output directories change.
CONFIG=(
  --python "$PYTHON"
  --task demand-forecasting
  --agent-type ResearchAgent
  --device 0
  --llm-name "$MODEL"
  --edit-script-llm-name "$MODEL"
  --fast-llm-name "$MODEL"
  --max-steps 50
  --agent-max-steps 35
  --max-retries 3
  --max-time 3600
  --edit-script-llm-max-tokens 8192
  --retrieval
  --max-steps-in-context 5
  --max-observation-steps-in-context 5
)

# Record configuration, source version and local edits without copying secrets.
printf '%q ' "${CONFIG[@]}" > "$LOG_ROOT/config.args" || exit 1
printf '\n' >> "$LOG_ROOT/config.args"
git rev-parse HEAD > "$LOG_ROOT/git_commit.txt" || exit 1
git diff HEAD > "$LOG_ROOT/local_changes.patch" || exit 1
"$PYTHON" -m pip freeze > "$LOG_ROOT/packages.txt" || exit 1
printf 'trajectory_id\tstarted_utc\tfinished_utc\texit_code\n' \
  > "$LOG_ROOT/summary.tsv" || exit 1

echo "Batch: $BATCH_ID"
echo "Logs:  $LOG_ROOT"

ANY_FAILED=0
trap 'echo "Batch interrupted."; exit 130' INT TERM

for ((i=1; i<=N; i++)); do
  TRAJECTORY_ID="$(printf 'trajectory_%03d' "$i")"
  LOG_DIR="$LOG_ROOT/$TRAJECTORY_ID"
  WORK_DIR="$WORK_ROOT/$TRAJECTORY_ID"
  mkdir "$LOG_DIR" "$WORK_DIR" || exit 1

  STARTED="$(date -u +%FT%TZ)"
  echo "[$i/$N] Starting $TRAJECTORY_ID"
  COMMAND=(
    "$PYTHON" -u -m MLAgentBench.runner
    "${CONFIG[@]}"
    --log-dir "$LOG_DIR"
    --work-dir "$WORK_DIR"
  )
  printf '%q ' "${COMMAND[@]}" > "$LOG_DIR/command.sh" || exit 1
  printf '\n' >> "$LOG_DIR/command.sh"

  "${COMMAND[@]}" > "$LOG_DIR/console.log" 2>&1
  CODE=$?

  FINISHED="$(date -u +%FT%TZ)"
  printf '%s\t%s\t%s\t%s\n' \
    "$TRAJECTORY_ID" "$STARTED" "$FINISHED" "$CODE" \
    >> "$LOG_ROOT/summary.tsv" || exit 1
  echo "[$i/$N] Finished $TRAJECTORY_ID: exit code $CODE"

  if ((CODE == 130 || CODE == 143)); then
    exit "$CODE"
  fi
  if ((CODE != 0)); then
    ANY_FAILED=1
  fi
done

echo "Completed. Summary: $LOG_ROOT/summary.tsv"
exit "$ANY_FAILED"
