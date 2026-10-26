#!/usr/bin/env bash
# Same CLI as the SLURM scripts, without sbatch. For Colab / a rented GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_local.sh <step> [args...]" >&2
  echo "  step: prepare | asr | merge | vote | confidence | cer | markers | calibration | stats | report | all" >&2
  exit 1
fi

STEP="$1"
shift

run_step() {
  echo "+ python -m wscaudit $*"
  python -m wscaudit "$@"
}

case "$STEP" in
  all)
    run_step prepare
    run_step asr --model sensevoice "$@"
    run_step asr --model firered "$@"
    run_step asr --model qwen3asr "$@"
    run_step merge --pipeline P0
    run_step vote --pipeline P0-vote
    run_step confidence --pipeline P0
    run_step cer
    run_step markers
    run_step calibration
    run_step stats
    run_step report --env
    ;;
  *)
    run_step "$STEP" "$@"
    ;;
esac
