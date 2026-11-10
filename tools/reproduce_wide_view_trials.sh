#!/usr/bin/env bash
set -euo pipefail

TRIAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRIAL_ROOT"
TRIAL_PHASE="${1:-all}"
TRIAL_OUTPUT="${2:-outputs/wide_view_trials_20261023}"
TRIAL_PYTHON="$TRIAL_ROOT/envs/viewcrafter/bin/python"
mkdir -p "$TRIAL_OUTPUT/logs"

for TRIAL_SCENE in parking buildings; do
  if [[ "$TRIAL_SCENE" == parking ]]; then
    TRIAL_IMAGE="outputs/open_source_trials_20261023/inputs/0000002_00005_d_0000014.jpg"
  else
    TRIAL_IMAGE="outputs/open_source_trials_20261023/inputs/0000001_02999_d_0000005.jpg"
  fi
  for TRIAL_MOTION in phi30 theta_minus20; do
    if [[ "$TRIAL_MOTION" == phi30 ]]; then
      TRIAL_PHI=30
      TRIAL_THETA=0
    else
      TRIAL_PHI=0
      TRIAL_THETA=-20
    fi
    "$TRIAL_PYTHON" -u tools/run_viewcrafter_trial.py \
      --image "$TRIAL_IMAGE" --output "$TRIAL_OUTPUT/${TRIAL_SCENE}_${TRIAL_MOTION}" \
      --phi "$TRIAL_PHI" --theta "$TRIAL_THETA" --elevation 35 \
      --seed 123 --steps 50 --phase "$TRIAL_PHASE" \
      > "$TRIAL_OUTPUT/logs/${TRIAL_SCENE}_${TRIAL_MOTION}_${TRIAL_PHASE}.log" 2>&1
  done
done
