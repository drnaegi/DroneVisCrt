#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"

TRIAL_OUTPUT="${1:-outputs/open_source_trials_20261023}"
TRIAL_INPUT="outputs/open_source_trials_20261023/inputs"
TRIAL_PYTHON="$TASK_ROOT/envs/nvs_trials/bin/python"
export PATH="$TASK_ROOT/envs/nvs_trials/bin:$PATH"
export MAX_JOBS=6
export TORCH_CUDA_ARCH_LIST=8.9
mkdir -p "$TRIAL_OUTPUT/logs"

"$TRIAL_PYTHON" -u tools/run_nvs_trials.py \
  --method sharp --input "$TRIAL_INPUT" --output "$TRIAL_OUTPUT/sharp" \
  --checkpoint checkpoints/SHARP/sharp_2572gikvuh.pt \
  > "$TRIAL_OUTPUT/logs/sharp_all.log" 2>&1

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
"$TRIAL_PYTHON" -u tools/run_nvs_trials.py \
  --method moge2 --input "$TRIAL_INPUT" --output "$TRIAL_OUTPUT/moge2" \
  --checkpoint checkpoints/MoGe2/moge-2-vitb-normal.pt \
  > "$TRIAL_OUTPUT/logs/moge2_all.log" 2>&1

"$TRIAL_PYTHON" tools/summarize_nvs_trials.py --root "$TRIAL_OUTPUT"
