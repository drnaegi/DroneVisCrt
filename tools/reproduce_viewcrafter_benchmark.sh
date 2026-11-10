#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$workspace"
output=${1:-outputs/viewcrafter_benchmark_20261026}
dataset=${2:-datasets/viewcrafter_tnt_subset_20261026}
python_bin=envs/viewcrafter/bin/python
mkdir -p "$output/logs"

"$python_bin" tools/prepare_viewcrafter_benchmark.py --output "$dataset"
for clip in train_stride1 train_stride3 truck_stride1 truck_stride3; do
    if [[ ! -f "$output/$clip/geometry/run.json" ]]; then
        "$python_bin" tools/run_viewcrafter_benchmark.py \
            --images "$dataset/clips/$clip" --output "$output/$clip" --phase geometry \
            > "$output/logs/${clip}_geometry.log" 2>&1
    fi
    for length in 16 25; do
        if [[ ! -f "$output/$clip/viewcrafter$length/run.json" ]]; then
            "$python_bin" tools/run_viewcrafter_benchmark.py \
                --images "$dataset/clips/$clip" --output "$output/$clip" --phase "$length" \
                > "$output/logs/${clip}_${length}.log" 2>&1
        fi
    done
done
"$python_bin" tools/evaluate_viewcrafter_benchmark.py --root "$output" \
    > "$output/logs/evaluate.log" 2>&1
"$python_bin" tools/validate_viewcrafter_benchmark.py --root "$output" \
    > "$output/logs/validate.log" 2>&1
