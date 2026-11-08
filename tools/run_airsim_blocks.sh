#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
installation_dir="/opt/airsim"
renderer="${AIRSIM_RENDERER:-software}"
mapfile -t launchers < <(find "$installation_dir/Blocks-v1.8.1" -type f -name Blocks.sh)
if [[ ${#launchers[@]} -ne 1 ]]; then
    echo "Expected one installed Blocks.sh under $installation_dir/Blocks-v1.8.1" >&2
    exit 1
fi
if [[ "$(id -u)" -ne 0 ]]; then
    echo "This container launcher must run as root; Unreal runs as the airsim service user." >&2
    exit 1
fi
install -m 644 -o airsim -g airsim "$workspace_dir/configs/airsim_blocks_cv.json" \
    "$installation_dir/home/Documents/AirSim/settings.json"

render_environment=()
render_arguments=()
case "$renderer" in
    software)
        render_environment+=(VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json LP_NUM_THREADS=8)
        render_arguments+=(-AllowSoftwareRendering)
        ;;
    gpu)
        ;;
    *)
        echo "AIRSIM_RENDERER must be software or gpu" >&2
        exit 1
        ;;
esac

ulimit -c 0
exec runuser -u airsim -- env \
    XDG_RUNTIME_DIR="$installation_dir/runtime" \
    "${render_environment[@]}" \
    xvfb-run -a -s "-screen 0 640x480x24" \
    "${launchers[0]}" -vulkan "${render_arguments[@]}" -windowed \
    -ResX=320 -ResY=240 -nosound -unattended -NoSplash \
    -settings="$installation_dir/home/Documents/AirSim/settings.json" "$@"
