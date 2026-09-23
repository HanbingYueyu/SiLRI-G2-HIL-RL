#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-m" && "${2:-}" == "g2_local.real_train" ]]; then
    python_bin="${project_root}/.venv/bin/python"
    lerobot_src="${project_root}/lerobot/src"
    if [[ ! -x "${python_bin}" && -x "${project_root}/../../.venv/bin/python" ]]; then
        python_bin="${project_root}/../../.venv/bin/python"
    fi
    if [[ ! -d "${lerobot_src}/lerobot/configs" && -d "${project_root}/../../lerobot/src/lerobot/configs" ]]; then
        lerobot_src="${project_root}/../../lerobot/src"
    fi
    export PYTHONPATH="${lerobot_src}:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
    cd "${project_root}"
    exec "${python_bin}" "$@"
fi
runtime_root="${G2_GDK_RUNTIME:-/home/flyfuture/.cache/agibot/app}"
source "${runtime_root}/env.sh" "${runtime_root}"
export PYTHONPATH="${project_root}/lerobot/src:${project_root}:${runtime_root}/gdk/lib"
mkdir -p "${project_root}/runtime/gdk_logs"
export GLOG_log_dir="${project_root}/runtime/gdk_logs"
cd "${project_root}"
exec "${project_root}/.venv/bin/python" "$@"
