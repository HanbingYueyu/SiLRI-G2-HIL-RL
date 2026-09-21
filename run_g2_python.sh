#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${G2_GDK_RUNTIME:-/home/flyfuture/.cache/agibot/app}"
source "${runtime_root}/env.sh" "${runtime_root}"
export PYTHONPATH="${project_root}/lerobot/src:${project_root}:${runtime_root}/gdk/lib"
mkdir -p "${project_root}/runtime/gdk_logs"
export GLOG_log_dir="${project_root}/runtime/gdk_logs"
cd "${project_root}"
exec "${project_root}/.venv/bin/python" "$@"
