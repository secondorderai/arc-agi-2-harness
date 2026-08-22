#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="${project_root}/.runtime/Bonsai-demo"

if [[ ! -x "${runtime_dir}/scripts/start_mlx_server.sh" ]]; then
  echo "Run scripts/setup_bonsai_mlx.sh first." >&2
  exit 1
fi

cd "${runtime_dir}"
export BONSAI_FAMILY=bonsai
export BONSAI_MODEL=27B
export BONSAI_CTX=8192
exec ./scripts/start_mlx_server.sh

