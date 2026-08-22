#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="${project_root}/.runtime/Bonsai-demo"

if [[ ! -x "${runtime_dir}/scripts/start_llama_server.sh" ]]; then
  echo "Run scripts/setup_bonsai_llama.sh first." >&2
  exit 1
fi

cd "${runtime_dir}"
export BONSAI_FAMILY=bonsai
export BONSAI_MODEL=27B
export BONSAI_CTX=8192
export BONSAI_SPECULATIVE=0
exec ./scripts/start_llama_server.sh \
  --reasoning-budget "${BONSAI_REASONING_BUDGET:-512}" \
  --parallel 1 \
  --cache-ram 1024
