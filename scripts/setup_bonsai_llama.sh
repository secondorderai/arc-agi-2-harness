#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="${project_root}/.runtime/Bonsai-demo"
demo_commit="70ae3e442c3e2f8c928b8dbe18b618bbd3365f62"

if [[ ! -d "${runtime_dir}/.git" ]]; then
  mkdir -p "$(dirname "${runtime_dir}")"
  git clone https://github.com/PrismML-Eng/Bonsai-demo.git "${runtime_dir}"
fi

cd "${runtime_dir}"
git checkout --detach "${demo_commit}"
export BONSAI_FAMILY=bonsai
export BONSAI_MODEL=27B
export BONSAI_SKIP_MLX=1
export BONSAI_OPENWEBUI=0
export BONSAI_CODE_INTERPRETER=0
./setup.sh

echo "Bonsai llama.cpp fallback installed at ${runtime_dir}"
