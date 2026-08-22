#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
data_dir="${project_root}/data/ARC-AGI-2"

if [[ -d "${data_dir}/data/training" ]]; then
  echo "ARC-AGI-2 data already exists at ${data_dir}"
  exit 0
fi

mkdir -p "$(dirname "${data_dir}")"
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git "${data_dir}"
echo "ARC-AGI-2 data downloaded to ${data_dir}"

