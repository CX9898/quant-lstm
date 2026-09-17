#!/usr/bin/env bash
set -euo pipefail

# 只处理入库的 C/C++/CUDA 源文件，避免格式化生成目录。
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root_dir}"
mapfile -t files < <(
  git ls-files '*.cc' '*.cu' '*.cuh' '*.h'
)

if [[ "${#files[@]}" -eq 0 ]]; then
  exit 0
fi
if [[ "${1:---write}" == "--check" ]]; then
  clang-format --dry-run --Werror "${files[@]}"
else
  clang-format -i "${files[@]}"
fi
