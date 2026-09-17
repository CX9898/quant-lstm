#!/usr/bin/env bash
set -euo pipefail

# 在独立构建树中验证 CPU-only 编译、测试、安装和外部消费。
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${root_dir}/build-stage7-cpu-package}"
install_dir="${build_dir}/install"
build_type="${2:-Release}"
consumer_build_dir="${build_dir}/consumer"

cmake -S "${root_dir}" -B "${build_dir}" \
  -DQUANT_LSTM_ENABLE_CUDA=OFF \
  -DQUANT_LSTM_BUILD_TESTS=ON \
  -DQUANT_LSTM_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE="${build_type}"
cmake --build "${build_dir}" -j2
ctest --test-dir "${build_dir}" --output-on-failure
cmake --install "${build_dir}" --prefix "${install_dir}"

cmake -S "${root_dir}/tests/package/consumer" \
  -B "${consumer_build_dir}" \
  -DCMAKE_PREFIX_PATH="${install_dir}" \
  -DCMAKE_BUILD_TYPE="${build_type}"
cmake --build "${consumer_build_dir}" -j2

"${install_dir}/bin/lstm_int32_example"
"${consumer_build_dir}/quant_lstm_cpu_consumer"

if ldd "${consumer_build_dir}/quant_lstm_cpu_consumer" | \
    grep -Eiq 'cuda|cublas|cudart'; then
  echo "CPU-only consumer 意外链接 CUDA 动态库" >&2
  exit 1
fi
