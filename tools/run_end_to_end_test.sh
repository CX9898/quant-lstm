#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build"
build_type="${QUANT_LSTM_BUILD_TYPE:-Release}"
python_bin="${PYTHON:-python3}"
run_cuda_validation=0
device=0

default_jobs=2
if command -v nproc >/dev/null 2>&1; then
  default_jobs="$(nproc)"
fi
jobs="${QUANT_LSTM_BUILD_JOBS:-${default_jobs}}"

usage() {
  cat <<'EOF'
Usage: tools/run_end_to_end_test.sh [options]

Build and test the native CUDA library and both PyTorch extensions.

Options:
  --build-type TYPE          CMake build type (default: Release)
  --jobs COUNT              Parallel build jobs (default: nproc)
  --with-cuda-validation    Run Stage 4 profiling and Stage 9 thresholds
  --device INDEX            GPU index for optional CUDA validation (default: 0)
  -h, --help                Show this help

Environment:
  PYTHON                    Python executable (default: python3)
  QUANT_LSTM_BUILD_TYPE     Default value for --build-type
  QUANT_LSTM_BUILD_JOBS     Default value for --jobs
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

run_step() {
  local description="$1"
  shift
  printf '\n==> %s\n' "${description}"
  "$@"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --build-type)
      [[ "$#" -ge 2 ]] || die "--build-type requires a value"
      build_type="$2"
      shift 2
      ;;
    --jobs)
      [[ "$#" -ge 2 ]] || die "--jobs requires a value"
      jobs="$2"
      shift 2
      ;;
    --with-cuda-validation)
      run_cuda_validation=1
      shift
      ;;
    --device)
      [[ "$#" -ge 2 ]] || die "--device requires a value"
      device="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "${build_type}" ]] || die "build type must not be empty"
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || die "jobs must be a positive integer"
[[ "${device}" =~ ^[0-9]+$ ]] || die "device must be a non-negative integer"

for command in cmake nvcc "${python_bin}"; do
  command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
done

run_step "Check Python and CUDA dependencies" "${python_bin}" - <<'PY'
from importlib import import_module

required = ("jsonschema", "onnx", "onnxruntime", "referencing", "torch")
missing = []
for name in required:
    try:
        import_module(name)
    except ModuleNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit("missing Python dependencies: " + ", ".join(missing))

import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access a CUDA device")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

run_step "Configure CUDA build" \
  cmake --fresh -S "${root_dir}" -B "${build_dir}" \
    -DQUANT_LSTM_ENABLE_CUDA=ON \
    -DQUANT_LSTM_BUILD_TESTS=ON \
    -DQUANT_LSTM_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE="${build_type}"

run_step "Build native library and tests" \
  cmake --build "${build_dir}" --parallel "${jobs}"

run_step "Run native CTest suite" \
  ctest --test-dir "${build_dir}" --output-on-failure

printf '\n==> Build PyTorch extensions\n'
(
  cd "${root_dir}/pytorch"
  "${python_bin}" setup.py build_ext --inplace --force
  "${python_bin}" tests/setup_test_extension.py build_ext --inplace --force
)

export PYTHONPATH="${root_dir}/pytorch${PYTHONPATH:+:${PYTHONPATH}}"

printf '\n==> Run PyTorch functional and strict suites\n'
(
  cd "${root_dir}/pytorch"
  QUANT_LSTM_TEST_SUITE=basic \
    "${python_bin}" tests/test_float_reference.py
  QUANT_LSTM_TEST_SUITE=strict \
    "${python_bin}" tests/test_float_reference.py
  "${python_bin}" -m unittest -v \
    tests.test_quantized_interface \
    tests.test_bidirectional_interface \
    tests.test_backward \
    tests.test_onnx_export
)

if [[ "${run_cuda_validation}" -eq 1 ]]; then
  artifact_root="${build_dir}/stage4-validation-results"
  marker="$(mktemp "${build_dir}/.stage4-e2e-marker.XXXXXX")"
  cleanup_marker() {
    rm -f "${marker}"
  }
  trap cleanup_marker EXIT

  run_step "Run Stage 4 CUDA profiling validation" \
    "${python_bin}" "${root_dir}/tools/run_stage4_cuda_validation.py" \
      --build-dir "${build_dir}" \
      --device "${device}" \
      --artifacts-root build

  mapfile -d '' validation_reports < <(
    find "${artifact_root}" -mindepth 2 -maxdepth 2 -type f \
      -name stage4_cuda_validation_report.json \
      -newer "${marker}" -print0
  )
  [[ "${#validation_reports[@]}" -eq 1 ]] || \
    die "expected one new Stage 4 report, found ${#validation_reports[@]}"

  benchmark_report="$(
    "${python_bin}" - "${validation_reports[0]}" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(report["artifacts"]["benchmark_report"])
PY
  )"
  run_step "Run Stage 9 CUDA performance thresholds" \
    "${python_bin}" "${root_dir}/tools/check_stage9_cuda_performance.py" \
      --report "${benchmark_report}"

  cleanup_marker
  trap - EXIT
fi

printf '\nQuantLSTM end-to-end tests passed.\n'
