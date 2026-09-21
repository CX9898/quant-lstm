#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python3}"
cache_root="${QUANT_LSTM_SPEECH_COMMANDS_CACHE:-${HOME}/.cache/quant-lstm/speech_commands_v0.02}"
dataset_root="${QUANT_LSTM_SPEECH_COMMANDS_ROOT:-}"
download=0

usage() {
  cat <<'EOF'
Usage: tests/real_network/run_speech_commands_lstm_test.sh [options]

Run matched torch.nn.LSTM, 8-bit QuantLSTM QAT, and 16-bit QuantLSTM QAT
training on a deterministic subset of Google's Speech Commands v0.02 dataset.

Options:
  --dataset-root PATH  Extracted dataset containing validation_list.txt
  --cache-root PATH    Download/extraction cache root
  --download           Download the official 2.3 GiB archive when absent
  -h, --help           Show this help

Environment:
  PYTHON                              Python executable (default: python3)
  QUANT_LSTM_SPEECH_COMMANDS_ROOT     Same as --dataset-root
  QUANT_LSTM_SPEECH_COMMANDS_CACHE    Same as --cache-root
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      [[ "$#" -ge 2 ]] || die "--dataset-root requires a value"
      dataset_root="$2"
      shift 2
      ;;
    --cache-root)
      [[ "$#" -ge 2 ]] || die "--cache-root requires a value"
      cache_root="$2"
      shift 2
      ;;
    --download)
      download=1
      shift
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

command -v "${python_bin}" >/dev/null 2>&1 || die "Python not found: ${python_bin}"

export PYTHONPATH="${root_dir}/pytorch:${root_dir}/tests/real_network${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${dataset_root}" ]]; then
  prepare_args=(
    "${root_dir}/tests/real_network/speech_commands_lstm_training.py"
    --cache-root "${cache_root}"
    --prepare-only
  )
  if [[ "${download}" -eq 1 ]]; then
    prepare_args+=(--download)
  fi
  dataset_root="$(${python_bin} "${prepare_args[@]}")"
fi

[[ -f "${dataset_root}/validation_list.txt" ]] || \
  die "dataset root does not contain validation_list.txt: ${dataset_root}"

"${python_bin}" - <<'PY'
from importlib import import_module

for name in ("torch", "torchaudio", "_quant_lstm"):
    import_module(name)
PY

export QUANT_LSTM_SPEECH_COMMANDS_ROOT="${dataset_root}"
"${python_bin}" "${root_dir}/tests/real_network/test_speech_commands_lstm_training.py"
