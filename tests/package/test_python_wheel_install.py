"""验证 quant-lstm wheel 可以脱离源码树安装和导入。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str],
) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args()
    source_root = arguments.source_root.resolve()
    environment = _isolated_environment()

    with tempfile.TemporaryDirectory(prefix="quant-lstm-wheel-test-") as temporary:
        temporary_root = Path(temporary)
        wheel_dir = temporary_root / "wheel"
        environment_dir = temporary_root / "venv"
        run_dir = temporary_root / "run"
        wheel_dir.mkdir()
        run_dir.mkdir()

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
                str(source_root / "pytorch"),
            ],
            environment=environment,
        )
        wheels = list(wheel_dir.glob("quant_lstm-*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"expected one quant-lstm wheel, found {wheels}")

        _run(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                str(environment_dir),
            ],
            environment=environment,
        )
        environment_python = environment_dir / "bin" / "python"
        _run(
            [
                str(environment_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                str(wheels[0]),
            ],
            environment=environment,
        )

        check = """
from pathlib import Path
import sys

import quant_lstm
import torch
from quant_lstm import QuantLSTM

source_root = Path(sys.argv[1]).resolve()
installed_module = Path(quant_lstm.__file__).resolve()
if installed_module.is_relative_to(source_root):
    raise AssertionError(f"quant_lstm imported from source tree: {installed_module}")

if not torch.cuda.is_available():
    raise AssertionError("standalone wheel test requires an available CUDA device")

module = QuantLSTM(2, 3, batch_first=True, device="cuda").eval()
config = module.get_quant_config()
if config["schema_version"] != 1:
    raise AssertionError("installed default config has the wrong schema version")
if config["operators"]["input"]["bitwidth"] != 8:
    raise AssertionError("installed default config has the wrong input bitwidth")
inputs = torch.randn(2, 4, 2, device="cuda")
output, (hidden, cell) = module(inputs)
if output.shape != (2, 4, 3):
    raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
if hidden.shape != (1, 2, 3) or cell.shape != (1, 2, 3):
    raise AssertionError("unexpected final-state shape")
print(f"standalone quant-lstm import passed: {installed_module}")
"""
        subprocess.run(
            [str(environment_python), "-I", "-c", check, str(source_root)],
            cwd=run_dir,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
