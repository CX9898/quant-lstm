"""单层 LSTM 的浮点与 CUDA FP32 q-carrier PyTorch 接口。"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

try:
    import _quant_lstm
except ImportError as exc:
    raise ImportError(
        "_quant_lstm 扩展未找到；请先构建 CMake 核心并运行 "
        "python setup.py build_ext --inplace"
    ) from exc

from lstm_autograd import float_lstm, quantized_lstm
from lstm_onnx import (
    ensure_quant_lstm_onnx_registered,
    onnx_lstm,
)


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "defaults"
    / "lstm_quant_default_v1.json"
)
_PARAMETER_OPERATORS = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
_CALIBRATION_METHODS = {"minmax", "sqnr", "percentile"}
_MATH_MODES = {"pedantic", "tf32"}


def _strict_json_loads(text: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 包含重复 key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("JSON 根必须是 object")
    return value


def _json_source(value: dict[str, Any] | str | Path | None) -> str:
    if value is None:
        return '{"schema_version":1}'
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    stripped = value.lstrip()
    if stripped.startswith("{"):
        return value
    return Path(value).read_text(encoding="utf-8")


def _resolved_override(resolved: dict[str, Any]) -> dict[str, Any]:
    operators = {}
    for name, config in resolved["operators"].items():
        fields = {
            "bitwidth": config["bitwidth"],
            "is_unsigned": config["is_unsigned"],
            "is_symmetric": config["is_symmetric"],
        }
        if name in _PARAMETER_OPERATORS:
            fields["granularity"] = config["granularity"]
        operators[name] = fields
    return {
        "schema_version": 1,
        "scale_mode": resolved["scale_mode"],
        "operators": operators,
    }


class _UnidirectionalQuantLSTM(nn.Module):
    """与单层单向 nn.LSTM 对齐的双模式模块。

    use_quantization=False 使用浮点路径。校准并设置
    use_quantization=True 后，唯一量化后端是 CUDA FP32 q-carrier。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        device=None,
        dtype=None,
        use_quantization: bool = False,
        quant_config: dict[str, Any] | str | Path | None = None,
        calibration_method: str = "minmax",
        cublas_math_mode: str = "pedantic",
        require_exact_accumulation: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size 和 hidden_size 必须为正数")
        if num_layers != 1:
            raise ValueError("当前仅支持 num_layers=1")
        if dropout != 0.0:
            raise ValueError("当前仅支持 dropout=0")
        if bidirectional:
            raise ValueError("阶段 6 尚不支持 bidirectional=True")
        if dtype not in (None, torch.float32):
            raise ValueError("当前仅支持 torch.float32")
        if calibration_method not in _CALIBRATION_METHODS:
            raise ValueError(
                "calibration_method 必须是 minmax、sqnr 或 percentile"
            )
        if cublas_math_mode not in _MATH_MODES:
            raise ValueError("cublas_math_mode 必须是 pedantic 或 tf32")
        if not _DEFAULT_CONFIG_PATH.is_file():
            raise RuntimeError(f"找不到默认量化配置: {_DEFAULT_CONFIG_PATH}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_quantization = bool(use_quantization)
        self.calibrating = False
        self.require_exact_accumulation = bool(require_exact_accumulation)
        self._calibration_method = calibration_method
        self._cublas_math_mode = cublas_math_mode

        factory_kwargs = {
            "device": device,
            "dtype": torch.float32 if dtype is None else dtype,
        }
        self.weight_ih_l0 = nn.Parameter(
            torch.empty((4 * hidden_size, input_size), **factory_kwargs)
        )
        self.weight_hh_l0 = nn.Parameter(
            torch.empty((4 * hidden_size, hidden_size), **factory_kwargs)
        )
        if bias:
            self.bias_ih_l0 = nn.Parameter(
                torch.empty(4 * hidden_size, **factory_kwargs)
            )
            self.bias_hh_l0 = nn.Parameter(
                torch.empty(4 * hidden_size, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_ih_l0", None)
            self.register_parameter("bias_hh_l0", None)
        self.reset_parameters()

        defaults = _DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        override_json = _json_source(quant_config)
        self._resolved_config_json = _quant_lstm.resolve_quant_config(
            defaults, override_json
        )
        self._override_config = _resolved_override(
            _strict_json_loads(self._resolved_config_json)
        )
        self._calibration_session = None
        self._quant_params_bundle_json: Optional[str] = None
        self._last_safety_report: Optional[dict[str, Any]] = None
        self._qat_saved_state: Optional[dict[str, Any]] = None

    @property
    def calibration_method(self) -> str:
        return self._calibration_method

    @calibration_method.setter
    def calibration_method(self, value: str) -> None:
        if value not in _CALIBRATION_METHODS:
            raise ValueError(
                "calibration_method 必须是 minmax、sqnr 或 percentile"
            )
        if getattr(self, "_calibration_method", value) != value:
            self.reset_calibration()
        self._calibration_method = value

    @property
    def cublas_math_mode(self) -> str:
        return self._cublas_math_mode

    @cublas_math_mode.setter
    def cublas_math_mode(self, value: str) -> None:
        if value not in _MATH_MODES:
            raise ValueError("cublas_math_mode 必须是 pedantic 或 tf32")
        self._cublas_math_mode = value

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.uniform_(-bound, bound)

    def _invalidate_quant_params(self) -> None:
        self._calibration_session = None
        self._quant_params_bundle_json = None
        self._last_safety_report = None
        self._qat_saved_state = None

    def _apply_override(self, override: dict[str, Any]) -> None:
        defaults = _DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        encoded = json.dumps(override, separators=(",", ":"), ensure_ascii=False)
        self._resolved_config_json = _quant_lstm.resolve_quant_config(
            defaults, encoded
        )
        self._override_config = _resolved_override(
            _strict_json_loads(self._resolved_config_json)
        )
        self._invalidate_quant_params()

    def get_quant_config(self, operator: Optional[str] = None) -> dict[str, Any]:
        """返回 C++ resolver 产生的完整 canonical resolved config。"""
        resolved = _strict_json_loads(self._resolved_config_json)
        if operator is None:
            return resolved
        if operator not in resolved["operators"]:
            raise ValueError(f"未知量化点: {operator}")
        return json.loads(json.dumps(resolved["operators"][operator]))

    def adjust_quant_config(self, operator: str, **changes: Any) -> None:
        """稀疏修改一个真实量化点，并立即通过 C++ resolver 校验。"""
        allowed = {
            "bitwidth",
            "is_unsigned",
            "is_symmetric",
            "granularity",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"未知配置字段: {sorted(unknown)}")
        if not changes:
            raise ValueError("至少需要提供一个配置字段")
        resolved = self.get_quant_config()
        if operator not in resolved["operators"]:
            raise ValueError(f"未知量化点: {operator}")
        override = _resolved_override(resolved)
        override["operators"][operator].update(changes)
        self._apply_override(override)

    def set_all_bitwidth(self, bitwidth: int = 8) -> None:
        """一次设置全部 18 个真实量化点的位宽。"""
        resolved = self.get_quant_config()
        override = _resolved_override(resolved)
        for operator in override["operators"].values():
            operator["bitwidth"] = bitwidth
        self._apply_override(override)

    def _ensure_calibration_session(self):
        if self._calibration_session is None:
            self._calibration_session = _quant_lstm.CalibrationSession(
                self._resolved_config_json,
                self.input_size,
                self.hidden_size,
                self.bias,
                self.calibration_method,
            )
        return self._calibration_session

    @staticmethod
    def _cpu_tensor(value: Optional[Tensor]) -> Optional[Tensor]:
        if value is None:
            return None
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()

    def _collect_calibration(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> None:
        if hx is None:
            h0 = c0 = None
        else:
            if len(hx) != 2:
                raise ValueError("hx 必须是 (h_0, c_0)")
            h0, c0 = hx
        session = self._ensure_calibration_session()
        session.collect(
            self._cpu_tensor(input),
            self._cpu_tensor(self.weight_ih_l0),
            self._cpu_tensor(self.weight_hh_l0),
            self._cpu_tensor(self.bias_ih_l0),
            self._cpu_tensor(self.bias_hh_l0),
            self._cpu_tensor(h0),
            self._cpu_tensor(c0),
            self.batch_first,
        )

    def reset_calibration(self) -> None:
        self._invalidate_quant_params()

    def finalize_calibration(self) -> dict[str, Any]:
        if self._calibration_session is None:
            raise RuntimeError(
                "未收集校准数据；请先设置 calibrating=True 并执行 forward"
            )
        result = self._calibration_session.finalize(
            self.require_exact_accumulation
        )
        self._quant_params_bundle_json = result["bundle_json"]
        self._resolved_config_json = result["resolved_config_json"]
        self._last_safety_report = result["safety"]
        if self._last_safety_report["has_precision_risk"]:
            warnings.warn(
                "量化配置包含 FP32 precision_risk；已保留在 safety report 中",
                RuntimeWarning,
                stacklevel=2,
            )
        return {
            "batch_count": result["batch_count"],
            "method": result["method"],
            "safety": self._last_safety_report,
        }

    def is_calibrated(self) -> bool:
        return self._quant_params_bundle_json is not None

    def calibration_state(self) -> str:
        if self._calibration_session is None:
            return "empty"
        return self._calibration_session.state

    def _float_forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if hx is None:
            h0 = c0 = None
        else:
            if len(hx) != 2:
                raise ValueError("hx 必须是 (h_0, c_0)")
            h0, c0 = hx
        output, final_hidden, final_cell = float_lstm(
            input,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            h0,
            c0,
            self.batch_first,
        )
        return output, (final_hidden, final_cell)

    def qat_saved_state(self) -> Optional[dict[str, Any]]:
        return self._qat_saved_state

    def _quantized_forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self._quant_params_bundle_json is None:
            raise RuntimeError(
                "量化推理需要先校准并调用 finalize_calibration，"
                "或加载参数包"
            )
        if not input.is_cuda:
            raise RuntimeError("量化 FP 载体主路径只支持 CUDA input")
        if hx is None:
            h0 = c0 = None
        else:
            if len(hx) != 2:
                raise ValueError("hx 必须是 (h_0, c_0)")
            h0, c0 = hx
        save_checkpoints = self.training or torch.is_grad_enabled()
        output, final_hidden, final_cell, qat_state, safety = quantized_lstm(
            input,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            h0,
            c0,
            self.batch_first,
            self._quant_params_bundle_json,
            self.cublas_math_mode,
            self.require_exact_accumulation,
            save_checkpoints,
        )
        self._last_safety_report = safety
        self._qat_saved_state = qat_state if self.training else None
        return output, (final_hidden, final_cell)

    def forward(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not input.is_cuda:
            raise RuntimeError(
                "QuantLSTM PyTorch 执行路径只支持 CUDA input；"
                "CPU 实现仅用于 C++ reference model"
            )
        if self.calibrating:
            self._collect_calibration(input, hx)
            return self._float_forward(input, hx)
        if self.use_quantization:
            return self._quantized_forward(input, hx)
        return self._float_forward(input, hx)

    def export_quant_params(
        self, destination: str | Path | None = None
    ) -> dict[str, Any]:
        if self._quant_params_bundle_json is None:
            raise RuntimeError("模块尚未校准，不能导出量化参数")
        resolved = self.get_quant_config()
        document = {
            "schema_version": 1,
            "execution_metadata": {
                "carrier": "cuda_fp32_qcarrier",
                "activation_mode": "real_sigmoid_tanh",
                "cublas_math_mode": self.cublas_math_mode,
                "standard_scale_mode": resolved["scale_mode"],
            },
            "quant_params": _strict_json_loads(
                self._quant_params_bundle_json
            ),
        }
        if destination is not None:
            Path(destination).write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return document

    def load_quant_params(
        self, source: dict[str, Any] | str | Path
    ) -> dict[str, Any]:
        if isinstance(source, dict):
            document = source
        else:
            document = _strict_json_loads(_json_source(source))
        if set(document) != {
            "schema_version",
            "execution_metadata",
            "quant_params",
        }:
            raise ValueError("PyTorch 参数文档字段不完整或包含未知字段")
        if document["schema_version"] != 1:
            raise ValueError("只支持 PyTorch 参数文档 schema_version=1")
        metadata = document["execution_metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "carrier",
            "activation_mode",
            "cublas_math_mode",
            "standard_scale_mode",
        }:
            raise ValueError("execution_metadata 字段不完整或非法")
        if metadata["carrier"] != "cuda_fp32_qcarrier":
            raise ValueError("只支持 cuda_fp32_qcarrier")
        if metadata["activation_mode"] != "real_sigmoid_tanh":
            raise ValueError("只支持 real_sigmoid_tanh activation mode")
        if metadata["cublas_math_mode"] not in _MATH_MODES:
            raise ValueError("cublas_math_mode 非法")
        bundle_text = json.dumps(
            document["quant_params"],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        audited = _quant_lstm.audit_quant_params_bundle(
            bundle_text, self.require_exact_accumulation
        )
        if (
            audited["input_size"] != self.input_size
            or audited["hidden_size"] != self.hidden_size
            or audited["bias_enabled"] != self.bias
        ):
            raise ValueError("量化参数 shape 或 bias 与模块不匹配")
        resolved = _strict_json_loads(audited["resolved_config_json"])
        if metadata["standard_scale_mode"] != resolved["scale_mode"]:
            raise ValueError("standard_scale_mode 与参数包不一致")
        self._quant_params_bundle_json = audited["bundle_json"]
        self._resolved_config_json = audited["resolved_config_json"]
        self._override_config = _resolved_override(resolved)
        self._last_safety_report = audited["safety"]
        self._calibration_session = None
        self._qat_saved_state = None
        self.cublas_math_mode = metadata["cublas_math_mode"]
        if self._last_safety_report["has_precision_risk"]:
            warnings.warn(
                "导入配置包含 FP32 precision_risk；已保留在 safety report 中",
                RuntimeWarning,
                stacklevel=2,
            )
        return audited["safety"]

    import_quant_params = load_quant_params

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, bias={self.bias}, "
            f"batch_first={self.batch_first}, "
            f"use_quantization={self.use_quantization}"
        )


class QuantLSTM(_UnidirectionalQuantLSTM):
    """与单层 nn.LSTM 对齐的单向或双向量化模块。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        device=None,
        dtype=None,
        use_quantization: bool = False,
        quant_config: dict[str, Any] | str | Path | None = None,
        calibration_method: str = "minmax",
        cublas_math_mode: str = "pedantic",
        require_exact_accumulation: bool = False,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
            use_quantization=use_quantization,
            quant_config=quant_config,
            calibration_method=calibration_method,
            cublas_math_mode=cublas_math_mode,
            require_exact_accumulation=require_exact_accumulation,
        )
        self.bidirectional = bool(bidirectional)
        if self.bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(
                torch.empty_like(self.weight_ih_l0)
            )
            self.weight_hh_l0_reverse = nn.Parameter(
                torch.empty_like(self.weight_hh_l0)
            )
            if self.bias:
                self.bias_ih_l0_reverse = nn.Parameter(
                    torch.empty_like(self.bias_ih_l0)
                )
                self.bias_hh_l0_reverse = nn.Parameter(
                    torch.empty_like(self.bias_hh_l0)
                )
            else:
                self.register_parameter("bias_ih_l0_reverse", None)
                self.register_parameter("bias_hh_l0_reverse", None)
            bound = 1.0 / math.sqrt(self.hidden_size)
            with torch.no_grad():
                for name in (
                    "weight_ih_l0_reverse",
                    "weight_hh_l0_reverse",
                    "bias_ih_l0_reverse",
                    "bias_hh_l0_reverse",
                ):
                    parameter = getattr(self, name, None)
                    if parameter is not None:
                        parameter.uniform_(-bound, bound)

        self._reverse_calibration_session = None
        self._reverse_quant_params_bundle_json: Optional[str] = None
        self._reverse_safety_report: Optional[dict[str, Any]] = None
        self.export_mode = False
        empty = torch.empty(
            0,
            device=self.weight_ih_l0.device,
            dtype=self.weight_ih_l0.dtype,
        )
        self.register_buffer(
            "_onnx_export_weight_ih", empty, persistent=False
        )
        self.register_buffer(
            "_onnx_export_weight_hh", empty.clone(), persistent=False
        )
        self.register_buffer(
            "_onnx_export_bias", empty.clone(), persistent=False
        )

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    def _invalidate_quant_params(self) -> None:
        super()._invalidate_quant_params()
        if hasattr(self, "_reverse_calibration_session"):
            self._reverse_calibration_session = None
            self._reverse_quant_params_bundle_json = None
            self._reverse_safety_report = None

    def _time_dimension(self) -> int:
        return 1 if self.batch_first else 0

    def _reverse_sequence(self, value: Tensor) -> Tensor:
        return torch.flip(value, dims=(self._time_dimension(),))

    def _split_bidirectional_state(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]],
    ) -> tuple[
        tuple[Optional[Tensor], Optional[Tensor]],
        tuple[Optional[Tensor], Optional[Tensor]],
    ]:
        if hx is None:
            return (None, None), (None, None)
        if len(hx) != 2:
            raise ValueError("hx 必须是 (h_0, c_0)")
        hidden, cell = hx
        if hidden is None or cell is None:
            raise ValueError("h_0 和 c_0 必须同时提供")
        batch = input.size(0 if self.batch_first else 1)
        expected = (2, batch, self.hidden_size)
        if tuple(hidden.shape) != expected or tuple(cell.shape) != expected:
            raise RuntimeError("双向 h_0/c_0 shape 必须为 [2,B,H]")
        return (
            hidden[0:1].contiguous(),
            cell[0:1].contiguous(),
        ), (
            hidden[1:2].contiguous(),
            cell[1:2].contiguous(),
        )

    def _ensure_reverse_calibration_session(self):
        if self._reverse_calibration_session is None:
            self._reverse_calibration_session = _quant_lstm.CalibrationSession(
                self._resolved_config_json,
                self.input_size,
                self.hidden_size,
                self.bias,
                self.calibration_method,
            )
        return self._reverse_calibration_session

    def _collect_calibration(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> None:
        if not self.bidirectional:
            return super()._collect_calibration(input, hx)
        forward_state, reverse_state = self._split_bidirectional_state(input, hx)
        super()._collect_calibration(input, forward_state)
        reverse_session = self._ensure_reverse_calibration_session()
        reverse_session.collect(
            self._cpu_tensor(self._reverse_sequence(input)),
            self._cpu_tensor(self.weight_ih_l0_reverse),
            self._cpu_tensor(self.weight_hh_l0_reverse),
            self._cpu_tensor(self.bias_ih_l0_reverse),
            self._cpu_tensor(self.bias_hh_l0_reverse),
            self._cpu_tensor(reverse_state[0]),
            self._cpu_tensor(reverse_state[1]),
            self.batch_first,
        )

    def finalize_calibration(self) -> dict[str, Any]:
        if not self.bidirectional:
            return super().finalize_calibration()
        if self._reverse_calibration_session is None:
            raise RuntimeError(
                "双向模块尚未收集 reverse 方向校准数据"
            )
        forward_report = super().finalize_calibration()
        reverse_result = self._reverse_calibration_session.finalize(
            self.require_exact_accumulation
        )
        if reverse_result["resolved_config_json"] != self._resolved_config_json:
            raise RuntimeError("双向 resolved config 不一致")
        forward_bundle = _strict_json_loads(
            self._quant_params_bundle_json
        )
        reverse_bundle = _strict_json_loads(
            reverse_result["bundle_json"]
        )
        if (
            forward_bundle["operators"]["input"]
            != reverse_bundle["operators"]["input"]
        ):
            raise RuntimeError("双向校准必须共享完全相同的 input 量化网格")
        if reverse_result["batch_count"] != forward_report["batch_count"]:
            raise RuntimeError("双向校准 batch 数不一致")
        self._reverse_quant_params_bundle_json = reverse_result["bundle_json"]
        self._reverse_safety_report = reverse_result["safety"]
        forward_safety = self._last_safety_report
        self._last_safety_report = {
            "forward": forward_safety,
            "reverse": self._reverse_safety_report,
        }
        if self._reverse_safety_report["has_precision_risk"]:
            warnings.warn(
                "reverse 方向包含 FP32 precision_risk",
                RuntimeWarning,
                stacklevel=2,
            )
        return {
            "batch_count": forward_report["batch_count"],
            "method": forward_report["method"],
            "safety": self._last_safety_report,
        }

    def is_calibrated(self) -> bool:
        if not self.bidirectional:
            return super().is_calibrated()
        return (
            self._quant_params_bundle_json is not None
            and self._reverse_quant_params_bundle_json is not None
        )

    def calibration_state(self) -> str | dict[str, str]:
        if not self.bidirectional:
            return super().calibration_state()
        return {
            "forward": super().calibration_state(),
            "reverse": (
                "empty"
                if self._reverse_calibration_session is None
                else self._reverse_calibration_session.state
            ),
        }

    def _run_float_direction(
        self,
        input: Tensor,
        state: tuple[Optional[Tensor], Optional[Tensor]],
        reverse: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        direction_input = self._reverse_sequence(input) if reverse else input
        suffix = "_reverse" if reverse else ""
        output, hidden, cell = float_lstm(
            direction_input,
            getattr(self, f"weight_ih_l0{suffix}"),
            getattr(self, f"weight_hh_l0{suffix}"),
            getattr(self, f"bias_ih_l0{suffix}"),
            getattr(self, f"bias_hh_l0{suffix}"),
            state[0],
            state[1],
            self.batch_first,
        )
        if reverse:
            output = self._reverse_sequence(output)
        return output, hidden, cell

    def _float_forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not self.bidirectional:
            return super()._float_forward(input, hx)
        forward_state, reverse_state = self._split_bidirectional_state(input, hx)
        forward = self._run_float_direction(input, forward_state, False)
        reverse = self._run_float_direction(input, reverse_state, True)
        return torch.cat((forward[0], reverse[0]), dim=-1), (
            torch.cat((forward[1], reverse[1]), dim=0),
            torch.cat((forward[2], reverse[2]), dim=0),
        )

    def _run_quantized_direction(
        self,
        input: Tensor,
        state: tuple[Optional[Tensor], Optional[Tensor]],
        reverse: bool,
    ):
        direction_input = self._reverse_sequence(input) if reverse else input
        suffix = "_reverse" if reverse else ""
        bundle_json = (
            self._reverse_quant_params_bundle_json
            if reverse
            else self._quant_params_bundle_json
        )
        output, hidden, cell, qat_state, safety = quantized_lstm(
            direction_input,
            getattr(self, f"weight_ih_l0{suffix}"),
            getattr(self, f"weight_hh_l0{suffix}"),
            getattr(self, f"bias_ih_l0{suffix}"),
            getattr(self, f"bias_hh_l0{suffix}"),
            state[0],
            state[1],
            self.batch_first,
            bundle_json,
            self.cublas_math_mode,
            self.require_exact_accumulation,
            self.training or torch.is_grad_enabled(),
        )
        output = self._reverse_sequence(output) if reverse else output
        return output, hidden, cell, qat_state, safety

    def _quantized_forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not self.bidirectional:
            return super()._quantized_forward(input, hx)
        if not self.is_calibrated():
            raise RuntimeError(
                "双向量化推理需要两个方向都已完成校准或参数导入"
            )
        if not input.is_cuda:
            raise RuntimeError("量化 FP 载体主路径只支持 CUDA input")
        forward_state, reverse_state = self._split_bidirectional_state(input, hx)
        forward = self._run_quantized_direction(
            input, forward_state, False
        )
        reverse = self._run_quantized_direction(
            input, reverse_state, True
        )
        self._last_safety_report = {
            "forward": forward[4],
            "reverse": reverse[4],
        }
        if self.training:
            self._qat_saved_state = {
                "forward": forward[3],
                "reverse": reverse[3],
            }
        else:
            self._qat_saved_state = None
        return torch.cat((forward[0], reverse[0]), dim=-1), (
            torch.cat((forward[1], reverse[1]), dim=0),
            torch.cat((forward[2], reverse[2]), dim=0),
        )

    def export_quant_params(
        self, destination: str | Path | None = None
    ) -> dict[str, Any]:
        if not self.bidirectional:
            return super().export_quant_params(destination)
        if not self.is_calibrated():
            raise RuntimeError("双向模块尚未完成两个方向的校准")
        resolved = self.get_quant_config()
        document = {
            "schema_version": 2,
            "execution_metadata": {
                "carrier": "cuda_fp32_qcarrier",
                "activation_mode": "real_sigmoid_tanh",
                "cublas_math_mode": self.cublas_math_mode,
                "standard_scale_mode": resolved["scale_mode"],
                "bidirectional": True,
            },
            "quant_params": _strict_json_loads(
                self._quant_params_bundle_json
            ),
            "quant_params_reverse": _strict_json_loads(
                self._reverse_quant_params_bundle_json
            ),
        }
        if destination is not None:
            Path(destination).write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return document

    def load_quant_params(
        self, source: dict[str, Any] | str | Path
    ) -> dict[str, Any]:
        if not self.bidirectional:
            return super().load_quant_params(source)
        document = (
            source
            if isinstance(source, dict)
            else _strict_json_loads(_json_source(source))
        )
        if set(document) != {
            "schema_version",
            "execution_metadata",
            "quant_params",
            "quant_params_reverse",
        }:
            raise ValueError("双向参数文档字段不完整或包含未知字段")
        if document["schema_version"] != 2:
            raise ValueError("双向参数文档必须使用 schema_version=2")
        metadata = document["execution_metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "carrier",
            "activation_mode",
            "cublas_math_mode",
            "standard_scale_mode",
            "bidirectional",
        }:
            raise ValueError("双向 execution_metadata 字段不完整或非法")
        if metadata["carrier"] != "cuda_fp32_qcarrier":
            raise ValueError("只支持 cuda_fp32_qcarrier")
        if metadata["activation_mode"] != "real_sigmoid_tanh":
            raise ValueError("只支持 real_sigmoid_tanh activation mode")
        if metadata["cublas_math_mode"] not in _MATH_MODES:
            raise ValueError("cublas_math_mode 非法")
        if metadata["bidirectional"] is not True:
            raise ValueError("双向参数文档必须标记 bidirectional=true")

        audited = []
        for field in ("quant_params", "quant_params_reverse"):
            result = _quant_lstm.audit_quant_params_bundle(
                json.dumps(
                    document[field],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                self.require_exact_accumulation,
            )
            if (
                result["input_size"] != self.input_size
                or result["hidden_size"] != self.hidden_size
                or result["bias_enabled"] != self.bias
            ):
                raise ValueError(f"{field} shape 或 bias 与模块不匹配")
            audited.append(result)
        if (
            audited[0]["resolved_config_json"]
            != audited[1]["resolved_config_json"]
        ):
            raise ValueError("双向参数包的 resolved config 不一致")
        forward_bundle = _strict_json_loads(audited[0]["bundle_json"])
        reverse_bundle = _strict_json_loads(audited[1]["bundle_json"])
        if (
            forward_bundle["operators"]["input"]
            != reverse_bundle["operators"]["input"]
        ):
            raise ValueError("双向参数包没有共享 input 量化网格")
        resolved = _strict_json_loads(
            audited[0]["resolved_config_json"]
        )
        if metadata["standard_scale_mode"] != resolved["scale_mode"]:
            raise ValueError("standard_scale_mode 与参数包不一致")

        self._quant_params_bundle_json = audited[0]["bundle_json"]
        self._reverse_quant_params_bundle_json = audited[1]["bundle_json"]
        self._resolved_config_json = audited[0]["resolved_config_json"]
        self._override_config = _resolved_override(resolved)
        self._calibration_session = None
        self._reverse_calibration_session = None
        self._reverse_safety_report = audited[1]["safety"]
        self._last_safety_report = {
            "forward": audited[0]["safety"],
            "reverse": audited[1]["safety"],
        }
        self._qat_saved_state = None
        self.cublas_math_mode = metadata["cublas_math_mode"]
        if any(
            report["has_precision_risk"]
            for report in self._last_safety_report.values()
        ):
            warnings.warn(
                "导入的双向配置包含 FP32 precision_risk",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._last_safety_report

    @staticmethod
    def _reorder_onnx_gates(value: Tensor) -> Tensor:
        """将 PyTorch (i,f,g,o) 重排为 ONNX LSTM (i,o,f,c)。"""
        input_gate, forget_gate, cell_gate, output_gate = value.chunk(4, dim=0)
        return torch.cat((input_gate, output_gate, forget_gate, cell_gate))

    def _pack_onnx_direction(
        self, input: Tensor, reverse: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        suffix = "_reverse" if reverse else ""
        weight_ih = getattr(self, f"weight_ih_l0{suffix}").detach().to(
            device=input.device, dtype=input.dtype
        )
        weight_hh = getattr(self, f"weight_hh_l0{suffix}").detach().to(
            device=input.device, dtype=input.dtype
        )
        weight_ih = self._reorder_onnx_gates(weight_ih)
        weight_hh = self._reorder_onnx_gates(weight_hh)
        if self.bias:
            bias_ih = getattr(self, f"bias_ih_l0{suffix}").detach().to(
                device=input.device, dtype=input.dtype
            )
            bias_hh = getattr(self, f"bias_hh_l0{suffix}").detach().to(
                device=input.device, dtype=input.dtype
            )
            bias_ih = self._reorder_onnx_gates(bias_ih)
            bias_hh = self._reorder_onnx_gates(bias_hh)
        else:
            bias_ih = input.new_zeros(4 * self.hidden_size)
            bias_hh = input.new_zeros(4 * self.hidden_size)
        return weight_ih, weight_hh, torch.cat((bias_ih, bias_hh))

    def _forward_onnx(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "export_mode=True 仅用于 torch.onnx.export(..., dynamo=False)"
            )
        ensure_quant_lstm_onnx_registered(opset=18)
        input_time = (
            input.transpose(0, 1).contiguous()
            if self.batch_first
            else input.contiguous()
        )
        batch = input_time.size(1)
        state_shape = (self.num_directions, batch, self.hidden_size)
        if hx is None:
            initial_hidden = input.new_zeros(state_shape)
            initial_cell = input.new_zeros(state_shape)
        else:
            if len(hx) != 2:
                raise ValueError("hx 必须是 (h_0, c_0)")
            initial_hidden, initial_cell = hx

        packed = [
            self._pack_onnx_direction(input_time, reverse=False)
        ]
        if self.bidirectional:
            packed.append(
                self._pack_onnx_direction(input_time, reverse=True)
            )
        self._onnx_export_weight_ih = torch.stack(
            [item[0] for item in packed], dim=0
        ).contiguous()
        self._onnx_export_weight_hh = torch.stack(
            [item[1] for item in packed], dim=0
        ).contiguous()
        self._onnx_export_bias = torch.stack(
            [item[2] for item in packed], dim=0
        ).contiguous()

        output, final_hidden, final_cell = onnx_lstm(
            input_time,
            initial_hidden.contiguous(),
            initial_cell.contiguous(),
            self._onnx_export_weight_ih,
            self._onnx_export_weight_hh,
            self._onnx_export_bias,
            self.hidden_size,
            self.bidirectional,
        )
        if self.batch_first:
            output = output.transpose(0, 1).contiguous()
        return output, (final_hidden, final_cell)

    def forward(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.export_mode:
            return self._forward_onnx(input, hx)
        return super().forward(input, hx)


    import_quant_params = load_quant_params

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, bias={self.bias}, "
            f"batch_first={self.batch_first}, "
            f"bidirectional={self.bidirectional}, "
            f"use_quantization={self.use_quantization}"
        )
