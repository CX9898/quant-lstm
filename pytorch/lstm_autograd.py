"""QuantLSTM 浮点 backward 与 FP32 q-carrier QAT 自动求导桥接。"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

import _quant_lstm


_CHECKPOINT_NAMES = (
    "weight_ih_linear",
    "weight_hh_linear",
    "gate_inputs",
    "gate_outputs",
    "cell_states",
    "cell_tanh_outputs",
    "hidden_outputs",
)


def _as_time_major(value: Tensor, batch_first: bool) -> Tensor:
    return value.transpose(0, 1).contiguous() if batch_first else value.contiguous()


def _zero_state(input: Tensor, weight_hh: Tensor, batch_first: bool) -> Tensor:
    batch = input.size(0 if batch_first else 1)
    return input.new_zeros((1, batch, weight_hh.size(1)))


def _gradient_or_zeros(value: Optional[Tensor], reference: Tensor) -> Tensor:
    return torch.zeros_like(reference) if value is None else value.contiguous()


class _FloatLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Optional[Tensor],
        bias_hh: Optional[Tensor],
        initial_hidden: Optional[Tensor],
        initial_cell: Optional[Tensor],
        batch_first: bool,
    ):
        ctx.has_native_trace = any(ctx.needs_input_grad[:7])
        trace = ()
        if ctx.has_native_trace:
            result = _quant_lstm.lstm_forward_training(
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                initial_hidden,
                initial_cell,
                batch_first,
            )
            output, hidden, cell = result[:3]
            trace = result[3:]
        else:
            output, hidden, cell = _quant_lstm.lstm_forward(
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                initial_hidden,
                initial_cell,
                batch_first,
            )
        empty = input.new_empty(0)
        ctx.save_for_backward(
            input,
            weight_ih,
            weight_hh,
            empty if bias_ih is None else bias_ih,
            empty if bias_hh is None else bias_hh,
            empty if initial_hidden is None else initial_hidden,
            empty if initial_cell is None else initial_cell,
            *trace,
        )
        ctx.batch_first = bool(batch_first)
        ctx.has_bias = bias_ih is not None
        ctx.has_state = initial_hidden is not None
        return output, hidden, cell

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        saved = ctx.saved_tensors
        input, weight_ih, weight_hh, bias_ih, bias_hh, initial_hidden, initial_cell = (
            saved[:7]
        )
        zero_state = _zero_state(input, weight_hh, ctx.batch_first)
        hidden = initial_hidden if ctx.has_state else zero_state
        cell = initial_cell if ctx.has_state else zero_state
        if not ctx.has_native_trace:
            raise RuntimeError("浮点 backward 缺少 native checkpoint")
        gate_outputs, cell_states, cell_tanh_outputs, hidden_outputs = saved[7:]
        grad_output_time = _gradient_or_zeros(
            None if grad_output is None else _as_time_major(grad_output, ctx.batch_first),
            hidden_outputs,
        )
        grad_hidden_value = _gradient_or_zeros(grad_hidden, hidden)[0]
        grad_cell_value = _gradient_or_zeros(grad_cell, cell)[0]
        gradients = _quant_lstm.lstm_backward_float(
            input,
            weight_ih,
            weight_hh,
            bias_ih if ctx.has_bias else None,
            bias_hh if ctx.has_bias else None,
            initial_hidden if ctx.has_state else None,
            initial_cell if ctx.has_state else None,
            ctx.batch_first,
            gate_outputs,
            cell_states,
            cell_tanh_outputs,
            hidden_outputs,
            grad_output_time,
            grad_hidden_value,
            grad_cell_value,
        )
        return (
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3] if ctx.has_bias else None,
            gradients[4] if ctx.has_bias else None,
            gradients[5].unsqueeze(0) if ctx.has_state else None,
            gradients[6].unsqueeze(0) if ctx.has_state else None,
            None,
        )


class _QuantizedLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Optional[Tensor],
        bias_hh: Optional[Tensor],
        initial_hidden: Optional[Tensor],
        initial_cell: Optional[Tensor],
        batch_first: bool,
        bundle_json: str,
        math_mode: str,
        require_exact_accumulation: bool,
        save_checkpoints: bool,
        state_sink: dict[str, Any],
    ):
        output, hidden, cell, checkpoints, safety = (
            _quant_lstm.lstm_forward_quantized(
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                initial_hidden,
                initial_cell,
                batch_first,
                bundle_json,
                math_mode,
                require_exact_accumulation,
                save_checkpoints,
            )
        )
        state_sink["safety"] = safety
        ctx.has_checkpoints = bool(save_checkpoints)
        if save_checkpoints:
            qat_state = {
                "quantized_master": checkpoints["quantized_master"],
                "master_clamp_masks": checkpoints["master_clamp_masks"],
                "checkpoints": checkpoints["values"],
                "checkpoint_clamp_masks": checkpoints["clamp_masks"],
            }
            state_sink["qat_saved_state"] = qat_state
            masters = qat_state["quantized_master"]
            master_masks = qat_state["master_clamp_masks"]
            values = qat_state["checkpoints"]
            checkpoint_masks = qat_state["checkpoint_clamp_masks"]
            empty = input.new_empty(0)
            empty_mask = torch.empty(0, dtype=torch.bool, device=input.device)
            ctx.save_for_backward(
                masters["input"],
                masters["weight_ih"],
                masters["weight_hh"],
                masters.get("bias_ih", empty),
                masters.get("bias_hh", empty),
                masters["h_0"],
                masters["c_0"],
                master_masks["input"],
                master_masks["weight_ih"],
                master_masks["weight_hh"],
                master_masks.get("bias_ih", empty_mask),
                master_masks.get("bias_hh", empty_mask),
                master_masks["h_0"],
                master_masks["c_0"],
                values["gate_outputs"],
                values["cell_states"],
                values["cell_tanh_outputs"],
                values["hidden_outputs"],
                *(checkpoint_masks[name] for name in _CHECKPOINT_NAMES),
            )
            ctx.bundle_json = bundle_json
        ctx.batch_first = bool(batch_first)
        ctx.has_bias = bias_ih is not None
        ctx.has_state = initial_hidden is not None
        return output, hidden, cell

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        if not ctx.has_checkpoints:
            raise RuntimeError("量化 backward 缺少训练态 q-carrier checkpoint")
        saved = ctx.saved_tensors
        masters = saved[:7]
        master_masks = saved[7:14]
        values = saved[14:18]
        checkpoint_masks = dict(zip(_CHECKPOINT_NAMES, saved[18:25]))
        grad_output_time = _gradient_or_zeros(
            None if grad_output is None else _as_time_major(grad_output, ctx.batch_first),
            values[3],
        )
        grad_hidden = _gradient_or_zeros(grad_hidden, masters[5])[0]
        grad_cell = _gradient_or_zeros(grad_cell, masters[6])[0]
        gradients = list(
            _quant_lstm.lstm_backward_qat(
                list(masters),
                ctx.batch_first,
                ctx.bundle_json,
                list(values),
                grad_output_time,
                grad_hidden,
                grad_cell,
                list(master_masks),
                [checkpoint_masks[name] for name in _CHECKPOINT_NAMES],
            )
        )
        return (
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3] if ctx.has_bias else None,
            gradients[4] if ctx.has_bias else None,
            gradients[5].unsqueeze(0) if ctx.has_state else None,
            gradients[6].unsqueeze(0) if ctx.has_state else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def float_lstm(
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor],
    bias_hh: Optional[Tensor],
    initial_hidden: Optional[Tensor],
    initial_cell: Optional[Tensor],
    batch_first: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return _FloatLSTMFunction.apply(
        input,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        initial_hidden,
        initial_cell,
        batch_first,
    )


def quantized_lstm(
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor],
    bias_hh: Optional[Tensor],
    initial_hidden: Optional[Tensor],
    initial_cell: Optional[Tensor],
    batch_first: bool,
    bundle_json: str,
    math_mode: str,
    require_exact_accumulation: bool,
    save_checkpoints: bool,
) -> tuple[Tensor, Tensor, Tensor, Optional[dict[str, Any]], dict[str, Any]]:
    state: dict[str, Any] = {}
    output, hidden, cell = _QuantizedLSTMFunction.apply(
        input,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        initial_hidden,
        initial_cell,
        batch_first,
        bundle_json,
        math_mode,
        require_exact_accumulation,
        save_checkpoints,
        state,
    )
    return output, hidden, cell, state.get("qat_saved_state"), state["safety"]
