"""QuantLSTM 浮点 backward 与 FP32 q-carrier QAT 自动求导桥接。"""

from __future__ import annotations

import json
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

import _quant_lstm


_GATE_OUTPUT_OPERATORS = (
    "input_gate_output",
    "forget_gate_output",
    "cell_gate_output",
    "output_gate_output",
)
_CHECKPOINT_NAMES = (
    "weight_ih_linear",
    "weight_hh_linear",
    "gate_inputs",
    "gate_outputs",
    "cell_states",
    "cell_tanh_outputs",
    "hidden_outputs",
)


def _quant_range(operator: dict[str, Any]) -> tuple[int, int]:
    bitwidth = operator["bitwidth"]
    if operator["is_unsigned"]:
        return 0, (1 << bitwidth) - 1
    if operator["is_symmetric"]:
        maximum = (1 << (bitwidth - 1)) - 1
        return -maximum, maximum
    return -(1 << (bitwidth - 1)), (1 << (bitwidth - 1)) - 1


def _operator_tensors(
    operator: dict[str, Any], value: Tensor, per_channel: bool
) -> tuple[Tensor, Tensor]:
    scales = torch.tensor(
        [float(item) for item in operator["scales"]],
        dtype=torch.float64,
        device=value.device,
    )
    zero_points = torch.tensor(
        operator["zero_points"], dtype=torch.float64, device=value.device
    )
    if per_channel:
        shape = [len(scales)] + [1] * (value.dim() - 1)
        scales = scales.reshape(shape)
        zero_points = zero_points.reshape(shape)
    return scales, zero_points


def _quantize_tensor(
    value: Tensor,
    operator: dict[str, Any],
    per_channel: bool = False,
) -> tuple[Tensor, Tensor]:
    qmin, qmax = _quant_range(operator)
    scales, zero_points = _operator_tensors(operator, value, per_channel)
    translated = value.detach().double() / scales + zero_points
    clamped = (translated < qmin) | (translated > qmax)
    quantized = torch.round(translated).clamp(qmin, qmax).float()
    return quantized, clamped


def _dequantize_tensor(
    value: Tensor,
    operator: dict[str, Any],
    per_channel: bool = False,
) -> Tensor:
    scales, zero_points = _operator_tensors(operator, value, per_channel)
    return ((value.double() - zero_points) * scales).float()


def _dequantize_gates(
    value: Tensor,
    bundle: dict[str, Any],
    operators: tuple[str, str, str, str],
) -> Tensor:
    chunks = value.chunk(4, dim=-1)
    return torch.cat(
        tuple(
            _dequantize_tensor(chunk, bundle["operators"][name])
            for chunk, name in zip(chunks, operators)
        ),
        dim=-1,
    )


def _as_time_major(value: Tensor, batch_first: bool) -> Tensor:
    return value.transpose(0, 1).contiguous() if batch_first else value.contiguous()


def _float_trace(
    input_time: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor],
    bias_hh: Optional[Tensor],
    initial_hidden: Tensor,
    initial_cell: Tensor,
) -> dict[str, Tensor]:
    hidden = initial_hidden
    cell = initial_cell
    gate_outputs = []
    cell_states = []
    cell_tanh_outputs = []
    hidden_outputs = []
    for input_step in input_time.unbind(0):
        gates = F.linear(input_step, weight_ih, bias_ih)
        gates = gates + F.linear(hidden, weight_hh, bias_hh)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_gate
        cell_tanh = torch.tanh(cell)
        hidden = output_gate * cell_tanh
        gate_outputs.append(
            torch.cat((input_gate, forget_gate, cell_gate, output_gate), dim=-1)
        )
        cell_states.append(cell)
        cell_tanh_outputs.append(cell_tanh)
        hidden_outputs.append(hidden)
    return {
        "gate_outputs": torch.stack(gate_outputs),
        "cell_states": torch.stack(cell_states),
        "cell_tanh_outputs": torch.stack(cell_tanh_outputs),
        "hidden_outputs": torch.stack(hidden_outputs),
    }


def _keep_gradient(mask: Optional[Tensor], reference: Tensor) -> Tensor | float:
    if mask is None:
        return 1.0
    return (mask == 0).to(dtype=reference.dtype)


def _lstm_backward(
    input_time: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    initial_hidden: Tensor,
    initial_cell: Tensor,
    trace: dict[str, Tensor],
    grad_output: Tensor,
    grad_final_hidden: Tensor,
    grad_final_cell: Tensor,
    masks: Optional[dict[str, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """以 real-domain checkpoint 执行单层单向 LSTM backward。"""
    masks = {} if masks is None else masks
    steps = input_time.size(0)
    hidden_size = weight_hh.size(1)
    grad_input = torch.zeros_like(input_time)
    grad_weight_ih = torch.zeros_like(weight_ih)
    grad_weight_hh = torch.zeros_like(weight_hh)
    grad_bias_ih = input_time.new_zeros(4 * hidden_size)
    grad_bias_hh = input_time.new_zeros(4 * hidden_size)
    grad_hidden = grad_final_hidden
    grad_cell = grad_final_cell

    for time in range(steps - 1, -1, -1):
        hidden_previous = (
            initial_hidden if time == 0 else trace["hidden_outputs"][time - 1]
        )
        cell_previous = initial_cell if time == 0 else trace["cell_states"][time - 1]
        input_gate, forget_gate, cell_gate, output_gate = trace[
            "gate_outputs"
        ][time].chunk(4, dim=-1)
        cell_tanh = trace["cell_tanh_outputs"][time]

        grad_hidden = grad_hidden + grad_output[time]
        grad_hidden = grad_hidden * _keep_gradient(
            masks.get("hidden_outputs", None)[time]
            if "hidden_outputs" in masks
            else None,
            grad_hidden,
        )
        grad_output_gate = grad_hidden * cell_tanh
        grad_cell_tanh = grad_hidden * output_gate
        grad_cell_tanh = grad_cell_tanh * _keep_gradient(
            masks.get("cell_tanh_outputs", None)[time]
            if "cell_tanh_outputs" in masks
            else None,
            grad_cell_tanh,
        )
        grad_cell = grad_cell + grad_cell_tanh * (1.0 - cell_tanh.square())
        grad_cell = grad_cell * _keep_gradient(
            masks.get("cell_states", None)[time]
            if "cell_states" in masks
            else None,
            grad_cell,
        )

        grad_forget_gate = grad_cell * cell_previous
        grad_input_gate = grad_cell * cell_gate
        grad_cell_gate = grad_cell * input_gate
        grad_cell = grad_cell * forget_gate
        grad_gate_outputs = torch.cat(
            (
                grad_input_gate,
                grad_forget_gate,
                grad_cell_gate,
                grad_output_gate,
            ),
            dim=-1,
        )
        grad_gate_outputs = grad_gate_outputs * _keep_gradient(
            masks.get("gate_outputs", None)[time]
            if "gate_outputs" in masks
            else None,
            grad_gate_outputs,
        )
        grad_input_gate, grad_forget_gate, grad_cell_gate, grad_output_gate = (
            grad_gate_outputs.chunk(4, dim=-1)
        )
        grad_gate_inputs = torch.cat(
            (
                grad_input_gate * input_gate * (1.0 - input_gate),
                grad_forget_gate * forget_gate * (1.0 - forget_gate),
                grad_cell_gate * (1.0 - cell_gate.square()),
                grad_output_gate * output_gate * (1.0 - output_gate),
            ),
            dim=-1,
        )
        grad_gate_inputs = grad_gate_inputs * _keep_gradient(
            masks.get("gate_inputs", None)[time]
            if "gate_inputs" in masks
            else None,
            grad_gate_inputs,
        )
        grad_input_linear = grad_gate_inputs * _keep_gradient(
            masks.get("weight_ih_linear", None)[time]
            if "weight_ih_linear" in masks
            else None,
            grad_gate_inputs,
        )
        grad_recurrent_linear = grad_gate_inputs * _keep_gradient(
            masks.get("weight_hh_linear", None)[time]
            if "weight_hh_linear" in masks
            else None,
            grad_gate_inputs,
        )

        grad_input[time] = grad_input_linear.matmul(weight_ih)
        grad_hidden = grad_recurrent_linear.matmul(weight_hh)
        grad_weight_ih.add_(grad_input_linear.transpose(0, 1).matmul(input_time[time]))
        grad_weight_hh.add_(
            grad_recurrent_linear.transpose(0, 1).matmul(hidden_previous)
        )
        grad_bias_ih.add_(grad_input_linear.sum(dim=0))
        grad_bias_hh.add_(grad_recurrent_linear.sum(dim=0))

    return (
        grad_input,
        grad_weight_ih,
        grad_weight_hh,
        grad_bias_ih,
        grad_bias_hh,
        grad_hidden,
        grad_cell,
    )


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
        ctx.native_cuda_backward = input.is_cuda and any(
            ctx.needs_input_grad[:7]
        )
        trace = ()
        if ctx.native_cuda_backward:
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
        if ctx.native_cuda_backward:
            gate_outputs, cell_states, cell_tanh_outputs, hidden_outputs = saved[7:]
            grad_output_time = _gradient_or_zeros(
                None
                if grad_output is None
                else _as_time_major(grad_output, ctx.batch_first),
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
        input_time = _as_time_major(input, ctx.batch_first)
        trace = _float_trace(
            input_time,
            weight_ih,
            weight_hh,
            bias_ih if ctx.has_bias else None,
            bias_hh if ctx.has_bias else None,
            hidden[0],
            cell[0],
        )
        grad_output_time = _gradient_or_zeros(
            None if grad_output is None else _as_time_major(grad_output, ctx.batch_first),
            trace["hidden_outputs"],
        )
        grad_hidden = _gradient_or_zeros(grad_hidden, hidden)[0]
        grad_cell = _gradient_or_zeros(grad_cell, cell)[0]
        gradients = _lstm_backward(
            input_time,
            weight_ih,
            weight_hh,
            hidden[0],
            cell[0],
            trace,
            grad_output_time,
            grad_hidden,
            grad_cell,
        )
        grad_input = (
            gradients[0].transpose(0, 1) if ctx.batch_first else gradients[0]
        )
        return (
            grad_input,
            gradients[1],
            gradients[2],
            gradients[3] if ctx.has_bias else None,
            gradients[4] if ctx.has_bias else None,
            gradients[5].unsqueeze(0) if ctx.has_state else None,
            gradients[6].unsqueeze(0) if ctx.has_state else None,
            None,
        )


def _build_qat_state(
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor],
    bias_hh: Optional[Tensor],
    initial_hidden: Optional[Tensor],
    initial_cell: Optional[Tensor],
    batch_first: bool,
    bundle: dict[str, Any],
    checkpoints: dict[str, Any],
) -> dict[str, Any]:
    zero_state = _zero_state(input, weight_hh, batch_first)
    tensors = {
        "input": (input, "input", False),
        "weight_ih": (weight_ih, "weight_ih", True),
        "weight_hh": (weight_hh, "weight_hh", True),
        "h_0": (
            zero_state if initial_hidden is None else initial_hidden,
            "output",
            False,
        ),
        "c_0": (
            zero_state if initial_cell is None else initial_cell,
            "cell_state",
            False,
        ),
    }
    if bias_ih is not None:
        tensors["bias_ih"] = (bias_ih, "bias_ih", True)
        tensors["bias_hh"] = (bias_hh, "bias_hh", True)
    quantized = {}
    masks = {}
    for name, (tensor, operator_name, per_channel) in tensors.items():
        quantized[name], masks[name] = _quantize_tensor(
            tensor, bundle["operators"][operator_name], per_channel
        )
    return {
        "quantized_master": quantized,
        "master_clamp_masks": masks,
        "checkpoints": checkpoints.get("values", {}),
        "checkpoint_clamp_masks": checkpoints.get("clamp_masks", {}),
    }


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
            bundle = json.loads(bundle_json)
            qat_state = _build_qat_state(
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                initial_hidden,
                initial_cell,
                batch_first,
                bundle,
                checkpoints,
            )
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
