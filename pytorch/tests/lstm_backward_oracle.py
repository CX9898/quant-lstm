"""Test-only PyTorch oracle for validating the native QAT backward implementation."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


GATE_OUTPUT_OPERATORS = (
    "input_gate_output",
    "forget_gate_output",
    "cell_gate_output",
    "output_gate_output",
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
        dtype=torch.float32,
        device=value.device,
    ).double()
    zero_points = torch.tensor(
        operator["zero_points"], dtype=torch.float64, device=value.device
    )
    if per_channel:
        shape = [len(scales)] + [1] * (value.dim() - 1)
        scales = scales.reshape(shape)
        zero_points = zero_points.reshape(shape)
    return scales, zero_points


def quantize_tensor(
    value: Tensor,
    operator: dict[str, Any],
    per_channel: bool = False,
) -> tuple[Tensor, Tensor]:
    qmin, qmax = _quant_range(operator)
    scales, zero_points = _operator_tensors(operator, value, per_channel)
    translated = value.detach().double() / scales + zero_points
    clamped = (translated < qmin) | (translated > qmax)
    return torch.round(translated).clamp(qmin, qmax).float(), clamped


def dequantize_tensor(
    value: Tensor,
    operator: dict[str, Any],
    per_channel: bool = False,
) -> Tensor:
    scales, zero_points = _operator_tensors(operator, value, per_channel)
    return ((value.double() - zero_points) * scales).float()


def dequantize_gates(value: Tensor, bundle: dict[str, Any]) -> Tensor:
    chunks = value.chunk(4, dim=-1)
    return torch.cat(
        tuple(
            dequantize_tensor(chunk, bundle["operators"][name])
            for chunk, name in zip(chunks, GATE_OUTPUT_OPERATORS)
        ),
        dim=-1,
    )


def keep_gradient(mask: Optional[Tensor], reference: Tensor) -> Tensor | float:
    if mask is None:
        return 1.0
    return (mask == 0).to(dtype=reference.dtype)


def lstm_backward(
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

        grad_hidden = (grad_hidden + grad_output[time]) * keep_gradient(
            masks.get("hidden_outputs", None)[time]
            if "hidden_outputs" in masks
            else None,
            grad_hidden,
        )
        grad_output_gate = grad_hidden * cell_tanh
        grad_cell_tanh = grad_hidden * output_gate
        grad_cell_tanh = grad_cell_tanh * keep_gradient(
            masks.get("cell_tanh_outputs", None)[time]
            if "cell_tanh_outputs" in masks
            else None,
            grad_cell_tanh,
        )
        grad_cell = (
            grad_cell + grad_cell_tanh * (1.0 - cell_tanh.square())
        ) * keep_gradient(
            masks.get("cell_states", None)[time]
            if "cell_states" in masks
            else None,
            grad_cell,
        )

        grad_gate_outputs = torch.cat(
            (
                grad_cell * cell_gate,
                grad_cell * cell_previous,
                grad_cell * input_gate,
                grad_output_gate,
            ),
            dim=-1,
        ) * keep_gradient(
            masks.get("gate_outputs", None)[time]
            if "gate_outputs" in masks
            else None,
            grad_output_gate,
        )
        grad_cell = grad_cell * forget_gate
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
        ) * keep_gradient(
            masks.get("gate_inputs", None)[time]
            if "gate_inputs" in masks
            else None,
            grad_gate_outputs,
        )
        grad_input_linear = grad_gate_inputs * keep_gradient(
            masks.get("weight_ih_linear", None)[time]
            if "weight_ih_linear" in masks
            else None,
            grad_gate_inputs,
        )
        grad_recurrent_linear = grad_gate_inputs * keep_gradient(
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
