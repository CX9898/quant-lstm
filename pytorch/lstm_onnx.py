"""QuantLSTM 单节点标准 ONNX LSTM 导出桥接。"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.onnx import register_custom_op_symbolic, symbolic_helper


QUANT_LSTM_ONNX_DOMAIN = "quant_lstm_onnx"
_QUANT_LSTM_ONNX_LIB = None


def _runtime_result(
    input_time: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    directions = hidden.size(0)
    output = input_time.new_zeros(
        input_time.size(0),
        input_time.size(1),
        directions * hidden.size(2),
    )
    return output, torch.zeros_like(hidden), torch.zeros_like(hidden)


def ensure_quant_lstm_onnx_registered(opset: int = 18) -> None:
    """注册临时 runtime op 及其标准 ONNX LSTM symbolic。"""
    opset = int(opset)
    if opset < 13:
        raise ValueError("QuantLSTM ONNX 导出仅支持 opset>=13")

    global _QUANT_LSTM_ONNX_LIB
    if _QUANT_LSTM_ONNX_LIB is None:
        _QUANT_LSTM_ONNX_LIB = torch.library.Library(
            QUANT_LSTM_ONNX_DOMAIN, "FRAGMENT"
        )

    if not getattr(ensure_quant_lstm_onnx_registered, "_runtime_done", False):
        schema = (
            "(Tensor input, Tensor h0, Tensor c0, Tensor W, Tensor R, Tensor B, "
            "int hidden_size) -> (Tensor, Tensor, Tensor)"
        )
        _QUANT_LSTM_ONNX_LIB.define("lstm" + schema)
        _QUANT_LSTM_ONNX_LIB.define("bilstm" + schema)

        def runtime(input, h0, c0, W, R, B, hidden_size):
            del c0, W, R, B, hidden_size
            return _runtime_result(input, h0)

        for name in ("lstm", "bilstm"):
            _QUANT_LSTM_ONNX_LIB.impl(
                name, runtime, dispatch_key="CompositeExplicitAutograd"
            )
        ensure_quant_lstm_onnx_registered._runtime_done = True

    registered = getattr(
        ensure_quant_lstm_onnx_registered, "_registered_opsets", set()
    )
    if opset in registered:
        return

    def symbolic(g, x, h0, c0, W, R, B, hidden_size, bidirectional):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        if hidden_size_i is None:
            raise RuntimeError("QuantLSTM ONNX hidden_size 必须为常量")
        optional = symbolic_helper._optional_input_placeholder_tensor
        attributes = {"hidden_size_i": int(hidden_size_i)}
        if bidirectional:
            attributes["direction_s"] = "bidirectional"
        y, y_h, y_c = g.op(
            "LSTM",
            x,
            W,
            R,
            B,
            optional(g),
            h0,
            c0,
            optional(g),
            outputs=3,
            **attributes,
        )
        if bidirectional:
            transposed = g.op("Transpose", y, perm_i=[0, 2, 1, 3])
            shape = g.op(
                "Constant",
                value_t=torch.tensor([0, 0, -1], dtype=torch.long),
            )
            output = g.op("Reshape", transposed, shape)
        else:
            axes = g.op(
                "Constant", value_t=torch.tensor([1], dtype=torch.long)
            )
            output = g.op("Squeeze", y, axes)
        return output, y_h, y_c

    register_custom_op_symbolic(
        f"{QUANT_LSTM_ONNX_DOMAIN}::lstm",
        lambda g, *args: symbolic(g, *args, False),
        opset,
    )
    register_custom_op_symbolic(
        f"{QUANT_LSTM_ONNX_DOMAIN}::bilstm",
        lambda g, *args: symbolic(g, *args, True),
        opset,
    )
    registered.add(opset)
    ensure_quant_lstm_onnx_registered._registered_opsets = registered


def onnx_lstm(
    input_time: Tensor,
    initial_hidden: Tensor,
    initial_cell: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias: Tensor,
    hidden_size: int,
    bidirectional: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """调用导出期临时 op；symbolic 会把它替换为标准 ONNX LSTM。"""
    namespace = getattr(torch.ops, QUANT_LSTM_ONNX_DOMAIN)
    operator = namespace.bilstm if bidirectional else namespace.lstm
    return operator(
        input_time,
        initial_hidden,
        initial_cell,
        weight_ih,
        weight_hh,
        bias,
        int(hidden_size),
    )
