"""阶段 1 的单层、单向 FP32 LSTM PyTorch 接口。"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

try:
    import _quant_lstm
except ImportError as exc:
    raise ImportError(
        "_quant_lstm 扩展未找到；请先构建 CMake 核心并运行 "
        "`python setup.py build_ext --inplace`"
    ) from exc


class QuantLSTM(nn.Module):
    """与 ``nn.LSTM`` 单层单向子集对齐的 FP32 前向模块。"""

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
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size 和 hidden_size 必须为正数")
        if num_layers != 1:
            raise ValueError("阶段 1 仅支持 num_layers=1")
        if dropout != 0.0:
            raise ValueError("阶段 1 仅支持 dropout=0")
        if bidirectional:
            raise ValueError("阶段 1 尚不支持 bidirectional=True")
        if dtype not in (None, torch.float32):
            raise ValueError("阶段 1 仅支持 torch.float32")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        factory_kwargs = {"device": device, "dtype": torch.float32 if dtype is None else dtype}
        self.weight_ih_l0 = nn.Parameter(
            torch.empty((4 * hidden_size, input_size), **factory_kwargs)
        )
        self.weight_hh_l0 = nn.Parameter(
            torch.empty((4 * hidden_size, hidden_size), **factory_kwargs)
        )
        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs))
            self.bias_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias_ih_l0", None)
            self.register_parameter("bias_hh_l0", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.uniform_(-bound, bound)

    def forward(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if hx is None:
            initial_hidden = None
            initial_cell = None
        else:
            if len(hx) != 2:
                raise ValueError("hx 必须是 (h_0, c_0)")
            initial_hidden, initial_cell = hx

        output, final_hidden, final_cell = _quant_lstm.lstm_forward(
            input,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
            initial_hidden,
            initial_cell,
            self.batch_first,
        )
        return output, (final_hidden, final_cell)

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, bias={self.bias}, "
            f"batch_first={self.batch_first}"
        )
