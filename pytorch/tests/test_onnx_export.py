"""阶段 9 标准 ONNX LSTM 单节点与 ONNX Runtime 语义验收。"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn

from quant_lstm import QuantLSTM, ensure_quant_lstm_onnx_registered
from tests.test_quantized_interface import deterministic_tensor, metrics


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/precision/results/stage9_onnx_report.json"
OPSET = 18


class ExportWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, initial_hidden, initial_cell):
        output, (hidden, cell) = self.module(
            input, (initial_hidden, initial_cell)
        )
        return output, hidden, cell


def initialize_parameters(module):
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            parameter.copy_(
                deterministic_tensor(
                    parameter.shape,
                    -0.17 + 0.006 * index,
                    0.19 - 0.004 * index,
                    device=parameter.device,
                )
            )


def reorder_gates(value):
    input_gate, forget_gate, cell_gate, output_gate = np.split(value, 4)
    return np.concatenate(
        (input_gate, output_gate, forget_gate, cell_gate), axis=0
    )


def tensor_value(model, name):
    values = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    if name in values:
        return values[name]
    for node in model.graph.node:
        if node.op_type != "Constant" or name not in node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                return numpy_helper.to_array(attribute.t)
    raise KeyError(name)


def direction_attribute(node):
    for attribute in node.attribute:
        if attribute.name == "direction":
            return attribute.s.decode("utf-8")
    return "forward"


class OnnxExportTest(unittest.TestCase):
    records = []

    @classmethod
    def setUpClass(cls):
        ensure_quant_lstm_onnx_registered(OPSET)

    @classmethod
    def tearDownClass(cls):
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(
                {
                    "stage": 9,
                    "opset": OPSET,
                    "onnx_version": onnx.__version__,
                    "onnxruntime_version": ort.__version__,
                    "validation_scope": "float_semantic",
                    "records": cls.records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def assert_metrics(self, case, name, actual, expected):
        mae, mse, cosine = metrics(actual, expected)
        self.records.append(
            {
                "case": case,
                "tensor": name,
                "mae": mae,
                "mse": mse,
                "cosine": cosine,
                "cosine_status": "valid",
            }
        )
        self.assertLess(mae, 0.003, f"{name}: MAE={mae}")
        self.assertLess(mse, 1.0e-5, f"{name}: MSE={mse}")
        self.assertGreaterEqual(cosine, 0.9999, f"{name}: cosine={cosine}")
        self.assertTrue(
            torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5),
            f"{name}: max_abs={(actual - expected).abs().max().item()}",
        )

    def export_case(self, directory, bidirectional, bias, batch_first):
        case = (
            f"bi{int(bidirectional)}_bias{int(bias)}_"
            f"bf{int(batch_first)}"
        )
        module = QuantLSTM(
            3,
            4,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        ).eval()
        initialize_parameters(module)
        reference = nn.LSTM(
            3,
            4,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        ).eval()
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                getattr(reference, name).copy_(parameter)
        input_time = deterministic_tensor((5, 2, 3), -0.28, 0.31)
        input_value = (
            input_time.transpose(0, 1).contiguous()
            if batch_first
            else input_time
        )
        directions = 2 if bidirectional else 1
        hidden = deterministic_tensor(
            (directions, 2, 4), -0.09, 0.12
        )
        cell = deterministic_tensor(
            (directions, 2, 4), -0.16, 0.18
        )
        with torch.no_grad():
            expected = reference(input_value, (hidden, cell))

        path = Path(directory) / f"{case}.onnx"
        module.export_mode = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                ExportWrapper(module),
                (input_value, hidden, cell),
                path,
                opset_version=OPSET,
                dynamo=False,
                input_names=["input", "h_0", "c_0"],
                output_names=["output", "h_n", "c_n"],
            )
        module.export_mode = False

        model = onnx.load(path)
        onnx.checker.check_model(model)
        lstm_nodes = [
            node for node in model.graph.node if node.op_type == "LSTM"
        ]
        self.assertEqual(len(lstm_nodes), 1)
        self.assertFalse(
            any(node.domain == "quant_lstm_onnx" for node in model.graph.node)
        )
        self.assertEqual(
            direction_attribute(lstm_nodes[0]),
            "bidirectional" if bidirectional else "forward",
        )
        self.assertEqual(
            sum(node.op_type == "LSTM" for node in model.graph.node), 1
        )

        node = lstm_nodes[0]
        actual_weight_ih = tensor_value(model, node.input[1])
        actual_weight_hh = tensor_value(model, node.input[2])
        actual_bias = tensor_value(model, node.input[3])
        suffixes = ("", "_reverse") if bidirectional else ("",)
        expected_weight_ih = []
        expected_weight_hh = []
        expected_bias = []
        for suffix in suffixes:
            expected_weight_ih.append(
                reorder_gates(
                    getattr(module, f"weight_ih_l0{suffix}")
                    .detach()
                    .numpy()
                )
            )
            expected_weight_hh.append(
                reorder_gates(
                    getattr(module, f"weight_hh_l0{suffix}")
                    .detach()
                    .numpy()
                )
            )
            if bias:
                bias_ih = reorder_gates(
                    getattr(module, f"bias_ih_l0{suffix}")
                    .detach()
                    .numpy()
                )
                bias_hh = reorder_gates(
                    getattr(module, f"bias_hh_l0{suffix}")
                    .detach()
                    .numpy()
                )
            else:
                bias_ih = np.zeros(16, dtype=np.float32)
                bias_hh = np.zeros(16, dtype=np.float32)
            expected_bias.append(np.concatenate((bias_ih, bias_hh)))
        np.testing.assert_array_equal(
            actual_weight_ih, np.stack(expected_weight_ih)
        )
        np.testing.assert_array_equal(
            actual_weight_hh, np.stack(expected_weight_hh)
        )
        np.testing.assert_array_equal(actual_bias, np.stack(expected_bias))

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        actual_values = session.run(
            ["output", "h_n", "c_n"],
            {
                "input": input_value.numpy(),
                "h_0": hidden.numpy(),
                "c_0": cell.numpy(),
            },
        )
        expected_values = (expected[0], expected[1][0], expected[1][1])
        for name, actual, expected_value in zip(
            ("output", "h_n", "c_n"), actual_values, expected_values
        ):
            self.assert_metrics(
                case, name, torch.from_numpy(actual), expected_value
            )

    def test_standard_lstm_structure_and_runtime_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            for bidirectional in (False, True):
                for bias in (False, True):
                    for batch_first in (False, True):
                        with self.subTest(
                            bidirectional=bidirectional,
                            bias=bias,
                            batch_first=batch_first,
                        ):
                            self.export_case(
                                directory,
                                bidirectional,
                                bias,
                                batch_first,
                            )

    def test_export_mode_rejects_eager_execution(self):
        module = QuantLSTM(3, 4)
        module.export_mode = True
        with self.assertRaisesRegex(RuntimeError, "仅用于"):
            module(torch.zeros((2, 1, 3)))


if __name__ == "__main__":
    unittest.main()
