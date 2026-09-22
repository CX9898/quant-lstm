"""阶段 7 QuantLSTM 双向接口验收。"""

import json
import unittest
from pathlib import Path

import jsonschema
import torch
from torch import nn

import _quant_lstm
from quant_lstm import QuantLSTM
from tests.test_quantized_interface import (
    calibrate,
    copy_parameters,
    deterministic_tensor,
    flatten_result,
    metrics,
)


ROOT = Path(__file__).resolve().parents[2]
BIDIRECTIONAL_SCHEMA = json.loads(
    (
        ROOT
        / "config/schema/lstm_pytorch_bidirectional_quant_params.schema.json"
    ).read_text()
)


def initialize_bidirectional(module):
    with torch.no_grad():
        for index, (_, parameter) in enumerate(module.named_parameters()):
            start = -0.16 + 0.01 * index
            stop = 0.19 - 0.008 * index
            parameter.copy_(
                deterministic_tensor(
                    parameter.shape,
                    start,
                    stop,
                    device=parameter.device,
                )
            )


def copy_to_native(custom, native):
    with torch.no_grad():
        for name, parameter in custom.named_parameters():
            getattr(native, name).copy_(parameter)


class BidirectionalInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def assert_float_metrics(self, name, actual, expected, device):
        mae, mse, cosine = metrics(actual, expected)
        self.assertLess(mae, 0.003, f"{name}: MAE={mae}")
        self.assertLess(mse, 1.0e-5, f"{name}: MSE={mse}")
        self.assertGreaterEqual(cosine, 0.9999, f"{name}: cosine={cosine}")
        tolerance = 5.0e-4
        self.assertTrue(
            torch.allclose(actual, expected, atol=tolerance, rtol=tolerance),
            f"{name}: max_abs={(actual - expected).abs().max().item()}",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_float_matches_pytorch_order_and_shapes(self):
        device = torch.device("cuda")
        for batch_first in (False, True):
            with self.subTest(batch_first=batch_first):
                custom = QuantLSTM(
                    3,
                    4,
                    bidirectional=True,
                    batch_first=batch_first,
                    device=device,
                )
                native = nn.LSTM(
                    3,
                    4,
                    bidirectional=True,
                    batch_first=batch_first,
                    device=device,
                )
                initialize_bidirectional(custom)
                copy_to_native(custom, native)
                input_time = deterministic_tensor(
                    (5, 2, 3), -0.28, 0.33, device=device
                )
                input_tensor = (
                    input_time.transpose(0, 1).contiguous()
                    if batch_first
                    else input_time
                )
                state = (
                    deterministic_tensor((2, 2, 4), -0.11, 0.13, device=device),
                    deterministic_tensor((2, 2, 4), -0.18, 0.21, device=device),
                )
                custom.eval()
                native.eval()
                with torch.no_grad():
                    actual = custom(input_tensor, state)
                    expected = native(input_tensor, state)
                self.assertEqual(
                    actual[0].shape,
                    (2, 5, 8) if batch_first else (5, 2, 8),
                )
                self.assertEqual(actual[1][0].shape, (2, 2, 4))
                self.assertEqual(actual[1][1].shape, (2, 2, 4))
                for name, actual_value, expected_value in zip(
                    ("output", "h_n", "c_n"),
                    (actual[0], *actual[1]),
                    (expected[0], *expected[1]),
                ):
                    self.assert_float_metrics(name, actual_value, expected_value, device)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_batch_first_layout_is_exactly_equivalent(self):
        device = torch.device("cuda")
        time_module = QuantLSTM(3, 4, bidirectional=True, device=device)
        batch_module = QuantLSTM(
            3, 4, bidirectional=True, batch_first=True, device=device
        )
        initialize_bidirectional(time_module)
        copy_parameters(time_module, batch_module)
        input_time = deterministic_tensor(
            (4, 2, 3), -0.25, 0.31, device=device
        )
        state = (
            deterministic_tensor((2, 2, 4), -0.1, 0.12, device=device),
            deterministic_tensor((2, 2, 4), -0.2, 0.17, device=device),
        )
        time_module.eval()
        batch_module.eval()
        with torch.no_grad():
            time_result = time_module(input_time, state)
            batch_result = batch_module(
                input_time.transpose(0, 1).contiguous(), state
            )
        for time_value, batch_value in zip(
            flatten_result(time_result, False),
            flatten_result(batch_result, True),
        ):
            self.assertTrue(torch.equal(time_value, batch_value))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_quantized_directions_roundtrip_and_direct_core(self):
        time_module = QuantLSTM(3, 4, bidirectional=True, device="cuda")
        batch_module = QuantLSTM(
            3, 4, bidirectional=True, batch_first=True, device="cuda"
        )
        initialize_bidirectional(time_module)
        copy_parameters(time_module, batch_module)
        input_time = deterministic_tensor(
            (4, 2, 3), -0.3, 0.36, device="cuda"
        )
        input_batch = input_time.transpose(0, 1).contiguous()
        state = (
            deterministic_tensor(
                (2, 2, 4), -0.09, 0.12, device="cuda"
            ),
            deterministic_tensor(
                (2, 2, 4), -0.16, 0.18, device="cuda"
            ),
        )
        calibrate(time_module, input_time, state)
        calibrate(batch_module, input_batch, state)
        document = time_module.export_quant_params()
        jsonschema.Draft202012Validator(
            BIDIRECTIONAL_SCHEMA
        ).validate(document)
        self.assertEqual(
            set(document),
            {
                "schema_version",
                "model_info",
                "execution_metadata",
                "operators",
                "operators_reverse",
            },
        )
        self.assertTrue(document["model_info"]["bidirectional"])
        self.assertEqual(
            document["operators"]["input"],
            document["operators_reverse"]["input"],
        )
        self.assertNotEqual(
            document["operators"]["weight_ih"]["scale"],
            document["operators_reverse"]["weight_ih"]["scale"],
        )
        batch_document = batch_module.export_quant_params()
        self.assertFalse(document["model_info"]["batch_first"])
        self.assertTrue(batch_document["model_info"]["batch_first"])
        self.assertEqual(document["operators"], batch_document["operators"])
        self.assertEqual(
            document["operators_reverse"],
            batch_document["operators_reverse"],
        )

        time_module.use_quantization = True
        batch_module.use_quantization = True
        time_module.eval()
        batch_module.eval()
        with torch.no_grad():
            time_result = time_module(input_time, state)
            batch_result = batch_module(input_batch, state)
        for time_value, batch_value in zip(
            flatten_result(time_result, False),
            flatten_result(batch_result, True),
        ):
            self.assertTrue(torch.equal(time_value, batch_value))

        forward = _quant_lstm.lstm_forward_quantized(
            input_time,
            time_module.weight_ih_l0,
            time_module.weight_hh_l0,
            time_module.bias_ih_l0,
            time_module.bias_hh_l0,
            state[0][0:1],
            state[1][0:1],
            False,
            time_module._quant_params_bundle_json,
            "pedantic",
            False,
            False,
        )
        reverse_input = torch.flip(input_time, dims=(0,))
        reverse = _quant_lstm.lstm_forward_quantized(
            reverse_input,
            time_module.weight_ih_l0_reverse,
            time_module.weight_hh_l0_reverse,
            time_module.bias_ih_l0_reverse,
            time_module.bias_hh_l0_reverse,
            state[0][1:2],
            state[1][1:2],
            False,
            time_module._reverse_quant_params_bundle_json,
            "pedantic",
            False,
            False,
        )
        expected_output = torch.cat(
            (forward[0], torch.flip(reverse[0], dims=(0,))), dim=-1
        )
        expected_hidden = torch.cat((forward[1], reverse[1]), dim=0)
        expected_cell = torch.cat((forward[2], reverse[2]), dim=0)
        self.assertTrue(torch.equal(time_result[0], expected_output))
        self.assertTrue(torch.equal(time_result[1][0], expected_hidden))
        self.assertTrue(torch.equal(time_result[1][1], expected_cell))

        time_module.train()
        with torch.no_grad():
            training_result = time_module(input_time, state)
        for expected, actual in zip(
            flatten_result(time_result, False),
            flatten_result(training_result, False),
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            set(time_module.qat_saved_state()), {"forward", "reverse"}
        )

        imported = QuantLSTM(3, 4, bidirectional=True, device="cuda")
        copy_parameters(time_module, imported)
        imported.load_quant_params(document)
        imported.use_quantization = True
        imported.eval()
        with torch.no_grad():
            imported_result = imported(input_time, state)
        for expected, actual in zip(
            flatten_result(time_result, False),
            flatten_result(imported_result, False),
        ):
            self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_bias_disabled_document_omits_both_direction_biases(self):
        module = QuantLSTM(
            2, 3, bias=False, bidirectional=True, device="cuda"
        )
        initialize_bidirectional(module)
        input_tensor = deterministic_tensor((3, 2, 2), device="cuda")
        state = (
            deterministic_tensor((2, 2, 3), -0.1, 0.1, device="cuda"),
            deterministic_tensor((2, 2, 3), -0.2, 0.2, device="cuda"),
        )
        calibrate(module, input_tensor, state)

        document = module.export_quant_params()
        jsonschema.Draft202012Validator(BIDIRECTIONAL_SCHEMA).validate(document)
        for field in ("operators", "operators_reverse"):
            self.assertNotIn("bias_ih", document[field])
            self.assertNotIn("bias_hh", document[field])

        imported = QuantLSTM(2, 3, bias=False, bidirectional=True)
        imported.load_quant_params(document)
        self.assertTrue(imported.is_calibrated())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_bidirectional_contract_errors(self):
        module = QuantLSTM(3, 4, bidirectional=True, device="cuda")
        input_tensor = torch.zeros((3, 2, 3), device="cuda")
        invalid_state = (
            torch.zeros((1, 2, 4), device="cuda"),
            torch.zeros((1, 2, 4), device="cuda"),
        )
        with self.assertRaisesRegex(RuntimeError, "\\[2,B,H\\]"):
            module(input_tensor, invalid_state)

        unidirectional = QuantLSTM(3, 4)
        initialize_bidirectional(module)
        calibrate(
            module,
            input_tensor,
            (
                torch.zeros((2, 2, 4), device="cuda"),
                torch.zeros((2, 2, 4), device="cuda"),
            ),
        )
        document = module.export_quant_params()
        with self.assertRaises(ValueError):
            unidirectional.load_quant_params(document)

        tampered = json.loads(json.dumps(document))
        tampered["operators_reverse"]["input"]["scale"] = 0.5
        with self.assertRaises(ValueError):
            QuantLSTM(3, 4, bidirectional=True).load_quant_params(tampered)


if __name__ == "__main__":
    unittest.main()
