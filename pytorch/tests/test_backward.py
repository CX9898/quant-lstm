"""阶段 8 QuantLSTM 浮点 backward 与 QAT 验收。"""

import json
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import lstm_autograd
from quant_lstm import QuantLSTM
from tests.test_quantized_interface import (
    calibrate,
    copy_parameters,
    deterministic_tensor,
    metrics,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/precision/results/stage8_backward_report.json"


def initialize_parameters(module):
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            parameter.copy_(
                deterministic_tensor(
                    parameter.shape,
                    -0.13 + 0.004 * index,
                    0.15 - 0.003 * index,
                    device=parameter.device,
                )
            )


def copy_to_native(source, target):
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            getattr(target, name).copy_(parameter)


def objective(result):
    output, (hidden, cell) = result
    return (
        output.square().mean()
        + 0.37 * hidden.square().mean()
        + 0.19 * cell.square().mean()
        + 0.01 * output.mean()
    )


def gradient_items(module, input_tensor, state):
    result = [
        ("input", input_tensor.grad),
        ("h_0", state[0].grad),
        ("c_0", state[1].grad),
    ]
    result.extend(
        (name, parameter.grad) for name, parameter in module.named_parameters()
    )
    return result


def prepare_quantized(module, calibration_input, calibration_state):
    module.set_all_bitwidth(16)
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        module.adjust_quant_config(name, granularity="per_tensor")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        calibrate(module, calibration_input, calibration_state)
    module.use_quantization = True
    module.train()


def qat_backward_reference(
    module, grad_output, grad_hidden, grad_cell
):
    state = module.qat_saved_state()
    bundle = json.loads(module._quant_params_bundle_json)
    operators = bundle["operators"]
    masters = state["quantized_master"]
    master_masks = state["master_clamp_masks"]
    checkpoint_values = state["checkpoints"]
    checkpoint_masks = state["checkpoint_clamp_masks"]

    input_value = lstm_autograd._dequantize_tensor(
        masters["input"], operators["input"]
    )
    weight_ih = lstm_autograd._dequantize_tensor(
        masters["weight_ih"], operators["weight_ih"], True
    )
    weight_hh = lstm_autograd._dequantize_tensor(
        masters["weight_hh"], operators["weight_hh"], True
    )
    initial_hidden = lstm_autograd._dequantize_tensor(
        masters["h_0"], operators["output"]
    )
    initial_cell = lstm_autograd._dequantize_tensor(
        masters["c_0"], operators["cell_state"]
    )
    trace = {
        "gate_outputs": lstm_autograd._dequantize_gates(
            checkpoint_values["gate_outputs"],
            bundle,
            lstm_autograd._GATE_OUTPUT_OPERATORS,
        ),
        "cell_states": lstm_autograd._dequantize_tensor(
            checkpoint_values["cell_states"], operators["cell_state"]
        ),
        "cell_tanh_outputs": lstm_autograd._dequantize_tensor(
            checkpoint_values["cell_tanh_outputs"],
            operators["cell_tanh_output"],
        ),
        "hidden_outputs": lstm_autograd._dequantize_tensor(
            checkpoint_values["hidden_outputs"], operators["output"]
        ),
    }
    gradients = list(
        lstm_autograd._lstm_backward(
            lstm_autograd._as_time_major(input_value, module.batch_first),
            weight_ih,
            weight_hh,
            initial_hidden[0],
            initial_cell[0],
            trace,
            lstm_autograd._as_time_major(grad_output, module.batch_first),
            grad_hidden[0],
            grad_cell[0],
            checkpoint_masks,
        )
    )
    if module.batch_first:
        gradients[0] = gradients[0].transpose(0, 1)
    master_mask_order = (
        "input",
        "weight_ih",
        "weight_hh",
        "bias_ih",
        "bias_hh",
        "h_0",
        "c_0",
    )
    for index, name in enumerate(master_mask_order):
        if name not in master_masks:
            continue
        mask = master_masks[name]
        if name in ("h_0", "c_0"):
            mask = mask[0]
        gradients[index] *= lstm_autograd._keep_gradient(
            mask, gradients[index]
        )
    return gradients


class BackwardTest(unittest.TestCase):
    records = []

    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(
                {
                    "stage": 8,
                    "validation_scope": "synthetic_numeric",
                    "real_data_status": "not_configured",
                    "records": cls.records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def assert_gradient_metrics(
        self, path, name, actual, expected, *, quantized=False
    ):
        self.assertIsNotNone(actual, name)
        self.assertIsNotNone(expected, name)
        mae, mse, cosine = metrics(actual, expected)
        self.records.append(
            {
                "path": path,
                "tensor": name,
                "mae": mae,
                "mse": mse,
                "cosine": cosine,
                "cosine_status": "valid",
            }
        )
        if quantized:
            self.assertLess(mae, 0.015, f"{name}: MAE={mae}")
            self.assertLess(mse, 1.0e-4, f"{name}: MSE={mse}")
            self.assertGreaterEqual(cosine, 0.999, f"{name}: cosine={cosine}")
        else:
            self.assertLess(mae, 0.003, f"{name}: MAE={mae}")
            self.assertLess(mse, 1.0e-5, f"{name}: MSE={mse}")
            self.assertGreaterEqual(cosine, 0.9999, f"{name}: cosine={cosine}")
            tolerance = 1.0e-5 if actual.device.type == "cpu" else 5.0e-4
            self.assertTrue(
                torch.allclose(
                    actual, expected, atol=tolerance, rtol=tolerance
                ),
                f"{name}: max_abs={(actual - expected).abs().max().item()}",
            )

    def test_float_backward_matches_pytorch(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for bidirectional in (False, True):
                for bias in (False, True):
                    for batch_first in (False, True):
                        case = (
                            f"float_{device.type}_bi{int(bidirectional)}_"
                            f"bias{int(bias)}_bf{int(batch_first)}"
                        )
                        with self.subTest(case=case):
                            custom = QuantLSTM(
                                3,
                                4,
                                bias=bias,
                                batch_first=batch_first,
                                bidirectional=bidirectional,
                                device=device,
                            )
                            native = nn.LSTM(
                                3,
                                4,
                                bias=bias,
                                batch_first=batch_first,
                                bidirectional=bidirectional,
                                device=device,
                            )
                            initialize_parameters(custom)
                            copy_to_native(custom, native)
                            input_time = deterministic_tensor(
                                (4, 2, 3), -0.31, 0.29, device=device
                            )
                            input_value = (
                                input_time.transpose(0, 1).contiguous()
                                if batch_first
                                else input_time
                            )
                            directions = 2 if bidirectional else 1
                            hidden = deterministic_tensor(
                                (directions, 2, 4),
                                -0.12,
                                0.11,
                                device=device,
                            )
                            cell = deterministic_tensor(
                                (directions, 2, 4),
                                -0.21,
                                0.18,
                                device=device,
                            )
                            custom_input = input_value.clone().requires_grad_()
                            native_input = input_value.clone().requires_grad_()
                            custom_state = (
                                hidden.clone().requires_grad_(),
                                cell.clone().requires_grad_(),
                            )
                            native_state = (
                                hidden.clone().requires_grad_(),
                                cell.clone().requires_grad_(),
                            )
                            objective(
                                custom(custom_input, custom_state)
                            ).backward()
                            objective(
                                native(native_input, native_state)
                            ).backward()
                            actual = gradient_items(
                                custom, custom_input, custom_state
                            )
                            expected = gradient_items(
                                native, native_input, native_state
                            )
                            self.assertEqual(
                                [item[0] for item in actual],
                                [item[0] for item in expected],
                            )
                            for (name, value), (_, reference) in zip(
                                actual, expected
                            ):
                                self.assert_gradient_metrics(
                                    case, name, value, reference
                                )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_float_cuda_backward_bypasses_python_reference(self):
        module = QuantLSTM(3, 4, device="cuda")
        initialize_parameters(module)
        input_value = deterministic_tensor(
            (4, 2, 3), -0.31, 0.29, device="cuda"
        ).requires_grad_()
        with mock.patch.object(
            lstm_autograd,
            "_lstm_backward",
            side_effect=AssertionError("CUDA float backward used Python fallback"),
        ):
            output, _ = module(input_value)
            output.square().mean().backward()
        self.assertIsNotNone(input_value.grad)
        self.assertGreater(input_value.grad.abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_gradients_match_float_surrogate(self):
        device = torch.device("cuda")
        for bidirectional, bias in ((False, True), (True, False)):
            case = f"qat_bi{int(bidirectional)}_bias{int(bias)}"
            with self.subTest(case=case):
                qat = QuantLSTM(
                    3,
                    4,
                    bias=bias,
                    bidirectional=bidirectional,
                    device=device,
                )
                initialize_parameters(qat)
                input_value = deterministic_tensor(
                    (4, 2, 3), -0.20, 0.25, device=device
                )
                directions = 2 if bidirectional else 1
                hidden = deterministic_tensor(
                    (directions, 2, 4), -0.05, 0.06, device=device
                )
                cell = deterministic_tensor(
                    (directions, 2, 4), -0.10, 0.09, device=device
                )
                prepare_quantized(
                    qat,
                    input_value * 1.5,
                    (hidden * 1.5, cell * 1.5),
                )
                reference = QuantLSTM(
                    3,
                    4,
                    bias=bias,
                    bidirectional=bidirectional,
                    device=device,
                )
                copy_parameters(qat, reference)
                reference.train()

                qat_input = input_value.clone().requires_grad_()
                reference_input = input_value.clone().requires_grad_()
                qat_state = (
                    hidden.clone().requires_grad_(),
                    cell.clone().requires_grad_(),
                )
                reference_state = (
                    hidden.clone().requires_grad_(),
                    cell.clone().requires_grad_(),
                )
                objective(qat(qat_input, qat_state)).backward()
                objective(
                    reference(reference_input, reference_state)
                ).backward()
                actual = gradient_items(qat, qat_input, qat_state)
                expected = gradient_items(
                    reference, reference_input, reference_state
                )
                for (name, value), (_, expected_value) in zip(
                    actual, expected
                ):
                    self.assert_gradient_metrics(
                        case,
                        name,
                        value,
                        expected_value,
                        quantized=True,
                    )
                saved = qat.qat_saved_state()
                if bidirectional:
                    self.assertEqual(set(saved), {"forward", "reverse"})
                else:
                    self.assertIn("checkpoint_clamp_masks", saved)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_native_backward_matches_python_ste_oracle(self):
        device = torch.device("cuda")
        module = QuantLSTM(3, 4, batch_first=True, device=device)
        initialize_parameters(module)
        calibration_input = deterministic_tensor(
            (2, 4, 3), -0.04, 0.05, device=device
        )
        calibration_state = (
            deterministic_tensor((1, 2, 4), -0.03, 0.03, device=device),
            deterministic_tensor((1, 2, 4), -0.04, 0.04, device=device),
        )
        module.set_all_bitwidth(8)
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            module.adjust_quant_config(name, granularity="per_tensor")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibrate(module, calibration_input, calibration_state)
        module.use_quantization = True
        module.train()

        input_value = (calibration_input * 24.0).clone().requires_grad_()
        state = (
            (calibration_state[0] * 16.0).clone().requires_grad_(),
            (calibration_state[1] * 16.0).clone().requires_grad_(),
        )
        output, (hidden, cell) = module(input_value, state)
        grad_output = deterministic_tensor(
            output.shape, -0.08, 0.09, device=device
        )
        grad_hidden = deterministic_tensor(
            hidden.shape, -0.07, 0.06, device=device
        )
        grad_cell = deterministic_tensor(
            cell.shape, -0.05, 0.08, device=device
        )
        reference = qat_backward_reference(
            module, grad_output, grad_hidden, grad_cell
        )
        masks = module.qat_saved_state()["checkpoint_clamp_masks"]
        self.assertGreaterEqual(
            sum(bool(mask.any()) for mask in masks.values()), 5
        )
        torch.autograd.backward(
            (output, hidden, cell),
            (grad_output, grad_hidden, grad_cell),
        )
        actual = [
            input_value.grad,
            module.weight_ih_l0.grad,
            module.weight_hh_l0.grad,
            module.bias_ih_l0.grad,
            module.bias_hh_l0.grad,
            state[0].grad[0],
            state[1].grad[0],
        ]
        for name, value, expected in zip(
            (
                "input",
                "weight_ih",
                "weight_hh",
                "bias_ih",
                "bias_hh",
                "h_0",
                "c_0",
            ),
            actual,
            reference,
        ):
            self.assertTrue(
                torch.allclose(value, expected, atol=5.0e-5, rtol=5.0e-5),
                f"{name}: max_abs={(value - expected).abs().max().item()}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_cuda_backward_bypasses_python_reference(self):
        module = QuantLSTM(3, 4, device="cuda")
        initialize_parameters(module)
        calibration_input = deterministic_tensor(
            (4, 2, 3), -0.20, 0.25, device="cuda"
        )
        prepare_quantized(module, calibration_input, None)
        input_value = calibration_input.clone().requires_grad_()
        with mock.patch.object(
            lstm_autograd,
            "_lstm_backward",
            side_effect=AssertionError("CUDA QAT used Python fallback"),
        ):
            output, _ = module(input_value)
            output.square().mean().backward()
        self.assertIsNotNone(input_value.grad)
        self.assertGreater(input_value.grad.abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_single_step_and_loss_decline(self):
        device = torch.device("cuda")
        module = QuantLSTM(2, 3, device=device)
        initialize_parameters(module)
        input_value = deterministic_tensor(
            (5, 2, 2), -0.18, 0.22, device=device
        )
        prepare_quantized(module, input_value * 1.5, None)
        target = torch.zeros((5, 2, 3), device=device)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.8)
        initial_weight = module.weight_ih_l0.detach().clone()
        losses = []
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            output, _ = module(input_value)
            loss = torch.nn.functional.mse_loss(output, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(initial_weight, module.weight_ih_l0))
        self.assertLess(losses[-1], losses[0] * 0.8)
        self.records.append(
            {
                "path": "qat_optimization",
                "tensor": "loss",
                "initial": losses[0],
                "final": losses[-1],
                "steps": len(losses),
            }
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_clamp_mask_blocks_only_saturated_master_gradient(self):
        device = torch.device("cuda")
        module = QuantLSTM(3, 4, device=device)
        initialize_parameters(module)
        calibration_input = deterministic_tensor(
            (4, 2, 3), -0.20, 0.25, device=device
        )
        prepare_quantized(module, calibration_input, None)
        input_value = (calibration_input * 0.5).clone()
        input_value[0, 0, 0] = 1.0
        input_value.requires_grad_()
        output, (hidden, cell) = module(input_value)
        (output.sum() + hidden.sum() + 0.1 * cell.sum()).backward()

        mask = module.qat_saved_state()["master_clamp_masks"]["input"]
        self.assertTrue(mask[0, 0, 0])
        self.assertEqual(input_value.grad[0, 0, 0].item(), 0.0)
        unclamped = ~mask
        self.assertTrue(unclamped.any())
        self.assertGreater(input_value.grad[unclamped].abs().max().item(), 0.0)
        self.records.append(
            {
                "path": "qat_clamp_ste",
                "tensor": "input",
                "clamped_count": int(mask.sum().item()),
                "unclamped_count": int(unclamped.sum().item()),
                "clamped_gradient": input_value.grad[0, 0, 0].item(),
                "max_unclamped_gradient": input_value.grad[unclamped]
                .abs()
                .max()
                .item(),
            }
        )


if __name__ == "__main__":
    unittest.main()
