"""阶段 8 QuantLSTM 浮点 backward 与 QAT 验收。"""

import json
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import _quant_lstm
from quant_lstm import QuantLSTM
from tests import lstm_backward_oracle as oracle
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
            tolerance = 5.0e-4
            self.assertTrue(
                torch.allclose(
                    actual, expected, atol=tolerance, rtol=tolerance
                ),
                f"{name}: max_abs={(actual - expected).abs().max().item()}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_float_backward_matches_pytorch(self):
        device = torch.device("cuda")
        for bidirectional in (False, True):
            for bias in (False, True):
                for batch_first in (False, True):
                    case = (
                        f"float_cuda_bi{int(bidirectional)}_"
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
                        objective(custom(custom_input, custom_state)).backward()
                        objective(native(native_input, native_state)).backward()
                        actual = gradient_items(custom, custom_input, custom_state)
                        expected = gradient_items(native, native_input, native_state)
                        self.assertEqual(
                            [item[0] for item in actual],
                            [item[0] for item in expected],
                        )
                        for (name, value), (_, reference) in zip(actual, expected):
                            self.assert_gradient_metrics(case, name, value, reference)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_float_cuda_backward_bypasses_python_reference(self):
        module = QuantLSTM(3, 4, device="cuda")
        initialize_parameters(module)
        input_value = deterministic_tensor(
            (4, 2, 3), -0.31, 0.29, device="cuda"
        ).requires_grad_()
        with mock.patch.object(
            _quant_lstm,
            "lstm_backward_float",
            wraps=_quant_lstm.lstm_backward_float,
        ) as native_backward:
            output, _ = module(input_value)
            output.square().mean().backward()
        native_backward.assert_called_once()
        self.assertIsNotNone(input_value.grad)
        self.assertGreater(input_value.grad.abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_float_training_tracks_pytorch_for_100_steps(self):
        device = torch.device("cuda")
        custom = QuantLSTM(2, 3, device=device, use_quantization=False)
        native = nn.LSTM(2, 3, device=device)
        initialize_parameters(custom)
        copy_to_native(custom, native)
        custom.train()
        native.train()

        input_value = deterministic_tensor(
            (5, 2, 2), -0.31, 0.29, device=device
        )
        target = deterministic_tensor(
            (5, 2, 3), -0.17, 0.23, device=device
        )
        custom_optimizer = torch.optim.SGD(custom.parameters(), lr=0.4)
        native_optimizer = torch.optim.SGD(native.parameters(), lr=0.4)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in custom.named_parameters()
        }
        custom_losses = []
        max_loss_difference = 0.0
        minimum_output_cosine = 1.0

        for _ in range(100):
            custom_optimizer.zero_grad(set_to_none=True)
            native_optimizer.zero_grad(set_to_none=True)
            custom_output, _ = custom(input_value)
            native_output, _ = native(input_value)
            custom_loss = torch.nn.functional.mse_loss(custom_output, target)
            native_loss = torch.nn.functional.mse_loss(native_output, target)
            custom_losses.append(custom_loss.item())
            max_loss_difference = max(
                max_loss_difference,
                abs(custom_loss.item() - native_loss.item()),
            )
            minimum_output_cosine = min(
                minimum_output_cosine,
                metrics(custom_output, native_output)[2],
            )
            custom_loss.backward()
            native_loss.backward()
            custom_optimizer.step()
            native_optimizer.step()

        parameter_mse = {
            name: torch.mean(
                (parameter.detach() - getattr(native, name).detach()).square()
            ).item()
            for name, parameter in custom.named_parameters()
        }
        parameter_update_norm = sum(
            (parameter.detach() - initial_parameters[name]).square().sum().item()
            for name, parameter in custom.named_parameters()
        ) ** 0.5

        self.assertLess(custom_losses[-1], custom_losses[0] * 0.8)
        self.assertGreater(parameter_update_norm, 0.0)
        self.assertGreaterEqual(minimum_output_cosine, 0.9999)
        self.assertLess(max(parameter_mse.values()), 1.0e-5)
        self.assertLess(max_loss_difference, 1.0e-4)
        self.records.append(
            {
                "path": "float_optimization_100_steps",
                "tensor": "training",
                "initial_loss": custom_losses[0],
                "final_loss": custom_losses[-1],
                "max_loss_difference": max_loss_difference,
                "minimum_output_cosine": minimum_output_cosine,
                "max_parameter_mse": max(parameter_mse.values()),
                "parameter_update_norm": parameter_update_norm,
                "steps": len(custom_losses),
            }
        )

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
    def test_qat_calibrated_parameter_endpoints_keep_ste_gradients(self):
        device = torch.device("cuda")
        input_value = deterministic_tensor(
            (4, 2, 3), -0.20, 0.25, device=device
        )
        hidden = deterministic_tensor(
            (1, 2, 4), -0.05, 0.06, device=device
        )
        cell = deterministic_tensor(
            (1, 2, 4), -0.10, 0.09, device=device
        )

        forward_mae = {}
        for bitwidth in (8, 16):
            with self.subTest(bitwidth=bitwidth):
                qat = QuantLSTM(3, 4, device=device)
                initialize_parameters(qat)
                qat.set_all_bitwidth(bitwidth)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    calibrate(qat, input_value, (hidden, cell))
                qat.use_quantization = True
                qat.train()

                reference = QuantLSTM(3, 4, device=device)
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
                qat_result = qat(qat_input, qat_state)
                reference_result = reference(reference_input, reference_state)
                forward_mae[bitwidth] = metrics(
                    qat_result[0], reference_result[0]
                )[0]
                objective(qat_result).backward()
                objective(reference_result).backward()

                saved = qat.qat_saved_state()
                for name in (
                    "weight_ih",
                    "weight_hh",
                    "bias_ih",
                    "bias_hh",
                ):
                    self.assertFalse(
                        saved["master_clamp_masks"][name].any(), name
                    )

                for name in ("bias_ih_l0", "bias_hh_l0"):
                    actual = getattr(qat, name).grad
                    expected = getattr(reference, name).grad
                    _, _, cosine = metrics(actual, expected)
                    self.assertGreaterEqual(
                        cosine, 0.98, f"{name}: cosine={cosine}"
                    )

        self.assertLess(forward_mae[16], forward_mae[8] * 0.1)

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
        reference = oracle.qat_backward_reference(
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
            _quant_lstm,
            "lstm_backward_qat",
            wraps=_quant_lstm.lstm_backward_qat,
        ) as native_backward:
            output, _ = module(input_value)
            output.square().mean().backward()
        native_backward.assert_called_once()
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

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_bias_clamp_lifecycle_matches_round_clamp(self):
        device = torch.device("cuda")
        input_value = deterministic_tensor(
            (4, 2, 3), -0.20, 0.25, device=device
        )
        first_clamped_steps = {}
        for bitwidth in (8, 16):
            with self.subTest(bitwidth=bitwidth):
                module = QuantLSTM(3, 4, device=device)
                initialize_parameters(module)
                module.set_all_bitwidth(bitwidth)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    calibrate(module, input_value, None)
                module.use_quantization = True
                module.train()
                bundle = json.loads(module._quant_params_bundle_json)
                optimizer = torch.optim.SGD(module.parameters(), lr=0.8)
                first_clamped_step = {"bias_ih": None, "bias_hh": None}

                for step in range(6):
                    optimizer.zero_grad(set_to_none=True)
                    output, (hidden, cell) = module(input_value)
                    saved = module.qat_saved_state()
                    for name in first_clamped_step:
                        parameter = getattr(module, f"{name}_l0")
                        _, expected_mask = oracle.quantize_tensor(
                            parameter, bundle["operators"][name], True
                        )
                        actual_mask = saved["master_clamp_masks"][name]
                        self.assertTrue(
                            torch.equal(actual_mask, expected_mask), name
                        )
                        if (
                            actual_mask.any()
                            and first_clamped_step[name] is None
                        ):
                            first_clamped_step[name] = step

                    loss = output.sum() + hidden.sum() + 0.1 * cell.sum()
                    loss.backward()
                    optimizer.step()

                self.assertEqual(
                    first_clamped_step, {"bias_ih": 1, "bias_hh": 1}
                )
                first_clamped_steps[str(bitwidth)] = first_clamped_step
        self.records.append(
            {
                "path": "qat_bias_clamp_lifecycle",
                "tensor": "bias",
                "first_clamped_step": first_clamped_steps,
            }
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_qat_bias_ste_blocks_only_saturated_channels(self):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibrate(module, calibration_input, calibration_state)
        bundle = json.loads(module._quant_params_bundle_json)

        with torch.no_grad():
            for name in ("bias_ih", "bias_hh"):
                parameter = getattr(module, f"{name}_l0")
                operator = bundle["operators"][name]
                scales = torch.tensor(
                    [float(value) for value in operator["scales"]],
                    device=device,
                )
                qmax = (1 << (operator["bitwidth"] - 1)) - 1
                parameter.mul_(0.5)
                parameter[0] = (qmax + 2) * scales[0]
                parameter[1] = (-qmax - 2) * scales[1]

        module.use_quantization = True
        module.train()
        input_value = calibration_input.clone().requires_grad_()
        state = tuple(value.clone().requires_grad_() for value in calibration_state)
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
        expected_gradients = oracle.qat_backward_reference(
            module, grad_output, grad_hidden, grad_cell
        )
        torch.autograd.backward(
            (output, hidden, cell),
            (grad_output, grad_hidden, grad_cell),
        )

        for gradient_index, name in enumerate(("bias_ih", "bias_hh"), 3):
            parameter = getattr(module, f"{name}_l0")
            mask = module.qat_saved_state()["master_clamp_masks"][name]
            _, expected_mask = oracle.quantize_tensor(
                parameter, bundle["operators"][name], True
            )
            self.assertTrue(torch.equal(mask, expected_mask), name)
            self.assertEqual(mask.nonzero().flatten().tolist(), [0, 1])
            self.assertTrue(
                torch.equal(parameter.grad[mask], torch.zeros(2, device=device))
            )
            self.assertGreater(parameter.grad[~mask].abs().max().item(), 0.0)
            self.assertTrue(
                torch.allclose(
                    parameter.grad,
                    expected_gradients[gradient_index],
                    atol=5.0e-5,
                    rtol=5.0e-5,
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()
