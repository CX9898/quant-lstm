"""阶段 1 浮点 LSTM 的基础与严格精度测试。"""

import json
import os
import unittest

import torch
from torch import nn

import _quant_lstm_test
from quant_lstm import QuantLSTM


PROFILES = {
    "cpu_basic": ((8, 4, 16, 32), 3001),
    "cuda_gru_basic": ((50, 64, 128, 256), 3002),
    "minimal": ((1, 1, 1, 1), 3003),
    "short_recurrent": ((3, 1, 2, 2), 3004),
    "non_aligned": ((7, 3, 31, 33), 3005),
    "long_sequence": ((256, 1, 64, 128), 3006),
}

def generated(shape, seed, role, *, parameter=False, hidden_size=1):
    return _quant_lstm_test.generate_test_tensor(
        list(shape), seed, role, parameter, hidden_size
    )


def metrics(actual, expected):
    actual_64 = actual.detach().cpu().reshape(-1).double()
    expected_64 = expected.detach().cpu().reshape(-1).double()
    difference = actual_64 - expected_64
    mae = difference.abs().mean().item()
    mse = difference.square().mean().item()
    actual_norm = torch.linalg.vector_norm(actual_64).item()
    expected_norm = torch.linalg.vector_norm(expected_64).item()
    if actual_norm <= 1.0e-12 and expected_norm <= 1.0e-12:
        cosine = None
        cosine_status = "both_near_zero"
    elif actual_norm <= 1.0e-12 or expected_norm <= 1.0e-12:
        cosine = None
        cosine_status = "one_near_zero"
    else:
        cosine = torch.dot(actual_64, expected_64).item() / (actual_norm * expected_norm)
        cosine_status = "valid"
    return {
        "mae": mae,
        "mse": mse,
        "cosine": cosine,
        "cosine_status": cosine_status,
    }


class FloatReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def assert_metrics(self, name, actual, expected):
        self.assertEqual(actual.shape, expected.shape, f"{name} shape 不匹配")
        result = metrics(actual, expected)
        self.assertLess(result["mse"], 1.0e-5, f"{name}: {result}")
        self.assertLess(result["mae"], 0.003, f"{name}: {result}")
        self.assertNotEqual(result["cosine_status"], "one_near_zero", f"{name}: {result}")
        if result["cosine"] is not None:
            self.assertGreaterEqual(result["cosine"], 0.9999, f"{name}: {result}")
        self.assertTrue(
            torch.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5),
            f"{name}: max_abs={(actual - expected).abs().max().item()}",
        )
        return result

    def assert_layout_equivalent(self, time_major_result, batch_major_result):
        time_output, time_hidden, time_cell = time_major_result
        batch_output, batch_hidden, batch_cell = batch_major_result
        self.assertTrue(torch.equal(time_output, batch_output.transpose(0, 1)))
        self.assertTrue(torch.equal(time_hidden, batch_hidden))
        self.assertTrue(torch.equal(time_cell, batch_cell))

    def run_case(self, profile_name, device, bias, batch_first, state_profile):
        (steps, batch, input_size, hidden_size), parameter_seed = PROFILES[profile_name]
        input_time_major = generated(
            (steps, batch, input_size), 0 if "basic" in profile_name else 2001, "input"
        )
        input_tensor = (
            input_time_major.transpose(0, 1).contiguous()
            if batch_first
            else input_time_major
        ).to(device)

        native = nn.LSTM(
            input_size, hidden_size, bias=bias, batch_first=batch_first, device=device
        )
        custom = QuantLSTM(
            input_size, hidden_size, bias=bias, batch_first=batch_first, device=device
        )
        parameters = {
            "weight_ih_l0": generated(
                (4 * hidden_size, input_size),
                parameter_seed,
                "weight_ih",
                parameter=True,
                hidden_size=hidden_size,
            ),
            "weight_hh_l0": generated(
                (4 * hidden_size, hidden_size),
                parameter_seed,
                "weight_hh",
                parameter=True,
                hidden_size=hidden_size,
            ),
        }
        if bias:
            parameters["bias_ih_l0"] = generated(
                (4 * hidden_size,),
                parameter_seed,
                "bias_ih",
                parameter=True,
                hidden_size=hidden_size,
            )
            parameters["bias_hh_l0"] = generated(
                (4 * hidden_size,),
                parameter_seed,
                "bias_hh",
                parameter=True,
                hidden_size=hidden_size,
            )
        with torch.no_grad():
            for name, value in parameters.items():
                getattr(native, name).copy_(value.to(device))
                getattr(custom, name).copy_(value.to(device))

        if state_profile == "omitted":
            state = None
        elif state_profile == "explicit_zero":
            state = (
                torch.zeros((1, batch, hidden_size), device=device),
                torch.zeros((1, batch, hidden_size), device=device),
            )
        elif state_profile == "typical_random":
            state = (
                generated((1, batch, hidden_size), 2001, "h0").to(device),
                generated((1, batch, hidden_size), 2001, "c0").to(device),
            )
        else:
            raise AssertionError(f"未知 state_profile: {state_profile}")

        native.eval()
        custom.eval()
        with torch.no_grad():
            expected_output, (expected_hidden, expected_cell) = native(input_tensor, state)
            actual_output, (actual_hidden, actual_cell) = custom(input_tensor, state)

        report = {
            "rng_version": _quant_lstm_test.rng_version,
            "rng_stream_registry": _quant_lstm_test.rng_stream_registry,
            "distribution_profile": _quant_lstm_test.distribution_profile,
            "profile": profile_name,
            "shape": [steps, batch, input_size, hidden_size],
            "device": str(device),
            "bias": bias,
            "batch_first": batch_first,
            "state_profile": state_profile,
            "output": self.assert_metrics("output", actual_output, expected_output),
            "h_n": self.assert_metrics("h_n", actual_hidden, expected_hidden),
            "c_n": self.assert_metrics("c_n", actual_cell, expected_cell),
        }

        normalized_actual = actual_output.transpose(0, 1) if batch_first else actual_output
        normalized_expected = expected_output.transpose(0, 1) if batch_first else expected_output
        time_metrics = [
            metrics(normalized_actual[time], normalized_expected[time]) for time in range(steps)
        ]
        report["time_worst_mae"] = max(item["mae"] for item in time_metrics)
        report["time_worst_mse"] = max(item["mse"] for item in time_metrics)
        valid_cosines = [item["cosine"] for item in time_metrics if item["cosine"] is not None]
        report["time_worst_cosine"] = min(valid_cosines) if valid_cosines else None
        print(json.dumps(report, sort_keys=True))
        return actual_output, actual_hidden, actual_cell

    def test_basic(self):
        devices_and_profiles = [(torch.device("cpu"), "cpu_basic")]
        if torch.cuda.is_available():
            devices_and_profiles.append((torch.device("cuda"), "cuda_gru_basic"))
        for device, profile in devices_and_profiles:
            for bias in (False, True):
                layout_results = {}
                for batch_first in (False, True):
                    with self.subTest(
                        device=device, profile=profile, bias=bias, batch_first=batch_first
                    ):
                        omitted = self.run_case(
                            profile, device, bias, batch_first, "omitted"
                        )
                        explicit = self.run_case(
                            profile, device, bias, batch_first, "explicit_zero"
                        )
                        for omitted_tensor, explicit_tensor in zip(omitted, explicit):
                            self.assertTrue(torch.equal(omitted_tensor, explicit_tensor))
                        layout_results[batch_first] = omitted
                self.assert_layout_equivalent(layout_results[False], layout_results[True])

    def test_strict(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for profile in ("minimal", "short_recurrent", "non_aligned", "long_sequence"):
                for bias in (False, True):
                    layout_results = {}
                    for batch_first in (False, True):
                        state_profile = (
                            "typical_random"
                            if profile in ("short_recurrent", "long_sequence")
                            else "omitted"
                        )
                        with self.subTest(
                            device=device,
                            profile=profile,
                            bias=bias,
                            batch_first=batch_first,
                            state=state_profile,
                        ):
                            layout_results[batch_first] = self.run_case(
                                profile, device, bias, batch_first, state_profile
                            )
                    self.assert_layout_equivalent(layout_results[False], layout_results[True])

    def test_contract_errors(self):
        with self.assertRaises(ValueError):
            QuantLSTM(2, 3, num_layers=2)
        with self.assertRaises(ValueError):
            QuantLSTM(2, 3, dtype=torch.float64)

        module = QuantLSTM(2, 3)
        input_tensor = torch.zeros((4, 1, 2))
        with self.assertRaises(RuntimeError):
            module(input_tensor.double())
        with self.assertRaises(RuntimeError):
            module(input_tensor, (torch.zeros((1, 1, 3)), None))
        with self.assertRaises(RuntimeError):
            invalid_state = torch.zeros((2, 1, 3))
            module(input_tensor, (invalid_state, invalid_state))

        module.weight_hh_l0 = nn.Parameter(torch.empty((11, 3)))
        with self.assertRaises(RuntimeError):
            module(input_tensor)

        module = QuantLSTM(2, 3)
        module.weight_ih_l0 = nn.Parameter(torch.empty((11, 2)))
        with self.assertRaises(RuntimeError):
            module(input_tensor)

        module = QuantLSTM(2, 3)
        module.bias_ih_l0 = nn.Parameter(torch.empty(11))
        with self.assertRaises(RuntimeError):
            module(input_tensor)

        if torch.cuda.is_available():
            cuda_module = QuantLSTM(2, 3, device="cuda")
            with self.assertRaises(RuntimeError):
                cuda_module(input_tensor)


if __name__ == "__main__":
    suite_name = os.environ.get("QUANT_LSTM_TEST_SUITE")
    if suite_name:
        names = {
            "basic": [
                "FloatReferenceTest.test_basic",
                "FloatReferenceTest.test_contract_errors",
            ],
            "strict": "FloatReferenceTest.test_strict",
        }
        unittest.main(defaultTest=names[suite_name])
    unittest.main()
