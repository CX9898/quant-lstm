"""阶段 6 QuantLSTM CUDA FP32 q-carrier 接口验收。"""

import json
import unittest
from pathlib import Path
from unittest import mock

import jsonschema
import torch

import _quant_lstm
from quant_lstm import QuantLSTM
from tests import lstm_backward_oracle as oracle


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = json.loads(
    (ROOT / "config/schema/lstm_pytorch_quant_params.schema.json").read_text()
)
def deterministic_tensor(shape, start=-0.35, stop=0.35, *, device="cpu"):
    count = 1
    for extent in shape:
        count *= extent
    return torch.linspace(start, stop, count, device=device).reshape(shape)


def initialize_module(module):
    with torch.no_grad():
        module.weight_ih_l0.copy_(
            deterministic_tensor(
                module.weight_ih_l0.shape,
                -0.18,
                0.21,
                device=module.weight_ih_l0.device,
            )
        )
        module.weight_hh_l0.copy_(
            deterministic_tensor(
                module.weight_hh_l0.shape,
                0.16,
                -0.14,
                device=module.weight_hh_l0.device,
            )
        )
        if module.bias:
            module.bias_ih_l0.copy_(
                deterministic_tensor(
                    module.bias_ih_l0.shape,
                    -0.03,
                    0.04,
                    device=module.bias_ih_l0.device,
                )
            )
            module.bias_hh_l0.copy_(
                deterministic_tensor(
                    module.bias_hh_l0.shape,
                    0.02,
                    -0.01,
                    device=module.bias_hh_l0.device,
                )
            )


def copy_parameters(source, target):
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            getattr(target, name).copy_(parameter)


def calibrate(module, input_tensor, state):
    module.calibrating = True
    with torch.no_grad():
        module(input_tensor, state)
        module(input_tensor * 0.75, state)
    module.calibrating = False
    return module.finalize_calibration()


def flatten_result(result, batch_first):
    output, (hidden, cell) = result
    normalized = output.transpose(0, 1) if batch_first else output
    return normalized, hidden, cell


def metrics(actual, expected):
    actual = actual.detach().double().reshape(-1)
    expected = expected.detach().double().reshape(-1)
    difference = actual - expected
    mae = difference.abs().mean().item()
    mse = difference.square().mean().item()
    denominator = (
        torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(expected)
    ).item()
    cosine = 1.0 if denominator <= 1.0e-12 else (
        torch.dot(actual, expected).item() / denominator
    )
    return mae, mse, cosine


class QuantizedInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def test_config_api_and_error_contracts(self):
        module = QuantLSTM(3, 4)
        resolved = module.get_quant_config()
        self.assertEqual(len(resolved["operators"]), 18)
        self.assertEqual(resolved["scale_mode"], "affine")

        module.set_all_bitwidth(16)
        self.assertTrue(
            all(
                value["bitwidth"] == 16
                for value in module.get_quant_config()["operators"].values()
            )
        )
        module.adjust_quant_config(
            "input", bitwidth=8, is_unsigned=True, is_symmetric=False
        )
        self.assertEqual(module.get_quant_config("input")["bitwidth"], 8)
        self.assertTrue(module.get_quant_config("input")["is_unsigned"])

        with self.assertRaises(ValueError):
            module.set_all_bitwidth(7)
        with self.assertRaises(ValueError):
            module.adjust_quant_config("missing", bitwidth=8)
        with self.assertRaises(ValueError):
            module.adjust_quant_config("input", granularity="per_channel")
        with self.assertRaises(ValueError):
            QuantLSTM(3, 4, calibration_method="unknown")
        with self.assertRaises(ValueError):
            QuantLSTM(3, 4, cublas_math_mode="fast")

        if torch.cuda.is_available():
            module = QuantLSTM(3, 4, use_quantization=True, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "校准"):
                module(torch.zeros((2, 1, 3), device="cuda"))
        with self.assertRaisesRegex(RuntimeError, "未收集校准数据"):
            module.finalize_calibration()

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_all_calibration_methods_and_manifest_contract(self):
        input_tensor = deterministic_tensor((3, 2, 3), device="cuda")
        state = (
            deterministic_tensor((1, 2, 4), -0.1, 0.12, device="cuda"),
            deterministic_tensor((1, 2, 4), -0.2, 0.18, device="cuda"),
        )
        for method in ("minmax", "sqnr", "percentile"):
            with self.subTest(method=method):
                module = QuantLSTM(3, 4, calibration_method=method, device="cuda")
                initialize_module(module)
                report = calibrate(module, input_tensor, state)
                self.assertEqual(report["batch_count"], 2)
                self.assertEqual(report["method"], method)
                self.assertEqual(module.calibration_state(), "locked")
                manifest = module.export_quant_params()
                jsonschema.Draft202012Validator(
                    MANIFEST_SCHEMA
                ).validate(manifest)
                self.assertEqual(
                    set(manifest),
                    {
                        "schema_version",
                        "model_info",
                        "execution_metadata",
                        "operators",
                    },
                )
                self.assertEqual(
                    manifest["model_info"],
                    {
                        "input_size": 3,
                        "hidden_size": 4,
                        "bias": True,
                        "batch_first": False,
                        "bidirectional": False,
                        "use_pot2_scale": False,
                    },
                )
                input_params = manifest["operators"]["input"]
                self.assertEqual(input_params["enc_type"], "PER_TENSOR")
                self.assertIsInstance(input_params["scale"], float)
                self.assertIsInstance(input_params["zero_point"], int)
                self.assertIsInstance(input_params["real_min"], float)
                self.assertIsInstance(input_params["real_max"], float)
                metadata = manifest["execution_metadata"]
                self.assertEqual(
                    metadata["carrier"], "cuda_fp32_qcarrier"
                )
                self.assertEqual(
                    metadata["activation_mode"], "real_sigmoid_tanh"
                )
                for name in (
                    "weight_ih",
                    "weight_hh",
                    "bias_ih",
                    "bias_hh",
                ):
                    params = manifest["operators"][name]
                    self.assertEqual(
                        set(params),
                        {
                            "dtype",
                            "symmetric",
                            "scale",
                            "zero_point",
                            "enc_type",
                            "real_min",
                            "real_max",
                        },
                    )
                    self.assertEqual(len(params["scale"]), 16)
                    self.assertEqual(len(params["zero_point"]), 16)
                    self.assertTrue(
                        all(isinstance(value, float) for value in params["scale"])
                    )
                    self.assertTrue(
                        all(isinstance(value, int) for value in params["zero_point"])
                    )
                    self.assertEqual(params["dtype"], "INT8")
                    self.assertTrue(params["symmetric"])
                    self.assertIn(
                        params["enc_type"],
                        {"PER_TENSOR", "PER_GATE", "PER_CHANNEL"},
                    )
                encoded = json.dumps(manifest)
                self.assertNotIn('"quant_params"', encoded)
                self.assertNotIn('"bitwidth"', encoded)
                self.assertNotIn('"scales"', encoded)
                self.assertNotIn("multiplier", encoded)
                self.assertNotIn("raw_ratio", encoded)

                imported = QuantLSTM(3, 4)
                imported.load_quant_params(manifest)
                self.assertTrue(imported.is_calibrated())
                self.assertEqual(
                    imported.get_quant_config(),
                    module.get_quant_config(),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_calibration_uses_single_cuda_forward(self):
        module = QuantLSTM(3, 4, device="cuda")
        initialize_module(module)
        input_tensor = deterministic_tensor((3, 2, 3), device="cuda")
        state = (
            deterministic_tensor((1, 2, 4), -0.1, 0.12, device="cuda"),
            deterministic_tensor((1, 2, 4), -0.2, 0.18, device="cuda"),
        )
        with torch.no_grad():
            expected = module(input_tensor, state)
            module.calibrating = True
            with mock.patch.object(
                module,
                "_float_forward",
                side_effect=AssertionError("calibration ran a second float forward"),
            ):
                actual = module(input_tensor, state)
        module.calibrating = False
        self.assertTrue(actual[0].is_cuda)
        for actual_value, expected_value in zip(
            (actual[0], *actual[1]), (expected[0], *expected[1])
        ):
            self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertEqual(module.finalize_calibration()["batch_count"], 1)

        session = _quant_lstm.CalibrationSession(
            module._resolved_config_json, 3, 4, True, "minmax"
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            session.collect(
                input_tensor.cpu(),
                module.weight_ih_l0,
                module.weight_hh_l0,
                module.bias_ih_l0,
                module.bias_hh_l0,
                state[0],
                state[1],
                False,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_gru_compatible_dtype_and_granularity_fields_roundtrip(self):
        module = QuantLSTM(2, 3, device="cuda")
        module.adjust_quant_config(
            "input", bitwidth=16, is_unsigned=True, is_symmetric=False
        )
        module.adjust_quant_config("weight_ih", granularity="per_gate")
        module.adjust_quant_config("weight_hh", granularity="per_tensor")
        initialize_module(module)
        calibrate(module, deterministic_tensor((2, 1, 2), device="cuda"), None)

        document = module.export_quant_params()
        self.assertEqual(document["operators"]["input"]["dtype"], "UINT16")
        self.assertFalse(document["operators"]["input"]["symmetric"])
        self.assertEqual(
            document["operators"]["weight_ih"]["enc_type"], "PER_GATE"
        )
        self.assertEqual(
            document["operators"]["weight_hh"]["enc_type"], "PER_TENSOR"
        )
        self.assertEqual(len(document["operators"]["weight_ih"]["scale"]), 12)
        self.assertEqual(len(document["operators"]["weight_hh"]["scale"]), 12)

        imported = QuantLSTM(2, 3)
        imported.load_quant_params(document)
        self.assertEqual(imported.get_quant_config(), module.get_quant_config())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_gru_compatible_pot2_metadata_roundtrip(self):
        module = QuantLSTM(2, 3, device="cuda")
        initialize_module(module)
        calibrate(module, deterministic_tensor((2, 1, 2), device="cuda"), None)
        document = module.export_quant_params()
        document["model_info"]["use_pot2_scale"] = True
        document["execution_metadata"]["standard_scale_mode"] = "pot2"

        for operator in document["operators"].values():
            count = len(operator["scale"]) if isinstance(operator["scale"], list) else 1
            is_unsigned = operator["dtype"].startswith("UINT")
            minimum = 0 if is_unsigned else -127
            maximum = 255 if is_unsigned else 127
            scales = [0.125] * count
            zero_points = [0] * count
            real_minimums = [minimum * 0.125] * count
            real_maximums = [maximum * 0.125] * count
            if count == 1:
                operator["scale"] = scales[0]
                operator["zero_point"] = zero_points[0]
                operator["real_min"] = real_minimums[0]
                operator["real_max"] = real_maximums[0]
            else:
                operator["scale"] = scales
                operator["zero_point"] = zero_points
                operator["real_min"] = real_minimums
                operator["real_max"] = real_maximums

        imported = QuantLSTM(2, 3)
        imported.load_quant_params(document)
        self.assertEqual(imported.get_quant_config()["scale_mode"], "pot2")
        self.assertEqual(imported.export_quant_params(), document)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_bias_disabled_bundle_omits_bias_operators(self):
        module = QuantLSTM(2, 3, bias=False, device="cuda")
        initialize_module(module)
        input_tensor = deterministic_tensor((2, 1, 2), device="cuda")
        calibrate(module, input_tensor, None)
        document = module.export_quant_params()
        jsonschema.Draft202012Validator(MANIFEST_SCHEMA).validate(document)
        operators = document["operators"]
        self.assertNotIn("bias_ih", operators)
        self.assertNotIn("bias_hh", operators)
        self.assertEqual(len(operators["weight_ih"]["scale"]), 12)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_layout_direct_binding_roundtrip_and_qat_state(self):
        time_module = QuantLSTM(3, 4, batch_first=False, device="cuda")
        batch_module = QuantLSTM(3, 4, batch_first=True, device="cuda")
        initialize_module(time_module)
        copy_parameters(time_module, batch_module)

        input_time = deterministic_tensor(
            (3, 2, 3), -0.3, 0.4, device="cuda"
        )
        input_batch = input_time.transpose(0, 1).contiguous()
        state = (
            deterministic_tensor(
                (1, 2, 4), -0.09, 0.11, device="cuda"
            ),
            deterministic_tensor(
                (1, 2, 4), -0.17, 0.19, device="cuda"
            ),
        )
        calibrate(time_module, input_time, state)
        calibrate(batch_module, input_batch, state)
        time_document = time_module.export_quant_params()
        batch_document = batch_module.export_quant_params()
        self.assertFalse(time_document["model_info"]["batch_first"])
        self.assertTrue(batch_document["model_info"]["batch_first"])
        self.assertEqual(time_document["operators"], batch_document["operators"])

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

        bundle_json = time_module._quant_params_bundle_json
        direct = _quant_lstm.lstm_forward_quantized(
            input_time,
            time_module.weight_ih_l0,
            time_module.weight_hh_l0,
            time_module.bias_ih_l0,
            time_module.bias_hh_l0,
            state[0],
            state[1],
            False,
            bundle_json,
            "pedantic",
            False,
            True,
        )
        for module_value, direct_value in zip(
            flatten_result(time_result, False), direct[:3]
        ):
            self.assertTrue(torch.equal(module_value, direct_value))
        checkpoints = direct[3]
        self.assertEqual(
            set(checkpoints),
            {
                "values",
                "clamp_masks",
                "quantized_master",
                "master_clamp_masks",
            },
        )
        self.assertEqual(
            set(checkpoints["values"]),
            {
                "weight_ih_linear",
                "weight_hh_linear",
                "gate_inputs",
                "gate_outputs",
                "cell_states",
                "cell_tanh_outputs",
                "hidden_outputs",
            },
        )
        self.assertEqual(
            checkpoints["values"]["gate_inputs"].shape, (3, 2, 16)
        )
        for mask in checkpoints["clamp_masks"].values():
            self.assertEqual(mask.dtype, torch.uint8)
            self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        bundle = json.loads(bundle_json)
        master_cases = (
            ("input", input_time, "input", False),
            ("weight_ih", time_module.weight_ih_l0, "weight_ih", True),
            ("weight_hh", time_module.weight_hh_l0, "weight_hh", True),
            ("bias_ih", time_module.bias_ih_l0, "bias_ih", True),
            ("bias_hh", time_module.bias_hh_l0, "bias_hh", True),
            ("h_0", state[0], "output", False),
            ("c_0", state[1], "cell_state", False),
        )
        for name, source, operator, per_channel in master_cases:
            expected, expected_mask = oracle.quantize_tensor(
                source, bundle["operators"][operator], per_channel
            )
            self.assertTrue(
                torch.equal(checkpoints["quantized_master"][name], expected), name
            )
            self.assertTrue(
                torch.equal(
                    checkpoints["master_clamp_masks"][name], expected_mask
                ),
                f"{name} mask",
            )

        time_module.train()
        with torch.no_grad():
            training_result = time_module(input_time, state)
        for expected, actual in zip(
            flatten_result(time_result, False),
            flatten_result(training_result, False),
        ):
            self.assertTrue(torch.equal(expected, actual))
        saved = time_module.qat_saved_state()
        self.assertEqual(
            saved["quantized_master"]["weight_ih"].dtype, torch.float32
        )
        self.assertEqual(
            saved["master_clamp_masks"]["input"].dtype, torch.bool
        )
        self.assertEqual(
            set(saved["checkpoint_clamp_masks"]),
            set(checkpoints["clamp_masks"]),
        )

        with torch.no_grad():
            time_module(input_time * 100.0, state)
        saturated = time_module.qat_saved_state()
        self.assertTrue(saturated["master_clamp_masks"]["input"].any())
        self.assertTrue(
            any(
                mask.any()
                for mask in saturated["checkpoint_clamp_masks"].values()
        )
        )

        imported = QuantLSTM(3, 4, device="cuda")
        copy_parameters(time_module, imported)
        imported.load_quant_params(time_module.export_quant_params())
        imported.use_quantization = True
        imported.eval()
        with torch.no_grad():
            imported_result = imported(input_time, state)
        for expected, actual in zip(
            flatten_result(time_result, False),
            flatten_result(imported_result, False),
        ):
            self.assertTrue(torch.equal(expected, actual))

        float_module = QuantLSTM(3, 4, device="cuda")
        copy_parameters(time_module, float_module)
        float_module.eval()
        with torch.no_grad():
            float_result = float_module(input_time, state)
        for name, actual, expected in zip(
            ("output", "h_n", "c_n"),
            flatten_result(time_result, False),
            flatten_result(float_result, False),
        ):
            mae, mse, cosine = metrics(actual, expected)
            self.assertLess(mae, 0.05, f"{name}: MAE={mae}")
            self.assertLess(mse, 0.001, f"{name}: MSE={mse}")
            self.assertGreaterEqual(
                cosine, 0.999, f"{name}: cosine={cosine}"
            )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_import_rejects_metadata_and_compact_parameter_vectors(self):
        module = QuantLSTM(2, 3, device="cuda")
        initialize_module(module)
        calibrate(module, deterministic_tensor((2, 1, 2), device="cuda"), None)
        manifest = module.export_quant_params()

        invalid_metadata = json.loads(json.dumps(manifest))
        invalid_metadata["execution_metadata"]["carrier"] = "cpu_int32"
        with self.assertRaises(ValueError):
            QuantLSTM(2, 3).load_quant_params(invalid_metadata)

        invalid_version = json.loads(json.dumps(manifest))
        invalid_version["schema_version"] = 1.0
        with self.assertRaises(ValueError):
            QuantLSTM(2, 3).load_quant_params(invalid_version)

        compact = json.loads(json.dumps(manifest))
        compact_weight = compact["operators"]["weight_ih"]
        for field in ("scale", "zero_point", "real_min", "real_max"):
            compact_weight[field] = compact_weight[field][:1]
        with self.assertRaises(ValueError):
            QuantLSTM(2, 3).load_quant_params(compact)

        wrong_shape = QuantLSTM(3, 3)
        with self.assertRaises(ValueError):
            wrong_shape.load_quant_params(manifest)


if __name__ == "__main__":
    unittest.main()
