"""Arbitrary-width LSTM configuration and FP-carrier execution contracts."""

import copy
import math
import unittest
import warnings

import torch
import _quant_lstm
from quant_lstm import QuantLSTM


class LstmBitwidthTest(unittest.TestCase):
    def setUp(self):
        cpu_only = not getattr(_quant_lstm, "supports_cuda", True)
        if not cpu_only and not torch.cuda.is_available():
            self.skipTest("LSTM execution requires a CPU wheel or CUDA")
        self.device = "cpu" if cpu_only else "cuda"
        torch.manual_seed(71)
        self.x = torch.randn(7, 2, 3, device=self.device) * .3 + .1

    def model(self, bits, *, bidirectional=False, method="minmax", granularity="per_channel"):
        operators = {name: {"bitwidth": bits, "granularity": granularity}
                     for name in ("weight_ih", "weight_hh")}
        return QuantLSTM(3, 4, bidirectional=bidirectional, device=self.device, use_quantization=True,
                         calibration_method=method,
                         quant_config={"schema_version": 1, "operators": operators}).eval()

    def calibrate(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with model.calibration_context(), torch.no_grad():
                model(self.x)
                model(self.x * 1.1)

    def assert_result_equal(self, actual, expected):
        for value, reference in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
            self.assertTrue(torch.isfinite(value).all())
            torch.testing.assert_close(value, reference, atol=0, rtol=0)

    def assert_po2(self, document):
        for direction in ("operators", "operators_reverse"):
            for record in document.get(direction, {}).values():
                scales = record["scale"]
                for scale in scales if isinstance(scales, list) else [scales]:
                    self.assertEqual(math.frexp(scale)[0], 0.5)

    def test_all_parameter_widths_export_and_reload(self):
        for bits in range(2, 33):
            with self.subTest(bits=bits):
                model = self.model(bits)
                self.calibrate(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model.enable_pot2()
                    self.assert_po2(model.export_quant_params())
                    document = model.export_quant_params_to_aimet_format({}, module_name="lstm")
                    for name in ("weight_ih", "weight_hh"):
                        record = document["param_encodings"][f"lstm.{name}.weight"]
                        self.assertEqual((record["dtype"], record["bitwidth"]), (f"INT{bits}", bits))
                    target = self.model(8)
                    target.load_state_dict(model.state_dict())
                    self.assertTrue(target.load_quant_params_from_aimet_format(document, module_name="lstm"))
                with torch.no_grad():
                    self.assert_result_equal(target(self.x), model(self.x))

    def test_i8w4_calibration_granularities_directions_and_native_format(self):
        for method in ("minmax", "sqnr", "percentile"):
            for granularity in ("per_tensor", "per_gate", "per_channel"):
                for bidirectional in (False, True):
                    with self.subTest(method=method, granularity=granularity,
                                      bidirectional=bidirectional):
                        model = self.model(4, bidirectional=bidirectional,
                                           method=method, granularity=granularity)
                        self.calibrate(model)
                        model.enable_pot2()
                        self.assert_po2(model.export_quant_params())
                        native = model.export_quant_params()
                        for direction in ("operators", "operators_reverse")[:2 if bidirectional else 1]:
                            for record in native[direction].values():
                                self.assertEqual(set(record), {
                                    "dtype", "symmetric", "scale", "zero_point", "enc_type",
                                    "real_min", "real_max",
                                })
                        target = self.model(8, bidirectional=bidirectional)
                        target.load_state_dict(model.state_dict())
                        target.load_quant_params(copy.deepcopy(native))
                        with torch.no_grad():
                            self.assert_result_equal(target(self.x), model(self.x))

    def test_high_width_fp_profiles_and_small_cell_ratios(self):
        for bits in (24, 32):
            with self.subTest(bits=bits):
                model = self.model(bits)
                operators = {}
                for name, config in model.get_quant_config()["operators"].items():
                    width = min(bits, 31) if config.get("is_unsigned", False) else bits
                    operators[name] = {"bitwidth": width}
                operators["cell_state"] = {"bitwidth": 4}
                model.load_quant_config({"schema_version": 1, "operators": operators})
                self.calibrate(model)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model.enable_pot2()
                    self.assert_po2(model.export_quant_params())
                    document = model.export_quant_params_to_aimet_format({}, module_name="lstm")
                    target = self.model(8)
                    target.load_state_dict(model.state_dict())
                    target.load_quant_params_from_aimet_format(document, module_name="lstm")
                with torch.no_grad():
                    self.assert_result_equal(target(self.x), model(self.x))

    def test_one_bit_affine_input_and_invalid_type_combinations(self):
        for unsigned in (False, True):
            with self.subTest(unsigned=unsigned):
                model = self.model(4)
                model.adjust_quant_config("input", bitwidth=1, is_unsigned=unsigned,
                                          is_symmetric=unsigned)
                self.calibrate(model)
                model.enable_pot2()
                self.assert_po2(model.export_quant_params())
                document = model.export_quant_params_to_aimet_format({}, module_name="lstm")
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(model(self.x)[0]).all())
        for bits in (0, 1, 33, 256, True):
            with self.subTest(parameter_bits=bits), self.assertRaises(ValueError):
                self.model(bits)
        for changes in ({"bitwidth": 32, "is_unsigned": True},
                        {"bitwidth": 1, "is_unsigned": False, "is_symmetric": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.model(4).adjust_quant_config("input", **changes)

    def test_high_width_asymmetric_zero_point_roundtrip(self):
        for bits, unsigned in ((32, False), (31, True)):
            with self.subTest(bits=bits, unsigned=unsigned):
                model = self.model(4)
                model.adjust_quant_config("input", bitwidth=bits, is_unsigned=unsigned,
                                          is_symmetric=False)
                x = -self.x.abs() - .1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with model.calibration_context(), torch.no_grad():
                        model(x)
                        model(x * 1.1)
                    model.enable_pot2()
                    self.assert_po2(model.export_quant_params())
                    native = model.export_quant_params()
                    record = native["operators"]["input"]
                    record["zero_point"] = 2147483647
                    lower = 0 if unsigned else -(1 << (bits - 1))
                    upper = (1 << (bits if unsigned else bits - 1)) - 1
                    record["real_min"] = (lower - record["zero_point"]) * record["scale"]
                    record["real_max"] = (upper - record["zero_point"]) * record["scale"]
                    model.load_quant_params(native)
                    document = model.export_quant_params_to_aimet_format({}, module_name="lstm")
                    input_record = document["activation_encodings"]["lstm"]["input"]["0"]
                    self.assertEqual(input_record["zero_point"], 2147483647)
                    target = self.model(8)
                    target.load_state_dict(model.state_dict())
                    target.load_quant_params_from_aimet_format(document, module_name="lstm")
                with torch.no_grad():
                    self.assert_result_equal(target(x), model(x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA QAT required")
    def test_cuda_i8w4_qat_has_finite_gradients(self):
        if self.device != "cuda":
            self.skipTest("CPU LSTM does not support QAT")
        model = self.model(4, bidirectional=True)
        self.calibrate(model)
        model.enable_pot2()
        self.assert_po2(model.export_quant_params())
        model.train()
        x = self.x.detach().clone().requires_grad_()
        output, (hidden, cell) = model(x)
        (output.square().mean() + hidden.square().mean() + cell.square().mean()).backward()
        self.assertIsNotNone(model.qat_saved_state())
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
