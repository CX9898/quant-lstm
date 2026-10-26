"""Fixed bias grids must permit small QAT updates, including zero channels."""

import unittest

import torch

from quant_lstm import QuantLSTM


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BiasHeadroomTest(unittest.TestCase):
    def test_updates_preserve_fixed_encoding_and_nonzero_gradient(self):
        for bits in (8, 16):
            for pot2 in (False, True):
                for initial, step in ((0.0, 1e-4), (0.2, 0.05), (-0.2, -0.05)):
                    with self.subTest(bits=bits, pot2=pot2, initial=initial):
                        torch.manual_seed(42)
                        model = QuantLSTM(4, 4).cuda()
                        model.set_all_bitwidth(bits)
                        if pot2:
                            model.enable_pot2()
                        with torch.no_grad():
                            model.bias_ih_l0.fill_(initial)
                            model.bias_hh_l0.fill_(initial)
                        inputs = torch.randn(5, 2, 4, device="cuda") * 0.5
                        with torch.no_grad(), model.calibration_context():
                            model(inputs)
                        model.use_quantization = True
                        before = model.export_quant_params()
                        model.set_quant_params_locked(True)
                        scales = torch.as_tensor(
                            before["operators"]["bias_ih"]["scale"], device="cuda"
                        )
                        codes = torch.round(torch.tensor(initial + step, device="cuda") / scales)
                        self.assertTrue(torch.all(codes <= 2 ** (bits - 1) - 1))
                        self.assertTrue(torch.all(codes >= -(2 ** (bits - 1))))
                        if initial == 0:
                            self.assertTrue(torch.all(codes != 0))
                        with torch.no_grad():
                            model.bias_ih_l0.add_(step)
                            model.bias_hh_l0.add_(step)
                        model.train()
                        output, _ = model(inputs)
                        output.square().mean().backward()
                        self.assertGreater(model.bias_ih_l0.grad.abs().max().item(), 0.0)
                        self.assertEqual(before, model.export_quant_params())
                        clone = QuantLSTM(4, 4).cuda()
                        clone.load_state_dict(model.state_dict())
                        clone.load_quant_params(before)
                        clone.use_quantization = True
                        model.eval()
                        clone.eval()
                        with torch.no_grad():
                            torch.testing.assert_close(clone(inputs)[0], model(inputs)[0], rtol=0, atol=0)

    def test_zero_bias_all_calibration_methods_and_granularities(self):
        for method in ("minmax", "percentile", "sqnr"):
            for granularity in ("per_tensor", "per_gate", "per_channel"):
                with self.subTest(method=method, granularity=granularity):
                    torch.manual_seed(42)
                    model = QuantLSTM(4, 4, calibration_method=method).cuda()
                    for name in ("bias_ih", "bias_hh"):
                        model.adjust_quant_config(name, granularity=granularity)
                    with torch.no_grad():
                        model.bias_ih_l0.zero_()
                        model.bias_hh_l0.zero_()
                        with model.calibration_context():
                            model(torch.randn(5, 2, 4, device="cuda") * 0.5)
                    encodings = model.export_quant_params()
                    for name in ("bias_ih", "bias_hh"):
                        scales = torch.as_tensor(encodings["operators"][name]["scale"])
                        codes = torch.round(1e-4 / scales)
                        self.assertTrue(torch.all(codes > 0))
                        self.assertTrue(torch.all(codes <= 127))


if __name__ == "__main__":
    unittest.main()
