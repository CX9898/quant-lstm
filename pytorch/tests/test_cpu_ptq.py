"""CPU-only public calibration and PTQ contracts."""

import copy
import json
import unittest
from unittest import mock

import torch

import _quant_lstm
from quant_lstm import QuantLSTM


@unittest.skipUnless(getattr(_quant_lstm, "supports_cuda", True) is False,
                     "requires the CPU QuantLSTM extension")
class CpuPtqTest(unittest.TestCase):
    def module(self, *, bias=True, batch_first=False, bidirectional=False, method="minmax"):
        torch.manual_seed(17)
        return QuantLSTM(3, 4, bias=bias, batch_first=batch_first,
                         bidirectional=bidirectional, calibration_method=method,
                         device="cpu").eval()

    def inputs(self, *, batch_first=False, bidirectional=False):
        value = torch.linspace(-0.7, 0.9, 30).reshape(5, 2, 3)
        if batch_first:
            value = value.transpose(0, 1)
        state_shape = (2 if bidirectional else 1, 2, 4)
        hidden = torch.linspace(-0.2, 0.3, torch.tensor(state_shape).prod().item()).reshape(state_shape)
        return value, (hidden, hidden * -0.4)

    def assert_result_close(self, actual, expected, **kwargs):
        for value, reference in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
            torch.testing.assert_close(value, reference, **kwargs)
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.device.type, "cpu")

    def test_float_and_calibration_match_pytorch_without_second_forward(self):
        for bias, batch_first, bidirectional in ((True, False, False), (False, True, True)):
            with self.subTest(bias=bias, batch_first=batch_first, bidirectional=bidirectional):
                module = self.module(bias=bias, batch_first=batch_first, bidirectional=bidirectional)
                reference = torch.nn.LSTM(3, 4, bias=bias, batch_first=batch_first,
                                          bidirectional=bidirectional).eval()
                reference.load_state_dict(module.state_dict())
                value, state = self.inputs(batch_first=batch_first, bidirectional=bidirectional)
                with torch.inference_mode():
                    expected = reference(value, state)
                    actual = module(value, state)
                    self.assert_result_close(actual, expected, rtol=1e-5, atol=1e-7)
                    module.calibrating = True
                    with mock.patch.object(module, "_float_forward", side_effect=AssertionError):
                        calibrated = module(value, state)
                    self.assert_result_close(calibrated, actual, rtol=0, atol=0)
                module.calibrating = False
                self.assertEqual(module.finalize_calibration()["batch_count"], 1)
                self.assertTrue(module.is_calibrated())

    def test_ptq_roundtrip_methods_and_layouts(self):
        for method, batch_first, bidirectional in (("minmax", False, False),
                                                   ("sqnr", True, False),
                                                   ("percentile", True, True)):
            with self.subTest(method=method, bidirectional=bidirectional):
                module = self.module(method=method, batch_first=batch_first,
                                     bidirectional=bidirectional)
                module.adjust_quant_config("weight_ih", granularity="per_channel")
                module.adjust_quant_config("weight_hh", granularity="per_gate")
                module.adjust_quant_config("input", is_unsigned=True, is_symmetric=False)
                value, state = self.inputs(batch_first=batch_first, bidirectional=bidirectional)
                with torch.inference_mode():
                    module.calibrating = True
                    module(value, state)
                    module(value * 0.8)
                    module.calibrating = False
                    report = module.finalize_calibration()
                    self.assertEqual(report["method"], method)
                    self.assertEqual(report["batch_count"], 2)
                    module.use_quantization = True
                    expected = module(value, state)
                    self.assertIsNone(module.qat_saved_state())
                    self.assertTrue(all(torch.isfinite(t).all() for t in (expected[0], *expected[1])))
                    restored = self.module(batch_first=batch_first, bidirectional=bidirectional)
                    restored.load_state_dict(module.state_dict())
                    document = module.export_quant_params()
                    restored.load_quant_params(document)
                    restored.use_quantization = True
                    self.assert_result_close(restored(value, state), expected, rtol=0, atol=0)
                    self.assert_result_close(copy.deepcopy(module)(value, state), expected,
                                             rtol=0, atol=0)
                    self.assertEqual(restored.export_quant_params(), document)

    def test_bias_disabled_pot2_and_optional_state(self):
        module = self.module(bias=False)
        module.load_quant_config({"schema_version": 1, "scale_mode": "pot2"})
        value, _ = self.inputs()
        with torch.no_grad():
            module.calibrating = True
            module(value)
            module.calibrating = False
            module.finalize_calibration()
            module.use_quantization = True
            actual = module(value)
            zeros = torch.zeros(1, 2, 4)
            self.assert_result_close(module(value, (zeros, zeros)), actual, rtol=0, atol=0)
        bundle = json.loads(module._quant_params_bundle_json)
        self.assertEqual(bundle["scale_mode"], "pot2")
        self.assertNotIn("bias_ih", bundle["operators"])

    def test_training_and_qat_are_rejected(self):
        module = self.module()
        value, state = self.inputs()
        module.train()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "inference only"):
            module(value, state)
        with self.assertRaisesRegex(RuntimeError, "inference only"):
            module(value, state)
        module.eval()
        with torch.inference_mode():
            module.calibrating = True
            module(value)
            module.calibrating = False
            module.finalize_calibration()
        with self.assertRaisesRegex(RuntimeError, "QAT"):
            _quant_lstm.lstm_forward_quantized(
                value, module.weight_ih_l0, module.weight_hh_l0,
                module.bias_ih_l0, module.bias_hh_l0, state[0], state[1], False,
                module._quant_params_bundle_json, "pedantic", False, True,
            )

    def prepare_mode(self, mode, *, batch_first=False, bidirectional=False):
        module = self.module(batch_first=batch_first, bidirectional=bidirectional)
        value, state = self.inputs(batch_first=batch_first, bidirectional=bidirectional)
        if mode == "quantized":
            with torch.no_grad():
                module.calibrating = True
                module(value, state)
            module.calibrating = False
            module.finalize_calibration()
            module.use_quantization = True
        elif mode == "calibration":
            module.calibrating = True
        return module, value, state

    def test_eval_forward_allows_grad_enabled_in_all_cpu_modes(self):
        for mode in ("float", "calibration", "quantized"):
            for batch_first, bidirectional in ((False, False), (False, True),
                                                (True, False), (True, True)):
                with self.subTest(mode=mode, batch_first=batch_first,
                                  bidirectional=bidirectional):
                    module, value, state = self.prepare_mode(
                        mode, batch_first=batch_first, bidirectional=bidirectional)
                    with torch.no_grad():
                        expected = module(value, state)
                    saved_tensors = []
                    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
                            lambda tensor: saved_tensors.append(tensor) or tensor,
                            lambda tensor: tensor):
                        actual = module(value, state)
                    for tensor, reference in zip((actual[0], *actual[1]),
                                                 (expected[0], *expected[1])):
                        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
                        self.assertTrue(tensor.requires_grad)
                    self.assertEqual(saved_tensors, [])
                    self.assertIsNone(module.qat_saved_state())

    def test_backward_rejects_all_outputs_and_autograd_dependencies(self):
        for mode in ("float", "calibration", "quantized"):
            for bidirectional in (False, True):
                template, _, _ = self.prepare_mode(mode, bidirectional=bidirectional)
                dependencies = ("input", "hidden", "cell", *dict(template.named_parameters()))
                for dependency in dependencies:
                    for output_index in range(3):
                        with self.subTest(mode=mode, bidirectional=bidirectional,
                                          dependency=dependency, output=output_index):
                            module, value, state = self.prepare_mode(
                                mode, bidirectional=bidirectional)
                            module.requires_grad_(False)
                            if dependency == "input":
                                value.requires_grad_()
                            elif dependency == "hidden":
                                state[0].requires_grad_()
                            elif dependency == "cell":
                                state[1].requires_grad_()
                            else:
                                dict(module.named_parameters())[dependency].requires_grad_()
                            with torch.enable_grad():
                                output, (hidden, cell) = module(value, state)
                                selected = (output, hidden, cell)[output_index]
                                self.assertTrue(selected.requires_grad)
                                with self.assertRaisesRegex(RuntimeError,
                                                            "CPU QuantLSTM.*backward"):
                                    selected.sum().backward()
                            for tensor in (value, *state, *module.parameters()):
                                self.assertIsNone(tensor.grad)

    def test_mixed_linear_network_does_not_silently_detach(self):
        for mode in ("float", "calibration", "quantized"):
            for bidirectional in (False, True):
                with self.subTest(mode=mode, bidirectional=bidirectional):
                    module, value, state = self.prepare_mode(mode, bidirectional=bidirectional)
                    module.requires_grad_(False)
                    upstream = torch.nn.Linear(3, 3)
                    downstream = torch.nn.Linear(4 * module.num_directions, 2)
                    with torch.enable_grad():
                        output, _ = module(upstream(value), state)
                        loss = downstream(output).sum()
                        with self.assertRaisesRegex(RuntimeError, "CPU QuantLSTM.*backward"):
                            loss.backward()
                    self.assertIsNone(upstream.weight.grad)
                    self.assertIsNone(upstream.bias.grad)

    def test_inplace_output_operations_allow_forward_and_reject_backward(self):
        for mode in ("float", "calibration", "quantized"):
            for batch_first, bidirectional in ((False, False), (False, True),
                                                (True, False), (True, True)):
                for output_index in range(3):
                    with self.subTest(mode=mode, batch_first=batch_first,
                                      bidirectional=bidirectional, output=output_index):
                        module, value, state = self.prepare_mode(
                            mode, batch_first=batch_first, bidirectional=bidirectional)
                        with torch.enable_grad():
                            output, (hidden, cell) = module(value, state)
                            selected = (output, hidden, cell)[output_index]
                            activated = torch.nn.ReLU(inplace=True)(selected)
                            self.assertTrue(torch.isfinite(activated).all())
                            with self.assertRaisesRegex(RuntimeError,
                                                        "CPU QuantLSTM.*backward"):
                                activated.sum().backward()

    def test_frozen_eval_and_disabled_grad_need_no_backward_guard(self):
        for mode in ("float", "calibration", "quantized"):
            for bidirectional in (False, True):
                with self.subTest(mode=mode, bidirectional=bidirectional):
                    module, value, state = self.prepare_mode(mode, bidirectional=bidirectional)
                    module.requires_grad_(False)
                    with torch.enable_grad():
                        actual = module(value, state)
                    self.assertTrue(all(t.grad_fn is None and not t.requires_grad
                                        for t in (actual[0], *actual[1])))
                    module.requires_grad_(True)
                    value.requires_grad_()
                    for context in (torch.no_grad, torch.inference_mode):
                        with context():
                            disabled = module(value)
                        self.assertTrue(all(t.grad_fn is None and not t.requires_grad
                                            for t in (disabled[0], *disabled[1])))

    def test_export_mode_preserves_onnx_forward_dispatch(self):
        module = self.module()
        module.export_mode = True
        module.train()
        value, state = self.inputs()
        expected = (torch.empty(5, 2, 4), (torch.empty(1, 2, 4), torch.empty(1, 2, 4)))
        with mock.patch.object(module, "_forward_onnx", return_value=expected) as forward:
            self.assertIs(module(value, state), expected)
        forward.assert_called_once_with(value, state)

    def test_native_onnx_calibration_and_ptq_on_cpu_provider(self):
        try:
            import numpy as np
            import onnx
            import onnxruntime as ort
            from onnx import helper, numpy_helper
            from quant_lstm_onnx import OnnxLSTM, runtime_library
        except ImportError as error:
            self.skipTest(str(error))
        module = self.module()
        value, _ = self.inputs()

        def onnx_gates(tensor):
            input_gate, forget_gate, cell_gate, output_gate = np.split(tensor.detach().numpy(), 4)
            return np.concatenate((input_gate, output_gate, forget_gate, cell_gate), axis=0)

        node = helper.make_node("LSTM", ["x", "w", "r", "b"], ["y", "h", "c"],
                                name="lstm", hidden_size=4)
        arrays = {
            "w": onnx_gates(module.weight_ih_l0)[None],
            "r": onnx_gates(module.weight_hh_l0)[None],
            "b": np.concatenate((onnx_gates(module.bias_ih_l0),
                                  onnx_gates(module.bias_hh_l0)))[None],
        }
        initializers = {name: numpy_helper.from_array(array, name) for name, array in arrays.items()}
        context = OnnxLSTM(node, initializers, device="cpu")
        context.module.calibrating = True
        calibrated = context.run(value.numpy())
        with torch.inference_mode():
            expected = module(value)
        np.testing.assert_allclose(calibrated[0][:, 0], expected[0].numpy(), atol=1e-7, rtol=1e-5)
        context.module.calibrating = False
        context.module.finalize_calibration()
        expected_ptq = context.run(value.numpy())
        graph = helper.make_graph(
            [context.runtime_node(node)], "cpu-native-lstm",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [5, 2, 3])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [5, 1, 2, 4]),
             helper.make_tensor_value_info("h", onnx.TensorProto.FLOAT, [1, 2, 4]),
             helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [1, 2, 4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18),
                                                       helper.make_opsetid("rx.quant_lstm", 1)])
        model.ir_version = 10
        options = ort.SessionOptions()
        options.register_custom_ops_library(runtime_library())
        session = ort.InferenceSession(model.SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        for actual, reference in zip(session.run(None, {"x": value.numpy()}), expected_ptq):
            np.testing.assert_array_equal(actual, reference)


if __name__ == "__main__":
    unittest.main()
