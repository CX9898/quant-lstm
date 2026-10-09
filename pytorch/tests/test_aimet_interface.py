"""Public recurrent integration interface, usable without installing AIMET."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from quant_lstm import QuantLSTM, normalize_quant_lstm_onnx


class ConfigurationTest(unittest.TestCase):
    def test_stage_config_and_pot2_before_calibration(self):
        module = QuantLSTM(4, 2)
        config = {'LSTM_config': {'use_quantization': True,
                  'quant_config': {'schema_version': 1, 'operators': {
                      'weight_ih': {'bitwidth': 16, 'granularity': 'per_gate'}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stage.json'
            path.write_text(json.dumps(config))
            module.load_bitwidth_config(path)
        self.assertTrue(module.use_quantization)
        self.assertEqual(module.get_quant_config('weight_ih')['bitwidth'], 16)
        module.enable_pot2()
        self.assertEqual(module.get_quant_config()['scale_mode'], 'pot2')
        with self.assertRaises(ValueError):
            module.load_bitwidth_config({'LSTM_config': {'use_quantization': 'true'}})

    def test_normalize_shared_initializer_preserves_other_consumers(self):
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        import numpy as np
        weight = numpy_helper.from_array(np.zeros((1, 8, 2), dtype=np.float32), 'shared')
        recurrent = numpy_helper.from_array(np.zeros((1, 8, 2), dtype=np.float32), 'recurrent')
        nodes = [helper.make_node('LSTM', ['x', 'shared', 'recurrent'], ['y'],
                                  name='lstm#LSTM', hidden_size=2),
                 helper.make_node('Identity', ['shared'], ['other'], name='keep')]
        graph = helper.make_graph(nodes, 'shared',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 1, 2])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 1, 1, 2]),
             helper.make_tensor_value_info('other', TensorProto.FLOAT, [1, 8, 2])],
            initializer=[weight, recurrent])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.onnx'
            onnx.save(model, path)
            normalize_quant_lstm_onnx(path)
            normalized = onnx.load(path)
            onnx.checker.check_model(normalized)
            self.assertEqual(normalized.graph.node[0].input[1], 'lstm.weight_ih.weight')
            self.assertEqual(normalized.graph.node[1].input[0], 'shared')
            self.assertIn('shared', {item.name for item in normalized.graph.initializer})
            normalize_quant_lstm_onnx(path)  # Repeat export normalization is idempotent.


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class NativeAimetInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.x = torch.randn(2, 5, 4, device='cuda')

    def test_export_roundtrip_gate_order_and_direction(self):
        module = QuantLSTM(4, 2, batch_first=True, bidirectional=True,
                          quant_config={'schema_version': 1, 'operators': {
                              'weight_ih': {'granularity': 'per_gate'}}}).cuda()
        with module.calibration_context():
            module(self.x)
        doc = module.export_quant_params()
        for field, scales in [('operators', [0.01, 0.02, 0.03, 0.04]),
                              ('operators_reverse', [0.05, 0.06, 0.07, 0.08])]:
            scales = [value for value in scales for _ in range(2)]
            enc = doc[field]['weight_ih']
            enc.update(scale=scales, real_min=[-127 * s for s in scales],
                       real_max=[127 * s for s in scales])
        module.load_quant_params(doc)
        encodings = module.export_quant_params_to_aimet_format({}, module_name='rnn')
        actual = encodings['param_encodings']['rnn.weight_ih.weight']['scale']
        torch.testing.assert_close(torch.tensor(actual), torch.tensor([
            [0.01, 0.01, 0.04, 0.04, 0.02, 0.02, 0.03, 0.03],
            [0.05, 0.05, 0.08, 0.08, 0.06, 0.06, 0.07, 0.07]]))
        fresh = QuantLSTM(4, 2, batch_first=True, bidirectional=True).cuda()
        fresh.load_state_dict(module.state_dict())
        self.assertTrue(fresh.load_quant_params_from_aimet_format(encodings, module_name='rnn'))
        self.assertEqual(module.export_quant_params(), fresh.export_quant_params())
        self.assertFalse(fresh.load_quant_params_from_aimet_format(encodings, module_name='missing'))
        module.use_quantization = fresh.use_quantization = True
        torch.testing.assert_close(fresh(self.x)[0], module(self.x)[0])
        clone = copy.deepcopy(module)
        torch.testing.assert_close(clone(self.x)[0], module(self.x)[0])

    def test_stage_roundtrip_with_distinct_bias_bitwidths(self):
        module = QuantLSTM(4, 3, batch_first=True, quant_config={
            'schema_version': 1, 'operators': {'bias_ih': {'bitwidth': 16}}}).cuda()
        with module.calibration_context():
            module(self.x)
        stage = module.export_quant_params_to_aimet_format({}, module_name='rnn', for_onnx=False)
        fresh = QuantLSTM(4, 3, batch_first=True).cuda()
        self.assertTrue(fresh.load_quant_params_from_aimet_format(stage, module_name='rnn'))
        self.assertEqual(module.export_quant_params(), fresh.export_quant_params())
        with self.assertRaisesRegex(ValueError, 'incompatible encodings'):
            module.export_quant_params_to_aimet_format({}, module_name='rnn')

    def test_lifecycle_lock_and_pot2(self):
        module = QuantLSTM(4, 3, batch_first=True).cuda()
        with self.assertRaisesRegex(ValueError, 'callback'):
            with module.calibration_context():
                module(self.x)
                raise ValueError('callback')
        self.assertFalse(module.calibrating)
        with module.calibration_context():
            module(self.x)
        module.enable_pot2()
        before = module.export_quant_params()
        for encoding in before['operators'].values():
            scales = torch.as_tensor(encoding['scale'])
            torch.testing.assert_close(scales.log2(), scales.log2().round())
        module.use_quantization = True
        module.set_quant_params_locked()
        with module.calibration_context():
            module(self.x * 10)
        self.assertEqual(before, module.export_quant_params())
        module.reset_calibration()
        self.assertFalse(module.quant_params_locked())
        with module.calibration_context():
            pass
        self.assertFalse(module.is_calibrated())


if __name__ == '__main__':
    unittest.main()
