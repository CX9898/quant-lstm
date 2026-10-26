"""Public recurrent integration interface, usable without installing AIMET."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from quant_lstm import QuantLSTM, normalize_quant_lstm_onnx


def assert_rx_encoding(test, record):
    """Validate the deployment record contract, independent of native loading."""
    test.assertEqual(set(record), {'dtype', 'bitwidth', 'is_symmetric', 'enc_type',
                                   'scale', 'zero_point', 'real_min', 'real_max'})
    test.assertIn(record['dtype'], {'INT8', 'UINT8', 'INT16', 'UINT16'})
    test.assertIn(record['bitwidth'], (8, 16))
    test.assertIn(record['is_symmetric'], ('True', 'False'))
    fields = [record[key] for key in ('scale', 'zero_point', 'real_min', 'real_max')]
    if record['enc_type'] == 'PER_TENSOR':
        test.assertTrue(all(isinstance(value, (int, float)) for value in fields))
        fields = [[value] for value in fields]
    else:
        test.assertIn(record['enc_type'], ('PER_GATE', 'PER_CHANNEL'))
        test.assertTrue(all(isinstance(value, list) for value in fields))
        test.assertTrue(all(isinstance(item, (int, float)) for value in fields for item in value))
        test.assertEqual(len({len(value) for value in fields}), 1)
    bits = record['bitwidth']
    unsigned = record['dtype'].startswith('UINT')
    qmax = (1 << (bits if unsigned else bits - 1)) - 1
    qmin = 0 if unsigned else -qmax - 1
    for scale, zp, lo, hi in zip(*fields):
        test.assertGreater(scale, 0)
        test.assertIsInstance(zp, int)
        test.assertTrue(math.isfinite(scale))
        test.assertTrue(math.isclose(lo, scale * (qmin - zp), rel_tol=1e-6, abs_tol=1e-9))
        test.assertTrue(math.isclose(hi, scale * (qmax - zp), rel_tol=1e-6, abs_tol=1e-9))


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
            enc.update(scale=scales, real_min=[-128 * s for s in scales],
                       real_max=[127 * s for s in scales])
        module.load_quant_params(doc)
        encodings = module.export_quant_params_to_aimet_format({}, module_name='rnn')
        actual = encodings['param_encodings']['rnn.weight_ih.weight']['scale']
        torch.testing.assert_close(torch.tensor(actual), torch.tensor([
            0.01, 0.01, 0.04, 0.04, 0.02, 0.02, 0.03, 0.03,
            0.05, 0.05, 0.08, 0.08, 0.06, 0.06, 0.07, 0.07]))
        fresh = QuantLSTM(4, 2, batch_first=True, bidirectional=True).cuda()
        fresh.load_state_dict(module.state_dict())
        self.assertTrue(fresh.load_quant_params_from_aimet_format(encodings, module_name='rnn'))
        self.assertEqual(encodings, fresh.export_quant_params_to_aimet_format({}, module_name='rnn'))
        self.assertFalse(fresh.load_quant_params_from_aimet_format(encodings, module_name='missing'))
        module.use_quantization = fresh.use_quantization = True
        torch.testing.assert_close(fresh(self.x)[0], module(self.x)[0])
        clone = copy.deepcopy(module)
        torch.testing.assert_close(clone(self.x)[0], module(self.x)[0])

    def test_deployment_schema_all_parameter_granularities(self):
        parameter_names = {'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'}
        for bidirectional in (False, True):
            for granularity in ('per_tensor', 'per_gate', 'per_channel'):
                for bias in (False, True):
                    with self.subTest(bidirectional=bidirectional, granularity=granularity, bias=bias):
                        operators = {key: {'granularity': granularity} for key in parameter_names}
                        operators['cell_state'] = {'bitwidth': 16}
                        module = QuantLSTM(4, 2, batch_first=True, bidirectional=bidirectional,
                                          bias=bias, quant_config={'schema_version': 1,
                                                                    'operators': operators}).cuda()
                        with module.calibration_context():
                            module(self.x)
                        native = module.export_quant_params()
                        result = module.export_quant_params_to_aimet_format({}, module_name='rnn')
                        self.assertEqual(result['schema_version'], 3)
                        self.assertEqual(set(result), {'schema_version', 'activation_encodings', 'param_encodings'})
                        self.assertEqual(module.export_quant_params(), native)
                        self.assertEqual(result, module.export_quant_params_to_aimet_format({}, module_name='rnn', for_onnx=False))
                        layer = result['activation_encodings']['rnn']
                        for kind in ('input', 'output'):
                            self.assertIsInstance(layer[kind], list)
                            self.assertEqual(len(layer[kind]), 1)
                            for record in layer[kind]:
                                assert_rx_encoding(self, record)
                                self.assertEqual(record['enc_type'], 'PER_TENSOR')
                                self.assertEqual(record['scale'], native['operators'][kind]['scale'])
                        for native_key, key in [('operators', 'internal_ops'),
                                                ('operators_reverse', 'internal_ops_reverse')]:
                            if native_key not in native:
                                self.assertNotIn(key, layer)
                                continue
                            self.assertEqual(set(layer[key]), set(native[native_key]) - parameter_names)
                            for name, operation in layer[key].items():
                                self.assertEqual(set(operation), {'output'})
                                self.assertEqual(len(operation['output']), 1)
                                record = operation['output'][0]
                                assert_rx_encoding(self, record)
                                self.assertEqual(record['enc_type'], 'PER_TENSOR')
                                self.assertEqual(record['scale'], native[native_key][name]['scale'])
                        directions = 2 if bidirectional else 1
                        expected = {'rnn.weight_ih.weight': 8 * directions,
                                    'rnn.weight_hh.weight': 8 * directions}
                        if bias:
                            expected['rnn.bias'] = 16 * directions
                        self.assertEqual(set(result['param_encodings']), set(expected))
                        for name, length in expected.items():
                            record = result['param_encodings'][name]
                            assert_rx_encoding(self, record)
                            self.assertEqual(len(record['scale']), length)
                        fresh = QuantLSTM(4, 2, batch_first=True, bidirectional=bidirectional,
                                          bias=bias).cuda()
                        fresh.load_state_dict(module.state_dict())
                        self.assertTrue(fresh.load_quant_params_from_aimet_format(result, module_name='rnn'))
                        self.assertEqual(result, fresh.export_quant_params_to_aimet_format({}, module_name='rnn'))
                        restored = fresh.export_quant_params()
                        for field in ('operators', 'operators_reverse')[:directions]:
                            for parameter in parameter_names.intersection(native[field]):
                                for key in ('scale', 'zero_point', 'real_min', 'real_max'):
                                    values = native[field][parameter][key]
                                    expected_values = values if isinstance(values, list) else [values] * 8
                                    self.assertEqual(restored[field][parameter][key], expected_values)
                        module.use_quantization = fresh.use_quantization = True
                        module.eval()
                        fresh.eval()
                        torch.testing.assert_close(fresh(self.x), module(self.x))

    def test_distinct_bidirectional_state_grids_are_not_lost(self):
        module = QuantLSTM(4, 2, batch_first=True, bidirectional=True).cuda()
        with module.calibration_context():
            module(self.x)
        doc = module.export_quant_params()
        for key, hidden_scale, cell_scale in [('operators', 0.01, 0.02),
                                              ('operators_reverse', 0.03, 0.04)]:
            for name, scale in [('output', hidden_scale), ('cell_state', cell_scale)]:
                doc[key][name].update(scale=scale, zero_point=0,
                                      real_min=-128 * scale, real_max=127 * scale)
        module.load_quant_params(doc)
        result = module.export_quant_params_to_aimet_format({}, module_name='rnn')
        layer = result['activation_encodings']['rnn']
        self.assertEqual(len(layer['input']), 1)
        self.assertEqual(len(layer['output']), 1)
        self.assertAlmostEqual(layer['output'][0]['scale'], 0.01)
        for key, hidden_scale, cell_scale in [('internal_ops', 0.01, 0.02),
                                              ('internal_ops_reverse', 0.03, 0.04)]:
            for name, scale in [('output', hidden_scale), ('cell_state', cell_scale)]:
                record = layer[key][name]['output'][0]
                assert_rx_encoding(self, record)
                self.assertEqual(record['enc_type'], 'PER_TENSOR')
                self.assertAlmostEqual(record['scale'], scale)
        fresh = QuantLSTM(4, 2, batch_first=True, bidirectional=True).cuda()
        fresh.load_state_dict(module.state_dict())
        fresh.load_quant_params_from_aimet_format(result, module_name='rnn')
        restored = fresh.export_quant_params()
        native = module.export_quant_params()
        for key in ('operators', 'operators_reverse'):
            for name in ('output', 'cell_state'):
                self.assertEqual(restored[key][name], native[key][name])
        module.use_quantization = fresh.use_quantization = True
        module.eval()
        fresh.eval()
        h0 = torch.randn(2, 2, 2, device='cuda')
        c0 = torch.randn_like(h0)
        torch.testing.assert_close(fresh(self.x, (h0, c0)), module(self.x, (h0, c0)),
                                   atol=0, rtol=0)

    def test_native_bias_formats_stay_separate_from_public_exports(self):
        module = QuantLSTM(4, 3, batch_first=True, quant_config={
            'schema_version': 1, 'operators': {'bias_ih': {'bitwidth': 16}}}).cuda()
        with module.calibration_context():
            module(self.x)
        native = module.export_quant_params()
        fresh = QuantLSTM(4, 3, batch_first=True).cuda()
        fresh.load_quant_params(native)
        self.assertEqual(native, fresh.export_quant_params())
        for for_onnx in (False, True):
            existing = {'schema_version': 3, 'activation_encodings': {'other': {'input': []}}}
            before = copy.deepcopy(existing)
            with self.assertRaisesRegex(ValueError, 'incompatible encodings'):
                module.export_quant_params_to_aimet_format(existing, module_name='rnn', for_onnx=for_onnx)
            self.assertEqual(before, existing)

    def test_pot2_recomputes_asymmetric_zero_points_from_calibration(self):
        for bits in (8, 16):
            for unsigned in (False, True):
                for method in ('minmax', 'percentile', 'sqnr'):
                    with self.subTest(bits=bits, unsigned=unsigned, method=method):
                        scale = 2.0 ** (-4 if bits == 8 else -12)
                        # The original lower bound and the affine grid's lower
                        # bound fall on opposite sides of a Po2 rounding boundary.
                        lower = -16.51 * scale
                        x = torch.tensor([lower, lower + 10], device='cuda').reshape(2, 1, 1)
                        def make(mode):
                            return QuantLSTM(1, 1, bidirectional=True, device='cuda',
                                calibration_method=method, quant_config={
                                    'schema_version': 1, 'scale_mode': mode,
                                    'operators': {'input': {'bitwidth': bits,
                                        'is_symmetric': False, 'is_unsigned': unsigned}}})
                        affine, native = make('affine'), make('pot2')
                        native.load_state_dict(affine.state_dict())
                        for module in (affine, native):
                            with module.calibration_context():
                                module(x)
                        clone = copy.deepcopy(affine)
                        expected = native.export_quant_params()
                        for module in (affine, clone):
                            module.enable_pot2()
                            actual = module.export_quant_params()
                            for direction in ('operators', 'operators_reverse'):
                                self.assertEqual(actual[direction]['input'], expected[direction]['input'])
                            module.use_quantization = True
                            self.assertTrue(torch.isfinite(module(x)[0]).all())
                            module.enable_pot2()  # Conversion remains idempotent.
                            self.assertEqual(actual, module.export_quant_params())

    def test_pot2_loaded_asymmetric_grids_round_and_clamp(self):
        module = QuantLSTM(4, 2, batch_first=True).cuda()
        with module.calibration_context():
            module(self.x)
        baseline = module.export_quant_params()
        for bits in (8, 16):
            for unsigned in (False, True):
                qmin = 0 if unsigned else -(1 << (bits - 1))
                qmax = (1 << (bits if unsigned else bits - 1)) - 1
                for method, scale in (('round', 0.25), ('cover_range', 0.5)):
                    for old_zp in (qmin, qmin + 1, qmin + 2, qmax):
                        with self.subTest(bits=bits, unsigned=unsigned, method=method, zp=old_zp):
                            doc = copy.deepcopy(baseline)
                            lo, hi = (qmin - old_zp) * 0.3125, (qmax - old_zp) * 0.3125
                            doc['operators']['input'].update(
                                dtype=f"{'U' if unsigned else ''}INT{bits}", symmetric=False,
                                scale=0.3125, zero_point=old_zp, real_min=lo, real_max=hi)
                            module.load_quant_params(doc)
                            module.enable_pot2(method=method, tolerance=0)
                            enc = module.export_quant_params()['operators']['input']
                            self.assertEqual(enc['scale'], scale)
                            zp = max(qmin, min(qmax, round(qmin - lo / scale)))
                            self.assertEqual(enc['zero_point'], zp)
                            self.assertEqual(enc['real_min'], (qmin - zp) * scale)
                            self.assertEqual(enc['real_max'], (qmax - zp) * scale)

    def test_pot2_preserves_symmetric_calibration_span(self):
        module = QuantLSTM(4, 2, batch_first=True).cuda()
        with module.calibration_context():
            module(self.x)
        document = module.export_quant_params()
        scale = 1.018 / 254
        document['operators']['input'].update(
            scale=scale, zero_point=0, real_min=-128 * scale, real_max=127 * scale)
        module.load_quant_params(document)
        # Symmetric calibration span 1.018 is within 2% of 1.0. Including the
        # extra negative code would exceed that tolerance and double the scale.
        module.enable_pot2(method='cover_range', tolerance=0.02)
        encoding = module.export_quant_params()['operators']['input']
        self.assertEqual(encoding['scale'], 2.0 ** -8)
        self.assertEqual(encoding['real_min'], -0.5)
        self.assertEqual(encoding['real_max'], 127 / 256)

    def test_public_reload_pot2_and_invalid_records(self):
        module = QuantLSTM(4, 3, batch_first=True, bidirectional=True).cuda()
        with module.calibration_context():
            module(self.x)
        module.enable_pot2()
        public = module.export_quant_params_to_aimet_format({}, module_name='rnn')
        fresh = QuantLSTM(4, 3, batch_first=True, bidirectional=True).cuda()
        fresh.load_state_dict(module.state_dict())
        self.assertTrue(fresh.load_quant_params_from_aimet_format(public, module_name='rnn'))
        self.assertEqual(fresh.get_quant_config()['scale_mode'], 'pot2')
        before = fresh.export_quant_params()
        module.use_quantization = fresh.use_quantization = True
        torch.testing.assert_close(fresh(self.x), module(self.x))
        for case in ('missing_reverse', 'short_weight', 'missing_bias', 'wrong_bitwidth',
                     'wrong_port', 'expanded_states', 'per_channel_state'):
            with self.subTest(case=case):
                invalid = copy.deepcopy(public)
                if case == 'missing_reverse':
                    del invalid['activation_encodings']['rnn']['internal_ops_reverse']
                elif case == 'short_weight':
                    invalid['param_encodings']['rnn.weight_ih.weight']['scale'].pop()
                elif case == 'missing_bias':
                    del invalid['param_encodings']['rnn.bias']
                elif case == 'wrong_bitwidth':
                    invalid['param_encodings']['rnn.bias']['bitwidth'] = 16
                elif case == 'wrong_port':
                    invalid['activation_encodings']['rnn']['output'][0]['scale'] *= 2
                elif case == 'expanded_states':
                    layer = invalid['activation_encodings']['rnn']
                    layer['input'].extend(copy.deepcopy(layer['output']) * 2)
                else:
                    record = invalid['activation_encodings']['rnn']['internal_ops_reverse']['cell_state']['output'][0]
                    record['enc_type'] = 'PER_CHANNEL'
                    for field in ('scale', 'zero_point', 'real_min', 'real_max'):
                        record[field] = [record[field]] * 3
                with self.assertRaises(ValueError):
                    fresh.load_quant_params_from_aimet_format(invalid, module_name='rnn')
                self.assertEqual(before, fresh.export_quant_params())

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
