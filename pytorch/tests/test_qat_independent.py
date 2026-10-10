"""Check complete quantized trajectories and gradients without native masks."""

import itertools
import json
import unittest
import warnings

import torch

from quant_lstm import QuantLSTM
from tests.test_quantized_interface import calibrate
from tests.lstm_qat_forward_oracle import Reference
from tests import lstm_backward_oracle

def make_case(seed, bits=8, bias=True, bf=False, mode='affine', saturate=False, arbitrary=False, shape=(4, 2, 3, 2), granularity='per_channel', mixed=False, unsigned=False):
    torch.manual_seed(seed)
    T, B, I, H = shape
    m = QuantLSTM(I, H, bias=bias, batch_first=bf, device='cuda')
    m.set_all_bitwidth(bits)
    x = torch.randn((B, T, I) if bf else (T, B, I), device='cuda') * 0.7
    state = (torch.randn(1, B, H, device='cuda') * 0.3, torch.randn(1, B, H, device='cuda') * 0.7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        calibrate(m, x, state)
    d = m.export_quant_params()
    d['execution_metadata']['standard_scale_mode'] = mode
    d['model_info']['use_pot2_scale'] = mode == 'pot2'
    for index, (k, p) in enumerate(d['operators'].items()):
        point_bits = 8 if mixed and index % 2 else bits
        parameter = k.startswith(('weight_', 'bias_')) and 'linear' not in k
        scale = 2 ** (-4 if parameter else -5)
        if k.endswith('_output'):
            scale = 2 ** (-7)
        if saturate:
            scale /= 8
        if arbitrary:
            scale *= 1.073
        if point_bits == 16:
            scale /= 128
        if isinstance(p['scale'], list):
            p['scale'] = [scale * 2 ** (0 if granularity == 'per_tensor' else (i // H if granularity == 'per_gate' else i) % 3 - 1) for i in range(len(p['scale']))]
            p['enc_type'] = granularity.upper()
            p['zero_point'] = [0] * len(p['scale'])
        else:
            p['scale'] = scale
            p['zero_point'] = 0 if parameter else 3
        p['symmetric'] = parameter
        is_unsigned = unsigned and not parameter
        p['dtype'] = f"{'UINT' if is_unsigned else 'INT'}{point_bits}"
        lo = 0 if is_unsigned else -2 ** (point_bits - 1)
        hi = 2 ** (point_bits - (not is_unsigned)) - 1
        if is_unsigned:
            p['zero_point'] = (hi // 2) | 1
        if isinstance(p['scale'], list):
            p['real_min'] = [s * lo for s in p['scale']]
            p['real_max'] = [s * hi for s in p['scale']]
        else:
            p['real_min'] = p['scale'] * (lo - p['zero_point'])
            p['real_max'] = p['scale'] * (hi - p['zero_point'])
    m.load_quant_params(d)
    m.use_quantization = True
    return (m, x, state)

def check(seed, **kw):
    m, x, state = make_case(seed, **kw)
    x.requires_grad_()
    state = tuple((v.requires_grad_() for v in state))
    leaves = [x, m.weight_ih_l0, m.weight_hh_l0, m.bias_ih_l0 if m.bias else None, m.bias_hh_l0 if m.bias else None, *state]
    ref_leaves = [None if v is None else v.detach().cpu().requires_grad_() for v in leaves]
    reference = Reference(json.loads(m._quant_params_bundle_json))
    expected = reference.run(ref_leaves, m.batch_first)
    out, (h, c) = m(x, state)
    actual = (out, h, c)
    saved = m.qat_saved_state()
    for family, ref in [('quantized_master', reference.masters), ('master_clamp_masks', reference.master_masks), ('checkpoints', reference.values), ('checkpoint_clamp_masks', reference.masks)]:
        for k, v in ref.items():
            torch.testing.assert_close(saved[family][k].cpu(), v, atol=0, rtol=0, check_dtype=False, msg=lambda msg: f'{family}.{k}: {msg}')
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)
    losses = ('output', 'hidden', 'cell', 'combined')
    for loss in losses:
        active = [loss == 'output' or loss == 'combined', loss == 'hidden' or loss == 'combined', loss == 'cell' or loss == 'combined']
        factors = [torch.randn_like(v.detach().cpu()) for v in expected]
        ea = sum(((v * f).sum() for v, f, on in zip(expected, factors, active) if on))
        aa = sum(((v * f.cuda()).sum() for v, f, on in zip(actual, factors, active) if on))
        ga = torch.autograd.grad(aa, [v for v in leaves if v is not None], retain_graph=True)
        ge = torch.autograd.grad(ea, [v for v in ref_leaves if v is not None], retain_graph=True)
        for i, (a, b) in enumerate(zip(ga, ge)):
            torch.testing.assert_close(a.cpu(), b, atol=2e-06, rtol=2e-05, msg=lambda msg: f'{loss}.gradient[{i}]: {msg}')


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
class IndependentQatTest(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings("ignore", message=".*FP32 precision_risk.*", category=RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def test_independent_quantized_forward_masks_and_gradients(self):
        for bits in (8, 16):
            for bias in (False, True):
                for bf in (False, True):
                    for mode in ("affine", "pot2"):
                        for saturate in (False, True):
                            for seed in range(5):
                                options = dict(bits=bits, bias=bias, bf=bf, mode=mode, saturate=saturate)
                                with self.subTest(seed=seed, **options):
                                    check(seed, **options)

    def test_affine_encoded_ratios(self):
        for bits in (8, 16):
            for saturate in (False, True):
                for seed in range(10):
                    with self.subTest(bits=bits, saturate=saturate, seed=seed):
                        check(seed, bits=bits, saturate=saturate, arbitrary=True,
                              bf=bool(seed % 2), bias=bool(seed % 3))

    def test_mixed_unsigned_and_parameter_granularity(self):
        for granularity in ("per_tensor", "per_gate", "per_channel"):
            for unsigned in (False, True):
                for shape in ((1, 1, 1, 1), (3, 2, 4, 3)):
                    with self.subTest(granularity=granularity, unsigned=unsigned, shape=shape):
                        check(11, bits=16, mixed=True, unsigned=unsigned,
                              granularity=granularity, shape=shape)

    def test_graph_state_layout_and_stream_lifetimes(self):
        def outputs(result):
            return (result[0], *result[1])

        def objective(result):
            return sum((i + 1) * v.square().sum() for i, v in enumerate(outputs(result)))

        def close(actual, expected):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

        for bits, batch_first, quantized in itertools.product((8, 16), (False, True), (False, True)):
            with self.subTest(bits=bits, batch_first=batch_first, quantized=quantized):
                module, x, state = make_case(7, bits=bits, bf=batch_first)
                module.use_quantization = quantized
                x1 = x.detach().clone().requires_grad_()
                x2 = (x * 0.53).detach().requires_grad_()
                # Both graphs remain live: the latest trace must not replace the first.
                first = objective(module(x1, state))
                second = objective(module(x2, state))
                joint = torch.autograd.grad(first + second, [x1, x2, *module.parameters()])
                a = torch.autograd.grad(objective(module(x1, state)), [x1, *module.parameters()])
                b = torch.autograd.grad(objective(module(x2, state)), [x2, *module.parameters()])
                for actual, expected in zip(joint, [a[0], b[0], *[v + w for v, w in zip(a[1:], b[1:])]]):
                    close(actual, expected)

                # Noncontiguous input and initial states exercise binding copies.
                noncontiguous = lambda v: v.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
                xn = noncontiguous(x)
                sn = tuple(noncontiguous(v) for v in state)
                actual = module(xn, sn)
                expected = module(x, state)
                for v, w in zip(outputs(actual), outputs(expected)):
                    close(v, w)
                gn = torch.autograd.grad(objective(actual), [xn, *sn, *module.parameters()])
                xc = x.clone().requires_grad_()
                sc = tuple(v.clone().requires_grad_() for v in state)
                gc = torch.autograd.grad(objective(module(xc, sc)), [xc, *sc, *module.parameters()])
                for v, w in zip(gn, gc):
                    close(v, w)

                # Explicit state carry must preserve the whole sequence's forward.
                time_axis = 1 if batch_first else 0
                xa, xb = x.split(2, dim=time_axis)
                ya, sa = module(xa, state)
                yb, sb = module(xb, sa)
                close(torch.cat([ya, yb], time_axis), expected[0])
                for v, w in zip(sb, expected[1]):
                    close(v, w)

                module.eval()
                xe = x.clone().requires_grad_()
                evaluation = module(xe, state)
                for v, w in zip(outputs(evaluation), outputs(expected)):
                    close(v, w)
                ge = torch.autograd.grad(objective(evaluation), [xe, *module.parameters()])
                module.train()
                xt = x.clone().requires_grad_()
                gt = torch.autograd.grad(objective(module(xt, state)), [xt, *module.parameters()])
                for v, w in zip(ge, gt):
                    close(v, w)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    xs = (x + 0.0123).requires_grad_()
                    streamed = module(xs, state)
                    gs = torch.autograd.grad(objective(streamed), [xs, *module.parameters()])
                torch.cuda.current_stream().wait_stream(stream)
                xd = (x + 0.0123).requires_grad_()
                default = module(xd, state)
                gd = torch.autograd.grad(objective(default), [xd, *module.parameters()])
                for v, w in zip(outputs(streamed), outputs(default)):
                    close(v, w)
                for v, w in zip(gs, gd):
                    close(v, w)


class StandardQuantizationReferenceTest(unittest.TestCase):
    def test_reference_rounds_before_adding_zero_point(self):
        values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        rounded = torch.tensor([-2., -2., 0., 0., 2., 2.])
        for zero_point in (-3, -2, -1, 0, 1, 2, 3):
            with self.subTest(zero_point=zero_point):
                operator = dict(bitwidth=8, is_unsigned=False, is_symmetric=False,
                                scales=["1.0"], zero_points=[zero_point])
                actual, clamped = lstm_backward_oracle.quantize_tensor(values, operator)
                torch.testing.assert_close(actual, rounded + zero_point, atol=0, rtol=0)
                self.assertFalse(clamped.any())


if __name__ == "__main__":
    unittest.main()
