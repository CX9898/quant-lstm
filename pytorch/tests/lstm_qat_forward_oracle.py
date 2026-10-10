"""Independent FP32 q-carrier forward and QAT autograd reference.

Consumes only standard quantization parameters and FP32 master inputs. No native
checkpoint, mask, derived execution parameter, or backward helper is reused.
Affine multiplier/shift and cell Q31 ratios are encoded here independently.
Transcendentals use PyTorch CUDA FP32: CPU tanh may differ by an ULP at a half
quantization step. All other arithmetic and the autograd graph run on CPU.
The specified surrogate uses activation derivatives evaluated at dequantized q;
it intentionally does not differentiate the discrete rounded forward function.
"""

import torch

NAMES = ('input_gate', 'forget_gate', 'cell_gate', 'output_gate')
TRACE = ('weight_ih_linear', 'weight_hh_linear', 'gate_inputs', 'gate_outputs', 'cell_states', 'cell_tanh_outputs', 'hidden_outputs')

class Reference:

    def __init__(self, bundle):
        self.ops = bundle['operators']
        self.mode = bundle['scale_mode']
        self.values = {}
        self.masks = {}
        self.masters = {}
        self.master_masks = {}

    def point(self, name, value):
        p = self.ops[name]
        s = torch.tensor([float(v) for v in p['scales']], dtype=torch.float32)
        z = torch.tensor(p['zero_points'], dtype=torch.float32)
        if name.startswith(('weight_', 'bias_')) and 'linear' not in name:
            s = s.reshape((-1,) + (1,) * (value.ndim - 1))
            z = z.reshape(s.shape)
        hi = 2 ** (p['bitwidth'] - (not p['is_unsigned'])) - 1
        lo = 0 if p['is_unsigned'] else -hi - 1
        return (s, z, lo, hi)

    def clamp(self, q, name):
        s, z, lo, hi = self.point(name, q)
        return (q.clamp(lo, hi), (q < lo) | (q > hi))

    def quant(self, x, name):
        s, z, _, _ = self.point(name, x)
        return self.clamp(torch.round(x.detach().double() / s.double()).float() + z, name)

    def real(self, q, name):
        s, z, _, _ = self.point(name, q)
        return (q - z) * s

    def scale(self, name):
        return torch.tensor([float(v) for v in self.ops[name]['scales']], dtype=torch.float32)

    def zp(self, name):
        return torch.tensor(self.ops[name]['zero_points'], dtype=torch.float32)

    def rescale(self, x, source, dest):
        ratios = source.double() / dest.double()
        if self.mode == 'pot2':
            shifts = torch.round(torch.log2(ratios)).int()
            return torch.round(torch.ldexp(x, shifts))
        mantissa, exponent = torch.frexp(ratios)
        mult = torch.round(mantissa * 65536)
        overflow = mult == 65536
        mult = torch.where(overflow, mult / 2, mult).float()
        shift = exponent + overflow.int() - 16
        return torch.round(torch.ldexp(x * mult, shift))

    def attach(self, continuous, q, mask, name, derivative=None):
        """Use dequantized q for activation derivatives per spec section 9.4."""
        delta = continuous - continuous.detach()
        if derivative is not None:
            delta = delta * derivative.detach()
        return self.real(q, name) + delta * ~mask

    def master(self, x, name, key):
        q, m = self.quant(x, name)
        self.masters[key] = q
        self.master_masks[key] = m
        return (q, self.attach(x, q, m, name))

    def record(self, key, q, m):
        self.values.setdefault(key, []).append(q)
        self.masks.setdefault(key, []).append(m)

    def linear(self, qx, rx, qw, rw, qb, rb, source, weight, bias, target):
        acc = qx @ qw.T - qw.sum(1) * self.zp(source)
        accscale = self.scale(source) * self.scale(weight)
        if qb is not None:
            acc = acc + self.rescale(qb, self.scale(bias), accscale)
        q, m = self.clamp(self.rescale(acc, accscale, self.scale(target)) + self.zp(target), target)
        self.record(target, q, m)
        continuous = rx @ rw.T
        if rb is not None:
            continuous = continuous + rb
        return (q, self.attach(continuous, q, m, target))

    def run(self, leaves, batch_first=False):
        x, wi, wh, bi, bh, h, c = leaves
        qx, rx = self.master(x, 'input', 'input')
        qwi, rwi = self.master(wi, 'weight_ih', 'weight_ih')
        qwh, rwh = self.master(wh, 'weight_hh', 'weight_hh')
        qbi, rbi = (None, None) if bi is None else self.master(bi, 'bias_ih', 'bias_ih')
        qbh, rbh = (None, None) if bh is None else self.master(bh, 'bias_hh', 'bias_hh')
        qh, rh = self.master(h, 'output', 'h_0')
        qc, rc = self.master(c, 'cell_state', 'c_0')
        qh, rh, qc, rc = (qh[0], rh[0], qc[0], rc[0])
        if batch_first:
            qx, rx = (qx.transpose(0, 1), rx.transpose(0, 1))
        results = []
        for t in range(len(qx)):
            ql, rl = self.linear(qx[t], rx[t], qwi, rwi, qbi, rbi, 'input', 'weight_ih', 'bias_ih', 'weight_ih_linear')
            qr, rr = self.linear(qh, rh, qwh, rwh, qbh, rbh, 'output', 'weight_hh', 'bias_hh', 'weight_hh_linear')
            qis = []
            mis = []
            qos = []
            mos = []
            ros = []
            for g, name in enumerate(NAMES):
                inp, out = (name + '_input', name + '_output')
                li = ql.chunk(4, -1)[g] - self.zp('weight_ih_linear')
                lr = qr.chunk(4, -1)[g] - self.zp('weight_hh_linear')
                qi, mi = self.clamp(self.rescale(li, self.scale('weight_ih_linear'), self.scale(inp)) + self.rescale(lr, self.scale('weight_hh_linear'), self.scale(inp)) + self.zp(inp), inp)
                ri = self.attach(rl.chunk(4, -1)[g] + rr.chunk(4, -1)[g], qi, mi, inp)
                a = self.real(qi, inp).cuda()
                if g == 2:
                    activation = torch.tanh(a).cpu()
                else:
                    e = torch.exp(-a.abs())
                    activation = torch.where(a >= 0, 1 / (1 + e), e / (1 + e)).cpu()
                qo, mo = self.quant(activation, out)
                realout = self.real(qo, out)
                derivative = 1 - realout ** 2 if g == 2 else realout * (1 - realout)
                ro = self.attach(ri, qo, mo, out, derivative)
                qis.append(qi)
                mis.append(mi)
                qos.append(qo)
                mos.append(mo)
                ros.append(ro)
            self.record('gate_inputs', torch.cat(qis, -1), torch.cat(mis, -1))
            self.record('gate_outputs', torch.cat(qos, -1), torch.cat(mos, -1))
            centered = [q - self.zp(n + '_output') for q, n in zip(qos, NAMES)]
            pf = centered[1] * (qc - self.zp('cell_state'))
            pi = centered[0] * centered[2]
            ratiof = self.scale('forget_gate_output').double()
            ratioi = self.scale('input_gate_output').double() * self.scale('cell_gate_output').double() / self.scale('cell_state').double()
            mf = torch.round(ratiof * 2 ** 31).float() / 2 ** 31
            mi = torch.round(ratioi * 2 ** 31).float() / 2 ** 31
            qc, mc = self.clamp(torch.round(pf * mf + pi * mi) + self.zp('cell_state'), 'cell_state')
            rc = self.attach(ros[1] * rc + ros[0] * ros[2], qc, mc, 'cell_state')
            self.record('cell_states', qc, mc)
            qt, mt = self.quant(torch.tanh(self.real(qc, 'cell_state').cuda()).cpu(), 'cell_tanh_output')
            rt = self.attach(rc, qt, mt, 'cell_tanh_output', 1 - self.real(qt, 'cell_tanh_output') ** 2)
            self.record('cell_tanh_outputs', qt, mt)
            product = centered[3] * (qt - self.zp('cell_tanh_output'))
            pscale = self.scale('output_gate_output') * self.scale('cell_tanh_output')
            qh, mh = self.clamp(self.rescale(product, pscale, self.scale('output')) + self.zp('output'), 'output')
            rh = self.attach(ros[3] * rt, qh, mh, 'output')
            self.record('hidden_outputs', qh, mh)
            results.append(rh)
        self.values = {k: torch.stack(v) for k, v in self.values.items()}
        self.masks = {k: torch.stack(v) for k, v in self.masks.items()}
        output = torch.stack(results)
        if batch_first:
            output = output.transpose(0, 1)
        return (output, rh[None], rc[None])
