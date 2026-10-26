"""Standard ONNX LSTM layout adapter for operator-owned QuantLSTM PTQ.

Calibration, gate arithmetic, grid conversion and encoding packing remain in
QuantLSTM. The runtime graph is private; deployment retains the standard LSTM.
"""

from __future__ import annotations

import copy
import importlib
import itertools
from pathlib import Path
import weakref

import numpy as np
import onnx
import torch

from quant_lstm import QuantLSTM

DOMAIN = "rx.quant_lstm"
OP_TYPE = "NativeLSTM"
_contexts = weakref.WeakValueDictionary()
_identifiers = itertools.count(1)


def runtime_library() -> str:
    try:
        module = importlib.import_module("quant_lstm_onnx_runtime")
    except ImportError as exc:
        raise RuntimeError("ONNX LSTM PTQ requires the quant_lstm ONNX runtime extension; "
                           "rebuild/install the updated quant_lstm wheel") from exc
    return str(Path(module.__file__).resolve())


def _run(identifier, value, initial_h, initial_c):
    try:
        adapter = _contexts[identifier]
    except KeyError as exc:
        raise RuntimeError("ONNX LSTM session no longer owns its runtime context") from exc
    return adapter.run(value, initial_h, initial_c)


class OnnxLSTM:
    """Adapt standard ONNX weights/states to the existing native LSTM interface."""

    def __init__(self, node, initializers, *, device="cuda", method="minmax", percentile=99.99):
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        self.name = node.name
        prefix = f"LSTM {self.name}: "
        if not self.name:
            raise ValueError("ONNX LSTM PTQ requires a nonempty unique node name")
        direction = attrs.get("direction", b"forward").decode()
        if direction not in ("forward", "bidirectional"):
            raise NotImplementedError(prefix + f"unsupported direction={direction}")
        directions = 2 if direction == "bidirectional" else 1
        expected_activations = [b"Sigmoid", b"Tanh", b"Tanh"] * directions
        if attrs.get("activations", expected_activations) != expected_activations:
            raise NotImplementedError(prefix + "custom activations are not supported")
        if any(key in attrs for key in ("clip", "activation_alpha", "activation_beta")):
            raise NotImplementedError(prefix + "activation clipping/parameters are not supported")
        if attrs.get("input_forget", 0):
            raise NotImplementedError(prefix + "coupled input_forget is not supported")
        inputs = list(node.input) + [""] * (8 - len(node.input))
        if inputs[4]:
            raise NotImplementedError(prefix + "sequence_lens is not supported")
        if inputs[7]:
            raise NotImplementedError(prefix + "peepholes are not supported")
        self.layout = attrs.get("layout", 0)
        if self.layout not in (0, 1):
            raise ValueError(prefix + f"invalid layout={self.layout}")
        self.inputs, self.outputs = inputs, list(node.output)
        self.parameter_names = {name for name in inputs[1:4] if name}

        def read(name):
            if name not in initializers:
                raise NotImplementedError(prefix + f"constant initializer required: {name}")
            value = onnx.numpy_helper.to_array(initializers[name])
            if value.dtype != np.float32:
                raise TypeError(prefix + f"float32 weights required: {name}")
            if not np.isfinite(value).all():
                raise ValueError(prefix + f"nonfinite parameter: {name}")
            return value.copy()

        w, r = read(inputs[1]), read(inputs[2])
        hidden = attrs.get("hidden_size")
        if hidden is None or w.ndim != 3 or w.shape[:2] != (directions, 4 * hidden):
            raise ValueError(prefix + "invalid W shape or hidden_size")
        if r.shape != (directions, 4 * hidden, hidden):
            raise ValueError(prefix + "invalid R shape")
        bias = read(inputs[3]) if inputs[3] else None
        if bias is not None and bias.shape != (directions, 8 * hidden):
            raise ValueError(prefix + "invalid B shape")
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError(prefix + "native QuantLSTM calibration requires CUDA")
        self.module = QuantLSTM(w.shape[2], hidden, bias=bias is not None,
                                batch_first=self.layout == 1, bidirectional=directions == 2,
                                use_quantization=True, calibration_method=method, device=device).eval()
        self.module.set_module_name(self.name)
        self.module.percentile_value = percentile
        with torch.no_grad():
            for index in range(directions):
                suffix = "_reverse" if index else ""
                parameters = [("weight_ih", w[index]), ("weight_hh", r[index])]
                if bias is not None:
                    parameters.extend((("bias_ih", bias[index, :4 * hidden]),
                                       ("bias_hh", bias[index, 4 * hidden:])))
                for target, array in parameters:
                    # ONNX IOFC -> native IFGO; the candidate gate C is G.
                    converted = torch.cat([torch.from_numpy(array).chunk(4)[gate] for gate in (0, 2, 3, 1)])
                    getattr(self.module, f"{target}_l0{suffix}").copy_(converted)
        self.identifier = next(_identifiers)
        _contexts[self.identifier] = self

    def runtime_node(self, node):
        outputs = list(node.output) + [""] * (3 - len(node.output))
        return onnx.helper.make_node(OP_TYPE, [self.inputs[0], self.inputs[5], self.inputs[6]],
                                     outputs, name=node.name, domain=DOMAIN, context_id=self.identifier)

    def configure(self, config_file):
        self.module.load_bitwidth_config(str(config_file), verbose=False)

    def run(self, value, initial_h=None, initial_c=None):
        if value.dtype != np.float32 or value.ndim != 3:
            raise ValueError(f"LSTM {self.name}: expected float32 rank-3 input")
        device = self.module.weight_ih_l0.device
        x = torch.from_numpy(np.asarray(value).copy()).to(device)
        states = None
        if initial_h is not None or initial_c is not None:
            batch = value.shape[0 if self.layout else 1]
            shape = (self.module.num_directions, batch, self.module.hidden_size)
            states = []
            for state in (initial_h, initial_c):
                if state is None:
                    states.append(torch.zeros(shape, device=device))
                    continue
                expected_shape = (batch, shape[0], shape[2]) if self.layout else shape
                if state.dtype != np.float32 or state.shape != expected_shape:
                    raise ValueError(f"LSTM {self.name}: initial states must be float32 with shape {expected_shape}")
                tensor = torch.from_numpy(np.asarray(state).copy()).to(device)
                states.append(tensor.transpose(0, 1).contiguous() if self.layout else tensor)
            states = tuple(states)
        with torch.cuda.device(device), torch.no_grad():
            output, (hidden, cell) = self.module(x, states)
            output = output.reshape(*output.shape[:2], self.module.num_directions, self.module.hidden_size)
            if self.layout:
                hidden, cell = hidden.transpose(0, 1), cell.transpose(0, 1)
            else:
                output = output.permute(0, 2, 1, 3)
            return tuple(tensor.contiguous().cpu().numpy() for tensor in (output, hidden, cell))

    def export_encodings(self):
        if not self.module.use_quantization:
            return {"activation_encodings": {}, "param_encodings": {}, "excluded_layers": [self.name]}
        if not self.module.is_calibrated():
            raise RuntimeError(f"LSTM {self.name}: internal quantizers are not calibrated")
        payload = self.module.export_quant_params_to_aimet_format({}, module_name=self.name)
        params = {original: payload["param_encodings"][f"{self.name}.{suffix}"]
                  for suffix, original in zip(("weight_ih.weight", "weight_hh.weight", "bias"), self.inputs[1:4])
                  if original}
        return {"activation_encodings": payload["activation_encodings"], "param_encodings": params,
                "excluded_layers": []}

    def load_encodings(self, payload, *, allow_overwrite=False):
        if self.name in payload.get("excluded_layers", []):
            self.module.use_quantization = False
            return
        canonical = copy.deepcopy(payload)
        for suffix, original in zip(("weight_ih.weight", "weight_hh.weight", "bias"), self.inputs[1:4]):
            if original:
                canonical["param_encodings"][f"{self.name}.{suffix}"] = payload["param_encodings"][original]
        if not self.module.load_quant_params_from_aimet_format(canonical, module_name=self.name, verbose=False):
            raise ValueError(f"LSTM {self.name}: missing internal encodings")
        self.module.use_quantization = True
        self.module.set_quant_params_locked(not allow_overwrite)
        expected = {"activation_encodings": {self.name: payload["activation_encodings"][self.name]},
                    "param_encodings": {name: payload["param_encodings"][name] for name in self.parameter_names},
                    "excluded_layers": []}
        if self.export_encodings() != expected:
            raise ValueError(f"LSTM {self.name}: incomplete or inconsistent internal encodings")
