"""Operator-owned integration with AIMET-style encodings; no AIMET dependency.

The public methods on QuantLSTM mirror QuantGRU's configuration and encoding
interface. Calibration, scale conversion and ONNX packing stay in this library.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path


def _pot2_scale(scale, method, rmin, rmax, tolerance):
    exponent = math.log2(scale)
    span = abs(rmax - rmin)
    nearest_span = 2.0 ** round(math.log2(span)) if span > 0 else 0
    near_power = nearest_span > 0 and abs(span - nearest_span) / nearest_span < tolerance
    return 2.0 ** (round(exponent) if method == "round" or near_power else math.ceil(exponent))


_PARAMETER_OPERATORS = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
_ENCODING_FIELDS = ("scale", "zero_point", "real_min", "real_max")


def _aimet_encoding(native):
    """Use the same scalar/vector record as QuantGRU's AIMET export."""
    encoding = deepcopy(native)
    encoding["bitwidth"] = int(encoding["dtype"].removeprefix("UINT").removeprefix("INT"))
    encoding["is_symmetric"] = "True" if encoding.pop("symmetric") else "False"
    return encoding


def _internal_encodings(operators):
    # Parameter records belong to param_encodings. Each internal activation
    # follows GRU's operator -> output -> list[encoding] structure.
    return {name: {"output": [_aimet_encoding(encoding)]}
            for name, encoding in operators.items() if name not in _PARAMETER_OPERATORS}


def _state_encoding(document, operator):
    """Encode the concatenated directions as a flat per-channel record.

    Sequence and h/c states use direction-major hidden channels. Repeating each
    direction's scalar grid H times retains independent grids without a custom
    forward/reverse dictionary inside an encoding record.
    """
    forward = document["operators"][operator]
    if "operators_reverse" not in document:
        return _aimet_encoding(forward)
    reverse = document["operators_reverse"][operator]
    if any(forward[key] != reverse[key] for key in ("dtype", "symmetric")):
        raise ValueError(f"LSTM directions have incompatible state encodings: {operator}")
    hidden = document["model_info"]["hidden_size"]
    encoding = _aimet_encoding(forward)
    encoding["enc_type"] = "PER_CHANNEL"
    for field in _ENCODING_FIELDS:
        encoding[field] = [forward[field]] * hidden + [reverse[field]] * hidden
    return encoding


def _onnx_parameter_encoding(doc, names):
    """Expand quantization groups and reorder IFGO to ONNX IOFC per direction."""
    hidden = doc["model_info"]["hidden_size"]
    directions = [doc["operators"]]
    if "operators_reverse" in doc:
        directions.append(doc["operators_reverse"])
    template = directions[0][names[0]]
    for operators in directions:
        for name in names:
            if any(operators[name][key] != template[key] for key in ("dtype", "symmetric")):
                raise ValueError(f"ONNX packed parameter has incompatible encodings: {names}")
    result = {"dtype": template["dtype"], "symmetric": template["symmetric"], "enc_type": "PER_CHANNEL"}
    for field in _ENCODING_FIELDS:
        packed = []
        for operators in directions:
            rows = []
            for name in names:
                enc = operators[name]
                values = enc[field]
                # Native per-gate and per-channel documents already carry one
                # value per row; only per-tensor scalars need expansion.
                if not isinstance(values, list):
                    values = [values] * (4 * hidden)
                if len(values) != 4 * hidden:
                    raise ValueError(f"Invalid native LSTM parameter row count: {name}.{field}")
                rows.extend(values[i * hidden + j] for i in (0, 3, 1, 2) for j in range(hidden))
            packed.extend(rows)
        result[field] = packed
    return _aimet_encoding(result)


class LSTMAimetIntegration:
    """Shared recurrent-operator interface, implemented using native LSTM state."""

    owns_quantization = True

    def set_module_name(self, module_name: str) -> None:
        self._module_name = module_name

    def load_bitwidth_config(self, config_file, verbose: bool = False) -> None:
        """Read LSTM_config from a full stage config (mapping or JSON path)."""
        config = config_file if isinstance(config_file, dict) else json.loads(
            Path(config_file).read_text(encoding="utf-8"))
        section = config.get("LSTM_config")
        if section is None:
            return
        if not isinstance(section, dict):
            raise ValueError("LSTM_config must be an object")
        unknown = set(section) - {"use_quantization", "quant_config", "comment"}
        if unknown:
            raise ValueError(f"Unknown LSTM_config fields: {sorted(unknown)}")
        enabled = section.get("use_quantization", self.use_quantization)
        if not isinstance(enabled, bool):
            raise ValueError("LSTM_config.use_quantization must be boolean")
        if "quant_config" in section:
            self.load_quant_config(section["quant_config"])
        self.use_quantization = enabled
        if verbose:
            print(f"[QuantLSTM] {getattr(self, '_module_name', '')}: configuration loaded")

    @contextmanager
    def calibration_context(self):
        """Collect a fresh PTQ pass; preserve locks and restore flags on failure."""
        if self.quant_params_locked() or self.calibrating:
            yield
            return
        previous = self.calibrating
        self.reset_calibration()
        self.calibrating = True
        try:
            yield
            state = self.calibration_state()
            states = state.values() if isinstance(state, dict) else [state]
            if any(value != "empty" for value in states):
                self.finalize_calibration()
        finally:
            self.calibrating = previous

    def enable_pot2(self, method="cover_range", tolerance=0.02):
        """Convert calibrated grids and let the native loader re-audit all rescalings."""
        if method not in {"round", "cover_range"}:
            raise ValueError("method must be round or cover_range")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")

        if self.get_quant_config()["scale_mode"] == "pot2":
            return
        if self.quant_params_locked():
            raise RuntimeError("QuantLSTM encodings are locked; cannot change scale_mode")
        if not self.is_calibrated():
            config = self.get_quant_config()
            # Resolved configs contain derived fields; use the public sparse format.
            operators = {}
            for name, item in config["operators"].items():
                fields = {"bitwidth": item["bitwidth"]}
                if name in {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}:
                    fields["granularity"] = item["granularity"]
                else:
                    fields.update(is_unsigned=item["is_unsigned"], is_symmetric=item["is_symmetric"])
                operators[name] = fields
            self.load_quant_config({"schema_version": 1, "scale_mode": "pot2", "operators": operators})
            return
        document = self.export_quant_params()
        for field in ("operators", "operators_reverse"):
            for encoding in document.get(field, {}).values():
                bitwidth = int(encoding["dtype"].lstrip("UINT"))
                unsigned = encoding["dtype"].startswith("UINT")
                qmax = (1 << (bitwidth if unsigned else bitwidth - 1)) - 1
                qmin = 0 if unsigned else (-qmax if encoding["symmetric"] else -qmax - 1)
                vector = isinstance(encoding["scale"], list)
                values = {key: value if vector else [value] for key, value in encoding.items()
                          if key in {"scale", "zero_point", "real_min", "real_max"}}
                scales = [_pot2_scale(scale, method, lo, hi, tolerance)
                    for scale, lo, hi in zip(values["scale"], values["real_min"], values["real_max"])]
                for key, values_out in {
                    "scale": scales,
                    "real_min": [s * (qmin - z) for s, z in zip(scales, values["zero_point"])],
                    "real_max": [s * (qmax - z) for s, z in zip(scales, values["zero_point"])],
                }.items():
                    encoding[key] = values_out if vector else values_out[0]
        document["model_info"]["use_pot2_scale"] = True
        document["execution_metadata"]["standard_scale_mode"] = "pot2"
        self.load_quant_params(document)

    def export_quant_params_to_aimet_format(
        self, encodings_dict: dict, module_name: str = None, verbose: bool = False,
        *, for_onnx: bool = True,
    ) -> dict:
        """Merge native round-trip data and ONNX IOFC parameter encodings in place."""
        if not self.is_calibrated():
            if self.use_quantization:
                raise RuntimeError("QuantLSTM must be calibrated before exporting encodings")
            return encodings_dict
        name = module_name if module_name is not None else getattr(self, "_module_name", "lstm")
        doc = self.export_quant_params()
        ops = doc["operators"]
        hidden_encoding = _state_encoding(doc, "output")
        cell_encoding = _state_encoding(doc, "cell_state")
        entry = {
            "is_LSTM": True,
            # Logical module inputs x/h0/c0 and outputs sequence/h/c, as lists
            # of ordinary RX encoding records (the same container as GRU).
            "input": [_aimet_encoding(ops["input"]), deepcopy(hidden_encoding), deepcopy(cell_encoding)],
            "output": [hidden_encoding, deepcopy(hidden_encoding), cell_encoding],
            "internal_ops": _internal_encodings(ops),
        }
        if "operators_reverse" in doc:
            entry["internal_ops_reverse"] = _internal_encodings(doc["operators_reverse"])
        params = {}
        if for_onnx:
            for suffix, operators in (("weight_ih.weight", ("weight_ih",)),
                                      ("weight_hh.weight", ("weight_hh",)),
                                      ("bias", ("bias_ih", "bias_hh"))):
                if suffix == "bias" and not self.bias:
                    continue
                params[f"{name}.{suffix}"] = _onnx_parameter_encoding(doc, operators)
        # Match GRU's outer schema and retain an unmodified native checkpoint.
        # Validate packing first so a failed deployment export leaves the caller
        # dictionary unchanged (stage-only export permits distinct bias grids).
        encodings_dict.setdefault("schema_version", 3)
        encodings_dict.setdefault("quant_lstm_encodings", {})[name] = doc
        encodings_dict.setdefault("activation_encodings", {})[name] = entry
        encodings_dict.setdefault("param_encodings", {}).update(params)
        return encodings_dict

    def load_quant_params_from_aimet_format(
        self, encodings_dict: dict, module_name: str = None, verbose: bool = False
    ) -> bool:
        """Restore exact native grids; leave execution mode and locking to caller."""
        name = module_name if module_name is not None else getattr(self, "_module_name", "lstm")
        document = encodings_dict.get("quant_lstm_encodings", {}).get(name)
        if document is None:
            return False
        self.load_quant_params(document)
        return True


def normalize_quant_lstm_onnx(onnx_path) -> None:
    """Align standard ONNX LSTM node/initializer names with exported encodings.

    The caller must have assigned PyTorch module paths to ONNX nodes. Shared
    constants are cloned so other consumers retain their original inputs.
    """
    import onnx

    graph = onnx.load(onnx_path)
    initializers = {item.name: item for item in graph.graph.initializer}
    nodes = [node for node in graph.graph.node if node.op_type == "LSTM"]
    seen = set()
    replaced = set()
    for node in nodes:
        name = node.name.split("#", 1)[0]
        if name in seen:
            raise RuntimeError(f"Expected one ONNX LSTM for {name!r}")
        seen.add(name)
        for other in graph.graph.node:
            if other is not node and other.name == name:
                other.name = name + "#" + other.op_type
        node.name = name
        for index, suffix in ((1, "weight_ih.weight"), (2, "weight_hh.weight"), (3, "bias")):
            if index >= len(node.input) or not node.input[index]:
                continue
            old_name = node.input[index]
            new_name = f"{name}.{suffix}"
            if old_name == new_name:
                continue
            if old_name not in initializers:
                raise RuntimeError(f"ONNX LSTM parameter is not a constant initializer: {old_name}")
            if new_name in initializers:
                raise RuntimeError(f"ONNX LSTM initializer name collision: {new_name}")
            copied = deepcopy(initializers[old_name])
            copied.name = new_name
            graph.graph.initializer.append(copied)
            initializers[new_name] = copied
            node.input[index] = new_name
            replaced.add(old_name)
    # Remove only replaced, unused constants. Other ONNX graph data stays intact.
    def input_names(subgraph):
        for current in subgraph.node:
            yield from current.input
            for attribute in current.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from input_names(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        yield from input_names(nested)

    used = set(input_names(graph.graph))
    used.update(value.name for value in graph.graph.output)
    used.update(value.name for value in graph.graph.input)
    for item in list(graph.graph.initializer):
        if item.name not in used and item.name in replaced:
            graph.graph.initializer.remove(item)
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)
