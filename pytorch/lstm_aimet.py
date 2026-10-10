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


def _native_encoding(record, name):
    """Translate a public RX encoding record for the native loader."""
    if not isinstance(record, dict):
        raise ValueError(f"Invalid encoding record: {name}")
    dtype = record.get("dtype")
    if dtype not in {"INT8", "UINT8", "INT16", "UINT16"}:
        raise ValueError(f"Invalid encoding dtype: {name}")
    if record.get("bitwidth") != int(dtype.removeprefix("UINT").removeprefix("INT")):
        raise ValueError(f"Inconsistent encoding bitwidth: {name}")
    symmetric = record.get("is_symmetric")
    if symmetric not in ("True", "False"):
        raise ValueError(f"Invalid encoding symmetry: {name}")
    try:
        return {"dtype": dtype, "symmetric": symmetric == "True",
                "enc_type": record["enc_type"],
                **{field: deepcopy(record[field]) for field in _ENCODING_FIELDS}}
    except KeyError as error:
        raise ValueError(f"Missing encoding field for {name}: {error.args[0]}") from error


def _unpack_parameter(record, names, hidden, directions, label):
    """Split ONNX directions/biases and undo IOFC packing to native IFGO."""
    encoding = _native_encoding(record, label)
    rows = 4 * hidden
    length = directions * len(names) * rows
    fields = {}
    for field in _ENCODING_FIELDS:
        values = encoding[field]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"Invalid ONNX parameter encoding length: {label}.{field}")
        fields[field] = values
    result = [dict() for _ in range(directions)]
    for direction in range(directions):
        for index, name in enumerate(names):
            unpacked = {"dtype": encoding["dtype"], "symmetric": encoding["symmetric"],
                        "enc_type": "PER_CHANNEL"}
            start = (direction * len(names) + index) * rows
            for field, values in fields.items():
                unpacked[field] = [values[start + gate * hidden + channel]
                                   for gate in (0, 2, 3, 1) for channel in range(hidden)]
            result[direction][name] = unpacked
    return result


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
                qmin = 0 if unsigned else -qmax - 1
                vector = isinstance(encoding["scale"], list)
                values = {key: value if vector else [value] for key, value in encoding.items()
                          if key in {"scale", "zero_point", "real_min", "real_max"}}
                # CoverRange uses the symmetric calibration span. The exported
                # integer range additionally includes the most negative code.
                symmetric_signed = encoding["symmetric"] and not unsigned
                scales = [_pot2_scale(scale, method,
                                      -qmax * scale if symmetric_signed else lo,
                                      qmax * scale if symmetric_signed else hi, tolerance)
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
        """Merge GRU-compatible public encodings, with ONNX IOFC parameters.

        Like GRU, stage and ONNX exports use the same public encoding layout.
        Native checkpoints are available separately through export_quant_params().
        """
        if not self.is_calibrated():
            if self.use_quantization:
                raise RuntimeError("QuantLSTM must be calibrated before exporting encodings")
            return encodings_dict
        name = module_name if module_name is not None else getattr(self, "_module_name", "lstm")
        doc = self.export_quant_params()
        ops = doc["operators"]
        entry = {
            "is_LSTM": True,
            # Match GRU's single forward input/output records. These do not
            # enumerate ONNX ports. State grids remain scalar in each
            # direction's internal_ops (output for h, cell_state for c).
            "input": [_aimet_encoding(ops["input"])],
            "output": [_aimet_encoding(ops["output"])],
            "internal_ops": _internal_encodings(ops),
        }
        if "operators_reverse" in doc:
            entry["internal_ops_reverse"] = _internal_encodings(doc["operators_reverse"])
        params = {}
        for suffix, operators in (("weight_ih.weight", ("weight_ih",)),
                                  ("weight_hh.weight", ("weight_hh",)),
                                  ("bias", ("bias_ih", "bias_hh"))):
            if suffix == "bias" and not self.bias:
                continue
            params[f"{name}.{suffix}"] = _onnx_parameter_encoding(doc, operators)
        # schema_version=3 is the existing QuantGRU public encoding schema.
        # Validate packing before modifying the caller's dictionary.
        encodings_dict.setdefault("schema_version", 3)
        encodings_dict.setdefault("activation_encodings", {})[name] = entry
        encodings_dict.setdefault("param_encodings", {}).update(params)
        return encodings_dict

    def load_quant_params_from_aimet_format(
        self, encodings_dict: dict, module_name: str = None, verbose: bool = False
    ) -> bool:
        """Restore from the same activation/parameter containers used by GRU.

        ONNX parameters are represented per channel, including expanded tensor
        and gate grids. Execution settings remain properties of the target
        module; they are not added to compiler encodings.
        """
        name = module_name if module_name is not None else getattr(self, "_module_name", "lstm")
        entry = encodings_dict.get("activation_encodings", {}).get(name)
        if entry is None:
            return False
        if not isinstance(entry, dict) or entry.get("is_LSTM") is not True:
            raise ValueError(f"Expected LSTM activation encodings: {name}")
        directions = 2 if self.bidirectional else 1
        if not self.bidirectional and "internal_ops_reverse" in entry:
            raise ValueError(f"Unexpected reverse LSTM encodings: {name}")
        required = set(self.get_quant_config()["operators"]) - _PARAMETER_OPERATORS
        operators = []
        for field in ("internal_ops", "internal_ops_reverse")[:directions]:
            internal = entry.get(field)
            if not isinstance(internal, dict) or set(internal) != required:
                raise ValueError(f"Missing or unexpected LSTM internal encodings: {name}.{field}")
            values = {}
            for operator, operation in internal.items():
                records = operation.get("output") if isinstance(operation, dict) else None
                if not isinstance(records, list) or len(records) != 1:
                    raise ValueError(f"Expected one internal encoding: {name}.{field}.{operator}")
                values[operator] = _native_encoding(records[0], f"{name}.{field}.{operator}")
            operators.append(values)
        params = encodings_dict.get("param_encodings", {})
        for suffix, names in (("weight_ih.weight", ("weight_ih",)),
                              ("weight_hh.weight", ("weight_hh",)),
                              ("bias", ("bias_ih", "bias_hh"))):
            if suffix == "bias" and not self.bias:
                continue
            key = f"{name}.{suffix}"
            if key not in params:
                raise ValueError(f"Missing LSTM parameter encoding: {key}")
            unpacked = _unpack_parameter(params[key], names, self.hidden_size, directions, key)
            for values, parameters in zip(operators, unpacked):
                values.update(parameters)
        scales = [scale for values in operators for encoding in values.values()
                  for scale in (encoding["scale"] if isinstance(encoding["scale"], list)
                                else [encoding["scale"]])]
        pot2 = all(isinstance(scale, (int, float)) and scale > 0
                   and math.isfinite(scale) and math.frexp(scale)[0] == 0.5 for scale in scales)
        scale_mode = "pot2" if pot2 else "affine"
        # Construct a native document only in memory for the existing audited
        # loader. This metadata is never written into RX/compiler encodings.
        document = {
            "schema_version": 1,
            "model_info": {"input_size": self.input_size, "hidden_size": self.hidden_size,
                           "bias": self.bias, "batch_first": self.batch_first,
                           "bidirectional": self.bidirectional, "use_pot2_scale": pot2},
            "execution_metadata": {"carrier": "cuda_fp32_qcarrier",
                                   "activation_mode": "real_sigmoid_tanh",
                                   "cublas_math_mode": self.cublas_math_mode,
                                   "standard_scale_mode": scale_mode},
            "operators": operators[0],
        }
        if self.bidirectional:
            document["operators_reverse"] = operators[1]
        if (entry.get("input") != [_aimet_encoding(operators[0]["input"])]
                or entry.get("output") != [_aimet_encoding(operators[0]["output"])]):
            raise ValueError(f"LSTM input/output encodings disagree with internal grids: {name}")
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
