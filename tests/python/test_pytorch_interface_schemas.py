import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "config/schema"


class PytorchInterfaceSchemaTest(unittest.TestCase):
    def load(self, name):
        return json.loads((SCHEMA_DIR / name).read_text())

    def test_schemas_are_valid_draft_2020_12(self):
        for name in (
            "lstm_pytorch_quant_params.schema.json",
            "lstm_pytorch_bidirectional_quant_params.schema.json",
        ):
            jsonschema.Draft202012Validator.check_schema(self.load(name))

    def test_bidirectional_metadata_is_strict(self):
        validator = jsonschema.Draft202012Validator(
            self.load(
                "lstm_pytorch_bidirectional_quant_params.schema.json"
            )
        )
        operator = {
            "dtype": "INT8",
            "symmetric": True,
            "scale": 0.125,
            "zero_point": 0,
            "enc_type": "PER_TENSOR",
            "real_min": -15.875,
            "real_max": 15.875,
        }
        operator_names = (
            "input",
            "output",
            "cell_state",
            "weight_ih",
            "weight_hh",
            "bias_ih",
            "bias_hh",
            "weight_ih_linear",
            "weight_hh_linear",
            "input_gate_input",
            "forget_gate_input",
            "cell_gate_input",
            "output_gate_input",
            "input_gate_output",
            "forget_gate_output",
            "cell_gate_output",
            "output_gate_output",
            "cell_tanh_output",
        )
        operators = {name: dict(operator) for name in operator_names}
        document = {
            "schema_version": 3,
            "model_info": {
                "input_size": 3,
                "hidden_size": 4,
                "bias": True,
                "batch_first": False,
                "bidirectional": True,
                "use_pot2_scale": False,
            },
            "execution_metadata": {
                "carrier": "cuda_fp32_qcarrier",
                "activation_mode": "real_sigmoid_tanh",
                "cublas_math_mode": "pedantic",
                "standard_scale_mode": "affine",
            },
            "operators": operators,
            "operators_reverse": {
                name: dict(operator) for name in operator_names
            },
        }
        validator.validate(document)

        invalid = json.loads(json.dumps(document))
        invalid["model_info"]["bidirectional"] = False
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(invalid)

        invalid = json.loads(json.dumps(document))
        invalid["unexpected"] = True
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(invalid)


if __name__ == "__main__":
    unittest.main()
