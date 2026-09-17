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
        document = {
            "schema_version": 2,
            "execution_metadata": {
                "carrier": "cuda_fp32_qcarrier",
                "activation_mode": "real_sigmoid_tanh",
                "cublas_math_mode": "pedantic",
                "standard_scale_mode": "affine",
                "bidirectional": True,
            },
            "quant_params": {},
            "quant_params_reverse": {},
        }
        validator.validate(document)

        invalid = json.loads(json.dumps(document))
        invalid["execution_metadata"]["bidirectional"] = False
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(invalid)

        invalid = json.loads(json.dumps(document))
        invalid["unexpected"] = True
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(invalid)


if __name__ == "__main__":
    unittest.main()
