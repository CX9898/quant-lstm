import copy
import json
import pathlib
import unittest

from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "config/schema/lstm_quant_params_bundle.schema.json"
DEFAULT_PATH = ROOT / "config/defaults/lstm_quant_default_v1.json"


class QuantParamsBundleSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.default = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
        StrictDraft202012Validator.check_schema(cls.schema)

    def make_bundle(self, bias_enabled: bool = True) -> dict:
        operators = {}
        for name, config in self.default["operators"].items():
            if not bias_enabled and name in {"bias_ih", "bias_hh"}:
                continue
            count = 4 if name in {
                "weight_ih",
                "weight_hh",
                "bias_ih",
                "bias_hh",
            } else 1
            operators[name] = {
                **config,
                "scales": ["0.0078125"] * count,
                "zero_points": [0] * count,
            }
        return {
            "schema_version": 1,
            "input_size": 2,
            "hidden_size": 1,
            "bias_enabled": bias_enabled,
            "scale_mode": "affine",
            "operators": operators,
        }

    def test_accepts_complete_full_4h_bundle(self) -> None:
        StrictDraft202012Validator(self.schema).validate(self.make_bundle())
        StrictDraft202012Validator(self.schema).validate(
            self.make_bundle(bias_enabled=False)
        )

    def test_rejects_compact_execution_and_bias_fields(self) -> None:
        validator = StrictDraft202012Validator(self.schema)
        compact = self.make_bundle()
        compact["operators"]["weight_ih"]["scales"] = ["0.1"]
        compact["operators"]["weight_ih"]["zero_points"] = [0]
        self.assertFalse(validator.is_valid(compact))

        execution = self.make_bundle()
        execution["operators"]["input"]["raw_ratio"] = "1"
        self.assertFalse(validator.is_valid(execution))

        no_bias = self.make_bundle(bias_enabled=False)
        no_bias["operators"]["bias_ih"] = copy.deepcopy(
            self.make_bundle()["operators"]["bias_ih"]
        )
        self.assertFalse(validator.is_valid(no_bias))

    def test_rejects_non_string_scale_and_non_integer_zero_point(self) -> None:
        validator = StrictDraft202012Validator(self.schema)
        numeric_scale = self.make_bundle()
        numeric_scale["operators"]["input"]["scales"][0] = 0.1
        self.assertFalse(validator.is_valid(numeric_scale))

        float_zero_point = self.make_bundle()
        float_zero_point["operators"]["input"]["zero_points"][0] = 0.0
        self.assertFalse(validator.is_valid(float_zero_point))


if __name__ == "__main__":
    unittest.main()
