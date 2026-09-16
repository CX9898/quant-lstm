import copy
import json
import pathlib
import unittest

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config/defaults/lstm_quant_default_v1.json"
OVERRIDE_SCHEMA_PATH = ROOT / "config/schema/lstm_quant_override.schema.json"
RESOLVED_SCHEMA_PATH = ROOT / "config/schema/lstm_quant_resolved.schema.json"


class QuantConfigSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.default = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
        cls.override_schema = json.loads(
            OVERRIDE_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        cls.resolved_schema = json.loads(
            RESOLVED_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(cls.override_schema)
        jsonschema.Draft202012Validator.check_schema(cls.resolved_schema)

    def test_default_is_complete_resolved_config(self) -> None:
        jsonschema.validate(
            self.default,
            self.resolved_schema,
            cls=jsonschema.Draft202012Validator,
        )
        self.assertEqual(len(self.default["operators"]), 18)

    def test_sparse_override_variants(self) -> None:
        valid = [
            {"schema_version": 1},
            {"schema_version": 1, "scale_mode": "pot2"},
            {
                "schema_version": 1,
                "operators": {"weight_ih": {"granularity": "per_gate"}},
            },
            {
                "schema_version": 1,
                "operators": {
                    "input": {
                        "bitwidth": 16,
                        "is_unsigned": True,
                        "is_symmetric": False,
                    }
                },
            },
        ]
        for override in valid:
            with self.subTest(override=override):
                jsonschema.validate(
                    override,
                    self.override_schema,
                    cls=jsonschema.Draft202012Validator,
                )

    def test_invalid_override_matrix(self) -> None:
        invalid = [
            {},
            {"schema_version": 2},
            {"schema_version": 1, "unknown": 1},
            {"schema_version": 1, "scale_mode": None},
            {"schema_version": 1, "pot_scale_method": "floor"},
            {"schema_version": 1, "pot_scale_tolerance": 0.02},
            {
                "schema_version": 1,
                "operators": {"input": {"bitwidth": 4}},
            },
            {
                "schema_version": 1,
                "operators": {"input": {"granularity": "per_gate"}},
            },
            {
                "schema_version": 1,
                "operators": {"weight_ih": {"is_unsigned": True}},
            },
            {
                "schema_version": 1,
                "operators": {"weight_ih": {"is_symmetric": False}},
            },
            {
                "schema_version": 1,
                "operators": {"mul_output_cell": {"bitwidth": 8}},
            },
        ]
        validator = jsonschema.Draft202012Validator(self.override_schema)
        for override in invalid:
            with self.subTest(override=override):
                self.assertFalse(validator.is_valid(override))

    def test_resolved_requires_every_operator_and_field(self) -> None:
        missing_operator = copy.deepcopy(self.default)
        del missing_operator["operators"]["cell_tanh_output"]
        missing_field = copy.deepcopy(self.default)
        del missing_field["operators"]["input"]["granularity"]
        extra_field = copy.deepcopy(self.default)
        extra_field["operators"]["input"]["unexpected"] = 1
        validator = jsonschema.Draft202012Validator(self.resolved_schema)
        self.assertFalse(validator.is_valid(missing_operator))
        self.assertFalse(validator.is_valid(missing_field))
        self.assertFalse(validator.is_valid(extra_field))


if __name__ == "__main__":
    unittest.main()
