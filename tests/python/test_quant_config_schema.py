import copy
import json
import pathlib
import unittest

from tools.strict_jsonschema import StrictDraft202012Validator


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
        StrictDraft202012Validator.check_schema(cls.override_schema)
        StrictDraft202012Validator.check_schema(cls.resolved_schema)

    def test_default_is_complete_resolved_config(self) -> None:
        StrictDraft202012Validator(self.resolved_schema).validate(self.default)
        self.assertEqual(len(self.default["operators"]), 18)

    def test_only_parameters_expose_granularity(self) -> None:
        parameters = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
        validator = StrictDraft202012Validator(self.resolved_schema)
        for name, operator in self.default["operators"].items():
            with self.subTest(operator=name):
                self.assertEqual("granularity" in operator, name in parameters)
                if name not in parameters:
                    for granularity in ("per_tensor", "per_gate", "per_channel"):
                        invalid = copy.deepcopy(self.default)
                        invalid["operators"][name]["granularity"] = granularity
                        self.assertFalse(validator.is_valid(invalid))

    def test_parameter_fixed_flags_are_not_configuration_fields(self) -> None:
        validators = [StrictDraft202012Validator(schema) for schema in
                      (self.resolved_schema, self.override_schema)]
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            for field in ("is_unsigned", "is_symmetric"):
                self.assertNotIn(field, self.default["operators"][name])
                for value in (False, True):
                    invalid = copy.deepcopy(self.default)
                    invalid["operators"][name][field] = value
                    for validator in validators:
                        self.assertFalse(validator.is_valid(invalid))

    def test_operator_comments_are_optional_strings(self) -> None:
        for schema in (self.resolved_schema, self.override_schema):
            validator = StrictDraft202012Validator(schema)
            validator.validate(self.default)
            without_comments = copy.deepcopy(self.default)
            for name, operator in without_comments["operators"].items():
                self.assertIsInstance(operator.pop("comment"), str)
                for invalid_comment in (None, 1, {}, []):
                    invalid = copy.deepcopy(self.default)
                    invalid["operators"][name]["comment"] = invalid_comment
                    self.assertFalse(validator.is_valid(invalid))
            validator.validate(without_comments)
        self.assertFalse(StrictDraft202012Validator(self.override_schema).is_valid({
            "schema_version": 1, "operators": {"input": {"comment": "输入序列"}},
        }))

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
                StrictDraft202012Validator(self.override_schema).validate(override)

    def test_invalid_override_matrix(self) -> None:
        invalid = [
            {},
            {"schema_version": 2},
            {"schema_version": 1.0},
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
                "operators": {"input": {"bitwidth": 8.0}},
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
        validator = StrictDraft202012Validator(self.override_schema)
        for override in invalid:
            with self.subTest(override=override):
                self.assertFalse(validator.is_valid(override))

    def test_resolved_requires_every_operator_and_field(self) -> None:
        missing_operator = copy.deepcopy(self.default)
        del missing_operator["operators"]["cell_tanh_output"]
        missing_field = copy.deepcopy(self.default)
        del missing_field["operators"]["weight_ih"]["granularity"]
        extra_field = copy.deepcopy(self.default)
        extra_field["operators"]["input"]["unexpected"] = 1
        validator = StrictDraft202012Validator(self.resolved_schema)
        self.assertFalse(validator.is_valid(missing_operator))
        self.assertFalse(validator.is_valid(missing_field))
        self.assertFalse(validator.is_valid(extra_field))


if __name__ == "__main__":
    unittest.main()
