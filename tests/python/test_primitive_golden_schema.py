import copy
import json
import pathlib
import unittest

from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "config/schema/primitive_golden.schema.json"
GOLDEN_DIRECTORY = ROOT / "tests/golden/primitive"


class PrimitiveGoldenSchemaTest(unittest.TestCase):
    def test_all_primitive_goldens_are_valid_and_complete(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        StrictDraft202012Validator.check_schema(schema)
        validator = StrictDraft202012Validator(schema)
        expected_primitives = {
            "round_to_nearest_even",
            "quantized_range",
            "m_shift",
            "pot2_cover_range",
            "quant_dequant",
            "real_activation",
        }
        case_ids: set[str] = set()
        primitives: set[str] = set()
        paths = sorted(GOLDEN_DIRECTORY.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                validator.validate(document)
                self.assertNotIn(document["case_id"], case_ids)
                case_ids.add(document["case_id"])
                primitives.add(document["primitive"])
        self.assertEqual(primitives, expected_primitives)

    def test_schema_rejects_carrier_and_tensor_contract_errors(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = StrictDraft202012Validator(schema)
        activation = json.loads(
            (GOLDEN_DIRECTORY / "real_activation_boundaries.json").read_text(
                encoding="utf-8"
            )
        )
        invalid_common = copy.deepcopy(activation)
        invalid_common["execution_model"] = "common"
        invalid_kind = copy.deepcopy(activation)
        invalid_kind["inputs"]["kind"]["data"][0] = "relu"
        invalid_attribute = copy.deepcopy(activation)
        invalid_attribute["attributes"]["unknown"] = True
        invalid_shape = copy.deepcopy(activation)
        invalid_shape["inputs"]["kind"]["shape"] = [0]

        ranges = json.loads(
            (GOLDEN_DIRECTORY / "quantized_ranges.json").read_text(encoding="utf-8")
        )
        invalid_uint8 = copy.deepcopy(ranges)
        invalid_uint8["inputs"]["bitwidth"]["data"][0] = 256
        invalid_integer_spelling = copy.deepcopy(ranges)
        invalid_integer_spelling["inputs"]["bitwidth"]["data"][0] = 8.0
        invalid_dtype = copy.deepcopy(ranges)
        invalid_dtype["inputs"]["bitwidth"]["dtype"] = "string"
        invalid_dtype["inputs"]["bitwidth"]["data"] = ["8"] * 6

        for document in [
            invalid_common,
            invalid_kind,
            invalid_attribute,
            invalid_shape,
            invalid_uint8,
            invalid_integer_spelling,
            invalid_dtype,
        ]:
            with self.subTest(case=document["case_id"]):
                self.assertFalse(validator.is_valid(document))

        scalar_round = {
            "schema_version": 1,
            "case_id": "scalar_round",
            "execution_model": "common",
            "primitive": "round_to_nearest_even",
            "attributes": {},
            "inputs": {
                "values": {"dtype": "float64", "shape": [], "data": [0.5]}
            },
            "expected": {
                "checkpoints": {
                    "rounded": {"dtype": "int64", "shape": [], "data": [0]}
                },
                "diagnostics": {},
            },
        }
        validator.validate(scalar_round)


if __name__ == "__main__":
    unittest.main()
