import copy
import json
import pathlib
import tempfile
import unittest

from tools.generate_golden import load_documents, render, validate_tensor_contracts
from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tests/golden/schema/golden_case.schema.json"
GOLDEN_SPEC_DIRECTORY = ROOT / "tests/golden/spec"
PRIMITIVE_DIRECTORY = GOLDEN_SPEC_DIRECTORY / "primitive"


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
        paths = sorted(PRIMITIVE_DIRECTORY.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                validator.validate(document)
                validate_tensor_contracts(document)
                self.assertEqual(document["kind"], "primitive")
                self.assertNotIn(document["case_id"], case_ids)
                case_ids.add(document["case_id"])
                primitives.add(document["primitive"])
        self.assertEqual(primitives, expected_primitives)
        loaded_primitives = [
            document
            for document in load_documents(GOLDEN_SPEC_DIRECTORY, SCHEMA_PATH)
            if document["kind"] == "primitive"
        ]
        self.assertEqual(
            len(loaded_primitives), len(paths)
        )

    def test_schema_rejects_carrier_and_tensor_contract_errors(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = StrictDraft202012Validator(schema)
        activation = json.loads(
            (PRIMITIVE_DIRECTORY / "real_activation_boundaries.json").read_text(
                encoding="utf-8"
            )
        )
        invalid_common = copy.deepcopy(activation)
        invalid_common["execution_model"] = "common"
        invalid_case_kind = copy.deepcopy(activation)
        invalid_case_kind["kind"] = "cell"
        invalid_kind = copy.deepcopy(activation)
        invalid_kind["inputs"]["kind"]["data"][0] = "relu"
        invalid_attribute = copy.deepcopy(activation)
        invalid_attribute["attributes"]["unknown"] = True
        invalid_shape = copy.deepcopy(activation)
        invalid_shape["inputs"]["kind"]["shape"] = [0]

        ranges = json.loads(
            (PRIMITIVE_DIRECTORY / "quantized_ranges.json").read_text(encoding="utf-8")
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
            invalid_case_kind,
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
            "kind": "primitive",
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
        validate_tensor_contracts(scalar_round)

        invalid_count = copy.deepcopy(scalar_round)
        invalid_count["inputs"]["values"]["shape"] = [2]
        with self.assertRaises(ValueError):
            validate_tensor_contracts(invalid_count)

        noncanonical_float32 = copy.deepcopy(activation)
        noncanonical_float32["inputs"]["input_scale"]["data"][0] = "0.1250"
        validator.validate(noncanonical_float32)
        with self.assertRaises(ValueError):
            validate_tensor_contracts(noncanonical_float32)

    def test_generator_is_deterministic_and_does_not_change_expected(self) -> None:
        first = render(GOLDEN_SPEC_DIRECTORY, SCHEMA_PATH)
        second = render(GOLDEN_SPEC_DIRECTORY, SCHEMA_PATH)
        self.assertEqual(first, second)
        self.assertIn("kGoldenDocuments", first)
        expected = json.loads(
            (PRIMITIVE_DIRECTORY / "round_half_ties.json").read_text(encoding="utf-8")
        )["expected"]
        self.assertIn(
            json.dumps(expected, sort_keys=True, separators=(",", ":")), first
        )

    def test_generator_rejects_directory_kind_mismatch(self) -> None:
        source = {
            "schema_version": 1,
            "case_id": "cell_in_wrong_directory",
            "kind": "cell",
            "execution_model": "cpu_int32",
            "attributes": {},
            "inputs": {
                "input": {"dtype": "int32", "shape": [], "data": [0]}
            },
            "expected": {
                "checkpoints": {
                    "output": {"dtype": "int32", "shape": [], "data": [0]}
                },
                "diagnostics": {},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            spec = pathlib.Path(temporary)
            primitive = spec / "primitive"
            primitive.mkdir()
            (primitive / "wrong_kind.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_documents(spec, SCHEMA_PATH)


if __name__ == "__main__":
    unittest.main()
