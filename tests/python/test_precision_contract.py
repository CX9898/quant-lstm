"""阶段 3 synthetic precision 配置的 schema 与覆盖契约测试。"""

import copy
import itertools
import json
import pathlib
import unittest

from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "tests/precision/config"
SCHEMA_DIR = ROOT / "tests/precision/schema"
RESOLVED_SCHEMA_PATH = ROOT / "config/schema/lstm_quant_resolved.schema.json"

DOCUMENTS = {
    "metric_policy": "metric_policy.json",
    "synthetic_data": "synthetic_data.json",
    "strict_thresholds": "strict_thresholds.json",
    "strict_matrix": "strict_matrix_v1.json",
}

PARAMETERS = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
GRANULARITIES = ("per_tensor", "per_gate", "per_channel")
STRICT_SHAPES = ("minimal", "short_recurrent", "non_aligned", "long_sequence")
BITWIDTH_PROFILES = ("all_int8", "all_int16", "mixed_8_16")
BACKEND_CONTRACT = {
    "cpu_int32_reference": ("cpu_int32", "int32"),
    "cpu_fp32_reference": ("cpu_fp32", "float32_quantized_values"),
}
REQUIRED_DIRECTED_PROFILES = {
    "signed_symmetric_minimum_boundary",
    "signed_asymmetric_minimum_boundary",
    "unsigned_symmetric_boundary",
    "unsigned_asymmetric_boundary",
    "contribution_cancellation",
    "contribution_same_sign",
    "fp32_exact_2pow24",
}


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class PrecisionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = {
            name: load_json(CONFIG_DIR / filename)
            for name, filename in DOCUMENTS.items()
        }
        cls.schemas = {
            name: load_json(SCHEMA_DIR / f"{name}.schema.json")
            for name in DOCUMENTS
        }

    def test_documents_match_strict_schemas(self) -> None:
        for name, document in self.documents.items():
            with self.subTest(document=name):
                schema = self.schemas[name]
                StrictDraft202012Validator.check_schema(schema)
                StrictDraft202012Validator(schema).validate(document)

    def test_matrix_quant_configs_match_authoritative_resolved_schema(self) -> None:
        schema = load_json(RESOLVED_SCHEMA_PATH)
        StrictDraft202012Validator.check_schema(schema)
        validator = StrictDraft202012Validator(schema)
        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                validator.validate(case["resolved_quant_config"])

    def test_schemas_reject_unknown_fields(self) -> None:
        for name, document in self.documents.items():
            with self.subTest(document=name):
                invalid = copy.deepcopy(document)
                invalid["unexpected"] = True
                validator = StrictDraft202012Validator(self.schemas[name])
                self.assertFalse(validator.is_valid(invalid))

    def test_case_ids_and_stage_scope_are_fixed(self) -> None:
        matrix = self.documents["strict_matrix"]
        self.assertEqual(matrix["stage"], 3)
        self.assertEqual(matrix["validation_scope"], "synthetic_numeric")
        self.assertEqual(matrix["real_data_status"], "not_configured")
        self.assertEqual(
            matrix["expanded_case_id_format"], "{case_id}_{seed_role}_{seed}"
        )
        case_ids = [case["case_id"] for case in matrix["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_basic_and_strict_shape_coverage(self) -> None:
        cases = self.documents["strict_matrix"]["cases"]
        basic = [case for case in cases if case["tier"] == "basic"]
        strict = [case for case in cases if case["tier"] == "strict"]
        self.assertTrue(basic)
        self.assertTrue(strict)
        self.assertEqual({case["shape_profile"] for case in basic}, {"cpu_basic"})

        for backend in BACKEND_CONTRACT:
            backend_strict = [case for case in strict if case["backend"] == backend]
            self.assertEqual(
                {case["shape_profile"] for case in backend_strict},
                set(STRICT_SHAPES),
            )
            self.assertEqual(
                {case["bitwidth_profile"] for case in backend_strict},
                set(BITWIDTH_PROFILES),
            )
            self.assertEqual(
                {case["resolved_quant_config"]["scale_mode"] for case in backend_strict},
                {"affine", "pot2"},
            )
            self.assertEqual(
                {case["bias_profile"] for case in backend_strict},
                {"enabled_random", "disabled"},
            )
            self.assertEqual(
                {case["layout_profile"] for case in backend_strict},
                {"time_major", "batch_major"},
            )

        self.assertEqual(
            {case["state_profile"] for case in strict},
            set(self.documents["synthetic_data"]["state_profiles"]),
        )
        self.assertEqual(
            {case["activation_profile"] for case in strict},
            {
                "domain_default",
                "signed_symmetric",
                "signed_asymmetric",
                "unsigned_symmetric",
                "unsigned_asymmetric",
            },
        )

    def test_execution_model_and_carrier_mapping(self) -> None:
        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                expected_model, expected_carrier = BACKEND_CONTRACT[case["backend"]]
                self.assertEqual(case["execution_model"], expected_model)
                self.assertEqual(case["carrier"], expected_carrier)
                self.assertEqual(case["math_mode"], "not_applicable")

    def test_seed_roles_are_isolated_and_parameter_seed_is_shape_stable(self) -> None:
        synthetic = self.documents["synthetic_data"]
        calibration = synthetic["strict_seeds"]["calibration"]
        evaluation = synthetic["strict_seeds"]["evaluation"]
        self.assertTrue(set(calibration).isdisjoint(evaluation))

        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                seeds = case["data_seeds"]
                if case["tier"] == "basic":
                    self.assertEqual(seeds["calibration"], [synthetic["basic_seed"]])
                    self.assertEqual(seeds["evaluation"], [synthetic["basic_seed"]])
                    self.assertTrue(seeds["calibration_equals_evaluation"])
                else:
                    self.assertEqual(seeds["calibration"], calibration)
                    self.assertEqual(seeds["evaluation"], evaluation)
                    self.assertFalse(seeds["calibration_equals_evaluation"])
                    self.assertTrue(
                        set(seeds["calibration"]).isdisjoint(seeds["evaluation"])
                    )
                self.assertEqual(
                    case["parameter_seed"],
                    synthetic["parameter_seeds"][case["shape_profile"]],
                )

    def test_shape_and_layout_metadata_match_synthetic_contract(self) -> None:
        synthetic = self.documents["synthetic_data"]
        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(
                    case["shape"], synthetic["shape_profiles"][case["shape_profile"]]
                )
                layout = synthetic["layout_profiles"][case["layout_profile"]]
                self.assertEqual(case["batch_first"], layout["batch_first"])

    def test_parameter_constraints_and_pairwise_granularity(self) -> None:
        strict_enabled = [
            case
            for case in self.documents["strict_matrix"]["cases"]
            if case["tier"] == "strict" and case["bias_profile"] == "enabled_random"
        ]

        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                operators = case["resolved_quant_config"]["operators"]
                granularities = case["parameter_granularities"]
                for parameter in PARAMETERS:
                    operator = operators[parameter]
                    self.assertFalse(operator["is_unsigned"])
                    self.assertTrue(operator["is_symmetric"])
                    if case["bias_profile"] == "disabled" and parameter.startswith("bias"):
                        self.assertEqual(granularities[parameter], "not_applicable")
                    else:
                        self.assertEqual(
                            granularities[parameter], operator["granularity"]
                        )

        for left, right in itertools.combinations(PARAMETERS, 2):
            observed = {
                (
                    case["parameter_granularities"][left],
                    case["parameter_granularities"][right],
                )
                for case in strict_enabled
            }
            expected = set(itertools.product(GRANULARITIES, repeat=2))
            self.assertEqual(observed, expected, f"{left}/{right} 粒度对覆盖不完整")

    def test_bitwidth_profile_and_threshold_tier(self) -> None:
        thresholds = self.documents["strict_thresholds"]
        mapping = thresholds["quantized_profile_selection"]["mapping"]
        for case in self.documents["strict_matrix"]["cases"]:
            with self.subTest(case=case["case_id"]):
                bitwidths = {
                    operator["bitwidth"]
                    for operator in case["resolved_quant_config"]["operators"].values()
                }
                expected_bitwidth_profile = {
                    frozenset({8}): "all_int8",
                    frozenset({16}): "all_int16",
                    frozenset({8, 16}): "mixed_8_16",
                }[frozenset(bitwidths)]
                minimum = min(bitwidths)
                expected_threshold = mapping[str(minimum)]
                self.assertEqual(case["bitwidth_profile"], expected_bitwidth_profile)
                self.assertEqual(case["threshold_profile"], expected_threshold)
                self.assertIn(expected_threshold, thresholds["profiles"])

    def test_activation_probe_matches_declared_profile(self) -> None:
        expected_flags = {
            "signed_symmetric": (False, True),
            "signed_asymmetric": (False, False),
            "unsigned_symmetric": (True, True),
            "unsigned_asymmetric": (True, False),
        }
        for case in self.documents["strict_matrix"]["cases"]:
            if case["activation_profile"] == "domain_default":
                continue
            with self.subTest(case=case["case_id"]):
                probe = case["resolved_quant_config"]["operators"][
                    case["activation_probe"]
                ]
                self.assertEqual(
                    (probe["is_unsigned"], probe["is_symmetric"]),
                    expected_flags[case["activation_profile"]],
                )

    def test_mandatory_directed_profiles_are_present(self) -> None:
        strict = [
            case
            for case in self.documents["strict_matrix"]["cases"]
            if case["tier"] == "strict"
        ]
        observed = {
            profile for case in strict for profile in case["directed_profiles"]
        }
        self.assertTrue(REQUIRED_DIRECTED_PROFILES.issubset(observed))

        fp_boundary_cases = [
            case
            for case in strict
            if "fp32_exact_2pow24" in case["directed_profiles"]
        ]
        self.assertTrue(fp_boundary_cases)
        self.assertTrue(
            all(case["execution_model"] == "cpu_fp32" for case in fp_boundary_cases)
        )


if __name__ == "__main__":
    unittest.main()
