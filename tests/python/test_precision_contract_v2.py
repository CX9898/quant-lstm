"""阶段 4 CUDA FP32 precision matrix v2 的 schema 与覆盖契约测试。"""

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

STRICT_SHAPES = {
    "minimal",
    "short_recurrent",
    "non_aligned",
    "long_sequence",
}
BITWIDTH_PROFILES = {"all_int8", "all_int16", "mixed_8_16"}
GRANULARITIES = {"per_tensor", "per_gate", "per_channel"}
PARAMETERS = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
BACKEND_CONTRACT = {
    "cpu_int32_reference": ("cpu_int32", "int32", {"not_applicable"}),
    "cpu_fp32_reference": (
        "cpu_fp32",
        "float32_quantized_values",
        {"not_applicable"},
    ),
    "cuda_fp32": ("cpu_fp32", "float32_quantized_values", {"pedantic", "tf32"}),
}
PAIRWISE_FACTORS = {
    "shape_profile": STRICT_SHAPES,
    "bitwidth_profile": BITWIDTH_PROFILES,
    "scale_mode": {"affine", "pot2"},
    "math_mode": {"pedantic", "tf32"},
    "state_profile": {
        "omitted",
        "explicit_zero",
        "typical_random",
        "near_quant_boundary",
        "mixed_h_random_c_boundary",
    },
    "bias_profile": {"enabled_random", "disabled"},
    "layout_profile": {"time_major", "batch_major"},
    "activation_profile": {
        "domain_default",
        "signed_symmetric",
        "signed_asymmetric",
        "unsigned_symmetric",
        "unsigned_asymmetric",
    },
    "weight_ih": GRANULARITIES,
    "weight_hh": GRANULARITIES,
    "bias_ih": GRANULARITIES | {"not_applicable"},
    "bias_hh": GRANULARITIES | {"not_applicable"},
    "fp32_accumulation_class": {"exact_integer_range", "precision_risk"},
}
REQUIRED_CUDA_DIRECTED_PROFILES = {
    "signed_symmetric_minimum_boundary",
    "signed_asymmetric_minimum_boundary",
    "unsigned_symmetric_boundary",
    "unsigned_asymmetric_boundary",
    "contribution_cancellation",
    "contribution_same_sign",
    "fp32_exact_2pow24",
    "fp32_precision_risk",
}


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def legal_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    values = {left: left_value, right: right_value}
    bias_profile = values.get("bias_profile")
    bias_values = [
        values[name] for name in ("bias_ih", "bias_hh") if name in values
    ]
    if bias_profile == "disabled":
        return all(value == "not_applicable" for value in bias_values)
    if bias_profile == "enabled_random":
        return all(value != "not_applicable" for value in bias_values)
    if len(bias_values) == 2:
        return (bias_values[0] == "not_applicable") == (
            bias_values[1] == "not_applicable"
        )
    return True


def pairwise_row(case: dict) -> dict:
    row = {
        name: case[name]
        for name in (
            "shape_profile",
            "bitwidth_profile",
            "math_mode",
            "state_profile",
            "bias_profile",
            "layout_profile",
            "activation_profile",
            "fp32_accumulation_class",
        )
    }
    row["scale_mode"] = case["resolved_quant_config"]["scale_mode"]
    row.update(case["parameter_granularities"])
    return row


class PrecisionContractV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_json(CONFIG_DIR / "strict_matrix_v1.json")
        cls.v2 = load_json(CONFIG_DIR / "strict_matrix_v2.json")
        cls.schema = load_json(
            SCHEMA_DIR / "strict_matrix_v2.schema.json"
        )
        cls.synthetic = load_json(CONFIG_DIR / "synthetic_data.json")
        cls.thresholds = load_json(CONFIG_DIR / "strict_thresholds.json")
        cls.cuda = [
            case for case in cls.v2["cases"] if case["backend"] == "cuda_fp32"
        ]
        cls.cuda_strict = [
            case for case in cls.cuda if case["tier"] == "strict"
        ]

    def test_v2_matches_strict_schema_and_resolved_config_schema(self) -> None:
        StrictDraft202012Validator.check_schema(self.schema)
        StrictDraft202012Validator(self.schema).validate(self.v2)

        resolved_schema = load_json(RESOLVED_SCHEMA_PATH)
        StrictDraft202012Validator.check_schema(resolved_schema)
        validator = StrictDraft202012Validator(resolved_schema)
        for case in self.v2["cases"]:
            with self.subTest(case=case["case_id"]):
                validator.validate(case["resolved_quant_config"])

    def test_v2_inherits_v1_cpu_cases_without_changes(self) -> None:
        cpu_cases = [
            case for case in self.v2["cases"] if case["backend"] != "cuda_fp32"
        ]
        self.assertEqual(cpu_cases, self.v1["cases"])
        self.assertTrue(
            all("fp32_accumulation_class" not in case for case in cpu_cases)
        )

    def test_version_scope_and_ids_are_fixed(self) -> None:
        self.assertEqual(self.v2["schema_version"], 2)
        self.assertEqual(self.v2["matrix_version"], "strict_matrix_v2")
        self.assertEqual(self.v2["stage"], 4)
        self.assertEqual(self.v2["validation_scope"], "synthetic_numeric")
        self.assertEqual(self.v2["real_data_status"], "not_configured")
        self.assertEqual(
            self.v2["expanded_case_id_format"], "{case_id}_{seed_role}_{seed}"
        )
        case_ids = [case["case_id"] for case in self.v2["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_schema_rejects_unknown_fields_and_invalid_cuda_mapping(self) -> None:
        unknown = copy.deepcopy(self.v2)
        unknown["unexpected"] = True
        validator = StrictDraft202012Validator(self.schema)
        self.assertFalse(validator.is_valid(unknown))

        cuda_index = next(
            index
            for index, case in enumerate(self.v2["cases"])
            if case["backend"] == "cuda_fp32"
        )
        for field, invalid_value in (
            ("execution_model", "cpu_int32"),
            ("carrier", "int32"),
            ("math_mode", "not_applicable"),
        ):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.v2)
                invalid["cases"][cuda_index][field] = invalid_value
                self.assertFalse(validator.is_valid(invalid))

        missing_label = copy.deepcopy(self.v2)
        del missing_label["cases"][cuda_index]["fp32_accumulation_class"]
        self.assertFalse(validator.is_valid(missing_label))

    def test_backend_execution_carrier_and_math_mode_mapping(self) -> None:
        for case in self.v2["cases"]:
            with self.subTest(case=case["case_id"]):
                execution, carrier, math_modes = BACKEND_CONTRACT[case["backend"]]
                self.assertEqual(case["execution_model"], execution)
                self.assertEqual(case["carrier"], carrier)
                self.assertIn(case["math_mode"], math_modes)

        self.assertEqual(
            {case["math_mode"] for case in self.cuda}, {"pedantic", "tf32"}
        )

    def test_cuda_tiers_cover_basic_strict_and_performance_shapes(self) -> None:
        basic = {case["shape_profile"] for case in self.cuda if case["tier"] == "basic"}
        strict = {
            case["shape_profile"] for case in self.cuda if case["tier"] == "strict"
        }
        performance = {
            case["shape_profile"]
            for case in self.cuda
            if case["tier"] == "performance"
        }
        self.assertEqual(basic, {"cuda_gru_basic"})
        self.assertEqual(strict, STRICT_SHAPES)
        self.assertEqual(performance, {"large_batch"})

    def test_shape_layout_seed_and_split_contract(self) -> None:
        calibration = self.synthetic["strict_seeds"]["calibration"]
        evaluation = self.synthetic["strict_seeds"]["evaluation"]
        for case in self.v2["cases"]:
            with self.subTest(case=case["case_id"]):
                profile = case["shape_profile"]
                self.assertEqual(
                    case["shape"], self.synthetic["shape_profiles"][profile]
                )
                self.assertEqual(
                    case["parameter_seed"],
                    self.synthetic["parameter_seeds"][profile],
                )
                self.assertEqual(
                    case["batch_first"],
                    self.synthetic["layout_profiles"][case["layout_profile"]][
                        "batch_first"
                    ],
                )
                seeds = case["data_seeds"]
                if case["tier"] == "basic":
                    self.assertEqual(seeds["calibration"], [0])
                    self.assertEqual(seeds["evaluation"], [0])
                    self.assertTrue(seeds["calibration_equals_evaluation"])
                else:
                    self.assertEqual(seeds["calibration"], calibration)
                    self.assertEqual(seeds["evaluation"], evaluation)
                    self.assertFalse(seeds["calibration_equals_evaluation"])
                    self.assertTrue(
                        set(seeds["calibration"]).isdisjoint(seeds["evaluation"])
                    )

    def test_cuda_bitwidth_scale_bias_layout_state_and_safety_coverage(self) -> None:
        self.assertEqual(
            {case["bitwidth_profile"] for case in self.cuda_strict},
            BITWIDTH_PROFILES,
        )
        self.assertEqual(
            {
                case["resolved_quant_config"]["scale_mode"]
                for case in self.cuda_strict
            },
            {"affine", "pot2"},
        )
        self.assertEqual(
            {case["bias_profile"] for case in self.cuda_strict},
            {"enabled_random", "disabled"},
        )
        self.assertEqual(
            {case["layout_profile"] for case in self.cuda_strict},
            {"time_major", "batch_major"},
        )
        self.assertEqual(
            {case["state_profile"] for case in self.cuda_strict},
            set(self.synthetic["state_profiles"]),
        )
        self.assertEqual(
            {
                case["fp32_accumulation_class"]
                for case in self.cuda_strict
            },
            {"exact_integer_range", "precision_risk"},
        )

    def test_cuda_quantization_and_granularity_constraints(self) -> None:
        threshold_mapping = self.thresholds["quantized_profile_selection"]["mapping"]
        activation_flags = {
            "signed_symmetric": (False, True),
            "signed_asymmetric": (False, False),
            "unsigned_symmetric": (True, True),
            "unsigned_asymmetric": (True, False),
        }
        for case in self.cuda:
            with self.subTest(case=case["case_id"]):
                operators = case["resolved_quant_config"]["operators"]
                bitwidths = {operator["bitwidth"] for operator in operators.values()}
                expected_profile = {
                    frozenset({8}): "all_int8",
                    frozenset({16}): "all_int16",
                    frozenset({8, 16}): "mixed_8_16",
                }[frozenset(bitwidths)]
                self.assertEqual(case["bitwidth_profile"], expected_profile)
                self.assertEqual(
                    case["threshold_profile"],
                    threshold_mapping[str(min(bitwidths))],
                )

                for parameter in PARAMETERS:
                    operator = operators[parameter]
                    self.assertFalse(operator["is_unsigned"])
                    self.assertTrue(operator["is_symmetric"])
                    declared = case["parameter_granularities"][parameter]
                    if (
                        case["bias_profile"] == "disabled"
                        and parameter.startswith("bias")
                    ):
                        self.assertEqual(declared, "not_applicable")
                    else:
                        self.assertEqual(declared, operator["granularity"])

                for name, operator in operators.items():
                    if name not in PARAMETERS:
                        self.assertEqual(operator["granularity"], "per_tensor")

                profile = case["activation_profile"]
                if profile in activation_flags:
                    probe = operators[case["activation_probe"]]
                    self.assertEqual(
                        (probe["is_unsigned"], probe["is_symmetric"]),
                        activation_flags[profile],
                    )

    def test_cuda_strict_has_complete_constrained_pairwise_coverage(self) -> None:
        rows = [pairwise_row(case) for case in self.cuda_strict]
        for left, right in itertools.combinations(PAIRWISE_FACTORS, 2):
            observed = {(row[left], row[right]) for row in rows}
            expected = {
                (left_value, right_value)
                for left_value, right_value in itertools.product(
                    PAIRWISE_FACTORS[left], PAIRWISE_FACTORS[right]
                )
                if legal_pair(left, left_value, right, right_value)
            }
            self.assertEqual(
                observed,
                expected,
                f"{left}/{right} 的合法 pair 覆盖不完整",
            )

    def test_cuda_directed_and_exact_risk_labels_are_consistent(self) -> None:
        observed = {
            profile
            for case in self.cuda_strict
            for profile in case["directed_profiles"]
        }
        self.assertTrue(REQUIRED_CUDA_DIRECTED_PROFILES.issubset(observed))
        for case in self.cuda:
            with self.subTest(case=case["case_id"]):
                directed = set(case["directed_profiles"])
                if case["fp32_accumulation_class"] == "exact_integer_range":
                    self.assertIn("fp32_exact_2pow24", directed)
                    self.assertNotIn("fp32_precision_risk", directed)
                else:
                    self.assertIn("fp32_precision_risk", directed)
                    self.assertNotIn("fp32_exact_2pow24", directed)

    def test_cuda_runtime_gate_profiles_reuse_stage3_threshold_cases(self) -> None:
        expected = {
            "stage4_strict_cuda_fp32_minimal_int8": (
                "minimal",
                "pedantic",
                "exact_integer_range",
            ),
            "stage4_strict_cuda_fp32_short_int16": (
                "short_recurrent",
                "tf32",
                "precision_risk",
            ),
            "stage4_strict_cuda_fp32_nonaligned_mixed": (
                "non_aligned",
                "pedantic",
                "precision_risk",
            ),
            "stage4_strict_cuda_fp32_long_int8": (
                "long_sequence",
                "tf32",
                "exact_integer_range",
            ),
        }
        by_id = {case["case_id"]: case for case in self.cuda_strict}
        for case_id, contract in expected.items():
            with self.subTest(case=case_id):
                case = by_id[case_id]
                self.assertEqual(
                    (
                        case["shape_profile"],
                        case["math_mode"],
                        case["fp32_accumulation_class"],
                    ),
                    contract,
                )
                if contract[2] == "exact_integer_range" and contract[1] == "tf32":
                    bits = {
                        operator["bitwidth"]
                        for operator in case["resolved_quant_config"][
                            "operators"
                        ].values()
                    }
                    self.assertEqual(bits, {8})


if __name__ == "__main__":
    unittest.main()
