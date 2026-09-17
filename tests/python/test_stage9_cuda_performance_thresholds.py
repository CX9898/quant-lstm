"""Stage 9 CUDA 性能阈值 schema、环境隔离与回归门禁测试。"""

import copy
import unittest

import jsonschema

from tests.python.test_stage4_cuda_benchmark_contract import (
    benchmark_result,
)
from tools import check_stage9_cuda_performance as checker
from tools.strict_jsonschema import StrictDraft202012Validator


THRESHOLDS = checker.load_json(checker.DEFAULT_THRESHOLDS)
THRESHOLD_SCHEMA = checker.load_json(checker.THRESHOLD_SCHEMA)


def passing_report() -> dict:
    report = benchmark_result()
    report["environment"] = copy.deepcopy(THRESHOLDS["environment"])
    for case, profile in zip(
        report["cases"], THRESHOLDS["profiles"]
    ):
        case["case_id"] = profile["case_id"]
        case["shape_profile"] = profile["shape_profile"]
        case["shape"] = list(profile["shape"])
        case["math_mode"] = profile["math_mode"]
        case["timing_ms"]["end_to_end"]["p50"] = (
            profile["maximum_p50_ms"] * 0.9
        )
        case["timing_ms"]["end_to_end"]["p95"] = (
            profile["maximum_p95_ms"] * 0.9
        )
        case["throughput"]["sequences_per_second"] = (
            profile["minimum_sequences_per_second"] * 1.1
        )
    return report


class Stage9CudaPerformanceThresholdTest(unittest.TestCase):
    def test_threshold_schema_and_config_are_strict(self) -> None:
        StrictDraft202012Validator.check_schema(THRESHOLD_SCHEMA)
        StrictDraft202012Validator(
            THRESHOLD_SCHEMA,
            format_checker=jsonschema.FormatChecker(),
        ).validate(THRESHOLDS)

        unknown = copy.deepcopy(THRESHOLDS)
        unknown["profiles"][0]["unexpected"] = 1
        self.assertFalse(
            StrictDraft202012Validator(
                THRESHOLD_SCHEMA,
                format_checker=jsonschema.FormatChecker(),
            ).is_valid(unknown)
        )

    def test_matching_report_passes_without_mutation(self) -> None:
        report = passing_report()
        original_report = copy.deepcopy(report)
        original_thresholds = copy.deepcopy(THRESHOLDS)
        result = checker.check_performance(report, THRESHOLDS)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["profiles"]), 4)
        self.assertEqual(report, original_report)
        self.assertEqual(THRESHOLDS, original_thresholds)

    def test_environment_mismatch_fails_instead_of_reusing_thresholds(
        self,
    ) -> None:
        report = passing_report()
        report["environment"]["gpu_name"] = "Different GPU"
        with self.assertRaisesRegex(
            checker.PerformanceThresholdError, "环境字段"
        ):
            checker.check_performance(report, THRESHOLDS)

    def test_latency_throughput_and_measurement_contract_are_enforced(
        self,
    ) -> None:
        mutations = (
            (
                "P50",
                lambda report: report["cases"][0]["timing_ms"][
                    "end_to_end"
                ].__setitem__("p50", 99.0),
            ),
            (
                "P95",
                lambda report: report["cases"][0]["timing_ms"][
                    "end_to_end"
                ].__setitem__("p95", 99.0),
            ),
            (
                "吞吐",
                lambda report: report["cases"][0][
                    "throughput"
                ].__setitem__("sequences_per_second", 1.0),
            ),
            (
                "测量次数",
                lambda report: report["cases"][0].__setitem__(
                    "measured_iterations", 99
                ),
            ),
            (
                "cache hit",
                lambda report: report["cases"][0][
                    "static_parameter_cache"
                ].__setitem__("measured_region_all_hits", False),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                report = passing_report()
                mutate(report)
                with self.assertRaisesRegex(
                    checker.PerformanceThresholdError, expected
                ):
                    checker.check_performance(report, THRESHOLDS)

    def test_profile_mapping_and_case_set_are_exact(self) -> None:
        wrong_shape = passing_report()
        wrong_shape["cases"][0]["shape"][0] += 1
        with self.assertRaisesRegex(
            checker.PerformanceThresholdError, "shape"
        ):
            checker.check_performance(wrong_shape, THRESHOLDS)

        missing = passing_report()
        missing["cases"][-1]["case_id"] = "unexpected_case"
        with self.assertRaisesRegex(
            checker.PerformanceThresholdError, "case_id 集合"
        ):
            checker.check_performance(missing, THRESHOLDS)

    def test_reviewed_preoptimization_baseline_is_rejected(self) -> None:
        baseline = {
            "stage4_basic_cuda_fp32_pedantic_int8_bias_time": (
                1.450847983,
                1.458847999,
                44112.1335,
            ),
            "stage4_basic_cuda_fp32_tf32_int16_no_bias_batch": (
                1.203328013,
                1.209375978,
                53185.8307,
            ),
            "stage4_performance_cuda_fp32_large_batch_pedantic": (
                0.501183987,
                0.507200003,
                255395.2309,
            ),
            "stage4_performance_cuda_fp32_large_batch_tf32": (
                0.403584003,
                0.408735991,
                317158.2593,
            ),
        }
        report = passing_report()
        for case in report["cases"]:
            p50, p95, throughput = baseline[case["case_id"]]
            case["timing_ms"]["end_to_end"]["p50"] = p50
            case["timing_ms"]["end_to_end"]["p95"] = p95
            case["throughput"]["sequences_per_second"] = throughput
        with self.assertRaisesRegex(
            checker.PerformanceThresholdError,
            "性能回归门禁失败",
        ):
            checker.check_performance(report, THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
