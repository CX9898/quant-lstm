#!/usr/bin/env python3
"""只读检查 Stage 9 CUDA 性能报告是否满足版本化设备阈值。"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import jsonschema

try:
    from strict_jsonschema import StrictDraft202012Validator
except ModuleNotFoundError:
    from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = (
    ROOT
    / "tests/benchmarks/config/cuda_performance_thresholds_v1.json"
)
SCHEMA_DIR = ROOT / "tests/benchmarks/schema"
THRESHOLD_SCHEMA = (
    SCHEMA_DIR / "cuda_performance_thresholds.schema.json"
)
BENCHMARK_SCHEMA = SCHEMA_DIR / "cuda_benchmark_result.schema.json"


class PerformanceThresholdError(RuntimeError):
    """性能报告、环境或阈值不满足冻结契约。"""


def _reject_duplicate_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    document: Dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise PerformanceThresholdError(
                f"JSON 包含重复字段: {key}"
            )
        document[key] = value
    return document


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream, object_pairs_hook=_reject_duplicate_keys
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise PerformanceThresholdError(
            f"无法读取 JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PerformanceThresholdError(
            f"JSON 顶层必须是对象: {path}"
        )
    return value


def _validate_schema(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    description: str,
    *,
    check_format: bool = False,
) -> None:
    try:
        StrictDraft202012Validator.check_schema(schema)
        validator = StrictDraft202012Validator(
            schema,
            format_checker=(
                jsonschema.FormatChecker()
                if check_format
                else None
            ),
        )
        validator.validate(document)
    except jsonschema.SchemaError as exc:
        raise PerformanceThresholdError(
            f"{description} schema 无效: {exc.message}"
        ) from exc
    except jsonschema.ValidationError as exc:
        location = ".".join(
            str(part) for part in exc.absolute_path
        ) or "<root>"
        raise PerformanceThresholdError(
            f"{description} 不符合 schema，位置 {location}: "
            f"{exc.message}"
        ) from exc


def check_performance(
    report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    benchmark_schema = load_json(BENCHMARK_SCHEMA)
    threshold_schema = load_json(THRESHOLD_SCHEMA)
    _validate_schema(report, benchmark_schema, "benchmark report")
    _validate_schema(
        thresholds,
        threshold_schema,
        "performance thresholds",
        check_format=True,
    )

    if report["benchmark_id"] != thresholds["benchmark_id"]:
        raise PerformanceThresholdError(
            "benchmark_id 与性能阈值不匹配"
        )
    for name, expected in thresholds["environment"].items():
        actual = report["environment"].get(name)
        if actual != expected:
            raise PerformanceThresholdError(
                f"环境字段 {name} 不匹配: "
                f"expected={expected!r}, actual={actual!r}"
            )

    cases = {case["case_id"]: case for case in report["cases"]}
    profiles = {
        profile["case_id"]: profile
        for profile in thresholds["profiles"]
    }
    if len(cases) != len(report["cases"]):
        raise PerformanceThresholdError(
            "benchmark report 包含重复 case_id"
        )
    if len(profiles) != len(thresholds["profiles"]):
        raise PerformanceThresholdError(
            "performance thresholds 包含重复 case_id"
        )
    if cases.keys() != profiles.keys():
        raise PerformanceThresholdError(
            "benchmark report 与阈值 case_id 集合不一致"
        )

    contract = thresholds["measurement_contract"]
    failures = []
    results = []
    for case_id in sorted(profiles):
        case = cases[case_id]
        profile = profiles[case_id]
        for name in ("shape_profile", "shape", "math_mode"):
            if case[name] != profile[name]:
                failures.append(
                    f"{case_id} 的 {name} 与阈值 profile 不一致"
                )
        if (
            case["warmup_iterations"]
            < contract["minimum_warmup_iterations"]
        ):
            failures.append(f"{case_id} 的 warmup 次数不足")
        if (
            case["measured_iterations"]
            < contract["minimum_measured_iterations"]
        ):
            failures.append(f"{case_id} 的测量次数不足")
        if case["timing_scope"] != contract["timing_scope"]:
            failures.append(f"{case_id} 的 timing_scope 不匹配")
        if not case["static_parameter_cache"][
            "measured_region_all_hits"
        ]:
            failures.append(f"{case_id} 的测量区不是全 cache hit")

        p50 = case["timing_ms"]["end_to_end"]["p50"]
        p95 = case["timing_ms"]["end_to_end"]["p95"]
        throughput = case["throughput"]["sequences_per_second"]
        if profile["maximum_p50_ms"] > profile["maximum_p95_ms"]:
            raise PerformanceThresholdError(
                f"{case_id} 阈值要求 maximum_p50_ms<=maximum_p95_ms"
            )
        if p50 > profile["maximum_p50_ms"]:
            failures.append(
                f"{case_id} P50 {p50:.6f}ms 超过 "
                f"{profile['maximum_p50_ms']:.6f}ms"
            )
        if p95 > profile["maximum_p95_ms"]:
            failures.append(
                f"{case_id} P95 {p95:.6f}ms 超过 "
                f"{profile['maximum_p95_ms']:.6f}ms"
            )
        if throughput < profile["minimum_sequences_per_second"]:
            failures.append(
                f"{case_id} 吞吐 {throughput:.3f} 低于 "
                f"{profile['minimum_sequences_per_second']:.3f}"
            )
        results.append(
            {
                "case_id": case_id,
                "p50_ms": p50,
                "maximum_p50_ms": profile["maximum_p50_ms"],
                "p95_ms": p95,
                "maximum_p95_ms": profile["maximum_p95_ms"],
                "sequences_per_second": throughput,
                "minimum_sequences_per_second": profile[
                    "minimum_sequences_per_second"
                ],
            }
        )

    if failures:
        raise PerformanceThresholdError(
            "性能回归门禁失败: " + "; ".join(failures)
        )
    return {
        "status": "passed",
        "threshold_id": thresholds["threshold_id"],
        "benchmark_id": report["benchmark_id"],
        "environment": report["environment"],
        "profiles": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读检查 Stage 9 CUDA 性能回归阈值。"
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        required=True,
        help="cuda benchmark JSON report",
    )
    parser.add_argument(
        "--thresholds",
        type=pathlib.Path,
        default=DEFAULT_THRESHOLDS,
        help=f"版本化阈值配置（默认: {DEFAULT_THRESHOLDS}）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = check_performance(
            load_json(args.report), load_json(args.thresholds)
        )
    except PerformanceThresholdError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
