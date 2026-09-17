#!/usr/bin/env python3
"""运行 Stage 4 CUDA benchmark、诊断与 Nsight Systems 交叉验证。"""

import argparse
import csv
import datetime
import io
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, MutableSequence, Sequence, Tuple

import jsonschema
from referencing import Registry, Resource

try:
    from strict_jsonschema import StrictDraft202012Validator
except ModuleNotFoundError:
    from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tests/benchmarks/config/cuda_benchmark_v1.json"
SCHEMA_DIR = ROOT / "tests/benchmarks/schema"
CONFIG_SCHEMA = SCHEMA_DIR / "cuda_benchmark_config.schema.json"
BENCHMARK_SCHEMA = SCHEMA_DIR / "cuda_benchmark_result.schema.json"
REPORT_SCHEMA = SCHEMA_DIR / "stage4_cuda_validation_report.schema.json"
GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "compute_cap",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "power.limit",
)


class Stage4ValidationError(RuntimeError):
    """可预期且应向调用方明确报告的验证失败。"""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    document: Dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise Stage4ValidationError(f"JSON 包含重复字段: {key}")
        document[key] = value
    return document


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError as exc:
        raise Stage4ValidationError(f"JSON 文件不存在: {path}") from exc
    except PermissionError as exc:
        raise Stage4ValidationError(f"无权读取 JSON 文件: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage4ValidationError(
            f"JSON 解析失败 {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise Stage4ValidationError(f"读取 JSON 文件失败 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage4ValidationError(f"JSON 顶层必须是对象: {path}")
    return value


def _validate_document(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    description: str,
    *,
    registry: Registry = Registry(),
    check_format: bool = False,
) -> None:
    try:
        StrictDraft202012Validator.check_schema(schema)
        validator = StrictDraft202012Validator(
            schema,
            registry=registry,
            format_checker=jsonschema.FormatChecker() if check_format else None,
        )
        validator.validate(document)
    except jsonschema.SchemaError as exc:
        raise Stage4ValidationError(f"{description} schema 无效: {exc.message}") from exc
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise Stage4ValidationError(
            f"{description} 不符合 schema，位置 {location}: {exc.message}"
        ) from exc


def _write_text(path: pathlib.Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise Stage4ValidationError(f"写入产物失败 {path}: {exc}") from exc


def _write_json(path: pathlib.Path, document: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def _require_tool(tool_name: str) -> str:
    resolved = shutil.which(tool_name)
    if resolved is None:
        raise Stage4ValidationError(
            f"必需工具不可用或不在 PATH 中: {tool_name}"
        )
    path = pathlib.Path(resolved)
    if not path.is_file():
        raise Stage4ValidationError(f"工具路径不是普通文件: {path}")
    if not os.access(path, os.X_OK):
        raise Stage4ValidationError(f"工具没有执行权限: {path}")
    return str(path.resolve())


def _resolve_benchmark(build_dir: pathlib.Path, configured_path: str) -> str:
    relative = pathlib.Path(configured_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Stage4ValidationError(
            "benchmark.executable 必须是 build-dir 内不含 '..' 的相对路径"
        )
    candidate = (build_dir / relative).resolve()
    try:
        candidate.relative_to(build_dir)
    except ValueError as exc:
        raise Stage4ValidationError(
            f"benchmark 可执行文件逃逸 build-dir: {candidate}"
        ) from exc
    if not candidate.is_file():
        raise Stage4ValidationError(
            f"benchmark 可执行文件不存在: {candidate}；请按 --help 中的 CLI 契约构建"
        )
    if not os.access(candidate, os.X_OK):
        raise Stage4ValidationError(f"benchmark 没有执行权限: {candidate}")
    return str(candidate)


def _expand_argv(
    executable: str,
    template: Sequence[str],
    replacements: Mapping[str, str],
) -> List[str]:
    allowed = set(replacements)
    result = [executable]
    placeholder_pattern = re.compile(r"\{([a-z_]+)\}")
    for item in template:
        placeholders = set(placeholder_pattern.findall(item))
        unknown = placeholders - allowed
        if unknown:
            raise Stage4ValidationError(
                f"benchmark argv 含未知占位符: {sorted(unknown)}"
            )
        expanded = item
        for name in placeholders:
            expanded = expanded.replace("{" + name + "}", replacements[name])
        if "{" in expanded or "}" in expanded:
            raise Stage4ValidationError(
                f"benchmark argv 含未解析占位符或花括号: {item}"
            )
        if not expanded:
            raise Stage4ValidationError("benchmark argv 展开后不得为空")
        result.append(expanded)
    return result


def _run_command(
    step: str,
    argv: Sequence[str],
    timeout_seconds: int,
    artifact_dir: pathlib.Path,
    records: MutableSequence[Dict[str, Any]],
    *,
    cwd: pathlib.Path,
) -> Tuple[str, str]:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise Stage4ValidationError(f"{step} 命令必须是非空字符串 argv 数组")
    argv_list = list(argv)
    ordinal = len(records) + 1
    stdout_path = artifact_dir / f"{ordinal:02d}_{step}.stdout.txt"
    stderr_path = artifact_dir / f"{ordinal:02d}_{step}.stderr.txt"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv_list,
            shell=False,
            check=False,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise Stage4ValidationError(f"{step} 可执行文件不存在: {argv_list[0]}") from exc
    except PermissionError as exc:
        raise Stage4ValidationError(f"{step} 命令无执行权限: {argv_list[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        _write_text(stdout_path, stdout)
        _write_text(stderr_path, stderr)
        raise Stage4ValidationError(
            f"{step} 超时（{timeout_seconds} 秒）；日志: {stdout_path}, {stderr_path}"
        ) from exc
    except OSError as exc:
        raise Stage4ValidationError(f"{step} 启动失败: {exc}") from exc

    duration = time.monotonic() - started
    _write_text(stdout_path, completed.stdout)
    _write_text(stderr_path, completed.stderr)
    records.append(
        {
            "step": step,
            "argv": argv_list,
            "returncode": completed.returncode,
            "duration_seconds": duration,
            "stdout_artifact": str(stdout_path),
            "stderr_artifact": str(stderr_path),
        }
    )
    if completed.returncode != 0:
        raise Stage4ValidationError(
            f"{step} 失败，退出码 {completed.returncode}；"
            f"日志: {stdout_path}, {stderr_path}"
        )
    return completed.stdout, completed.stderr


def _parse_finite_number(value: str, field: str) -> float:
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise Stage4ValidationError(
            f"nvidia-smi 字段 {field} 不是数值: {value!r}"
        ) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise Stage4ValidationError(
            f"nvidia-smi 字段 {field} 不是有限数值: {value!r}"
        )
    return number


def _parse_gpu_environment(stdout: str, requested_device: int) -> Dict[str, Any]:
    rows = list(csv.reader(io.StringIO(stdout.strip())))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY_FIELDS):
        raise Stage4ValidationError(
            "nvidia-smi 返回格式未知：期望单行 11 列 GPU 环境数据"
        )
    values = [value.strip() for value in rows[0]]
    try:
        gpu_index = int(values[0])
    except ValueError as exc:
        raise Stage4ValidationError(
            f"nvidia-smi GPU index 无效: {values[0]!r}"
        ) from exc
    if gpu_index != requested_device:
        raise Stage4ValidationError(
            f"nvidia-smi 返回 GPU {gpu_index}，但请求的是 GPU {requested_device}"
        )
    text_fields = values[1:6]
    if any(not value or value.upper() in {"N/A", "NOT SUPPORTED"} for value in text_fields):
        raise Stage4ValidationError("nvidia-smi 未能完整提供 GPU/driver/compute/P-state")
    return {
        "gpu_index": gpu_index,
        "gpu_name": values[1],
        "gpu_uuid": values[2],
        "driver_version": values[3],
        "compute_capability": values[4],
        "pstate": values[5],
        "graphics_clock_mhz": _parse_finite_number(values[6], "graphics_clock_mhz"),
        "sm_clock_mhz": _parse_finite_number(values[7], "sm_clock_mhz"),
        "memory_clock_mhz": _parse_finite_number(values[8], "memory_clock_mhz"),
        "power_draw_w": _parse_finite_number(values[9], "power_draw_w"),
        "power_limit_w": _parse_finite_number(values[10], "power_limit_w"),
    }


def _parse_cublas_sgemm_count(stats_csv: str, name_pattern: str) -> int:
    pattern = re.compile(name_pattern)
    header: Dict[str, int] = {}
    total = 0
    matched_rows = 0
    for row in csv.reader(io.StringIO(stats_csv)):
        normalized = [cell.strip().lower() for cell in row]
        if "instances" in normalized and any(
            name in normalized for name in ("name", "range", "operation")
        ):
            name_header = next(
                name for name in ("name", "range", "operation") if name in normalized
            )
            header = {
                "instances": normalized.index("instances"),
                "name": normalized.index(name_header),
            }
            continue
        if not header or len(row) <= max(header.values()):
            continue
        call_name = row[header["name"]].strip()
        if pattern.fullmatch(call_name) is None:
            continue
        raw_instances = row[header["instances"]].strip().replace(",", "")
        try:
            instances = int(raw_instances)
        except ValueError as exc:
            raise Stage4ValidationError(
                f"nsys stats 的 cublasSgemm Instances 无效: {raw_instances!r}"
            ) from exc
        if instances < 1:
            raise Stage4ValidationError(
                f"nsys stats 的 cublasSgemm Instances 必须为正数: {instances}"
            )
        total += instances
        matched_rows += 1
    if not header:
        raise Stage4ValidationError(
            "nsys stats 输出格式未知：未找到含 Instances 与 Name/Range/Operation 的 CSV 表头"
        )
    if matched_rows == 0:
        raise Stage4ValidationError(
            "nsys stats 未发现 cublasSgemm/cublasSgemm_v2 调用"
        )
    return total


def _parse_nsys_sqlite_gemm_count(
    database: pathlib.Path, include_pattern: str, exclude_pattern: str
) -> int:
    include = re.compile(include_pattern)
    exclude = re.compile(exclude_pattern)
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise Stage4ValidationError(
            f"无法只读打开 Nsight SQLite: {database}: {exc}"
        ) from exc
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        if not required.issubset(tables):
            raise Stage4ValidationError(
                "Nsight SQLite 缺少 CUDA kernel/StringIds 表"
            )
        rows = connection.execute(
            """
            SELECT strings.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
            JOIN StringIds AS strings
              ON strings.id = kernels.demangledName
            """
        )
        count = 0
        for (name,) in rows:
            if (
                isinstance(name, str)
                and include.search(name) is not None
                and exclude.search(name) is None
            ):
                count += 1
    except sqlite3.Error as exc:
        raise Stage4ValidationError(
            f"解析 Nsight SQLite 失败: {exc}"
        ) from exc
    finally:
        connection.close()
    if count == 0:
        raise Stage4ValidationError(
            "Nsight CUDA kernel trace 未发现 SGEMM/GEMM 主 kernel"
        )
    return count


def _validate_benchmark_result(
    path: pathlib.Path,
    schema: Mapping[str, Any],
    benchmark_id: str,
    device: int,
    description: str,
) -> Dict[str, Any]:
    result = _load_json(path)
    _validate_document(result, schema, description)
    if result["benchmark_id"] != benchmark_id:
        raise Stage4ValidationError(
            f"{description} benchmark_id 不匹配: {result['benchmark_id']!r}"
        )
    if result["device"] != device:
        raise Stage4ValidationError(
            f"{description} device 不匹配: {result['device']!r}"
        )
    return result


def _require_iteration_contract(
    result: Mapping[str, Any], warmup: int, measured: int, description: str
) -> None:
    for case in result["cases"]:
        if (
            case["warmup_iterations"] != warmup
            or case["measured_iterations"] != measured
        ):
            raise Stage4ValidationError(
                f"{description} 的 {case['case_id']} 必须报告 "
                f"warmup_iterations={warmup}、measured_iterations={measured}"
            )
        shape = case["shape"]
        expected = 1 + shape[0]
        calls = case["gemm_calls"]
        if (
            calls["input_per_forward"] != 1
            or calls["recurrent_per_forward"] != shape[0]
            or calls["total_per_forward"] != expected
        ):
            raise Stage4ValidationError(
                f"{description} 的 {case['case_id']} GEMM 次数不符合 1+T"
            )


def _artifact_root(
    selection: str, build_dir: pathlib.Path, config: Mapping[str, Any]
) -> pathlib.Path:
    if selection == "results":
        root = (ROOT / config["artifacts"]["default_root"]).resolve()
        allowed = (ROOT / "tests/benchmarks/results").resolve()
    elif selection == "build":
        root = (build_dir / "stage4-validation-results").resolve()
        allowed = build_dir
    else:
        raise Stage4ValidationError(f"未知 artifacts-root: {selection}")
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise Stage4ValidationError(
            "产物目录必须位于 tests/benchmarks/results 或 build-dir"
        ) from exc
    return root


def _make_artifact_dir(root: pathlib.Path) -> pathlib.Path:
    run_name = (
        datetime.datetime.now(datetime.timezone.utc).strftime(
            "stage4_cuda_validation_%Y%m%dT%H%M%SZ"
        )
        + f"_{os.getpid()}"
    )
    artifact_dir = root / run_name
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise Stage4ValidationError(f"产物目录已存在: {artifact_dir}") from exc
    except PermissionError as exc:
        raise Stage4ValidationError(f"无权创建产物目录: {artifact_dir}") from exc
    except OSError as exc:
        raise Stage4ValidationError(f"创建产物目录失败 {artifact_dir}: {exc}") from exc
    return artifact_dir.resolve()


def run_validation(args: argparse.Namespace) -> pathlib.Path:
    config_path = args.config.resolve()
    config = _load_json(config_path)
    config_schema = _load_json(CONFIG_SCHEMA)
    benchmark_schema = _load_json(BENCHMARK_SCHEMA)
    report_schema = _load_json(REPORT_SCHEMA)
    _validate_document(config, config_schema, "Stage 4 benchmark 配置")

    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        raise Stage4ValidationError(f"build-dir 不存在或不是目录: {build_dir}")
    if args.device < 0:
        raise Stage4ValidationError("device 必须是非负整数")

    artifact_dir = _make_artifact_dir(
        _artifact_root(args.artifacts_root, build_dir, config)
    )
    failure_path = artifact_dir / "failure.json"
    started_at = _utc_now()
    records: List[Dict[str, Any]] = []

    try:
        tools = {
            name: _require_tool(configured)
            for name, configured in config["tools"].items()
        }
        executable = _resolve_benchmark(
            build_dir, config["benchmark"]["executable"]
        )
        timeouts = config["timeouts_seconds"]

        gpu_argv = [
            tools["nvidia_smi"],
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
            "--id=" + str(args.device),
        ]
        gpu_stdout, _ = _run_command(
            "gpu_environment",
            gpu_argv,
            timeouts["environment"],
            artifact_dir,
            records,
            cwd=build_dir,
        )
        environment = _parse_gpu_environment(gpu_stdout, args.device)

        benchmark_report = artifact_dir / "benchmark_report.json"
        benchmark_argv = _expand_argv(
            executable,
            config["benchmark"]["argv"],
            {
                "device": str(args.device),
                "benchmark_report": str(benchmark_report),
            },
        )
        _run_command(
            "benchmark",
            benchmark_argv,
            timeouts["benchmark"],
            artifact_dir,
            records,
            cwd=build_dir,
        )
        benchmark = _validate_benchmark_result(
            benchmark_report,
            benchmark_schema,
            config["benchmark"]["benchmark_id"],
            args.device,
            "benchmark 报告",
        )

        profile_template = config["benchmark"]["profile_argv"]
        for step, sanitizer_tool in (
            ("memcheck", "memcheck"),
            ("racecheck", "racecheck"),
        ):
            diagnostic_report = artifact_dir / f"{step}_benchmark_report.json"
            diagnostic_argv = _expand_argv(
                executable,
                profile_template,
                {
                    "device": str(args.device),
                    "profile_benchmark_report": str(diagnostic_report),
                },
            )
            sanitizer_argv = [
                tools["compute_sanitizer"],
                "--tool",
                sanitizer_tool,
                "--error-exitcode",
                "1",
                *diagnostic_argv,
            ]
            _run_command(
                step,
                sanitizer_argv,
                timeouts[step],
                artifact_dir,
                records,
                cwd=build_dir,
            )
            diagnostic_result = _validate_benchmark_result(
                diagnostic_report,
                benchmark_schema,
                config["benchmark"]["benchmark_id"],
                args.device,
                f"{step} benchmark 报告",
            )
            _require_iteration_contract(
                diagnostic_result, 0, 1, f"{step} benchmark"
            )

        profile_benchmark_report = artifact_dir / "profile_benchmark_report.json"
        profile_benchmark_argv = _expand_argv(
            executable,
            profile_template,
            {
                "device": str(args.device),
                "profile_benchmark_report": str(profile_benchmark_report),
            },
        )
        nsys_prefix = artifact_dir / "stage4_cuda_profile"
        nsys_report = pathlib.Path(str(nsys_prefix) + ".nsys-rep")
        profile_argv = [
            tools["nsys"],
            "profile",
            "--force-overwrite=true",
            "--trace=" + ",".join(config["nsys"]["trace"]),
            "--output=" + str(nsys_prefix),
            *profile_benchmark_argv,
        ]
        _run_command(
            "nsys_profile",
            profile_argv,
            timeouts["nsys_profile"],
            artifact_dir,
            records,
            cwd=build_dir,
        )
        if not nsys_report.is_file():
            raise Stage4ValidationError(
                f"nsys profile 成功但未生成预期报告: {nsys_report}"
            )
        profile_benchmark = _validate_benchmark_result(
            profile_benchmark_report,
            benchmark_schema,
            config["benchmark"]["benchmark_id"],
            args.device,
            "profile benchmark 报告",
        )
        _require_iteration_contract(
            profile_benchmark, 0, 1, "profile benchmark"
        )
        _require_iteration_contract(benchmark, 10, 100, "benchmark")
        benchmark_profiles = {
            (case["shape_profile"], case["math_mode"], tuple(case["shape"]))
            for case in benchmark["cases"]
        }
        profile_profiles = {
            (case["shape_profile"], case["math_mode"], tuple(case["shape"]))
            for case in profile_benchmark["cases"]
        }
        if benchmark_profiles != profile_profiles:
            raise Stage4ValidationError(
                "benchmark 与 profile benchmark 的 profile 集合不一致"
            )

        nsys_sqlite = artifact_dir / "stage4_cuda_profile.sqlite"
        export_argv = [
            tools["nsys"],
            "export",
            "--force-overwrite=true",
            "--type",
            "sqlite",
            "--output",
            str(nsys_sqlite),
            str(nsys_report),
        ]
        _run_command(
            "nsys_export",
            export_argv,
            timeouts["nsys_stats"],
            artifact_dir,
            records,
            cwd=build_dir,
        )
        if not nsys_sqlite.is_file():
            raise Stage4ValidationError(
                f"nsys export 未生成 SQLite: {nsys_sqlite}"
            )

        stats_argv = [
            tools["nsys"],
            "stats",
            "--force-export=true",
            "--report",
            ",".join(config["nsys"]["stats_reports"]),
            "--format",
            "csv",
            str(nsys_report),
        ]
        stats_stdout, _ = _run_command(
            "nsys_stats",
            stats_argv,
            timeouts["nsys_stats"],
            artifact_dir,
            records,
            cwd=build_dir,
        )
        trace_count = _parse_nsys_sqlite_gemm_count(
            nsys_sqlite,
            config["nsys"]["gemm_kernel_name_pattern"],
            config["nsys"]["gemm_kernel_exclude_pattern"],
        )

        expected_count = sum(
            case["gemm_calls"]["total_per_forward"]
            for case in profile_benchmark["cases"]
        )
        benchmark_count = sum(
            case["gemm_calls"]["total_per_forward"]
            for case in benchmark["cases"]
        )
        profile_count = profile_benchmark[
            "total_profiled_cublas_sgemm_calls"
        ]
        if profile_count != expected_count:
            raise Stage4ValidationError(
                "profile benchmark 报告的 cublasSgemm 次数不符合 1+T："
                f"报告 {profile_count}，期望 {expected_count}"
            )
        if trace_count != expected_count:
            raise Stage4ValidationError(
                "nsys trace 的 cublasSgemm 次数不符合 1+T："
                f"trace {trace_count}，期望 {expected_count}"
            )

        nsys_stats_path = pathlib.Path(records[-1]["stdout_artifact"])
        memcheck_log = pathlib.Path(records[2]["stderr_artifact"])
        racecheck_log = pathlib.Path(records[3]["stderr_artifact"])
        validation_report = artifact_dir / "stage4_cuda_validation_report.json"
        report = {
            "schema_version": 1,
            "config_id": config["config_id"],
            "stage": 4,
            "status": "passed",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "build_dir": str(build_dir),
            "artifact_dir": str(artifact_dir),
            "device": args.device,
            "environment": environment,
            "benchmark": benchmark,
            "profile_benchmark": profile_benchmark,
            "validation": {
                "expected_cublas_sgemm_calls": expected_count,
                "benchmark_cublas_sgemm_calls": benchmark_count,
                "trace_cublas_sgemm_calls": trace_count,
                "formula": "sum_cases(1+T)",
                "benchmark_matches_expected": True,
                "trace_matches_expected": True,
                "benchmark_trace_cross_check": benchmark_count == trace_count,
            },
            "commands": records,
            "artifacts": {
                "benchmark_report": str(benchmark_report),
                "profile_benchmark_report": str(profile_benchmark_report),
                "memcheck_log": str(memcheck_log),
                "racecheck_log": str(racecheck_log),
                "nsys_report": str(nsys_report),
                "nsys_sqlite": str(nsys_sqlite),
                "nsys_stats": str(nsys_stats_path),
                "validation_report": str(validation_report),
            },
        }
        registry = Registry().with_resource(
            benchmark_schema["$id"], Resource.from_contents(benchmark_schema)
        )
        _validate_document(
            report,
            report_schema,
            "Stage 4 validation 报告",
            registry=registry,
            check_format=True,
        )
        _write_json(validation_report, report)
        return validation_report
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "stage": 4,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "commands": records,
        }
        try:
            _write_json(failure_path, failure)
        except Stage4ValidationError:
            pass
        if isinstance(exc, Stage4ValidationError):
            raise
        raise Stage4ValidationError(
            f"未预期的 {type(exc).__name__}: {exc}；失败详情: {failure_path}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 Stage 4 CUDA benchmark、sanitizer 与 nsys 验证。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
benchmark CLI 契约
------------------
脚本只约定 config 中声明的 argv，不推断 benchmark 的其他参数或行为：

1. executable 是相对 --build-dir 的可执行文件路径。
2. argv/profile_argv 是直接传给 subprocess 的参数数组；不会经过 shell。
3. 支持占位符 {device}、{benchmark_report}、{profile_benchmark_report}。
4. benchmark 必须将 JSON 写到对应 report 占位符路径，成功返回 0，失败返回非 0。
5. JSON 必须符合 tests/benchmarks/schema/cuda_benchmark_result.schema.json。
6. profile_argv 必须让每个 case 产生 warmup_iterations=0、
   measured_iterations=1 的报告，以便 nsys CUDA kernel trace 的 GEMM 主
   kernel 数可与 sum_cases(1+T) 精确交叉检查。

产物只会写入 tests/benchmarks/results（默认）或 --build-dir 内。
必需工具、GPU 查询字段、执行权限或输出格式未知时，脚本明确失败且返回非 0。
""",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help=f"benchmark 配置（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        required=True,
        help="包含 benchmark 可执行文件的构建目录",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="CUDA/nvidia-smi 设备索引（非负整数）",
    )
    parser.add_argument(
        "--artifacts-root",
        choices=("results", "build"),
        default="results",
        help="产物根目录：仓库 ignored results 或 build-dir（默认: results）",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = run_validation(args)
    except Stage4ValidationError as exc:
        print(f"Stage 4 CUDA validation 失败: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stage 4 CUDA validation 被中断", file=sys.stderr)
        return 130
    print(f"Stage 4 CUDA validation 通过: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
