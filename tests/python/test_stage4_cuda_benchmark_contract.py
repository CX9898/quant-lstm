"""Stage 4 CUDA benchmark 配置、报告与编排器完整性测试。"""

import argparse
import ast
import copy
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

import jsonschema
from referencing import Registry, Resource

from tools import run_stage4_cuda_validation as runner
from tools.strict_jsonschema import StrictDraft202012Validator


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests/benchmarks/config/cuda_benchmark_v1.json"
SCHEMA_DIR = ROOT / "tests/benchmarks/schema"
CONFIG_SCHEMA_PATH = SCHEMA_DIR / "cuda_benchmark_config.schema.json"
BENCHMARK_SCHEMA_PATH = SCHEMA_DIR / "cuda_benchmark_result.schema.json"
REPORT_SCHEMA_PATH = SCHEMA_DIR / "stage4_cuda_validation_report.schema.json"
RUNNER_PATH = ROOT / "tools/run_stage4_cuda_validation.py"


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_result(*, warmup: int = 10, measured: int = 100) -> dict:
    def timing() -> dict:
        return {
            "minimum": 0.1,
            "maximum": 0.2,
            "mean": 0.15,
            "p50": 0.15,
            "p95": 0.19,
        }

    def metric() -> dict:
        return {
            "max_absolute_error": 0.01,
            "mae": 0.005,
            "mse": 0.0001,
            "cosine_similarity": 0.999,
            "sqnr_db": 40.0,
            "saturation_rate": 0.0,
        }

    workspace = {
        "quantized_input_bytes": 1,
        "quantized_weight_ih_bytes": 1,
        "quantized_weight_hh_bytes": 1,
        "quantized_bias_bytes": 1,
        "quantized_state_bytes": 1,
        "input_linear_bytes": 1,
        "recurrent_linear_bytes": 1,
        "weight_sum_bytes": 1,
        "device_parameter_bytes": 1,
        "alignment_padding_bytes": 0,
        "total_bytes": 10,
    }
    cases = []
    for shape_profile, shape in (
        ("cuda_gru_basic", [50, 64, 128, 256]),
        ("large_batch", [16, 128, 64, 128]),
    ):
        for math_mode in ("pedantic", "tf32"):
            steps = shape[0]
            cases.append(
                {
                    "case_id": f"stage4_{shape_profile}_{math_mode}",
                    "shape_profile": shape_profile,
                    "shape": shape,
                    "math_mode": math_mode,
                    "fp32_accumulation_class": "precision_risk",
                    "warmup_iterations": warmup,
                    "measured_iterations": measured,
                    "timing_scope": (
                        "cuda_event_quantize_core_dequantize_"
                        "cached_execution_params"
                    ),
                    "timing_ms": {
                        "end_to_end": timing(),
                        "quantization_overhead": timing(),
                        "quantized_core": timing(),
                    },
                    "throughput": {
                        "sequences_per_second": 1.0,
                        "time_steps_per_second": 2.0,
                        "output_elements_per_second": 3.0,
                    },
                    "workspace": workspace,
                    "gemm_calls": {
                        "input_per_forward": 1,
                        "recurrent_per_forward": steps,
                        "total_per_forward": 1 + steps,
                    },
                    "precision": {
                        "output": metric(),
                        "h_n": metric(),
                        "c_n": metric(),
                    },
                }
            )
    return {
        "schema_version": 2,
        "benchmark_id": "quantized_fp_cuda_lstm_v1",
        "device": 0,
        "matrix_version": "strict_matrix_v2",
        "environment": {
            "gpu_name": "Example GPU",
            "compute_capability": "9.0",
            "cuda_driver_version": 13000,
            "cuda_runtime_version": 13000,
            "cublas_version": 13000,
        },
        "total_profiled_cublas_sgemm_calls": (
            sum(case["gemm_calls"]["total_per_forward"] for case in cases)
            * (warmup + measured)
        ),
        "cases": cases,
    }


def command(step: str, ordinal: int) -> dict:
    return {
        "step": step,
        "argv": ["/usr/bin/tool", "--flag"],
        "returncode": 0,
        "duration_seconds": 0.1,
        "stdout_artifact": f"/tmp/stage4/{ordinal:02d}.stdout.txt",
        "stderr_artifact": f"/tmp/stage4/{ordinal:02d}.stderr.txt",
    }


def validation_report() -> dict:
    artifact_dir = "/tmp/stage4"
    return {
        "schema_version": 1,
        "config_id": "cuda_benchmark_v1",
        "stage": 4,
        "status": "passed",
        "started_at_utc": "2026-09-17T01:00:00+00:00",
        "finished_at_utc": "2026-09-17T01:01:00+00:00",
        "build_dir": "/tmp/build",
        "artifact_dir": artifact_dir,
        "device": 0,
        "environment": {
            "gpu_index": 0,
            "gpu_name": "Example GPU",
            "gpu_uuid": "GPU-0000",
            "driver_version": "999.0",
            "compute_capability": "9.0",
            "pstate": "P0",
            "graphics_clock_mhz": 1500.0,
            "sm_clock_mhz": 1500.0,
            "memory_clock_mhz": 9000.0,
            "power_draw_w": 100.0,
            "power_limit_w": 300.0,
        },
        "benchmark": benchmark_result(),
        "profile_benchmark": benchmark_result(warmup=0, measured=1),
        "validation": {
            "expected_cublas_sgemm_calls": 136,
            "benchmark_cublas_sgemm_calls": 136,
            "trace_cublas_sgemm_calls": 136,
            "formula": "sum_cases(1+T)",
            "benchmark_matches_expected": True,
            "trace_matches_expected": True,
            "benchmark_trace_cross_check": True,
        },
        "commands": [
            command("gpu_environment", 1),
            command("benchmark", 2),
            command("memcheck", 3),
            command("racecheck", 4),
            command("nsys_profile", 5),
            command("nsys_export", 6),
            command("nsys_stats", 7),
        ],
        "artifacts": {
            "benchmark_report": f"{artifact_dir}/benchmark_report.json",
            "profile_benchmark_report": (
                f"{artifact_dir}/profile_benchmark_report.json"
            ),
            "memcheck_log": f"{artifact_dir}/03_memcheck.stderr.txt",
            "racecheck_log": f"{artifact_dir}/04_racecheck.stderr.txt",
            "nsys_report": f"{artifact_dir}/stage4_cuda_profile.nsys-rep",
            "nsys_sqlite": f"{artifact_dir}/stage4_cuda_profile.sqlite",
            "nsys_stats": f"{artifact_dir}/07_nsys_stats.stdout.txt",
            "validation_report": (
                f"{artifact_dir}/stage4_cuda_validation_report.json"
            ),
        },
    }


class Stage4CudaBenchmarkContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_json(CONFIG_PATH)
        cls.config_schema = load_json(CONFIG_SCHEMA_PATH)
        cls.benchmark_schema = load_json(BENCHMARK_SCHEMA_PATH)
        cls.report_schema = load_json(REPORT_SCHEMA_PATH)
        cls.registry = Registry().with_resource(
            cls.benchmark_schema["$id"],
            Resource.from_contents(cls.benchmark_schema),
        )

    def test_all_schemas_are_valid_draft_2020_12(self) -> None:
        for name, schema in (
            ("config", self.config_schema),
            ("benchmark", self.benchmark_schema),
            ("report", self.report_schema),
        ):
            with self.subTest(schema=name):
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                StrictDraft202012Validator.check_schema(schema)

    def test_checked_in_config_matches_strict_schema(self) -> None:
        StrictDraft202012Validator(self.config_schema).validate(self.config)

    def test_config_schema_rejects_unknown_and_non_integer_fields(self) -> None:
        validator = StrictDraft202012Validator(self.config_schema)
        unknown = copy.deepcopy(self.config)
        unknown["benchmark"]["unexpected"] = True
        boolean_stage = copy.deepcopy(self.config)
        boolean_stage["stage"] = True
        float_timeout = copy.deepcopy(self.config)
        float_timeout["timeouts_seconds"]["benchmark"] = 600.0
        missing_device = copy.deepcopy(self.config)
        missing_device["benchmark"]["argv"].remove("{device}")
        unknown_placeholder = copy.deepcopy(self.config)
        unknown_placeholder["benchmark"]["profile_argv"].append("{unknown}")
        for invalid in (
            unknown,
            boolean_stage,
            float_timeout,
            missing_device,
            unknown_placeholder,
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(validator.is_valid(invalid))

    def test_argv_contract_is_complete_and_profile_is_single_iteration(self) -> None:
        benchmark = self.config["benchmark"]
        self.assertEqual(benchmark["argv"].count("{benchmark_report}"), 1)
        self.assertEqual(
            benchmark["profile_argv"].count("{profile_benchmark_report}"), 1
        )
        self.assertIn("{device}", benchmark["argv"])
        self.assertIn("{device}", benchmark["profile_argv"])

        profile_pairs = dict(
            zip(benchmark["profile_argv"][::2], benchmark["profile_argv"][1::2])
        )
        self.assertEqual(profile_pairs["--warmup-iterations"], "0")
        self.assertEqual(profile_pairs["--iterations"], "1")

    def test_benchmark_result_schema_is_strict(self) -> None:
        validator = StrictDraft202012Validator(self.benchmark_schema)
        valid = benchmark_result()
        validator.validate(valid)

        extra = copy.deepcopy(valid)
        extra["cases"][0]["timing_ms"]["unexpected"] = 1
        boolean_length = copy.deepcopy(valid)
        boolean_length["cases"][0]["shape"][0] = True
        float_iterations = copy.deepcopy(valid)
        float_iterations["cases"][0]["measured_iterations"] = 100.0
        for invalid in (extra, boolean_length, float_iterations):
            with self.subTest(invalid=invalid):
                self.assertFalse(validator.is_valid(invalid))

    def test_aggregate_report_matches_schema_and_rejects_unknown_fields(self) -> None:
        validator = StrictDraft202012Validator(
            self.report_schema,
            registry=self.registry,
            format_checker=jsonschema.FormatChecker(),
        )
        valid = validation_report()
        validator.validate(valid)

        unknown = copy.deepcopy(valid)
        unknown["environment"]["unexpected"] = "value"
        self.assertFalse(validator.is_valid(unknown))

        failed_check = copy.deepcopy(valid)
        failed_check["validation"]["trace_matches_expected"] = False
        self.assertFalse(validator.is_valid(failed_check))

    def test_config_requires_all_environment_and_validation_tools(self) -> None:
        validator = StrictDraft202012Validator(self.config_schema)
        for field in ("nvidia_smi", "compute_sanitizer", "nsys"):
            with self.subTest(tool=field):
                invalid = copy.deepcopy(self.config)
                del invalid["tools"][field]
                self.assertFalse(validator.is_valid(invalid))

    def test_runner_uses_subprocess_argv_with_shell_false(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "run"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsInstance(call.args[0], ast.Name)
        self.assertEqual(call.args[0].id, "argv_list")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertIn("shell", keywords)
        self.assertIsInstance(keywords["shell"], ast.Constant)
        self.assertIs(keywords["shell"].value, False)

    def test_gpu_environment_parser_requires_all_requested_fields(self) -> None:
        parsed = runner._parse_gpu_environment(
            (
                "0, Example GPU, GPU-0000, 999.0, 9.0, P0, "
                "1500, 1500, 9000, 100.5, 300.0\n"
            ),
            0,
        )
        self.assertEqual(parsed["compute_capability"], "9.0")
        self.assertEqual(parsed["power_draw_w"], 100.5)

        with self.assertRaisesRegex(runner.Stage4ValidationError, "完整提供"):
            runner._parse_gpu_environment(
                "0, Example GPU, GPU-0000, 999.0, N/A, P0, "
                "1500, 1500, 9000, 100.5, 300.0\n",
                0,
            )

    def test_nsys_csv_parser_counts_cublas_sgemm_instances(self) -> None:
        stats = (
            '"Time (%)","Total Time (ns)","Instances","Range"\n'
            '"30.0","1000","1","cublasSgemm"\n'
            '"70.0","2000","8","cublasSgemm_v2"\n'
        )
        self.assertEqual(
            runner._parse_cublas_sgemm_count(
                stats, self.config["nsys"]["cublas_sgemm_name_pattern"]
            ),
            9,
        )
        with self.assertRaisesRegex(
            runner.Stage4ValidationError, "未发现 cublasSgemm"
        ):
            runner._parse_cublas_sgemm_count(
                '"Instances","Range"\n"9","cublasGemmEx"\n',
                self.config["nsys"]["cublas_sgemm_name_pattern"],
            )

    def test_nsys_sqlite_parser_counts_main_gemm_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = pathlib.Path(temporary) / "trace.sqlite"
            connection = runner.sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    demangledName INTEGER
                );
                INSERT INTO StringIds VALUES
                    (1, 'cutlass::sgemm_kernel'),
                    (2, 'cublasLt::splitKreduce_gemm'),
                    (3, 'quant_lstm::pointwise');
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                    (1), (1), (2), (3);
                """
            )
            connection.commit()
            connection.close()
            self.assertEqual(
                runner._parse_nsys_sqlite_gemm_count(
                    database,
                    self.config["nsys"]["gemm_kernel_name_pattern"],
                    self.config["nsys"]["gemm_kernel_exclude_pattern"],
                ),
                2,
            )

    def test_unknown_argv_placeholder_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(runner.Stage4ValidationError, "未知占位符"):
            runner._expand_argv(
                "/tmp/benchmark",
                ["--output", "{unknown}"],
                {"device": "0"},
            )

    def test_artifact_roots_are_restricted_to_ignored_locations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build_dir = pathlib.Path(temporary).resolve()
            self.assertEqual(
                runner._artifact_root("build", build_dir, self.config),
                build_dir / "stage4-validation-results",
            )
        self.assertEqual(
            runner._artifact_root(
                "results", pathlib.Path("/tmp/build"), self.config
            ),
            (ROOT / "tests/benchmarks/results").resolve(),
        )

    def test_help_documents_benchmark_cli_contract(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            runner._build_parser().parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("benchmark CLI 契约", help_text)
        self.assertIn("{profile_benchmark_report}", help_text)
        self.assertIn("subprocess", help_text)
        self.assertIn("1+T", help_text)

    def test_missing_build_directory_fails_before_tool_execution(self) -> None:
        args = argparse.Namespace(
            config=CONFIG_PATH,
            build_dir=pathlib.Path("/definitely/missing/quant-lstm-build"),
            device=0,
            artifacts_root="results",
        )
        with self.assertRaisesRegex(runner.Stage4ValidationError, "build-dir"):
            runner.run_validation(args)


if __name__ == "__main__":
    unittest.main()
