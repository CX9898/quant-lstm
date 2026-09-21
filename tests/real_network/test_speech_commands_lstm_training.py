"""End-to-end training comparison on Speech Commands v0.02."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from speech_commands_lstm_training import (
    ExperimentConfig,
    _quant_param_range_summary,
    _tensor_error_metrics,
    run_training_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/results/speech_commands_lstm_training.json"
QUANT_OPERATORS = {
    "input",
    "output",
    "cell_state",
    "weight_ih",
    "weight_hh",
    "bias_ih",
    "bias_hh",
    "weight_ih_linear",
    "weight_hh_linear",
    "input_gate_input",
    "forget_gate_input",
    "cell_gate_input",
    "output_gate_input",
    "input_gate_output",
    "forget_gate_output",
    "cell_gate_output",
    "output_gate_output",
    "cell_tanh_output",
}


class SpeechCommandsDiagnosticsTest(unittest.TestCase):
    def test_tensor_error_metrics_report_tail_and_normalized_errors(self) -> None:
        reference = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
        actual = torch.tensor([[1.5, -1.5], [1.0, -1.0]])

        result = _tensor_error_metrics(actual, reference)

        self.assertAlmostEqual(result["mae"], 0.75)
        self.assertAlmostEqual(result["mse"], 0.625)
        self.assertAlmostEqual(result["rmse"], 0.625**0.5)
        self.assertAlmostEqual(result["normalized_mae"], 0.5)
        self.assertAlmostEqual(result["normalized_rmse"], (0.625 / 2.5) ** 0.5)
        self.assertAlmostEqual(result["p99_absolute_error"], 1.0)
        self.assertAlmostEqual(result["max_absolute_error"], 1.0)

    def test_quant_param_range_summary_reports_resolution(self) -> None:
        summary = _quant_param_range_summary(
            {
                "operators": {
                    "input": {
                        "bitwidth": 8,
                        "is_unsigned": False,
                        "is_symmetric": True,
                        "granularity": "per_tensor",
                        "scales": ["0.1"],
                        "zero_points": [0],
                    }
                }
            }
        )["input"]

        self.assertEqual(summary["quantized_levels"], 254)
        self.assertAlmostEqual(summary["quantization_step_min"], 0.1)
        self.assertAlmostEqual(summary["quantization_step_max"], 0.1)
        self.assertAlmostEqual(summary["representable_min"], -12.7)
        self.assertAlmostEqual(summary["representable_max"], 12.7)
        self.assertAlmostEqual(summary["representable_span_min"], 25.4)
        self.assertAlmostEqual(summary["representable_span_max"], 25.4)


class SpeechCommandsLstmTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_root = os.environ.get("QUANT_LSTM_SPEECH_COMMANDS_ROOT")
        if not dataset_root:
            raise unittest.SkipTest(
                "set QUANT_LSTM_SPEECH_COMMANDS_ROOT to run the real-data test"
            )
        cls.dataset_root = Path(dataset_root)
        if not cls.dataset_root.is_dir():
            raise RuntimeError(f"Speech Commands dataset not found: {cls.dataset_root}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("the QuantLSTM training comparison requires CUDA")

    def test_replacing_only_lstm_preserves_real_training_quality(self) -> None:
        report = run_training_comparison(
            ExperimentConfig(
                dataset_root=self.dataset_root,
                labels=("yes", "no", "up", "down"),
                train_samples_per_label=128,
                validation_samples_per_label=32,
                test_samples_per_label=32,
                hidden_size=64,
                batch_size=32,
                epochs=10,
                learning_rate=3.0e-3,
                calibration_batches=4,
                calibration_refresh_epochs=1,
                quant_bitwidths=(8, 16),
                seed=20260921,
                quality_gate_seeds=(20260921, 20260922, 20260923),
            )
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        baseline = report["training"]["torch_lstm"]
        native_float = report["training"]["quant_lstm_float"]
        quantized_variants = {
            bitwidth: report["training"][f"quant_lstm_qat_{bitwidth}bit"]
            for bitwidth in (8, 16)
        }
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(report["replacement"]["changed_module"], "lstm")
        self.assertEqual(report["quantization"]["bitwidths"], [8, 16])
        backward_oracle = report["quantization"]["real_batch_backward_oracle"]
        self.assertEqual(set(backward_oracle), {"8", "16"})
        for bitwidth, result in backward_oracle.items():
            with self.subTest(backward_oracle_bitwidth=bitwidth):
                self.assertEqual(result["sample_count"], 32)
                self.assertEqual(result["label_counts"], [8, 8, 8, 8])
                self.assertEqual(
                    set(result["gradients"]),
                    {
                        "input",
                        "weight_ih",
                        "weight_hh",
                        "bias_ih",
                        "bias_hh",
                        "h_0",
                        "c_0",
                    },
                )
                self.assertTrue(
                    all(
                        metrics["max_absolute_error"] <= 5.0e-6
                        and metrics["cosine"] >= 0.99999
                        for metrics in result["gradients"].values()
                    )
                )
        self.assertEqual(
            report["quantization"]["calibration_strategy"],
            {
                "selection": "balanced_round_robin",
                "refresh_interval_epochs": 1,
                "refresh_timing": "after_training_before_validation",
            },
        )
        self.assertEqual(
            report["replacement"]["initial_shared_state_max_abs_diff"],
            {
                "quant_lstm_float": 0.0,
                "quant_lstm_qat_8bit": 0.0,
                "quant_lstm_qat_16bit": 0.0,
            },
        )
        matrix = report["quantization"]["calibration_strategy_matrix"]
        self.assertEqual(matrix["source_model"], "quant_lstm_qat_8bit")
        self.assertTrue(matrix["fixed_weights"])
        self.assertEqual(set(matrix["methods"]), {"minmax", "percentile", "sqnr"})
        self.assertEqual(matrix["sample_counts"], [128, 512])
        self.assertEqual(matrix["bitwidths"], [8, 16])
        self.assertEqual(len(matrix["entries"]), 12)
        self.assertEqual(
            {
                (entry["method"], entry["sample_count"], entry["bitwidth"])
                for entry in matrix["entries"]
            },
            {
                (method, sample_count, bitwidth)
                for method in matrix["methods"]
                for sample_count in matrix["sample_counts"]
                for bitwidth in matrix["bitwidths"]
            },
        )
        for entry in matrix["entries"]:
            self.assertEqual(set(entry["operators"]), QUANT_OPERATORS)
            self.assertEqual(entry["unsafe_non_finite_count"], 0)
            self.assertIn("checkpoint_clamp_masks.cell_states", entry["clamp_rates"])
            self.assertGreaterEqual(entry["error"]["prediction_agreement"], 0.0)
            self.assertLessEqual(entry["error"]["prediction_agreement"], 1.0)
        matrix_by_case = {
            (entry["method"], entry["sample_count"], entry["bitwidth"]): entry
            for entry in matrix["entries"]
        }
        minmax_128 = matrix_by_case[("minmax", 128, 8)]
        minmax_512 = matrix_by_case[("minmax", 512, 8)]
        percentile_512 = matrix_by_case[("percentile", 512, 8)]
        self.assertGreater(
            minmax_512["operators"]["cell_state"]["quantization_step_max"],
            minmax_128["operators"]["cell_state"]["quantization_step_max"],
        )
        self.assertGreater(
            minmax_512["error"]["mae"], minmax_128["error"]["mae"]
        )
        self.assertLess(
            percentile_512["error"]["mae"],
            minmax_512["error"]["mae"] * 0.75,
        )
        self.assertTrue(
            all(
                entry["error"]["mae"] < 1.0e-4
                for entry in matrix["entries"]
                if entry["bitwidth"] == 16
            )
        )
        self.assertLess(baseline["final_train_loss"], baseline["initial_train_loss"])
        self.assertGreater(baseline["parameter_update_norm"], 0.0)
        self.assertGreaterEqual(baseline["best_validation_accuracy"], 0.55)
        self.assertGreaterEqual(baseline["final_test_accuracy"], 0.60)
        self.assertLess(
            native_float["final_train_loss"], native_float["initial_train_loss"]
        )
        self.assertGreater(native_float["parameter_update_norm"], 0.0)
        self.assertFalse(native_float["native_qat_checkpoint_observed"])
        self.assertGreaterEqual(native_float["best_validation_accuracy"], 0.55)
        self.assertGreaterEqual(native_float["final_test_accuracy"], 0.60)
        self.assertGreaterEqual(
            native_float["best_validation_accuracy"],
            baseline["best_validation_accuracy"] - 0.05,
        )
        quality_thresholds = {
            8: {
                "validation": 0.50,
                "testing": 0.50,
                "maximum_baseline_gap": 0.10,
            },
            16: {
                "validation": 0.60,
                "testing": 0.60,
                "maximum_baseline_gap": 0.05,
            },
        }
        for bitwidth, quantized in quantized_variants.items():
            with self.subTest(bitwidth=bitwidth):
                calibration = report["quantization"]["calibration"][
                    f"quant_lstm_qat_{bitwidth}bit"
                ]
                safety = calibration["safety"]
                self.assertEqual(safety["unsafe_non_finite_count"], 0)
                self.assertEqual(calibration["selection"], "balanced_round_robin")
                self.assertEqual(calibration["sample_count"], 128)
                self.assertEqual(calibration["label_counts"], [32, 32, 32, 32])
                self.assertLess(
                    quantized["final_train_loss"], quantized["initial_train_loss"]
                )
                self.assertGreater(quantized["parameter_update_norm"], 0.0)
                self.assertTrue(quantized["native_qat_checkpoint_observed"])
                self.assertGreaterEqual(
                    quantized["best_validation_accuracy"],
                    quality_thresholds[bitwidth]["validation"],
                )
                self.assertGreaterEqual(
                    quantized["final_test_accuracy"],
                    quality_thresholds[bitwidth]["testing"],
                )
                self.assertGreaterEqual(
                    quantized["best_validation_accuracy"],
                    baseline["best_validation_accuracy"]
                    - quality_thresholds[bitwidth]["maximum_baseline_gap"],
                )
                self.assertGreaterEqual(
                    quantized["final_test_accuracy"],
                    baseline["final_test_accuracy"]
                    - quality_thresholds[bitwidth]["maximum_baseline_gap"],
                )
                self.assertEqual(len(quantized["calibration_refreshes"]), 10)
                self.assertTrue(
                    all(
                        refresh["label_counts"] == [32, 32, 32, 32]
                        for refresh in quantized["calibration_refreshes"]
                    )
                )
                for epoch in quantized["epochs"]:
                    rates = epoch["qat_clamp_rates"]
                    self.assertLess(
                        rates["master_clamp_masks.weight_ih"], 0.10
                    )
                    self.assertLess(
                        rates["master_clamp_masks.weight_hh"], 0.10
                    )
                    bias_rates = epoch["qat_bias_clamp_rates"]
                    self.assertEqual(set(bias_rates), {"pre_step", "post_step"})
                    for timing in bias_rates.values():
                        self.assertEqual(set(timing), {"bias_ih", "bias_hh"})
                        self.assertTrue(
                            all(0.0 <= value <= 1.0 for value in timing.values())
                        )
                    ranges = epoch["qat_bias_ranges"]
                    self.assertEqual(
                        set(ranges), {"before_refresh", "after_refresh"}
                    )
                    for timing in ranges.values():
                        for name in ("bias_ih", "bias_hh"):
                            diagnostic = timing[name]
                            self.assertEqual(
                                len(diagnostic["channels"]), 4 * 64
                            )
                            self.assertLessEqual(
                                diagnostic["representable_min"],
                                diagnostic["representable_max"],
                            )
                            self.assertGreater(
                                diagnostic["quantization_step_min"], 0.0
                            )
                    for diagnostic in ranges["after_refresh"].values():
                        self.assertEqual(diagnostic["clamp_rate"], 0.0)

        quantized_8 = quantized_variants[8]
        quantized_16 = quantized_variants[16]
        trace = quantized_8["final_quantization_error"]["time_step_trace"]
        self.assertEqual(len(trace["steps"]), 49)
        self.assertEqual(trace["steps"][0]["time_step"], 0)
        self.assertEqual(trace["steps"][-1]["time_step"], 48)
        self.assertEqual(trace["tail"]["start_time_step"], 36)
        self.assertIn(
            trace["peak_mae_time_step"], range(len(trace["steps"]))
        )
        self.assertEqual(
            quantized_8["final_quantization_error"]["sample_count"], 128
        )
        self.assertGreaterEqual(
            quantized_8["final_quantization_error"]["prediction_agreement"],
            0.90,
        )
        ablation = quantized_8["operator_bitwidth_ablation"]
        self.assertEqual(set(ablation["operators"]), QUANT_OPERATORS)
        self.assertEqual(
            set(ablation["ranked_by_mae_improvement"]), QUANT_OPERATORS
        )
        self.assertEqual(ablation["base_bitwidth"], 8)
        self.assertEqual(ablation["promoted_bitwidth"], 16)
        self.assertTrue(
            all(
                result["calibration_unsafe_non_finite_count"] == 0
                for result in ablation["operators"].values()
            )
        )
        self.assertLess(
            ablation["all_promoted"]["mae"], ablation["baseline"]["mae"]
        )
        self.assertGreaterEqual(
            quantized_16["best_validation_accuracy"],
            quantized_8["best_validation_accuracy"] + 0.05,
        )
        self.assertGreaterEqual(
            quantized_16["final_test_accuracy"],
            quantized_8["final_test_accuracy"] + 0.05,
        )
        self.assertLess(
            quantized_16["final_quantization_error"]["mae"],
            quantized_8["final_quantization_error"]["mae"] * 0.10,
        )
        self.assertGreaterEqual(
            quantized_16["final_quantization_error"]["cosine"], 0.9999
        )

        multi_seed = report["multi_seed_quality"]
        self.assertEqual(
            multi_seed["seeds"], [20260921, 20260922, 20260923]
        )
        self.assertEqual(
            set(multi_seed["runs"]),
            {"20260921", "20260922", "20260923"},
        )
        for seed, run in multi_seed["runs"].items():
            with self.subTest(quality_seed=seed):
                for name in (
                    "torch_lstm",
                    "quant_lstm_qat_8bit",
                    "quant_lstm_qat_16bit",
                ):
                    self.assertLess(
                        run[name]["final_train_loss"],
                        run[name]["initial_train_loss"],
                    )
                self.assertGreaterEqual(
                    run["quant_lstm_qat_8bit"]["prediction_agreement"], 0.95
                )
                self.assertLess(
                    run["quant_lstm_qat_16bit"]["logit_mae"],
                    run["quant_lstm_qat_8bit"]["logit_mae"] * 0.10,
                )
        aggregate = multi_seed["aggregate"]
        self.assertGreaterEqual(
            aggregate["torch_lstm"]["minimum_test_accuracy"], 0.58
        )
        self.assertGreaterEqual(
            aggregate["quant_lstm_qat_8bit"]["minimum_test_accuracy"], 0.47
        )
        self.assertGreaterEqual(
            aggregate["quant_lstm_qat_16bit"]["minimum_test_accuracy"], 0.50
        )
        self.assertGreaterEqual(
            aggregate["quant_lstm_qat_16bit"]["mean_test_accuracy"],
            aggregate["quant_lstm_qat_8bit"]["mean_test_accuracy"] + 0.02,
        )


if __name__ == "__main__":
    unittest.main()
