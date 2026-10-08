"""End-to-end training comparison on Speech Commands v0.02."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from pretrained_quality_assertions import QUALITY_POLICY, assert_pretrained_quality

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
        torch.set_num_threads(8)
        cls.dataset_root = Path(dataset_root)
        if not cls.dataset_root.is_dir():
            raise RuntimeError(f"Speech Commands dataset not found: {cls.dataset_root}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("the QuantLSTM training comparison requires CUDA")

    def test_pretrained_qat_preserves_real_training_quality(self) -> None:
        report = run_training_comparison(
            ExperimentConfig(
                dataset_root=self.dataset_root,
                labels=("yes", "no", "up", "down"),
                train_samples_per_label=128,
                validation_samples_per_label=32,
                test_samples_per_label=32,
                hidden_size=64,
                batch_size=32,
                pretrain_epochs=50,
                epochs=20,
                learning_rate=3.0e-4,
                calibration_batches=16,
                quant_bitwidths=(8, 16),
                seed=20260921,
                quality_gate_seeds=(20260921, 20260922, 20260923, 20261008),
            )
        )
        report["quality_policy"] = QUALITY_POLICY["subset"]
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        assert_pretrained_quality(self, report)
        matrix = report["quantization"]["calibration_strategy_matrix"]
        self.assertTrue(matrix["fixed_weights"])
        self.assertEqual(set(matrix["methods"]), {"minmax", "percentile", "sqnr"})
        self.assertEqual(
            {(e["method"], e["sample_count"], e["bitwidth"]) for e in matrix["entries"]},
            {(m, n, b) for m in matrix["methods"] for n in matrix["sample_counts"] for b in (8, 16)},
        )
        for entry in matrix["entries"]:
            self.assertEqual(set(entry["operators"]), QUANT_OPERATORS)
            self.assertEqual(entry["unsafe_non_finite_count"], 0)
            # Calibration rankings/MAE ratios are descriptive, not correctness laws.
            self.assertGreaterEqual(entry["error"]["prediction_agreement"], 0)
            self.assertLessEqual(entry["error"]["prediction_agreement"], 1)
        for bits in (8, 16):
            result = report["training"][f"quant_lstm_qat_{bits}bit"]
            error = result["selected_quantization_error"]
            self.assertEqual(error["sample_count"], 128)
            self.assertEqual(len(error["time_step_trace"]["steps"]), 49)
        ablation = report["training"]["quant_lstm_qat_8bit"]["operator_bitwidth_ablation"]
        self.assertEqual(set(ablation["operators"]), QUANT_OPERATORS)
        self.assertTrue(all(v["calibration_unsafe_non_finite_count"] == 0 for v in ablation["operators"].values()))


if __name__ == "__main__":
    unittest.main()
