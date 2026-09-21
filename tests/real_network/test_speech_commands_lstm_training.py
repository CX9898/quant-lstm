"""End-to-end training comparison on Speech Commands v0.02."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from speech_commands_lstm_training import ExperimentConfig, run_training_comparison


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/results/speech_commands_lstm_training.json"


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
        self.assertEqual(report["schema_version"], 5)
        self.assertEqual(report["replacement"]["changed_module"], "lstm")
        self.assertEqual(report["quantization"]["bitwidths"], [8, 16])
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


if __name__ == "__main__":
    unittest.main()
