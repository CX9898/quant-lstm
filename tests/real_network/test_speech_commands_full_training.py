"""Full 35-class Speech Commands v0.02 training and evaluation gate."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from speech_commands_lstm_training import (
    FULL_SPEECH_COMMAND_LABELS,
    ExperimentConfig,
    run_training_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/results/speech_commands_lstm_full_training.json"
FEATURE_CACHE = ROOT / "tests/results/speech_commands_v0.02_full_mfcc.pt"


class SpeechCommandsFullTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("QUANT_LSTM_RUN_FULL_SPEECH_COMMANDS") != "1":
            raise unittest.SkipTest(
                "set QUANT_LSTM_RUN_FULL_SPEECH_COMMANDS=1 to run full training"
            )
        dataset_root = os.environ.get("QUANT_LSTM_SPEECH_COMMANDS_ROOT")
        if not dataset_root:
            raise RuntimeError("QUANT_LSTM_SPEECH_COMMANDS_ROOT is required")
        cls.dataset_root = Path(dataset_root)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("full QuantLSTM training requires CUDA")

    def test_all_official_samples_train_and_evaluate_all_lstm_modes(self) -> None:
        report = run_training_comparison(
            ExperimentConfig(
                dataset_root=self.dataset_root,
                labels=FULL_SPEECH_COMMAND_LABELS,
                train_samples_per_label=None,
                validation_samples_per_label=None,
                test_samples_per_label=None,
                dataset_profile="full",
                feature_chunk_size=512,
                feature_cache=FEATURE_CACHE,
                hidden_size=64,
                batch_size=256,
                epochs=10,
                learning_rate=3.0e-3,
                calibration_batches=4,
                calibration_refresh_epochs=1,
                quant_bitwidths=(8, 16),
                seed=20260921,
                quality_gate_seeds=(20260921,),
                extended_diagnostics=False,
            )
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        dataset = report["dataset"]
        self.assertEqual(dataset["profile"], "full")
        self.assertEqual(dataset["labels"], list(FULL_SPEECH_COMMAND_LABELS))
        self.assertEqual(dataset["training_samples"], 84_843)
        self.assertEqual(dataset["validation_samples"], 9_981)
        self.assertEqual(dataset["test_samples"], 11_005)
        self.assertEqual(dataset["feature_shape"], [49, 20])
        audit = dataset["audit"]
        self.assertEqual(audit["selected_word_sample_count"], 105_829)
        self.assertEqual(audit["omitted_word_sample_count"], 0)
        self.assertEqual(audit["split_overlap_count"], 0)

        expected_models = {
            "torch_lstm",
            "quant_lstm_float",
            "quant_lstm_qat_8bit",
            "quant_lstm_qat_16bit",
        }
        self.assertEqual(set(report["training"]), expected_models)
        minimum_accuracy = {
            "torch_lstm": 0.85,
            "quant_lstm_float": 0.85,
            "quant_lstm_qat_8bit": 0.80,
            "quant_lstm_qat_16bit": 0.82,
        }
        minimum_macro_f1 = {
            "torch_lstm": 0.84,
            "quant_lstm_float": 0.84,
            "quant_lstm_qat_8bit": 0.78,
            "quant_lstm_qat_16bit": 0.80,
        }
        minimum_class_f1 = {
            "torch_lstm": 0.65,
            "quant_lstm_float": 0.65,
            "quant_lstm_qat_8bit": 0.58,
            "quant_lstm_qat_16bit": 0.65,
        }
        for name, result in report["training"].items():
            with self.subTest(model=name):
                self.assertLess(result["final_train_loss"], result["initial_train_loss"])
                self.assertGreater(result["parameter_update_norm"], 0.0)
                self.assertGreaterEqual(
                    result["final_test_accuracy"], minimum_accuracy[name]
                )
                classification = result["final_test_classification"]
                self.assertEqual(classification["sample_count"], 11_005)
                self.assertEqual(
                    set(classification["per_class"]), set(FULL_SPEECH_COMMAND_LABELS)
                )
                self.assertTrue(
                    all(
                        metrics["support"]
                        == audit["per_label_split_counts"]["testing"][label]
                        for label, metrics in classification["per_class"].items()
                    )
                )
                self.assertGreaterEqual(
                    classification["macro_f1"], minimum_macro_f1[name]
                )
                self.assertGreaterEqual(
                    min(value["f1"] for value in classification["per_class"].values()),
                    minimum_class_f1[name],
                )
                self.assertEqual(len(classification["confusion_matrix"]), 35)
                self.assertTrue(
                    all(len(row) == 35 for row in classification["confusion_matrix"])
                )

        baseline = report["training"]["torch_lstm"]
        native_float = report["training"]["quant_lstm_float"]
        self.assertGreaterEqual(
            native_float["final_test_accuracy"], baseline["final_test_accuracy"] - 0.02
        )

        backward = report["quantization"]["real_batch_backward_oracle"]
        self.assertEqual(set(backward), {"8", "16"})
        for result in backward.values():
            self.assertEqual(result["sample_count"], 256)
            self.assertTrue(
                all(
                    gradient["max_absolute_error"] <= 5.0e-6
                    and gradient["cosine"] >= 0.99999
                    for gradient in result["gradients"].values()
                )
            )

        calibrations = report["quantization"]["calibration"]
        for bitwidth, name in (
            (8, "quant_lstm_qat_8bit"),
            (16, "quant_lstm_qat_16bit"),
        ):
            calibration = calibrations[name]
            self.assertEqual(calibration["sample_count"], 1_024)
            self.assertEqual(sum(calibration["label_counts"]), 1_024)
            self.assertLessEqual(
                max(calibration["label_counts"]) - min(calibration["label_counts"]), 1
            )
            self.assertEqual(calibration["safety"]["unsafe_non_finite_count"], 0)
            self.assertEqual(
                calibration["method"], {8: "sqnr", 16: "minmax"}[bitwidth]
            )

        int8 = report["training"]["quant_lstm_qat_8bit"]
        int16 = report["training"]["quant_lstm_qat_16bit"]
        self.assertTrue(int8["native_qat_checkpoint_observed"])
        self.assertTrue(int16["native_qat_checkpoint_observed"])
        self.assertEqual(int8["final_quantization_error"]["sample_count"], 11_005)
        self.assertEqual(int16["final_quantization_error"]["sample_count"], 11_005)
        self.assertGreaterEqual(
            int8["final_test_accuracy"], baseline["final_test_accuracy"] - 0.06
        )
        self.assertGreaterEqual(
            int16["final_test_accuracy"], baseline["final_test_accuracy"] - 0.04
        )
        self.assertGreaterEqual(
            int16["final_test_accuracy"], int8["final_test_accuracy"] + 0.01
        )
        self.assertGreaterEqual(
            int8["final_quantization_error"]["prediction_agreement"], 0.90
        )
        self.assertGreaterEqual(
            int16["final_quantization_error"]["prediction_agreement"], 0.99
        )
        self.assertLess(
            int16["final_quantization_error"]["mae"],
            int8["final_quantization_error"]["mae"] * 0.01,
        )
        self.assertLess(int8["final_quantization_error"]["mae"], 0.70)
        self.assertLess(int16["final_quantization_error"]["mae"], 0.005)
        self.assertGreaterEqual(
            int16["final_quantization_error"]["cosine"], 0.9999
        )


if __name__ == "__main__":
    unittest.main()
