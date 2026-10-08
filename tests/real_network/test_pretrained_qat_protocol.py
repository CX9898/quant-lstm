"""Exercise the real training/calibration ordering without downloading audio."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch
import speech_commands_lstm_training as task


@unittest.skipUnless(torch.cuda.is_available(), "requires native CUDA QAT")
class PretrainedQatProtocolTest(unittest.TestCase):
    def test_ptq_calibrates_the_selected_pretrained_weights(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        data = (torch.randn(8, 5, 20), torch.arange(8) % 2)
        features = {split: data for split in ("training", "validation", "testing")}
        config = task.ExperimentConfig(
            dataset_root=Path("unused"), labels=("no", "yes"), hidden_size=8,
            batch_size=4, epochs=2, pretrain_epochs=2, calibration_batches=2,
            minimum_pretrain_validation_accuracy=0.0, extended_diagnostics=False,
        )
        manifest = SimpleNamespace(audit={})
        original_train, original_calibrate = task._train, task._calibrate_quant_lstm
        pretrained = {}

        def train(name, model, *args, **kwargs):
            result = original_train(name, model, *args, **kwargs)
            if isinstance(model.lstm, torch.nn.LSTM):
                pretrained.update({k:v.detach().clone() for k,v in model.state_dict().items()})
            return result

        def calibrate(model, *args, **kwargs):
            self.assertTrue(pretrained, "PTQ was calibrated before float pretraining")
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, pretrained[name], atol=0, rtol=0)
            return original_calibrate(model, *args, **kwargs)

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore", RuntimeWarning)
            with patch.object(task, "load_feature_sets", return_value=(features, manifest, {})), \
                 patch.object(task, "_train", side_effect=train), \
                 patch.object(task, "_calibrate_quant_lstm", side_effect=calibrate):
                report = task.run_training_comparison(config)
        baseline = report["training"]["torch_lstm"]
        for name, result in report["training"].items():
            if name != "torch_lstm":
                self.assertEqual(result["initial_state_sha256"], baseline["selected_state_sha256"])
                self.assertGreater(result["best_trained_epoch"], 0)


    def test_checkpoint_selection_ignores_test_accuracy(self):
        device = torch.device("cuda")
        torch.manual_seed(71)
        data = {split: (torch.randn(8, 5, 20), torch.arange(8) % 2)
                for split in ("training", "validation", "testing")}
        config = task.ExperimentConfig(
            dataset_root=Path("unused"), labels=("no", "yes"),
            hidden_size=8, batch_size=4, epochs=2,
        )
        model = task.SpeechCommandsLstmClassifier(8, 2, use_quant_lstm=False, device=device)
        states = []

        def evaluate(model, values, *args):
            fingerprint = task._state_sha256(model)
            if values is data["training"]:
                return {"accuracy": 0.5, "loss": 1.0}
            if values is data["validation"]:
                if fingerprint not in states:
                    states.append(fingerprint)
                score = [0.5, 0.8, 0.7][states.index(fingerprint)]
            else:
                self.assertEqual(len(states), 3, "test set consulted before selection finished")
                # Test accuracy deliberately favors epoch 2 over validation-selected epoch 1.
                score = [0.5, 0.1, 0.95][states.index(fingerprint)]
            return {"accuracy": score, "loss": 1.0-score}

        with patch.object(task, "_evaluate", side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = task._train("selection", model, data, config, device, extended_diagnostics=False)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["best_trained_epoch"], 1)
        self.assertEqual(result["selected_test"]["accuracy"], 0.1)
        self.assertEqual(result["last_test"]["accuracy"], 0.95)
        self.assertEqual(task._state_sha256(model), states[1])

    def test_unready_float_baseline_stops_before_calibration(self):
        config = task.ExperimentConfig(dataset_root=Path("unused"))
        with patch.object(task, "_train", return_value={"selected_validation": {"accuracy": 0.1}}), \
             patch.object(task, "_calibrate_quant_lstm") as calibrate:
            with self.assertRaisesRegex(RuntimeError, "Float baseline validation accuracy"):
                task._run_quality_seed({}, config, torch.device("cuda"))
            calibrate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
