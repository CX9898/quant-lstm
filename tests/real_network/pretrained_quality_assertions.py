"""Acceptance policy for pretrained QAT, independent of diagnostic preferences.

Freeze these budgets before a run; do not fit them to the resulting scores.
Accuracy/F1 drops are absolute fractions (0.02 = two percentage points).
"""

QUALITY_POLICY = {
    "subset": {
        "minimum_float_accuracy": 0.80,
        "minimum_float_macro_f1": 0.78,
        "minimum_class_f1": 0.60,
        "maximum_drop": {8: 0.05, 16: 0.03},
    },
    "full": {
        "minimum_float_accuracy": 0.90,
        "minimum_float_macro_f1": 0.89,
        "minimum_class_f1": 0.65,
        "maximum_drop": {8: 0.02, 16: 0.01},
    },
}


def assert_pretrained_quality(test, report):
    test.assertEqual(report["schema_version"], 10)
    test.assertEqual(report["validation_scope"], "pretrained_fixed_quantizer_qat")
    test.assertEqual(report["selection"]["source"], "validation_only")
    test.assertTrue(report["selection"]["epoch_zero_eligible"])
    test.assertEqual(report["replacement"]["source_checkpoint"], "validation_selected_pretrained_float")
    test.assertEqual(report["quantization"]["calibration_strategy"]["quant_params_policy"], "fixed_after_ptq")
    policy = QUALITY_POLICY[report["dataset"]["profile"]]
    test.assertEqual(set(report["multi_seed_quality"]["runs"]), {str(s) for s in report["multi_seed_quality"]["seeds"]})
    for seed, run in report["multi_seed_quality"]["runs"].items():
        training = run["training"]
        baseline = training["torch_lstm"]
        base_test = baseline["selected_test"]
        with test.subTest(seed=seed, phase="float_pretraining"):
            test.assertGreater(baseline["selected_epoch"], 0)
            test.assertEqual(len(baseline["epochs"]), report["config"]["pretrain_epochs"])
            test.assertGreaterEqual(baseline["selected_validation"]["accuracy"], policy["minimum_float_accuracy"])
            test.assertGreaterEqual(base_test["accuracy"], policy["minimum_float_accuracy"])
            test.assertGreaterEqual(base_test["macro_f1"], policy["minimum_float_macro_f1"])
            test.assertGreaterEqual(min(m["f1"] for m in base_test["per_class"].values()), policy["minimum_class_f1"])
        for name, result in training.items():
            if name == "torch_lstm":
                continue
            with test.subTest(seed=seed, phase=name):
                test.assertEqual(result["initial_state_sha256"], baseline["selected_state_sha256"])
                test.assertEqual(run["initial_shared_state_max_abs_diff"][name], 0)
                test.assertEqual(len(result["epochs"]), report["config"]["epochs"])
                test.assertGreater(result["parameter_update_norm"], 0)
                test.assertGreater(result["best_trained_epoch"], 0)
                for field in ("starting_test", "selected_test", "best_trained_test", "last_test"):
                    metrics = result[field]
                    test.assertEqual(metrics["sample_count"], report["dataset"]["test_samples"])
                    test.assertEqual(set(metrics["per_class"]), set(report["dataset"]["labels"]))
                if name == "quant_lstm_float":
                    test.assertFalse(result["native_qat_checkpoint_observed"])
                    continue
                bits = int(name.removesuffix("bit").rsplit("_", 1)[1])
                test.assertTrue(result["native_qat_checkpoint_observed"])
                test.assertEqual(result["quant_params_policy"], "fixed_after_ptq")
                test.assertEqual(len(result["quant_params_sha256"]), 64)
                test.assertTrue(all(e["quant_params_sha256"] == result["quant_params_sha256"] for e in result["epochs"]))
                calibration = run["calibration"][name]
                test.assertEqual(calibration["safety"]["unsafe_non_finite_count"], 0)
                test.assertEqual(calibration["method"], {8:"sqnr",16:"minmax"}[bits])
                test.assertLessEqual(max(calibration["label_counts"])-min(calibration["label_counts"]), 1)
                for gradient in run["real_batch_backward_oracle"][str(bits)]["gradients"].values():
                    test.assertLessEqual(gradient["max_absolute_error"], 5e-6)
                    test.assertGreaterEqual(gradient["cosine"], 0.99999)
                # Enforce both deployable selection (may be PTQ) and actual QAT quality.
                # Final epoch is diagnostic: validation-based selection is the contract.
                for field in ("selected_test", "best_trained_test"):
                    with test.subTest(checkpoint=field):
                        metrics = result[field]
                        for metric in ("accuracy", "macro_f1"):
                            test.assertGreaterEqual(metrics[metric], base_test[metric]-policy["maximum_drop"][bits])
                        test.assertGreaterEqual(min(m["f1"] for m in metrics["per_class"].values()), policy["minimum_class_f1"])
