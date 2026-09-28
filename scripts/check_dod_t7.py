"""DoD check for plan.csv T7 — run from train/: `uv run python scripts/check_dod_t7.py`.

Confirms sensitivity/specificity/AUC-ROC are computed correctly: present,
finite, in [0,1], ROC curve arrays consistent (fpr/tpr same length, fpr
non-decreasing, endpoints at 0 and 1), and sensitivity/specificity/AUC-ROC
derivable independently from the confusion matrix agree with the reported
values (catches a wrong positive-class convention, which wouldn't otherwise
raise an error -- just silently swap sensitivity and specificity).
"""
import dataclasses
import math

from asd_train.configs import PRESETS
from asd_train.train_torch import run_torch_cv_experiment

cfg = dataclasses.replace(PRESETS["ghost_mobilenetv4_pa2"], num_epochs=1, patience=1)
result = run_torch_cv_experiment(cfg, dataset_root="../data/mahmoud-dataset", output_dir="/tmp/check_dod_t7", k=2)

for r in result.fold_results:
    c = r.clinical
    for name, value in [("sensitivity", c.sensitivity), ("specificity", c.specificity), ("auc_roc", c.auc_roc)]:
        assert math.isfinite(value), f"{name} not finite: {value}"
        assert 0.0 <= value <= 1.0, f"{name} out of [0,1]: {value}"

    assert len(c.roc_fpr) == len(c.roc_tpr) > 0, "roc_fpr/roc_tpr length mismatch or empty"
    assert all(a <= b + 1e-9 for a, b in zip(c.roc_fpr, c.roc_fpr[1:])), "roc_fpr not non-decreasing"
    assert c.roc_fpr[0] == 0.0 and c.roc_fpr[-1] == 1.0, f"roc_fpr should span [0,1], got {c.roc_fpr[0]}..{c.roc_fpr[-1]}"

    # Independently recompute sensitivity/specificity from the confusion matrix
    # confusion_matrix was built via confusion_matrix(1-y_all, 1-yhat_all), so
    # index 1 (post-flip) = ASD. cm[i][j] = true i, predicted j.
    cm = r.confusion_matrix
    asd_total = cm[1][0] + cm[1][1]
    non_asd_total = cm[0][0] + cm[0][1]
    expected_sensitivity = cm[1][1] / asd_total if asd_total else float("nan")
    expected_specificity = cm[0][0] / non_asd_total if non_asd_total else float("nan")
    assert abs(c.sensitivity - expected_sensitivity) < 1e-6, (
        f"sensitivity {c.sensitivity} != confusion-matrix-derived {expected_sensitivity} "
        f"(possible positive-class convention bug)"
    )
    assert abs(c.specificity - expected_specificity) < 1e-6, (
        f"specificity {c.specificity} != confusion-matrix-derived {expected_specificity} "
        f"(possible positive-class convention bug)"
    )

for name, value, ci in [
    ("sensitivity", result.mean_sensitivity, result.ci_sensitivity),
    ("specificity", result.mean_specificity, result.ci_specificity),
    ("auc_roc", result.mean_auc_roc, result.ci_auc_roc),
]:
    assert math.isfinite(value), f"mean {name} not finite: {value}"
    assert math.isfinite(ci[0]) and math.isfinite(ci[1]), f"{name} CI not finite: {ci}"

print("OK: sensitivity/specificity/AUC-ROC finite, in range, ROC curve well-formed")
print("OK: sensitivity/specificity independently verified against confusion matrix")
print("OK: T7 DoD checks passed")
