"""DoD check for plan.csv T6 — run from train/: `uv run python scripts/check_dod_t6.py`.

Confirms grouped k-fold CV (make_group_kfold_splits + run_torch_cv_experiment):
subject-independent within every fold, every participant used as test exactly
once across the whole k-fold partition, and mean/CI aggregation is finite.
Uses k=2 and 1 epoch/fold for speed — this checks correctness, not accuracy.
"""
import dataclasses
import math

from asd_train.configs import DATASET_PATH_DEFAULT, PRESETS
from asd_train.data import get_participant_groups, make_group_kfold_splits
from asd_train.train_torch import run_torch_cv_experiment
from torchvision import datasets

K = 2

# --- Split-level checks (cheap, no training) ---
dataset = datasets.ImageFolder(DATASET_PATH_DEFAULT)
groups = get_participant_groups(dataset)
splits = make_group_kfold_splits(dataset, k=K)

assert len(splits) == K, f"expected {K} folds, got {len(splits)}"

all_test_participants: list[int] = []
for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
    train_p = {groups[i] for i in train_idx}
    val_p = {groups[i] for i in val_idx}
    test_p = {groups[i] for i in test_idx}

    for a, b, name_a, name_b in [(train_p, val_p, "train", "val"), (train_p, test_p, "train", "test"), (val_p, test_p, "val", "test")]:
        overlap = a & b
        assert not overlap, f"fold {fold_i}: participants in both {name_a} and {name_b}: {overlap}"

    all_test_participants.extend(test_p)
    print(f"fold {fold_i}: train={len(train_idx)} img/{len(train_p)} pts, val={len(val_idx)} img/{len(val_p)} pts, test={len(test_idx)} img/{len(test_p)} pts")

unique_participants = set(groups)
assert sorted(all_test_participants) == sorted(unique_participants), (
    f"test-fold participants don't exactly partition all participants: "
    f"missing={unique_participants - set(all_test_participants)}, "
    f"duplicated={[p for p in unique_participants if all_test_participants.count(p) > 1]}"
)
print(f"OK: {len(unique_participants)} participants, each used as test exactly once across {K} folds")

# --- Full run_torch_cv_experiment smoke test (cheap config overrides) ---
cfg = dataclasses.replace(PRESETS["ghost_mobilenetv4_pa2"], num_epochs=1, patience=1)
result = run_torch_cv_experiment(cfg, dataset_root="../data/mahmoud-dataset", output_dir="/tmp/check_dod_t6", k=K)

assert len(result.fold_results) == K, f"expected {K} fold results, got {len(result.fold_results)}"
for metric_name, value, ci in [
    ("accuracy", result.mean_accuracy, result.ci_accuracy),
    ("precision", result.mean_precision, result.ci_precision),
    ("recall", result.mean_recall, result.ci_recall),
]:
    assert math.isfinite(value), f"mean {metric_name} is not finite: {value}"
    assert math.isfinite(ci[0]) and math.isfinite(ci[1]), f"{metric_name} CI is not finite: {ci}"

print("OK: run_torch_cv_experiment returns finite mean/CI for accuracy/precision/recall")
print("OK: T6 DoD checks passed")
