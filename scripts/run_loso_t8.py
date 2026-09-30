"""T8 — Leave-One-Subject-Out for the best config so far (ghost_mobilenetv4_pa2),
run from train/: `uv run python scripts/run_loso_t8.py`

Reuses run_torch_cv_experiment with k = number of participants -- verified
separately that make_group_kfold_splits(k=n_participants) puts exactly one
participant in each fold's test set, i.e. true LOSO, no new split logic
needed. Resume support (T14 fix) applies here too, which matters given ~54
folds is a long run.

Picks ghost_mobilenetv4_pa2 as "the best config so far" per the partial T14
grid results (Decision #14) -- flagged as provisional: if the full T14/T15
grid later shows a different config is genuinely best, LOSO should be re-run
on that one instead (see plan.csv T8 note: "chờ kết quả GĐ6").

Output: the usual {config}_cv.json/.pth per fold, plus a per-participant
breakdown (loso_by_participant.csv) -- this is T8's actual diagnostic value,
not just an aggregate accuracy number (which grouped 5-fold CV already
gives, T6).
"""
import csv

from asd_train.configs import DATASET_PATH_DEFAULT, PRESETS
from asd_train.data import get_participant_groups, make_group_kfold_splits
from asd_train.train_torch import run_torch_cv_experiment
from torchvision import datasets

CONFIG_NAME = "ghost_mobilenetv4_pa2"
DATASET_ROOT = "../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts/loso"

dataset = datasets.ImageFolder(DATASET_PATH_DEFAULT)
groups = get_participant_groups(dataset)
n_participants = len(set(groups))
print(f"LOSO: {n_participants} participants, config={CONFIG_NAME}")

splits = make_group_kfold_splits(dataset, k=n_participants, val_size=0.2)
fold_to_participant = {i: next(iter({groups[j] for j in test_idx})) for i, (_, _, test_idx) in enumerate(splits)}

cfg = PRESETS[CONFIG_NAME]
result = run_torch_cv_experiment(cfg, dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR, k=n_participants)

rows = []
for i, fold in enumerate(result.fold_results):
    pid = fold_to_participant[i]
    rows.append({
        "participant_id": pid,
        "n_images": None,  # filled below
        "accuracy": fold.test_accuracy,
        "sensitivity": fold.clinical.sensitivity,
        "specificity": fold.clinical.specificity,
    })

from collections import Counter
img_counts = Counter(groups)
for row in rows:
    row["n_images"] = img_counts[row["participant_id"]]

rows.sort(key=lambda r: r["accuracy"])

with open(f"{OUTPUT_DIR}/loso_by_participant.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["participant_id", "n_images", "accuracy", "sensitivity", "specificity"])
    writer.writeheader()
    writer.writerows(rows)

print("\nWorst 10 participants (lowest per-participant accuracy):")
for r in rows[:10]:
    print(f"  P{r['participant_id']}: acc={r['accuracy']:.3f} n_images={r['n_images']} sens={r['sensitivity']:.3f} spec={r['specificity']:.3f}")

print(f"\nOverall LOSO accuracy: {result.mean_accuracy:.4f} (95% CI {result.ci_accuracy[0]:.4f}-{result.ci_accuracy[1]:.4f})")
print(f"Full per-participant breakdown written to {OUTPUT_DIR}/loso_by_participant.csv")
