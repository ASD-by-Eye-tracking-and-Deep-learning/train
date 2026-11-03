"""T5 — pooled LOSO metrics, run from train/ after run_loso_t8.py:
`uv run python scripts/loso_pooled_t5.py --config ghost_convnexttiny_pa3`

Why: every LOSO fold's test set is one child, so it holds a single class. The
per-fold sensitivity/specificity/AUC that run_torch_cv_experiment averages are
undefined (or 0) for half the folds, and the printed CV means for them are
meaningless -- only per-fold accuracy is. The valid summary pools the held-out
predictions of all 54 folds and computes the metrics once.

Re-runs inference with each fold's saved best checkpoint on that fold's test
child (same split, same transform, same model code as training) and checks the
per-fold accuracy against {config}_cv.json before using it.

Reports:
- image level: pooled sensitivity/specificity (Wilson 95% CI), AUC;
- child level (primary, the unit of diagnosis): mean P(ASD) over a child's
  images, threshold 0.5 -> sensitivity/specificity (Wilson CI), AUC;
- AUC CIs: bootstrap over children (2000 resamples, seed 0), since images of
  one child are not independent.

Output: artifacts/loso/loso_pooled.md, loso_pooled_predictions.csv.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from asd_train.configs import PRESETS_WITH_INCEPTIONV3
from asd_train.data import build_torch_transform, get_participant_groups, make_group_kfold_splits
from asd_train.metrics import ASD_LABEL
from asd_train.train_torch import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="ghost_convnexttiny_pa3")
args = parser.parse_args()

DATASET_ROOT = "../../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts/loso"
cfg = PRESETS_WITH_INCEPTIONV3[args.config]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.path.join(DATASET_ROOT, os.path.basename(cfg.dataset_path.rstrip("/")))
transform = build_torch_transform(cfg.img_size, cfg.extra_transform_augmentation, cfg.grayscale)
dataset = datasets.ImageFolder(dataset_path, transform=transform)
groups = get_participant_groups(dataset)
k = len(set(groups))
splits = make_group_kfold_splits(dataset, k=k, val_size=0.2)
saved = json.load(open(f"{OUTPUT_DIR}/{cfg.name}_cv.json"))["folds"]
assert len(saved) == k, f"{len(saved)} folds in cv.json, expected {k}"


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


rows = []
mismatches = []
for fold_i, (_, _, test_idx) in enumerate(splits):
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/{cfg.name}_fold{fold_i}.pth", map_location=device))
    model.eval()
    loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            probs.extend(torch.sigmoid(model(images.to(device))).squeeze(1).cpu().numpy())
            labels.extend(y.numpy())
    probs, labels = np.array(probs), np.array(labels)
    acc = float(((probs > 0.5).astype(int) == labels).mean())
    if abs(acc - float(saved[fold_i]["test_accuracy"])) > 1e-6:
        mismatches.append((fold_i, acc, saved[fold_i]["test_accuracy"]))
    for idx, p, y in zip(test_idx, probs, labels):
        p_asd = 1 - p if ASD_LABEL == 0 else p
        rows.append({"fold": fold_i, "participant_id": groups[idx], "image": os.path.basename(dataset.samples[idx][0]),
                     "is_asd": int(y == ASD_LABEL), "p_asd": float(p_asd)})
    del model
    torch.cuda.empty_cache()

with open(f"{OUTPUT_DIR}/loso_pooled_predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

y = np.array([r["is_asd"] for r in rows])
p = np.array([r["p_asd"] for r in rows])
pid = np.array([r["participant_id"] for r in rows])
children = sorted(set(pid))
child_y = np.array([y[pid == c][0] for c in children])
child_p = np.array([p[pid == c].mean() for c in children])


def sens_spec(truth: np.ndarray, score: np.ndarray) -> dict:
    pred = score >= 0.5
    tp, fn = int((pred & (truth == 1)).sum()), int((~pred & (truth == 1)).sum())
    tn, fp = int((~pred & (truth == 0)).sum()), int((pred & (truth == 0)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "acc": (tp + tn) / len(truth), "acc_ci": wilson(tp + tn, len(truth)),
            "sens": tp / (tp + fn), "sens_ci": wilson(tp, tp + fn),
            "spec": tn / (tn + fp), "spec_ci": wilson(tn, tn + fp),
            "auc": roc_auc_score(truth, score)}


img, child = sens_spec(y, p), sens_spec(child_y, child_p)

rng = np.random.default_rng(0)
img_aucs, child_aucs = [], []
for _ in range(2000):
    pick = rng.choice(len(children), len(children), replace=True)
    if len(set(child_y[pick])) < 2:
        continue
    child_aucs.append(roc_auc_score(child_y[pick], child_p[pick]))
    mask = np.concatenate([np.where(pid == children[i])[0] for i in pick])
    img_aucs.append(roc_auc_score(y[mask], p[mask]))
img["auc_ci"] = tuple(np.percentile(img_aucs, [2.5, 97.5]))
child["auc_ci"] = tuple(np.percentile(child_aucs, [2.5, 97.5]))


def fmt(m: dict, key: str) -> str:
    lo, hi = m[f"{key}_ci"]
    return f"{m[key]:.4f} ({lo:.4f}–{hi:.4f})"


lines = [
    f"# LOSO pooled metrics — {cfg.name}",
    "",
    f"{k} folds (one child each), {len(rows)} images, {int(child_y.sum())} ASD / {int((1 - child_y).sum())} TD children.",
    "Positive class = ASD. Threshold 0.5 on P(ASD). Accuracy/sens/spec CI: Wilson 95%; AUC CI: bootstrap over children (2000).",
    f"Re-inference vs cv.json per-fold accuracy: {'all ' + str(k) + ' folds match' if not mismatches else f'MISMATCH {mismatches}'}.",
    "",
    "| Level | n | Accuracy | Sensitivity | Specificity | AUC-ROC | TP/FN/TN/FP |",
    "|---|---|---|---|---|---|---|",
    f"| Child (mean P(ASD)) — primary | {len(children)} | {fmt(child, 'acc')} | {fmt(child, 'sens')} | {fmt(child, 'spec')} | {fmt(child, 'auc')} | {child['tp']}/{child['fn']}/{child['tn']}/{child['fp']} |",
    f"| Image (pooled) | {len(rows)} | {fmt(img, 'acc')} | {fmt(img, 'sens')} | {fmt(img, 'spec')} | {fmt(img, 'auc')} | {img['tp']}/{img['fn']}/{img['tn']}/{img['fp']} |",
    "",
    "Mean of per-fold accuracy (each child weighted equally, as printed by run_loso_t8.py) is a different quantity from pooled image accuracy.",
]
with open(f"{OUTPUT_DIR}/loso_pooled.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
