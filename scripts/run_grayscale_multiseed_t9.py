"""T9 — RGB vs grayscale, repeated over several training seeds on the SAME
5-fold split, with a paired test per fold. Run from train/:

    uv run python scripts/run_grayscale_multiseed_t9.py [--base NAME] [--seeds 0 1 2]

The 1-seed T29 run (artifacts/ablation/ablation_summary.md, ablation_grayscale
row) is left untouched. Here the split seed stays fixed (random_state=42, same
folds as every other CV run) and only the training seed (weight init, data
order, FMMix sampling) changes, so each (seed, fold) gives one RGB/grayscale
pair evaluated on identical test participants.

Primary test: per fold, average the RGB-minus-grayscale difference over seeds
-> 5 paired differences over 5 disjoint test sets (independent). Secondary:
all seed x fold pairs (more pairs, but pairs from the same fold share test
participants, so this p-value is optimistic and is reported as such).

Writes artifacts/ablation/grayscale_paired_test.md and appends one
"ablation_grayscale_multiseed" row to artifacts/ablation/ablation_summary.md.
"""
import argparse
import os
import random
from dataclasses import replace

import numpy as np
import torch
from scipy import stats

from asd_train.configs import PRESETS_WITH_INCEPTIONV3
from asd_train.train_torch import run_torch_cv_experiment

DATASET_ROOT = "../../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts/ablation/multiseed"
REPORT_PATH = "artifacts/ablation/grayscale_paired_test.md"
SUMMARY_PATH = "artifacts/ablation/ablation_summary.md"
K = 5

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="ghost_mobilenetv4_pa2", help="config name in PRESETS_WITH_INCEPTIONV3")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = parser.parse_args()

base = PRESETS_WITH_INCEPTIONV3[args.base]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# metrics[arm] -> array of shape (n_seeds, K)
metrics = {arm: {"accuracy": [], "auc_roc": []} for arm in ("rgb", "gray")}
for seed in args.seeds:
    for arm in ("rgb", "gray"):
        cfg = replace(base, name=f"t9ms_{args.base}_{arm}_s{seed}", grayscale=(arm == "gray"))
        print(f"\n{'#' * 80}\n# {cfg.name}\n{'#' * 80}", flush=True)
        set_seed(seed)
        r = run_torch_cv_experiment(cfg, dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR, k=K)
        metrics[arm]["accuracy"].append([f.test_accuracy for f in r.fold_results])
        metrics[arm]["auc_roc"].append([f.clinical.auc_roc for f in r.fold_results])


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(x))
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)
    return m, m - h, m + h


def paired(rgb: np.ndarray, gray: np.ndarray) -> dict:
    d = rgb - gray
    m, lo, hi = mean_ci(d)
    t_p = float(stats.ttest_rel(rgb, gray).pvalue)
    try:
        w_p = float(stats.wilcoxon(rgb, gray).pvalue)
    except ValueError:  # all differences zero
        w_p = float("nan")
    return {"n": len(d), "rgb": float(rgb.mean()), "gray": float(gray.mean()), "diff": m, "lo": lo, "hi": hi, "t_p": t_p, "w_p": w_p}


rows = []
results = {}
for metric in ("accuracy", "auc_roc"):
    rgb = np.array(metrics["rgb"][metric])
    gray = np.array(metrics["gray"][metric])
    results[metric] = {
        "primary": paired(rgb.mean(axis=0), gray.mean(axis=0)),
        "secondary": paired(rgb.ravel(), gray.ravel()),
    }
    for level, label in (("primary", "fold-level, seed-averaged (primary)"), ("secondary", "all seed×fold pairs (optimistic)")):
        s = results[metric][level]
        rows.append(
            f"| {metric} | {label} | {s['n']} | {s['rgb']:.4f} | {s['gray']:.4f} "
            f"| {s['diff']:+.4f} ({s['lo']:+.4f} to {s['hi']:+.4f}) | {s['t_p']:.4f} | {s['w_p']:.4f} |"
        )

primary_sig = [m for m in ("accuracy", "auc_roc") if results[m]["primary"]["t_p"] < 0.05]
if primary_sig:
    verdict = (
        "RGB and grayscale differ significantly at the fold level (p < 0.05, paired t-test) on: "
        + ", ".join(primary_sig)
        + ". Direction is given by the sign of the mean difference (RGB − grayscale)."
    )
else:
    verdict = (
        "No significant RGB-vs-grayscale difference at the fold level (paired t-test p ≥ 0.05 for both "
        "accuracy and AUC-ROC). With 5 folds the test has low power, so this is 'no detectable difference', "
        "not evidence that the two inputs are equivalent."
    )

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(
        f"# T9 — RGB vs grayscale, multiseed paired test\n\n"
        f"Base config: `{args.base}`. Training seeds: {args.seeds}. Split: 5-fold grouped CV, "
        f"random_state=42 (same folds for both arms and every seed).\n\n"
        "| Metric | Pairing | n pairs | RGB mean | Gray mean | Diff RGB−Gray (95% CI) | p-value (paired t) | p-value (Wilcoxon) |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        f"**Conclusion:** {verdict}\n"
    )
print(open(REPORT_PATH, encoding="utf-8").read())

acc, auc = results["accuracy"]["primary"], results["auc_roc"]["primary"]
row = (
    f"| ablation_grayscale_multiseed ({args.base}, seeds {args.seeds}) "
    f"| gray acc {acc['gray']:.4f} vs RGB {acc['rgb']:.4f}, p={acc['t_p']:.4f} "
    f"| — | — "
    f"| gray AUC {auc['gray']:.4f} vs RGB {auc['rgb']:.4f}, p={auc['t_p']:.4f} |"
)
existing = open(SUMMARY_PATH, encoding="utf-8").read() if os.path.exists(SUMMARY_PATH) else (
    "# Ablation results (5-fold grouped CV)\n\n"
    "| Config | Accuracy | Sensitivity (ASD) | Specificity | AUC-ROC |\n|---|---|---|---|---|\n"
)
if "ablation_grayscale_multiseed" not in existing:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(existing.rstrip("\n") + "\n" + row + "\n")
    print(f"Appended multiseed row to {SUMMARY_PATH}")

print(f"\nOK: T9 multiseed paired test written to {REPORT_PATH}")
