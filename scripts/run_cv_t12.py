"""T12 — finish the 13-config grid (12 Ghost/CBAM + InceptionV3 PyTorch) under
the T3 protocol (5-fold grouped CV) and pick the best config. Run from train/:

    uv run python scripts/run_cv_t12.py

Resumable: run_torch_cv_experiment skips folds already recorded in
artifacts/{name}_cv.json, so configs that are complete only get reloaded and
an interrupted run continues from the next unfinished fold.

Writes artifacts/cv_summary_t12.md: mean ± 95% CI per config, sorted by mean
AUC-ROC (threshold-independent, so less sensitive to the fixed 0.5 cut-off
than accuracy on a 40/60 class split).
"""
from asd_train.configs import PRESETS_WITH_INCEPTIONV3
from asd_train.train_torch import run_torch_cv_experiment

DATASET_ROOT = "../../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts"
K = 5

cv_results = {}
for name, cfg in PRESETS_WITH_INCEPTIONV3.items():
    print(f"\n{'#' * 80}\n# {name}\n{'#' * 80}", flush=True)
    cv_results[name] = run_torch_cv_experiment(cfg, dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR, k=K)


def fmt(mean, ci):
    return f"{mean:.4f} ({ci[0]:.4f}–{ci[1]:.4f})"


ranked = sorted(cv_results.items(), key=lambda kv: kv[1].mean_auc_roc, reverse=True)
lines = [
    "| Config | Accuracy | Sensitivity (ASD) | Specificity | AUC-ROC |",
    "|---|---|---|---|---|",
]
for name, r in ranked:
    lines.append(
        f"| {name} "
        f"| {fmt(r.mean_accuracy, r.ci_accuracy)} "
        f"| {fmt(r.mean_sensitivity, r.ci_sensitivity)} "
        f"| {fmt(r.mean_specificity, r.ci_specificity)} "
        f"| {fmt(r.mean_auc_roc, r.ci_auc_roc)} |"
    )
summary = "\n".join(lines)
print("\n" + summary)

best_name, best = ranked[0]
with open(f"{OUTPUT_DIR}/cv_summary_t12.md", "w", encoding="utf-8") as f:
    f.write(
        "# T12 — 13-config results (5-fold grouped CV, mean ± 95% CI)\n\n"
        "Sorted by mean AUC-ROC.\n\n" + summary + "\n\n"
        f"Best by mean AUC-ROC: **{best_name}** "
        f"(AUC {best.mean_auc_roc:.4f}, accuracy {best.mean_accuracy:.4f}).\n"
    )

print(f"\nOK: T12 grid finished, summary written to {OUTPUT_DIR}/cv_summary_t12.md")
