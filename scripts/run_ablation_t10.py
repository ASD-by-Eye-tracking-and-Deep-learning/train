"""T10 — run the T9 ablation grid under the real T6 protocol (5-fold CV,
full epoch budget with early stopping). Run from train/:

    uv run python scripts/run_ablation_t10.py

Separate output_dir from the main 12-config grid (artifacts/): the base
config's name is literally "ghost_mobilenetv4_pa2", same as the real T14
preset -- sharing a directory would let this run's fold weights collide
with T14's later.
"""
from asd_train.configs import ABLATION_PRESETS
from asd_train.train_torch import run_torch_cv_experiment

DATASET_ROOT = "../../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts/ablation"
K = 5

cv_results = {}
for key, cfg in ABLATION_PRESETS.items():
    print(f"\n{'#' * 80}\n# {key} ({cfg.name})\n{'#' * 80}")
    cv_results[key] = run_torch_cv_experiment(cfg, dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR, k=K)

lines = [
    "| Config | Accuracy | Sensitivity (ASD) | Specificity | AUC-ROC |",
    "|---|---|---|---|---|",
]
for key, r in cv_results.items():
    lines.append(
        f"| {key} "
        f"| {r.mean_accuracy:.4f} ({r.ci_accuracy[0]:.4f}–{r.ci_accuracy[1]:.4f}) "
        f"| {r.mean_sensitivity:.4f} ({r.ci_sensitivity[0]:.4f}–{r.ci_sensitivity[1]:.4f}) "
        f"| {r.mean_specificity:.4f} ({r.ci_specificity[0]:.4f}–{r.ci_specificity[1]:.4f}) "
        f"| {r.mean_auc_roc:.4f} ({r.ci_auc_roc[0]:.4f}–{r.ci_auc_roc[1]:.4f}) |"
    )
summary = "\n".join(lines)
print("\n" + summary)

with open(f"{OUTPUT_DIR}/ablation_summary.md", "w") as f:
    f.write("# T10 — Ablation results (5-fold grouped CV, mean ± 95% CI)\n\n" + summary + "\n")

print(f"\nOK: T10 ablation grid finished, summary written to {OUTPUT_DIR}/ablation_summary.md")
