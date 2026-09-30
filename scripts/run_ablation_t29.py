"""T29 -- run the grayscale-input ablation arm under the same T6 protocol
(5-fold CV, full epoch budget with early stopping) T10 used for the other 4
ablation configs. Run from train/:

    uv run python scripts/run_ablation_t29.py

Same output_dir as T10 (artifacts/ablation) so this fold's weights/JSON sit
alongside the other 4 configs' artifacts -- but this script only APPENDS one
row to the existing ablation_summary.md, it never rewrites the file (T10's
run_ablation_t10.py truncates and rewrites the whole file; doing that here
would destroy the already-recorded ablation_base/fmmix_off/cbam_isolated/
frozen_backbone rows, which are a completed, "Done" task's result -- see
plan.csv T29 / the "Does this touch Done tasks" note in the approved plan).
"""
from asd_train.configs import T29_EXTRA_PRESETS
from asd_train.train_torch import run_torch_cv_experiment

DATASET_ROOT = "../data/mahmoud-dataset"
OUTPUT_DIR = "artifacts/ablation"
SUMMARY_PATH = f"{OUTPUT_DIR}/ablation_summary.md"
K = 5

for key, cfg in T29_EXTRA_PRESETS.items():
    print(f"\n{'#' * 80}\n# {key} ({cfg.name})\n{'#' * 80}")
    result = run_torch_cv_experiment(cfg, dataset_root=DATASET_ROOT, output_dir=OUTPUT_DIR, k=K)

    row = (
        f"| {key} "
        f"| {result.mean_accuracy:.4f} ({result.ci_accuracy[0]:.4f}–{result.ci_accuracy[1]:.4f}) "
        f"| {result.mean_sensitivity:.4f} ({result.ci_sensitivity[0]:.4f}–{result.ci_sensitivity[1]:.4f}) "
        f"| {result.mean_specificity:.4f} ({result.ci_specificity[0]:.4f}–{result.ci_specificity[1]:.4f}) "
        f"| {result.mean_auc_roc:.4f} ({result.ci_auc_roc[0]:.4f}–{result.ci_auc_roc[1]:.4f}) |"
    )
    print("\n" + row)

    with open(SUMMARY_PATH) as f:
        existing = f.read()
    if key in existing:
        print(f"'{key}' row already present in {SUMMARY_PATH} -- not appending a duplicate.")
        continue
    with open(SUMMARY_PATH, "a") as f:
        f.write(row + "\n")
    print(f"Appended '{key}' row to {SUMMARY_PATH}")

print("\nOK: T29 grayscale ablation arm finished")
