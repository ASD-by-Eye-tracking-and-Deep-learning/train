"""T27 — small lr/weight_decay grid search for ghost_mobilenetv4_pa2 (the
ablation base, T9/T10), run from train/: `uv run python scripts/tune_hparams_t27.py`.

Uses T5's single subject-independent hold-out split (load_torch_dataset), NOT
the T6 grouped k-fold splits -- selection is by val_loss on that one split
only, so this never touches the folds T6's CV protocol will later report
test numbers on. Winning hyperparameters must still be validated via a real
run_torch_cv_experiment (T6 protocol) before being trusted as an actual
improvement -- a single-split win can be noise on this small a dataset.
"""
import dataclasses
import json

import torch
from asd_train.configs import PRESETS
from asd_train.data import load_torch_dataset
from asd_train.train_torch import _train_and_evaluate

BASE = PRESETS["ghost_mobilenetv4_pa2"]
DATASET_PATH = "../data/mahmoud-dataset/Images"

LR_GRID = [5e-6, 1e-5, 2e-5]
WD_GRID = [0.0, 1e-4, 3e-4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_torch_dataset(
    DATASET_PATH, img_size=BASE.img_size, batch_size=BASE.batch_size,
    extra_transform_augmentation=BASE.extra_transform_augmentation,
)

results = []
for lr in LR_GRID:
    for wd in WD_GRID:
        cfg = dataclasses.replace(BASE, name=f"tune_lr{lr}_wd{wd}", lr=lr, weight_decay=wd)
        print(f"\n{'#' * 80}\n# lr={lr} weight_decay={wd}\n{'#' * 80}")
        result = _train_and_evaluate(cfg, bundle, device, f"/tmp/tune_t27_lr{lr}_wd{wd}.pth")
        results.append({
            "lr": lr, "weight_decay": wd,
            "best_val_loss": result.best_val_loss,
            "test_accuracy": result.test_accuracy,
            "sensitivity": result.clinical.sensitivity,
            "specificity": result.clinical.specificity,
        })

results.sort(key=lambda r: r["best_val_loss"])

print("\n" + "=" * 80)
print(f"{'lr':>10} {'wd':>10} {'val_loss':>10} {'test_acc':>10} {'sens':>8} {'spec':>8}")
for r in results:
    print(f"{r['lr']:>10} {r['weight_decay']:>10} {r['best_val_loss']:>10.4f} "
          f"{r['test_accuracy']:>10.4f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f}")

with open("artifacts/tune_t27_results.json", "w") as f:
    json.dump(results, f, indent=2)

best = results[0]
print(f"\nOK: best by val_loss: lr={best['lr']} weight_decay={best['weight_decay']} "
      f"(val_loss={best['best_val_loss']:.4f}) -- MUST re-validate via T6 CV before adopting")
