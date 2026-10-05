"""DoD check for plan.csv T9 — run from train/: `uv run python scripts/check_dod_t9.py`.

T9's deliverable is a list of ablation configs, each changing exactly one
factor off a shared base (so any measured delta in T10 can be attributed to
that one factor). This checks the structural property directly — every
ABLATION_PRESETS entry other than the base differs from it in exactly one
dataclass field — and smoke-runs each new toggle (use_fmmix=False,
freeze_backbone_always) once to confirm the plumbing actually executes,
since a field that's "different" but never read by the training loop would
pass the structural check while being a no-op ablation.
"""
import dataclasses
import math

from asd_train.configs import ABLATION_PRESETS
from asd_train.train_torch import run_torch_cv_experiment

base = ABLATION_PRESETS["ablation_base"]
fields = [f.name for f in dataclasses.fields(base) if f.name != "name"]

for key, cfg in ABLATION_PRESETS.items():
    if key == "ablation_base":
        continue
    diffs = [f for f in fields if getattr(cfg, f) != getattr(base, f)]
    assert len(diffs) == 1, f"{key}: expected exactly 1 field to differ from base, got {diffs}"
    print(f"OK: {key} differs from base in exactly one field: {diffs[0]}")

# --- Smoke-run each config (cheap: k=2, 1 epoch) to confirm the new toggles execute ---
K = 2
for key, cfg in ABLATION_PRESETS.items():
    cheap_cfg = dataclasses.replace(cfg, num_epochs=1, patience=1)
    result = run_torch_cv_experiment(
        cheap_cfg, dataset_root="../../data/mahmoud-dataset", output_dir="/tmp/check_dod_t9", k=K
    )
    assert len(result.fold_results) == K, f"{key}: expected {K} fold results, got {len(result.fold_results)}"
    assert math.isfinite(result.mean_accuracy), f"{key}: mean_accuracy not finite"
    print(f"OK: {key} smoke-trained without error, mean_accuracy={result.mean_accuracy:.4f}")

print("OK: T9 DoD checks passed")
