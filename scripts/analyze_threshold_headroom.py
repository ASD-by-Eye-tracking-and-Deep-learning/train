"""Diagnostic: how much headroom does threshold tuning offer over the fixed
0.5 decision threshold currently used? Run from train/:
`uv run python scripts/analyze_threshold_headroom.py`

Uses only the already-saved ROC fpr/tpr arrays in artifacts/*_cv.json (no
retraining, no GPU) -- computes the Youden's-J-optimal point on each fold's
TEST-set ROC curve. This is an OPTIMISTIC UPPER BOUND, not a valid estimate:
picking the threshold on the test set itself is a form of leakage. A
properly validated number (threshold chosen on val, applied to test) can
only be <= what this reports. See record.md Decision #15.
"""
import glob
import json

from asd_train.metrics import mean_ci

print(f"{'config':30s} {'acc@0.5':>10} {'mean_youdenJ':>14} {'balacc@best (optimistic)':>26}")
for path in sorted(glob.glob("artifacts/*_cv.json")):
    d = json.load(open(path))
    accs, js, balaccs = [], [], []
    for f in d["folds"]:
        c = f["clinical"]
        fpr, tpr = c["roc_fpr"], c["roc_tpr"]
        best_j, best_idx = -1, 0
        for idx, (fp, tp) in enumerate(zip(fpr, tpr)):
            j = tp - fp
            if j > best_j:
                best_j, best_idx = j, idx
        accs.append(f["test_accuracy"])
        js.append(best_j)
        balaccs.append((tpr[best_idx] + (1 - fpr[best_idx])) / 2)
    ma, _ = mean_ci(accs)
    mj, _ = mean_ci(js)
    mb, _ = mean_ci(balaccs)
    print(f"{d['config_name']:30s} {ma:>10.3f} {mj:>14.3f} {mb:>26.3f}")

print("\nConclusion: optimistic best-case balanced accuracy (~0.70-0.75) is barely above the")
print("default-threshold accuracy already achieved (~0.66-0.73) -- threshold tuning is NOT a")
print("meaningful lever here. Not pursuing a properly-validated (val-selected) version.")
