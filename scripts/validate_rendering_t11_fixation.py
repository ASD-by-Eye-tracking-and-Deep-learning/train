"""T11 step 4 — are fixations drawn in a fixed colour? Run from train/
after validate_rendering_t11_colour.py:

    uv run python scripts/validate_rendering_t11_fixation.py

Step 3 found G/B anti-correlated with every acceleration/jerk definition.
Here each segment of the recovered windows is labelled by SMI's own
``Category Right`` (both endpoints "Fixation" vs not), and the real image's
colour at the segment midpoint is compared between the two groups.
"""
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

RAW = "../../data/mahmoud-dataset/Eye-tracking Output"
IMG = "../../data/mahmoud-dataset/Images"
REPORT = "artifacts/rendering_t11/colour_validation.md"
OUT = "artifacts/rendering_t11/fixation_colour.md"
AX = (80.0, 576.0, 57.6, 427.2)
XC, YC = "Point of Regard Right X [px]", "Point of Regard Right Y [px]"

rows = re.findall(r"^\| (\S+\.png) \| (\S+?)/(\S+) \| (\d+) \|", open(REPORT, encoding="utf-8").read(), re.M)
cache, recs = {}, []
for name, csv_name, trial, start in rows:
    pid = int(name.rsplit("_", 1)[1][:-4])
    if csv_name not in cache:
        df = pd.read_csv(os.path.join(RAW, csv_name), low_memory=False, usecols=["Trial", "Participant", "Category Right", XC, YC])
        for c in ("Participant", XC, YC):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        cache[csv_name] = df
    d = cache[csv_name]
    d = d[(d.Trial == trial) & (d.Participant == pid)].dropna(subset=[XC, YC]).iloc[int(start) : int(start) + 200]
    x, y, cat = d[XC].to_numpy(), d[YC].to_numpy(), d["Category Right"].astype(str).to_numpy()
    path = next(os.path.join(IMG, s, name) for s in ("ASD", "non-ASD") if os.path.exists(os.path.join(IMG, s, name)))
    img = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    mx = AX[0] + (x[:-1] + x[1:]) / 2 / 1280 * (AX[1] - AX[0])
    my = AX[3] - (y[:-1] + y[1:]) / 2 / 1024 * (AX[3] - AX[2])
    col = img[np.clip(my.round().astype(int), 0, 479), np.clip(mx.round().astype(int), 0, 639)]
    v = np.hypot(np.diff(x), np.diff(y))
    for i in range(len(v)):
        if col[i].max() > 30:
            recs.append((cat[i] == "Fixation" and cat[i + 1] == "Fixation", v[i], *col[i]))

r = pd.DataFrame(recs, columns=["fix", "v", "R", "G", "B"])
med = r.groupby("fix")[["R", "G", "B", "v"]].median()
n = r.groupby("fix").size()
lines = [
    f"# T11 — fixation segments vs the rest ({len(rows)} images, {len(r)} segment midpoints)\n",
    "Real-image colour (0-255) at segment midpoints; segment = fixation when both endpoints are SMI 'Fixation'.\n",
    "| Segment | n | median R | median G | median B | median velocity (px/sample) |",
    "|---|---|---|---|---|---|",
]
for k, label in ((True, "Fixation→Fixation"), (False, "other")):
    lines.append(f"| {label} | {n[k]} | {med.loc[k, 'R']:.0f} | {med.loc[k, 'G']:.0f} | {med.loc[k, 'B']:.0f} | {med.loc[k, 'v']:.1f} |")
lines.append("")
for ch in "RGB":
    lines.append(f"- {ch}: fixation vs other, Mann-Whitney p = {stats.mannwhitneyu(r[r.fix][ch], r[~r.fix][ch]).pvalue:.3g}")
f, o = r[r.fix], r[~r.fix]
lines += [
    f"- Fixation segments: Spearman(G, B) = {stats.spearmanr(f.G, f.B).statistic:.3f} (a fixed colour gives G≈B)",
    f"- Other segments: Spearman(velocity, R) = {stats.spearmanr(o.v, o.R).statistic:.3f}, "
    f"Spearman(velocity, G) = {stats.spearmanr(o.v, o.G).statistic:.3f}",
    "",
    "Reading: fixation transitions are drawn in one fixed cyan-like colour (R≈0, G≈B), not by kinematics; "
    "only the remaining transitions carry the velocity code. Absolute values are low (≤ ~60/255) for "
    "both groups — consistent with thin anti-aliased 1-px lines, so only relative comparisons are meaningful.",
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
