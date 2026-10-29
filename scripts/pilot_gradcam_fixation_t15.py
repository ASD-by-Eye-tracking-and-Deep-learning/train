"""T15 — exploratory pilot (N=3 children: 11-ASD, 20-ASD, 50-TD): does
Grad-CAM attend to fixations or to saccades in the real gaze? Run from
train/ after run_explain_t13_t14.py has written the fold maps:

    uv run python scripts/pilot_gradcam_fixation_t15.py --config NAME

EXPLORATORY, NOT GENERALISABLE: three children.

For each image of the pilot children: take the child's best (file, trial)
segment from the T11 geometry report, find the MAX_POINTS-sample window that
reproduces the image (symmetric chamfer, same coordinate convention T11
confirmed), and read SMI's own ``Category Right`` label (Fixation/Saccade) for
each sample of that window. The Grad-CAM map of that image comes from the CV
fold in which the image was a TEST image (model never saw the child). Report,
per image, the mean Grad-CAM value at fixation samples vs at saccade samples,
plus an overlay figure.
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats
from scipy.ndimage import distance_transform_edt as edt
from torchvision import datasets

from asd_train.configs import PRESETS_WITH_INCEPTIONV3
from asd_train.rendering import MAX_POINTS

PILOT = (11, 20, 50)
RAW_DIR = "../../data/mahmoud-dataset/Eye-tracking Output"
GEOM_REPORT = "artifacts/rendering_t11/participant_geometry.md"
EXPLAIN_DIR = "artifacts/explain"
OUT_DIR = "artifacts/pilot_t15"
AX = (80.0, 576.0, 57.6, 427.2)  # Matplotlib default axes box on 640x480 (T11)
FG = 30

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = PRESETS_WITH_INCEPTIONV3[args.config]


def to_px(x, y):
    x0, x1, y0, y1 = AX
    return x0 + x / 1280 * (x1 - x0), y1 - y / 1024 * (y1 - y0)


def raster(px, py):
    im = Image.new("1", (640, 480), 0)
    ImageDraw.Draw(im).line(list(zip(px.tolist(), py.tolist())), fill=1, width=1)
    return np.asarray(im, dtype=bool)


def segment(csv_name, trial, pid):
    xc, yc = "Point of Regard Right X [px]", "Point of Regard Right Y [px]"
    df = pd.read_csv(os.path.join(RAW_DIR, csv_name), usecols=["Trial", "Participant", "Category Right", xc, yc], low_memory=False)
    for c in ("Participant", xc, yc):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["Trial"] == trial) & (df["Participant"] == pid)].dropna(subset=[xc, yc])
    df = df[(df[xc] > 0) & (df[yc] > 0)]
    px, py = to_px(df[xc].to_numpy(), df[yc].to_numpy())
    return px, py, df["Category Right"].astype(str).to_numpy()


geom = dict(
    (name, (csv_name, trial))
    for name, csv_name, trial in re.findall(r"^\| (\S+\.png) \| (\S+?)/(\S+) \| [\d.]+ \|$", open(GEOM_REPORT, encoding="utf-8").read(), re.M)
)

dataset = datasets.ImageFolder(os.path.join("../../data/mahmoud-dataset", os.path.basename(cfg.dataset_path.rstrip("/"))))
names = [os.path.basename(p) for p, _ in dataset.samples]
cam_of = {}
for fold in range(5):
    p = os.path.join(EXPLAIN_DIR, args.config, f"fold{fold}_cams.npz")
    if os.path.exists(p):
        z = np.load(p)
        for i, idx in enumerate(z["test_idx"]):
            cam_of[names[idx]] = (fold, z["cams"][i], int(z["labels"][i]), int(z["preds"][i]))

os.makedirs(OUT_DIR, exist_ok=True)
rows, pairs = [], []
for name in sorted(n for n in names if int(n.rsplit("_", 1)[1].split(".")[0]) in PILOT):
    pid = int(name.rsplit("_", 1)[1].split(".")[0])
    if name not in geom or name not in cam_of:
        rows.append(f"| {pid} | {name} | — | — | — | — | — | no {'segment' if name not in geom else 'Grad-CAM map'} |")
        continue
    real = np.asarray(Image.open(dataset.samples[names.index(name)][0]).convert("RGB"))
    real_m = real.max(axis=2) > FG
    real_dt = edt(~real_m)
    px, py, cat = segment(*geom[name], pid)
    best = (np.inf, 0)
    for s in range(0, max(len(px) - MAX_POINTS, 0) + 1, 5):
        r = raster(px[s : s + MAX_POINTS], py[s : s + MAX_POINTS])
        if r.any():
            best = min(best, ((real_dt[r].mean() + edt(~r)[real_m].mean()) / 2, s))
    d, s = best
    wx, wy, wc = px[s : s + MAX_POINTS], py[s : s + MAX_POINTS], cat[s : s + MAX_POINTS]

    fold, cam, label, pred = cam_of[name]
    cam_img = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((640, 480), Image.BILINEAR), dtype=float) / 255
    xi = np.clip(wx.round().astype(int), 0, 639)
    yi = np.clip(wy.round().astype(int), 0, 479)
    vals = cam_img[yi, xi]
    fix, sac = vals[wc == "Fixation"], vals[wc == "Saccade"]
    if len(fix) and len(sac):
        pairs.append((fix.mean(), sac.mean()))
    rows.append(
        f"| {pid} | {name} | {fold} | {'ASD' if label == 0 else 'TD'} / {'ASD' if pred == 0 else 'TD'} | {d:.2f} "
        f"| {fix.mean() if len(fix) else float('nan'):.3f} (n={len(fix)}) | {sac.mean() if len(sac) else float('nan'):.3f} (n={len(sac)}) | |"
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(real)
    ax[0].set_title(f"{name} (real)", fontsize=8)
    ax[1].imshow(real)
    ax[1].imshow(cam_img, cmap="jet", alpha=0.4)
    ax[1].scatter(wx[wc == "Fixation"], wy[wc == "Fixation"], s=4, c="white", label="Fixation")
    ax[1].scatter(wx[wc == "Saccade"], wy[wc == "Saccade"], s=4, c="magenta", label="Saccade")
    ax[1].legend(fontsize=6, loc="lower right")
    ax[1].set_title(f"Grad-CAM (fold {fold}) + SMI categories, chamfer {d:.1f}px", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name[:-4]}_overlay.png"), dpi=110)
    plt.close(fig)
    print(rows[-1], flush=True)

p = np.array(pairs)
if len(p):
    diff = p[:, 1] - p[:, 0]
    w = stats.wilcoxon(p[:, 1], p[:, 0]).pvalue if len(p) > 5 else float("nan")
    summary = (
        f"Summary over {len(p)} images with both labels: saccade > fixation in {int((diff > 0).sum())}; "
        f"median (saccade − fixation) = {np.median(diff):+.3f}; Wilcoxon p = {w:.3g} "
        "(images of the same child are not independent — descriptive only).\n"
    )
else:
    summary = "No image had both fixation and saccade samples.\n"
print(summary)

with open(os.path.join(OUT_DIR, "pilot_report.md"), "w", encoding="utf-8") as f:
    f.write(
        f"# T15 — pilot: Grad-CAM vs SMI fixation/saccade labels ({args.config})\n\n"
        "**Exploratory, N = 3 children (11-ASD, 20-ASD, 50-TD). Not generalisable.**\n\n"
        "Chamfer = how well the recovered 200-sample window reproduces the image (px at 640x480; "
        "large values mean the window, and so the fixation labels, are uncertain for that image).\n\n"
        "| Child | Image | Test fold | True / Pred | Window chamfer | Mean Grad-CAM at fixations | Mean Grad-CAM at saccades | Note |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n" + summary
    )
print(f"OK: {OUT_DIR}/pilot_report.md")
