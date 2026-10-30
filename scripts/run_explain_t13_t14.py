"""T13 + T14 — Grad-CAM/Grad-CAM++ with a mandatory model-randomization
sanity check, then the channel-value test inside high-activation regions.
Run from train/ after the CV runs that produced the fold weights:

    uv run python scripts/run_explain_t13_t14.py --configs NAME [NAME ...]

Per config and fold, uses the SAME grouped 5-fold split as training
(random_state=42) and that fold's ``artifacts/{name}_fold{i}.pth``, so every
map is computed on participants the model never saw.

T13 sanity check (Adebayo et al., 2018): Spearman correlation between each
image's map from the trained model and from a fully re-initialised copy.
A config PASSES when the median correlation is below SANITY_MAX_RHO — i.e.
the map depends on learned weights. Only passing configs go on to T14.

T14: on correctly classified test images, mean R/G/B (velocity/acceleration/
jerk) over trajectory pixels inside the top-15% Grad-CAM region; ASD vs TD by
Mann-Whitney U per channel, per fold and pooled. Deviation from the plan,
stated in the report: the plan said "across LOSO folds", but a LOSO test fold
holds one child (one class), so a between-group test cannot run inside it; the
5 grouped-CV folds each contain both classes and are used instead.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from asd_train.configs import PRESETS_WITH_INCEPTIONV3
from asd_train.data import build_torch_transform, get_participant_groups, make_group_kfold_splits
from asd_train.explain import (
    GradCAM,
    RegionChannelStats,
    mann_whitney_by_channel,
    randomize_weights,
    region_channel_means,
    spearman_per_image,
    upsample,
)
from asd_train.metrics import ASD_LABEL
from asd_train.train_torch import build_model

DATASET_ROOT = "../../data/mahmoud-dataset"
WEIGHTS_DIR = "artifacts"
OUT_DIR = "artifacts/explain"
K = 5
SANITY_MAX_RHO = 0.5
BATCH = 16

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", required=True)
parser.add_argument("--top-frac", type=float, default=0.15)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maps_for(model, x, plus_plus):
    cam = GradCAM(model, plus_plus=plus_plus)
    try:
        return cam(x.clone().requires_grad_(True))
    finally:
        cam.remove()


def save_examples(path, images, cams, labels, preds, n=6):
    idx = np.linspace(0, len(images) - 1, min(n, len(images))).astype(int)
    fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6))
    for col, i in enumerate(np.atleast_1d(idx)):
        img = images[i].transpose(1, 2, 0)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true={'ASD' if labels[i] == ASD_LABEL else 'TD'} pred={'ASD' if preds[i] == ASD_LABEL else 'TD'}", fontsize=8)
        axes[1, col].imshow(img)
        axes[1, col].imshow(cams[i], cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


report = ["# T13/T14 — Grad-CAM sanity check and channel test\n"]
summary = {}
for name in args.configs:
    cfg = PRESETS_WITH_INCEPTIONV3[name]
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    dataset_path = os.path.join(DATASET_ROOT, os.path.basename(cfg.dataset_path.rstrip("/")))
    dataset = datasets.ImageFolder(dataset_path, transform=build_torch_transform(cfg.img_size, False, cfg.grayscale))
    splits = make_group_kfold_splits(dataset, k=K, random_state=42)
    participant = np.array(get_participant_groups(dataset))

    rho = {"gradcam": [], "gradcampp": []}
    t14_folds, pooled_rgb, pooled_lab, pooled_pid, pooled_ctrl = [], [], [], [], []
    for fold, (_, _, test_idx) in enumerate(splits):
        wpath = os.path.join(WEIGHTS_DIR, f"{name}_fold{fold}.pth")
        if not os.path.exists(wpath):
            print(f"[{name}] fold {fold}: no weights at {wpath}, skipped")
            continue
        model = build_model(cfg, device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.eval()
        rand_model = build_model(cfg, device)
        rand_model.load_state_dict(model.state_dict())
        randomize_weights(rand_model, seed=fold)
        rand_model.eval()
        # Grad-CAM needs gradients w.r.t. the feature map only (the input
        # requires grad, so the path stays differentiable); computing every
        # parameter's gradient on two ConvNeXt copies ran the 8 GB GPU out of
        # memory on 2026-10-30.
        for m in (model, rand_model):
            m.requires_grad_(False)

        imgs, labs, cams, cams_pp, logits = [], [], [], [], []
        for s in range(0, len(test_idx), BATCH):
            batch = [dataset[i] for i in test_idx[s : s + BATCH]]
            x = torch.stack([b[0] for b in batch]).to(device)
            y = np.array([b[1] for b in batch])
            c, lg = maps_for(model, x, plus_plus=False)
            cpp, _ = maps_for(model, x, plus_plus=True)
            cr, _ = maps_for(rand_model, x, plus_plus=False)
            crpp, _ = maps_for(rand_model, x, plus_plus=True)
            rho["gradcam"] += spearman_per_image(c, cr).tolist()
            rho["gradcampp"] += spearman_per_image(cpp, crpp).tolist()
            imgs.append(x.detach().cpu().numpy())
            labs.append(y)
            cams.append(c)
            cams_pp.append(cpp)
            logits.append(lg)
        imgs, labs = np.concatenate(imgs), np.concatenate(labs)
        cams, cams_pp, logits = np.concatenate(cams), np.concatenate(cams_pp), np.concatenate(logits)
        preds = np.where(logits >= 0, 1 - ASD_LABEL, ASD_LABEL)
        cams_full = upsample(cams, cfg.img_size)
        np.savez_compressed(
            os.path.join(out, f"fold{fold}_cams.npz"),
            test_idx=np.array(test_idx), labels=labs, preds=preds, cams=cams, cams_pp=cams_pp,
        )
        save_examples(os.path.join(out, f"fold{fold}_examples.png"), imgs, cams_full, labs, preds)

        correct = preds == labs
        s = region_channel_means(imgs[correct], cams_full[correct], labs[correct], top_frac=args.top_frac)
        t14_folds.append({"fold": fold, "n_correct": int(correct.sum()), "tests": mann_whitney_by_channel(s)})
        pooled_rgb.append(s.rgb)
        pooled_lab.append(s.labels)
        pooled_pid.append(participant[np.array(test_idx)][correct])
        # Control: same statistic over the WHOLE trajectory (top_frac=1.0). If
        # ASD/TD already differ there, the Grad-CAM-region result reflects the
        # data, not where the model looks.
        pooled_ctrl.append(region_channel_means(imgs[correct], cams_full[correct], labs[correct], top_frac=1.0).rgb)
        print(f"[{name}] fold {fold}: {len(test_idx)} test images, {int(correct.sum())} correct", flush=True)
        del model, rand_model
        torch.cuda.empty_cache()

    if not rho["gradcam"]:
        report.append(f"## {name}\n\nNo fold weights found — not evaluated.\n")
        continue

    med = {k: float(np.median(v)) for k, v in rho.items()}
    passed = {k: v < SANITY_MAX_RHO for k, v in med.items()}
    entry = {"sanity_median_spearman": med, "sanity_pass": passed, "sanity_threshold": SANITY_MAX_RHO}
    report.append(
        f"## {name}\n\n### T13 — model randomization test\n\n"
        "| Method | Median Spearman (trained vs randomized) | Pass (< "
        f"{SANITY_MAX_RHO}) |\n|---|---|---|\n"
        + "\n".join(f"| {k} | {med[k]:.3f} | {'PASS' if passed[k] else 'FAIL'} |" for k in med)
        + "\n"
    )

    if passed["gradcam"]:
        rgb_all, lab_all, pid_all = np.concatenate(pooled_rgb), np.concatenate(pooled_lab), np.concatenate(pooled_pid)
        pooled = mann_whitney_by_channel(RegionChannelStats(rgb_all, lab_all))
        # Primary: one value per child (mean over that child's images) — images
        # of the same child are not independent, so the image-level p-value
        # overstates the evidence.
        pids = np.unique(pid_all)
        child_rgb = np.array([np.nanmean(rgb_all[pid_all == p], axis=0) for p in pids])
        child_lab = np.array([lab_all[pid_all == p][0] for p in pids])
        per_child = mann_whitney_by_channel(RegionChannelStats(child_rgb, child_lab))
        ctrl_all = np.concatenate(pooled_ctrl)
        child_ctrl = np.array([np.nanmean(ctrl_all[pid_all == p], axis=0) for p in pids])
        control = mann_whitney_by_channel(RegionChannelStats(child_ctrl, child_lab))
        entry["t14_per_child"] = per_child
        entry["t14_control_whole_trajectory"] = control
        entry["t14_pooled_images"] = pooled
        entry["t14_folds"] = t14_folds
        lines = [
            f"### T14 — channel values in top-{int(args.top_frac * 100)}% Grad-CAM region, correctly classified images\n",
        ]
        header = [
            "| Channel | n ASD | n TD | median ASD | median TD | p (Mann-Whitney) | rank-biserial |",
            "|---|---|---|---|---|---|---|",
        ]
        for title, rows_ in (
            ("**Primary — one value per child** (mean over the child's correctly classified test images):\n", per_child),
            ("\nControl — same per-child test over the whole trajectory, not just the Grad-CAM region:\n", control),
            ("\nSecondary — image level, pooled over folds (images of one child are not independent; p is optimistic):\n", pooled),
        ):
            lines += [title] + header
            for r in rows_:
                lines.append(
                    f"| {r['channel']} | {r['n_asd']} | {r['n_td']} | {r.get('median_asd', float('nan')):.4f} "
                    f"| {r.get('median_td', float('nan')):.4f} | {r['p']:.4g} | {r.get('rank_biserial', float('nan')):+.3f} |"
                )
        lines.append("\nPer-fold consistency (folds with p < 0.05 / folds tested, and the sign of the effect):\n")
        for c in ("R_velocity", "G_acceleration", "B_jerk"):
            per = [t for f in t14_folds for t in f["tests"] if t["channel"] == c and not np.isnan(t["p"])]
            sig = [t for t in per if t["p"] < 0.05]
            signs = "".join("+" if t["rank_biserial"] > 0 else "-" for t in per)
            lines.append(f"- {c}: {len(sig)}/{len(per)} folds significant; effect signs per fold: `{signs}`")
        lines.append(
            "\nNote: the plan said 'across LOSO folds', but each LOSO test fold contains one child (one class), "
            "so a between-group test cannot run within it; the 5 grouped-CV folds (both classes each) are used.\n"
        )
        report.append("\n".join(lines) + "\n")
    else:
        report.append("Grad-CAM failed the sanity check — maps not used as evidence, T14 not run for this config.\n")

    summary[name] = entry
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)

with open(os.path.join(OUT_DIR, "explain_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\n".join(report))
print(f"\nOK: report written to {OUT_DIR}/explain_report.md")
