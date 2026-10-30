"""Grad-CAM / Grad-CAM++ and the model-randomization sanity check (T13), plus
the channel-value test inside high-activation regions (T14).

Both model families call ``model.backbone.forward_features(x)``, whose output
IS the last convolutional feature map, for every architecture (MobileNetV4,
ConvNeXtV2, InceptionV3) — so that call is wrapped to capture it, with no
per-arch layer lookup. (A forward hook on ``model.backbone`` would never fire:
``forward_features`` bypasses the module's ``__call__``.)

Target logit: the models emit one logit for P(non-ASD) (ASD_LABEL = 0), so the
explained score is ``+logit`` for images predicted non-ASD and ``-logit`` for
images predicted ASD — the map always explains the class the model chose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .metrics import ASD_LABEL


class GradCAM:
    def __init__(self, model: torch.nn.Module, plus_plus: bool = False):
        self.model = model
        self.plus_plus = plus_plus
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        backbone = model.backbone
        original = backbone.forward_features

        def capture(x):
            out = original(x)
            self._acts = out
            out.register_hook(self._store_grad)
            return out

        backbone.forward_features = capture
        self._restore = lambda: delattr(backbone, "forward_features")

    def _store_grad(self, grad: torch.Tensor) -> None:
        self._grads = grad

    def remove(self) -> None:
        self._restore()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cams at feature-map resolution, shape (B, h, w), scaled to
        [0, 1] per image; logits, shape (B,))."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        sign = torch.where(logits >= 0, 1.0, -1.0)
        (logits * sign).sum().backward()
        acts, grads = self._acts.detach(), self._grads.detach()
        self._acts = self._grads = None  # drop graph references between batches
        if self.plus_plus:
            g2, g3 = grads.pow(2), grads.pow(3)
            denom = 2 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return cam.cpu().numpy(), logits.detach().cpu().numpy()


def randomize_weights(model: torch.nn.Module, seed: int = 0) -> None:
    """Full model randomization (Adebayo et al., 2018): re-initialise every
    layer that has parameters, so nothing learned (neither ImageNet nor ASD
    training) survives."""
    torch.manual_seed(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters") and any(True for _ in m.parameters(recurse=False)):
            m.reset_parameters()


def spearman_per_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rank correlation between two batches of maps, one value per image.
    NaN when either map is constant (common for a randomized net after ReLU:
    62% of InceptionV3 maps on 2026-10-30). Counting those as 0 would make
    the sanity check pass by construction, so callers take the median over
    the non-NaN values and report how many there were."""
    out = []
    for x, y in zip(a.reshape(len(a), -1), b.reshape(len(b), -1)):
        rho = stats.spearmanr(x, y).statistic if x.std() > 0 and y.std() > 0 else np.nan
        out.append(float(rho))
    return np.array(out)


def upsample(cams: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(cams).unsqueeze(1)
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).squeeze(1).numpy()


@dataclass
class RegionChannelStats:
    """Mean R (velocity), G (acceleration), B (jerk) over the trajectory pixels
    inside each image's top-``top_frac`` Grad-CAM region. NaN when that region
    holds no trajectory pixel (only black background)."""

    rgb: np.ndarray  # (N, 3)
    labels: np.ndarray  # (N,)


def region_channel_means(
    images: np.ndarray, cams_full: np.ndarray, labels: np.ndarray, top_frac: float = 0.15, bg_thresh: float = 0.02
) -> RegionChannelStats:
    """``images`` (N, 3, H, W) in [0, 1] (ToTensor output, no normalisation),
    ``cams_full`` (N, H, W) already upsampled to the image size. Background
    pixels (all channels <= ``bg_thresh``) are excluded: they encode no motion,
    so including them would just measure how much empty space the map covers."""
    out = np.full((len(images), 3), np.nan)
    for i, (img, cam) in enumerate(zip(images, cams_full)):
        region = cam >= np.quantile(cam, 1 - top_frac)
        fg = img.max(axis=0) > bg_thresh
        mask = region & fg
        if mask.any():
            out[i] = img[:, mask].mean(axis=1)
    return RegionChannelStats(rgb=out, labels=labels)


def mann_whitney_by_channel(s: RegionChannelStats) -> list[dict]:
    """ASD vs TD per channel, two-sided; also returns medians and the
    rank-biserial effect size (positive = ASD higher)."""
    rows = []
    for c, name in enumerate(("R_velocity", "G_acceleration", "B_jerk")):
        v = s.rgb[:, c]
        ok = ~np.isnan(v)
        asd, td = v[ok & (s.labels == ASD_LABEL)], v[ok & (s.labels != ASD_LABEL)]
        if len(asd) < 2 or len(td) < 2:
            rows.append({"channel": name, "n_asd": len(asd), "n_td": len(td), "p": float("nan")})
            continue
        res = stats.mannwhitneyu(asd, td, alternative="two-sided")
        rows.append({
            "channel": name,
            "n_asd": len(asd),
            "n_td": len(td),
            "median_asd": float(np.median(asd)),
            "median_td": float(np.median(td)),
            "U": float(res.statistic),
            "p": float(res.pvalue),
            "rank_biserial": float(2 * res.statistic / (len(asd) * len(td)) - 1),
        })
    return rows
