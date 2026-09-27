"""
FMMix1 feature-level augmentation (Nguyen, Tran & Dinh, IEEE Access 2024) and
the basic pre-augmentation used across all training notebooks.

Two distinct FMMix1 implementations existed in the original notebooks and are
kept separate here (not merged) since they produce different behavior and
each is tied to the runs that actually used them:

- ``fmmix1_multimode``: used by the InceptionV3 (TensorFlow) notebook. Supports
  4 mask-area modes; the notebook only ever used mode 1 (fixed box size = alpha).
- ``fmmix1``: used by all 12 Ghost/CBAM (PyTorch) configs. Single mode, box
  size randomized per-sample (``sqrt(alpha * rand(...))`` instead of a fixed
  ``alpha``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FMMixArgs:
    alpha: float = 0.05
    mask_area_mode: int = 1


def augment_pre_fmmix1(images: torch.Tensor, labels: torch.Tensor):
    """Random horizontal flip + brightness/contrast jitter, applied pre-FMMix1."""
    B = images.size(0)
    flip_mask = torch.rand(B, device=images.device) < 0.5
    for i in range(B):
        if flip_mask[i]:
            images[i] = torch.flip(images[i], dims=[2])
    brightness_delta = torch.empty(B, 1, 1, 1, device=images.device).uniform_(-0.1, 0.1)
    images = images + brightness_delta
    mean = images.mean(dim=(2, 3), keepdim=True)
    contrast_factor = torch.empty(B, 1, 1, 1, device=images.device).uniform_(0.9, 1.1)
    images = (images - mean) * contrast_factor + mean
    images = torch.clamp(images, 0.0, 1.0)
    return images, labels


def fmmix1(args: FMMixArgs, x: torch.Tensor, target: torch.Tensor):
    """FMMix1 as used by all 12 Ghost/CBAM PyTorch configs."""
    B, C, H, W = x.shape
    shuffle_idx = torch.randperm(B)
    x_shuffle = x[shuffle_idx]
    max_val_indices = torch.argmax(x_shuffle.view(B, C, -1), dim=2)
    max_indices_unraveled = torch.stack((max_val_indices // W, max_val_indices % W), dim=-1)

    lambdas = torch.sqrt(args.alpha * torch.rand(B, 1, 1, 1, device=x.device))
    Wb = (lambdas * W).squeeze(-1)
    Hb = (lambdas * H).squeeze(-1)

    x_range = torch.arange(W, device=x.device)[None, :].expand(B, C, H, W)
    y_range = torch.arange(H, device=x.device)[:, None].expand(B, C, H, W)
    x_min = torch.clamp(max_indices_unraveled[..., 1][..., None, None] - Wb[..., None] // 2, 0, W)
    y_min = torch.clamp(max_indices_unraveled[..., 0][..., None, None] - Hb[..., None] // 2, 0, H)
    x_max = torch.clamp(x_min + Wb[..., None], 0, W)
    y_max = torch.clamp(y_min + Hb[..., None], 0, H)
    masks = (x_range >= x_min) & (x_range < x_max) & (y_range >= y_min) & (y_range < y_max)
    x = x * ~masks + x_shuffle * masks

    target_shuffled = target[shuffle_idx]
    p_src = torch.sum(masks, dim=[1, 2, 3]) / (C * W * H)
    p_tar = 1 - p_src
    return x, [2, [p_tar, p_src], [target, target_shuffled]]


def fmmix1_multimode(args: FMMixArgs, x: torch.Tensor, target: torch.Tensor):
    """FMMix1 as used by the InceptionV3 (TensorFlow, via tf.numpy_function bridge)
    notebook. Supports 4 mask-area modes; only mode 1 (fixed box size) was
    actually used."""
    B, C, H, W = x.shape
    lam = args.alpha

    shuffle_idx = torch.randperm(B)
    x_shuffle = x[shuffle_idx]

    max_val_indices = torch.argmax(x_shuffle.view(B, C, -1), dim=2)
    max_indices_unraveled = torch.stack((max_val_indices // W, max_val_indices % W), dim=-1)

    if args.mask_area_mode in (0, 1):
        lambdas = torch.sqrt(lam * torch.ones(B, 1, 1, 1, device=x.device))
        Wb = (lambdas * W).squeeze(-1)
        Hb = (lambdas * H).squeeze(-1)
    else:
        Wb = torch.rand(B, 1, 1, device=x.device) * W
        if args.mask_area_mode == 2:
            lambdas = lam * torch.ones(B, 1, 1, device=x.device)
            Hb = (lambdas * W * H) / Wb
        else:
            lambdas = lam * torch.rand(B, 1, 1, device=x.device)
            max_Hb = (lambdas * W * H) / Wb
            Hb = torch.rand(B, 1, 1, device=x.device) * max_Hb
        Hb = torch.min(Hb, H * torch.ones_like(Hb))

    x_range = torch.arange(W, device=x.device)[None, :].expand(B, C, H, W)
    y_range = torch.arange(H, device=x.device)[:, None].expand(B, C, H, W)
    x_min = torch.clamp(max_indices_unraveled[..., 1][..., None, None] - Wb[..., None] // 2, 0, W)
    y_min = torch.clamp(max_indices_unraveled[..., 0][..., None, None] - Hb[..., None] // 2, 0, H)
    x_max = torch.clamp(x_min + Wb[..., None], 0, W)
    y_max = torch.clamp(y_min + Hb[..., None], 0, H)

    masks = (x_range >= x_min) & (x_range < x_max) & (y_range >= y_min) & (y_range < y_max)
    x = x * ~masks + x_shuffle * masks

    target_shuffled = target[shuffle_idx]
    p_src = torch.sum(masks, dim=[1, 2, 3]) / (C * W * H)
    p_tar = 1 - p_src

    return x, [2, [p_tar, p_src], [target, target_shuffled]]
