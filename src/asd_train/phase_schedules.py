"""
Per-backbone backbone-freezing / LR phase schedules, called once per epoch
from the training loop as ``schedule_fn(model, optimizer, epoch, base_lr)``.

These are kept as distinct functions rather than one generic parametrized
schedule because they encode genuinely different logic (not just different
constants) discovered by reading the original notebooks: MobileNetV4's
schedule hardcodes absolute LR values and selectively unfreezes named
blocks; ConvNeXtV2's schedule freezes/unfreezes the whole backbone at once
and scales a base LR. The ConvNeXtV2 ghost-only and ghost+CBAM notebooks
also used different breakpoints/multipliers from each other, so both are
kept.
"""
from __future__ import annotations

import torch


def update_training_phase_mobilenetv4(model, optimizer, epoch: int, base_lr: float) -> None:
    """Used by both ghost/mobilenetv4_small and ghost+cbam/mobilenetv4_small.
    ``base_lr`` is accepted for interface consistency but unused — this
    schedule sets absolute LR values, matching the original notebooks."""
    if epoch == 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        new_lr = 5e-5
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        print(f"\n>>> Phase 1: Warm-up | LR = {new_lr}")
    elif epoch == 11:
        for name, param in model.backbone.named_parameters():
            if any(f"blocks.{i}" in name for i in (5, 6)):
                param.requires_grad = True
        new_lr = 2e-5
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        print(f"\n>>> Phase 2: Deep Fine-tuning | LR = {new_lr}")
    elif epoch == 36:
        for param in model.backbone.parameters():
            param.requires_grad = True
        new_lr = 5e-6
        for group in optimizer.param_groups:
            group["lr"] = new_lr
        print(f"\n>>> Phase 3: Final Tweak | LR = {new_lr}")


def update_training_phase_convnextv2_ghost(model, optimizer, epoch: int, base_lr: float) -> float:
    """Used by ghost/convnexttiny_v2 only. Phase 3 LR is intentionally
    *higher* than phase 2 (0.3x vs 0.1x) — this matches the original
    notebook exactly, not a typo."""
    if epoch < 5:
        for param in model.backbone.parameters():
            param.requires_grad = False
        current_lr = base_lr
    elif epoch < 25:
        for param in model.backbone.parameters():
            param.requires_grad = True
        current_lr = base_lr * 0.1
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True
        current_lr = base_lr * 0.3

    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


def freeze_backbone_always(model, optimizer, epoch: int, base_lr: float) -> None:
    """T9 ablation baseline: backbone frozen for the entire run (linear
    probe on frozen ImageNet features), contrasted against the staged
    unfreeze schedules above to isolate whether fine-tuning the backbone
    at all contributes over just training the head."""
    if epoch == 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for group in optimizer.param_groups:
            group["lr"] = base_lr


def update_training_phase_convnextv2_ghostcbam(model, optimizer, epoch: int, base_lr: float) -> float:
    """Used by ghost+cbam/convnexttiny_v2 only — different breakpoints and
    multipliers from the ghost-only ConvNeXt schedule (monotonically
    decreasing, unlike the ghost-only version)."""
    if epoch < 5:
        for param in model.backbone.parameters():
            param.requires_grad = False
        current_lr = base_lr
    elif epoch < 20:
        for param in model.backbone.parameters():
            param.requires_grad = True
        current_lr = base_lr * 0.5
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True
        current_lr = base_lr * 0.1

    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


def _unfreeze_backbone_from(backbone, start_name: str) -> None:
    """Freeze every top-level child before ``start_name`` and unfreeze from
    it to the end. timm's InceptionV3 is a flat sequence of children
    (Conv2d_1a_3x3, ..., Mixed_7c), so ordering is by module registration
    order — no ``features`` container exists to index into."""
    children = list(backbone.named_children())
    names = [n for n, _ in children]
    if start_name not in names:
        raise ValueError(f"{start_name!r} not found in backbone children: {names}")
    start = names.index(start_name)
    for i, (_, module) in enumerate(children):
        for param in module.parameters():
            param.requires_grad = i >= start


def update_training_phase_inceptionv3_torch(model, optimizer, epoch: int, base_lr: float) -> float:
    """T12 InceptionV3-in-PyTorch schedule: linear probe on frozen ImageNet
    features -> unfreeze from ``model.unfreeze_from`` (Keras
    block5a/block6a/mixed7 equivalent) -> full fine-tune, LR decreasing at
    each transition. The original Keras-Tuner track searched this whole
    space per trial; here the breakpoints are fixed so the run is
    reproducible across the 5 CV folds."""
    if epoch < 5:
        for param in model.backbone.parameters():
            param.requires_grad = False
        current_lr = base_lr
    elif epoch < 20:
        _unfreeze_backbone_from(model.backbone, model.unfreeze_from)
        current_lr = base_lr * 0.5
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True
        current_lr = base_lr * 0.1

    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr
