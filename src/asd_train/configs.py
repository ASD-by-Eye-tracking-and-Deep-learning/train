"""
Every hyperparameter here was read directly out of the corresponding
original notebook (extracted cell-by-cell), not guessed or assumed uniform
across configs — several per-backbone/per-CBAM differences turned out to be
real (patience, max_norm, GhostModule kernel_size, whether pos_weight is
used inside the FMMix1 loss, and one structural difference: ghost-only
ConvNeXtV2 PA2 uses a single GhostBottleneck stage instead of two).

See train/README.md for how each preset maps back to its source notebook
and reported test accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from asd_train.phase_schedules import (
    freeze_backbone_always,
    update_training_phase_convnextv2_ghost,
    update_training_phase_convnextv2_ghostcbam,
    update_training_phase_mobilenetv4,
)

BACKBONES = {
    "mobilenetv4": "mobilenetv4_conv_small.e2400_r224_in1k",
    "convnextv2": "convnextv2_tiny.fcmae",
}

# Lives in the meta-repo's data/ (sibling of this train/ repo), not inside
# train/ itself — see ASD/.agents/record.md (T4) for how it's populated.
DATASET_PATH_DEFAULT = "../data/mahmoud-dataset/Images"


@dataclass
class GhostRunConfig:
    name: str
    backbone_key: Literal["mobilenetv4", "convnextv2"]
    variant: Literal["pa1", "pa2", "pa3"]
    use_cbam: bool
    phase_schedule: Callable

    # Optimizer / training loop
    lr: float
    weight_decay: float
    max_norm: float
    patience: int
    num_epochs: int = 100
    batch_size: int = 32
    img_size: int = 224
    pos_weight: float = 0.67
    use_pos_weight_in_fmmix_loss: bool = False

    # Augmentation
    use_fmmix: bool = True
    fmmix_alpha: float = 0.05
    extra_transform_augmentation: bool = False

    # Architecture
    kernel_size: int = 1
    ghost_ratio: int = 2
    bottleneck_expansion: int = 2
    bottleneck_stages: int = 2
    adapter_reduction: int = 4
    cbam_reduction: int = 16

    # LR scheduler — stepped once per epoch, *in addition to* the phase
    # schedule above (both were used together in the original notebooks).
    # "none": ghost+cbam/mobilenetv4_small used no scheduler at all.
    # "plateau": only ghostcbam_convnexttiny_pa2 used ReduceLROnPlateau
    # instead of cosine annealing — every other config used cosine.
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    scheduler_eta_min: float = 1e-6
    scheduler_plateau_factor: float = 0.5
    scheduler_plateau_patience: int = 5

    dataset_path: str = DATASET_PATH_DEFAULT

    @property
    def backbone_name(self) -> str:
        return BACKBONES[self.backbone_key]


def _mobilenetv4_ghost(variant: str, **overrides) -> GhostRunConfig:
    base = dict(
        backbone_key="mobilenetv4",
        variant=variant,
        use_cbam=False,
        phase_schedule=update_training_phase_mobilenetv4,
        lr=1e-5,
        weight_decay=1e-4,
        max_norm=0.1,
        patience=10,
        fmmix_alpha=0.05,
        extra_transform_augmentation=True,
        pos_weight=0.67,
        use_pos_weight_in_fmmix_loss=False,
        kernel_size=3,
        ghost_ratio=2,
    )
    base.update(overrides)
    return GhostRunConfig(name=f"ghost_mobilenetv4_{variant}", **base)


def _mobilenetv4_ghostcbam(variant: str, **overrides) -> GhostRunConfig:
    base = dict(
        backbone_key="mobilenetv4",
        variant=variant,
        use_cbam=True,
        phase_schedule=update_training_phase_mobilenetv4,
        lr=1e-5,
        weight_decay=1e-4,
        max_norm=5.0,
        patience=15,
        fmmix_alpha=0.05,
        extra_transform_augmentation=False,
        pos_weight=0.67,
        use_pos_weight_in_fmmix_loss=False,
        kernel_size=1,
        ghost_ratio=2,
        cbam_reduction=16,
        scheduler="none",
    )
    base.update(overrides)
    return GhostRunConfig(name=f"ghostcbam_mobilenetv4_{variant}", **base)


def _convnextv2_ghost(variant: str, **overrides) -> GhostRunConfig:
    base = dict(
        backbone_key="convnextv2",
        variant=variant,
        use_cbam=False,
        phase_schedule=update_training_phase_convnextv2_ghost,
        lr=1e-5,
        weight_decay=0.0,
        max_norm=0.5,
        patience=6,
        fmmix_alpha=0.01,
        extra_transform_augmentation=False,
        pos_weight=0.67,
        use_pos_weight_in_fmmix_loss=True,
        kernel_size=1,
        ghost_ratio=2,
    )
    base.update(overrides)
    return GhostRunConfig(name=f"ghost_convnexttiny_{variant}", **base)


def _convnextv2_ghostcbam(variant: str, **overrides) -> GhostRunConfig:
    base = dict(
        backbone_key="convnextv2",
        variant=variant,
        use_cbam=True,
        phase_schedule=update_training_phase_convnextv2_ghostcbam,
        lr=5e-6,
        weight_decay=2e-4,
        max_norm=0.3,
        patience=12,
        fmmix_alpha=0.01,
        extra_transform_augmentation=False,
        pos_weight=0.67,
        use_pos_weight_in_fmmix_loss=True,
        kernel_size=1,
        ghost_ratio=2,
        cbam_reduction=16,
    )
    base.update(overrides)
    return GhostRunConfig(name=f"ghostcbam_convnexttiny_{variant}", **base)


PRESETS: dict[str, GhostRunConfig] = {
    # --- ghost/mobilenetv4_small ---
    "ghost_mobilenetv4_pa1": _mobilenetv4_ghost("pa1"),
    "ghost_mobilenetv4_pa2": _mobilenetv4_ghost("pa2", bottleneck_expansion=4, bottleneck_stages=2),
    "ghost_mobilenetv4_pa3": _mobilenetv4_ghost("pa3", adapter_reduction=2),
    # --- ghost/convnexttiny_v2 ---
    "ghost_convnexttiny_pa1": _convnextv2_ghost("pa1", scheduler_eta_min=1e-7),
    # single-stage bottleneck: deliberate simplification in the original notebook
    "ghost_convnexttiny_pa2": _convnextv2_ghost("pa2", bottleneck_expansion=4, bottleneck_stages=1),
    "ghost_convnexttiny_pa3": _convnextv2_ghost("pa3", adapter_reduction=4),
    # --- ghost+cbam/mobilenetv4_small ---
    "ghostcbam_mobilenetv4_pa1": _mobilenetv4_ghostcbam("pa1"),
    "ghostcbam_mobilenetv4_pa2": _mobilenetv4_ghostcbam("pa2", bottleneck_expansion=2, bottleneck_stages=2),
    "ghostcbam_mobilenetv4_pa3": _mobilenetv4_ghostcbam("pa3", adapter_reduction=4),
    # --- ghost+cbam/convnexttiny_v2 ---
    "ghostcbam_convnexttiny_pa1": _convnextv2_ghostcbam("pa1", scheduler_eta_min=1e-7),
    "ghostcbam_convnexttiny_pa2": _convnextv2_ghostcbam(
        "pa2", bottleneck_expansion=2, bottleneck_stages=2,
        scheduler="plateau", scheduler_plateau_factor=0.5, scheduler_plateau_patience=5,
    ),
    "ghostcbam_convnexttiny_pa3": _convnextv2_ghostcbam("pa3", adapter_reduction=4),
}

# The best-accuracy run per notebook (see README) — what's kept on git/HF/Drive artifacts.
BEST_PRESETS = {
    "ghost_mobilenetv4": "ghost_mobilenetv4_pa2",       # 81.25%
    "ghost_convnexttiny": "ghost_convnexttiny_pa1",     # 82.81%
    "ghostcbam_mobilenetv4": "ghostcbam_mobilenetv4_pa2",   # 82.81%
    "ghostcbam_convnexttiny": "ghostcbam_convnexttiny_pa1",  # 78.12% (tied w/ pa3)
}


# T9 ablation grid (record.md Decision #10): each entry flips *exactly one*
# field off ``ghost_mobilenetv4_pa2`` (the base row) -- unlike the CBAM
# comparison you get from PRESETS' ghost_* vs ghostcbam_* pairs, which also
# change lr/max_norm/patience/scheduler/kernel_size at the same time and so
# can't isolate CBAM's own contribution. Variant (PA1/2/3) is not repeated
# here since PRESETS already varies it with everything else held fixed.
_ABLATION_BASE = PRESETS["ghost_mobilenetv4_pa2"]

ABLATION_PRESETS: dict[str, GhostRunConfig] = {
    "ablation_base": _ABLATION_BASE,
    "ablation_fmmix_off": replace(_ABLATION_BASE, name="ablation_fmmix_off", use_fmmix=False),
    "ablation_cbam_isolated": replace(_ABLATION_BASE, name="ablation_cbam_isolated", use_cbam=True),
    "ablation_frozen_backbone": replace(
        _ABLATION_BASE, name="ablation_frozen_backbone", phase_schedule=freeze_backbone_always
    ),
}


@dataclass
class InceptionV3Config:
    name: str = "inceptionv3"
    img_size: int = 256
    batch_size: int = 32
    tuner_max_trials: int = 30
    tuner_epochs: int = 75
    final_epochs: int = 100
    fmmix_alpha: float = 0.2
    dataset_path: str = DATASET_PATH_DEFAULT  # originally a stale "Images/Images" path; same dataset


INCEPTIONV3_CONFIG = InceptionV3Config()
