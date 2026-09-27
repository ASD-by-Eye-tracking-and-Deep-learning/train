"""
Unified model builder for the 12 Ghost/CBAM configurations explored in the
original per-notebook experiments (see train/README.md for the mapping).

There are 3 architectural variants ("Phuong an" in the original Vietnamese
notebooks) for inserting GhostModule into an ImageNet-pretrained timm
backbone, each optionally augmented with a CBAM attention block. CBAM is
inserted at a *different point* per variant — this was verified by reading
the original notebooks directly, not assumed:

- pa1 (head):        backbone -> [CBAM] -> GAP -> GhostModule head
- pa2 (bottleneck):  backbone -> GAP -> stacked GhostBottleneck (CBAM, if any,
                      lives *inside* each bottleneck, between its two
                      GhostModules) -> Linear head
- pa3 (adapter):     backbone -> GhostModule residual adapter -> [CBAM] -> GAP
                      -> Linear head
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from asd_train.layers.cbam import CBAM
from asd_train.layers.ghost import GhostModule


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=2, kernel_size=1, ratio=2):
        super().__init__()
        mid_ch = in_ch * expansion
        self.ghost1 = GhostModule(in_ch, mid_ch, kernel_size=kernel_size, ratio=ratio)
        self.ghost2 = GhostModule(mid_ch, out_ch, kernel_size=kernel_size, ratio=ratio, relu=False)

    def forward(self, x):
        return self.ghost2(self.ghost1(x))


class GhostBottleneckCBAM(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=2, kernel_size=1, ratio=2, cbam_reduction=16):
        super().__init__()
        mid_ch = in_ch * expansion
        self.ghost1 = GhostModule(in_ch, mid_ch, kernel_size=kernel_size, ratio=ratio)
        self.cbam = CBAM(mid_ch, reduction=cbam_reduction)
        self.ghost2 = GhostModule(mid_ch, out_ch, kernel_size=kernel_size, ratio=ratio, relu=False)

    def forward(self, x):
        x = self.ghost1(x)
        x = self.cbam(x)
        return self.ghost2(x)


class GhostAdapter(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=1, ratio=2):
        super().__init__()
        mid_channels = in_channels // reduction
        self.adapter = nn.Sequential(
            GhostModule(in_channels, mid_channels, kernel_size=kernel_size, ratio=ratio),
            nn.ReLU(inplace=True),
            GhostModule(mid_channels, in_channels, kernel_size=kernel_size, ratio=ratio, relu=False),
        )

    def forward(self, x):
        return x + self.adapter(x)


class GhostVariantModel(nn.Module):
    """One model covering all 3 variants x {with, without} CBAM."""

    def __init__(
        self,
        backbone_name: str,
        variant: str,  # "pa1" | "pa2" | "pa3"
        use_cbam: bool,
        kernel_size: int = 1,
        ghost_ratio: int = 2,
        bottleneck_expansion: int = 2,
        bottleneck_stages: int = 2,
        adapter_reduction: int = 4,
        cbam_reduction: int = 16,
    ):
        super().__init__()
        if variant not in ("pa1", "pa2", "pa3"):
            raise ValueError(f"Unknown variant: {variant!r}")
        self.variant = variant
        self.use_cbam = use_cbam

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        self.gap = nn.AdaptiveAvgPool2d(1)
        in_features = self.backbone.num_features

        if variant == "pa1":
            self.cbam = CBAM(in_features, reduction=cbam_reduction) if use_cbam else None
            mid_features = in_features // 2
            self.ghost_head = nn.Sequential(
                GhostModule(in_features, mid_features, kernel_size=kernel_size, ratio=ghost_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(mid_features, 1),
            )
        elif variant == "pa2":
            if bottleneck_stages not in (1, 2):
                raise ValueError("bottleneck_stages must be 1 or 2")
            bottleneck_cls = GhostBottleneckCBAM if use_cbam else GhostBottleneck
            kwargs = {"kernel_size": kernel_size, "ratio": ghost_ratio}
            if use_cbam:
                kwargs["cbam_reduction"] = cbam_reduction
            if bottleneck_stages == 1:
                # ghost/convnexttiny_v2 (no CBAM) only: single stage, documented
                # in the original notebook as a deliberate simplification.
                self.ghost_bottleneck = nn.Sequential(
                    bottleneck_cls(in_features, in_features // 2, expansion=bottleneck_expansion, **kwargs),
                    nn.ReLU(inplace=True),
                )
                self.head = nn.Linear(in_features // 2, 1)
            else:
                self.ghost_bottleneck = nn.Sequential(
                    bottleneck_cls(in_features, in_features // 2, expansion=bottleneck_expansion, **kwargs),
                    nn.ReLU(inplace=True),
                    bottleneck_cls(in_features // 2, in_features // 4, expansion=bottleneck_expansion, **kwargs),
                    nn.ReLU(inplace=True),
                )
                self.head = nn.Linear(in_features // 4, 1)
        else:  # pa3
            self.ghost_adapter = GhostAdapter(in_features, reduction=adapter_reduction, kernel_size=kernel_size, ratio=ghost_ratio)
            self.cbam = CBAM(in_features, reduction=cbam_reduction) if use_cbam else None
            self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)

        if self.variant == "pa1":
            if self.cbam is not None:
                x = self.cbam(x)
            x = self.gap(x).flatten(1)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.ghost_head[0](x)
            x = x.flatten(1)
            x = self.ghost_head[1](x)
            x = self.ghost_head[2](x)
            return x

        if self.variant == "pa2":
            x = self.gap(x).flatten(1)
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.ghost_bottleneck(x).flatten(1)
            return self.head(x)

        # pa3
        x = self.ghost_adapter(x)
        if self.cbam is not None:
            x = self.cbam(x)
        x = self.gap(x).flatten(1)
        return self.head(x)
