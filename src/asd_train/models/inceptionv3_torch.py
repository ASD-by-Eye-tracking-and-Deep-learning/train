"""InceptionV3 track (paper's accuracy-focused baseline), reimplemented in
PyTorch/timm so it runs through the same subject-independent 5-fold CV
protocol as the 12 Ghost/CBAM configs (T12).

The original is a Keras-Tuner hypermodel (``models/inceptionv3_tf.py``):
ImageNet InceptionV3 -> GAP -> 1..3 Dense layers with dropout -> sigmoid,
with progressive unfreezing starting at one of
``block5a_expand`` / ``block6a_expand`` / ``mixed7``. This module keeps the
same head shape and the same 3 unfreeze points, mapped onto timm's module
names (timm's InceptionV3 is a flat sequence, there is no ``features``
container):

    Keras block5a_expand -> timm Mixed_5b
    Keras block6a_expand -> timm Mixed_6a
    Keras mixed7         -> timm Mixed_7a

What is **not** reproduced: the 30-trial Keras-Tuner search itself (too
expensive to repeat for every CV fold). Its role is played by explicit
config fields, so whatever the tuner picks can be written down as a config
and re-run deterministically (see ``INCEPTIONV3_TORCH_PRESET``).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Keras block name -> timm top-level module name (see module docstring).
UNFREEZE_POINTS = {
    "block5a_expand": "Mixed_5b",
    "block6a_expand": "Mixed_6a",
    "mixed7": "Mixed_7a",
}

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class InceptionV3TorchModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "inception_v3",
        num_dense_layers: int = 1,
        units: int = 256,
        dropout: float = 0.3,
        activation: str = "relu",
        unfreeze_from: str = "Mixed_6a",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation!r}")
        if unfreeze_from not in set(UNFREEZE_POINTS.values()):
            raise ValueError(
                f"unfreeze_from must be one of {sorted(set(UNFREEZE_POINTS.values()))}, "
                f"got {unfreeze_from!r}"
            )
        if not 1 <= num_dense_layers <= 3:
            raise ValueError("num_dense_layers must be 1..3 (Keras tuner range)")

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.unfreeze_from = unfreeze_from

        in_features = self.backbone.num_features
        blocks: list[nn.Module] = []
        for i in range(num_dense_layers):
            blocks += [
                nn.Linear(in_features if i == 0 else units, units),
                _ACTIVATIONS[activation](),
                nn.Dropout(dropout),
            ]
        blocks.append(nn.Linear(units, 1))  # single logit, same as sigmoid output
        self.head = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)
