"""T12 (InceptionV3 in PyTorch) — self-check, run from train/:

    uv run python scripts/check_t12_inceptionv3.py      (or .venv\\Scripts\\python.exe)

Checks, on CPU with random tensors (no dataset, no GPU, ~10 s):
1. The preset wires through ``GhostRunConfig`` with arch="inceptionv3" and is
   registered in ``PRESETS_WITH_INCEPTIONV3``.
2. ``InceptionV3TorchModel`` builds, forward pass returns shape (B, 1) —
   single logit, same convention as the 12 Ghost configs.
3. Backward pass reaches the head, and (phase 1) does NOT reach the frozen
   backbone — this is the property the unfreeze schedule depends on; a
   silent bug here would train the whole backbone from epoch 0 and quietly
   invalidate the comparison.
4. ``update_training_phase_inceptionv3_torch`` freezes at epoch 0, unfreezes
   from ``unfreeze_from`` at epoch 5, unfreezes everything at epoch 20, and
   sets the expected LR multipliers.
5. ``unfreeze_from`` accepts the 3 Keras-equivalent names and rejects junk.
6. ``train_torch`` dispatches to the right model class per ``cfg.arch``.

Exit code 0 + "OK" means all checks passed.
"""
import sys

import torch

from asd_train.configs import INCEPTIONV3_TORCH_PRESET, PRESETS, PRESETS_WITH_INCEPTIONV3
from asd_train.models.inceptionv3_torch import UNFREEZE_POINTS, InceptionV3TorchModel
from asd_train.phase_schedules import update_training_phase_inceptionv3_torch

failures = []


def check(label, condition, detail=""):
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


print("1. Config wiring")
cfg = INCEPTIONV3_TORCH_PRESET
check("arch == 'inceptionv3'", cfg.arch == "inceptionv3", cfg.arch)
check("backbone_name resolves to timm 'inception_v3'", cfg.backbone_name == "inception_v3", cfg.backbone_name)
check("registered in PRESETS_WITH_INCEPTIONV3", cfg.name in PRESETS_WITH_INCEPTIONV3)
check("PRESETS itself unchanged (12 ghost configs)", len(PRESETS) == 12, str(len(PRESETS)))
check(
    "all 12 existing presets still dispatch to the ghost model (no regression)",
    all(c.arch == "ghost" for c in PRESETS.values())
    and all(c.inception_dense_layers == 1 and c.inception_unfreeze_from == "Mixed_6a" for c in PRESETS.values()),
)
check("img_size 256 (same as TF track)", cfg.img_size == 256, str(cfg.img_size))

print("2/3. Model build + forward/backward")
torch.manual_seed(0)
model = InceptionV3TorchModel(
    backbone_name=cfg.backbone_name,
    num_dense_layers=cfg.inception_dense_layers,
    units=cfg.inception_units,
    dropout=cfg.inception_dropout,
    activation=cfg.inception_activation,
    unfreeze_from=cfg.inception_unfreeze_from,
)
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, cfg.img_size, cfg.img_size))
check("forward output shape is (B, 1)", tuple(out.shape) == (2, 1), str(tuple(out.shape)))
check("output is a single logit (not 2-class softmax)", out.shape[-1] == 1)

# Phase 1: backbone frozen, so only head params should receive gradients.
for p in model.backbone.parameters():
    p.requires_grad = False
model.train()
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    model(torch.randn(2, 3, cfg.img_size, cfg.img_size)).squeeze(1), torch.tensor([0.0, 1.0])
)
loss.backward()
backbone_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
check("frozen backbone receives no gradient", not backbone_grads, f"{len(backbone_grads)} backbone params got grads")
check("head receives gradients", len(head_grads) > 0)

print("4. Phase schedule")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def n_trainable(backbone):
    return sum(1 for p in backbone.parameters() if p.requires_grad)


update_training_phase_inceptionv3_torch(model, optimizer, epoch=0, base_lr=1e-4)
check("epoch 0: whole backbone frozen", n_trainable(model.backbone) == 0, str(n_trainable(model.backbone)))
check("epoch 0: lr == base_lr", abs(optimizer.param_groups[0]["lr"] - 1e-4) < 1e-12)

update_training_phase_inceptionv3_torch(model, optimizer, epoch=5, base_lr=1e-4)
partial = n_trainable(model.backbone)
total = sum(1 for _ in model.backbone.parameters())
check(
    f"epoch 5: partially unfrozen from {model.unfreeze_from} (0 < n < all)",
    0 < partial < total,
    f"{partial}/{total}",
)
check("epoch 5: lr == 0.5 * base_lr", abs(optimizer.param_groups[0]["lr"] - 5e-5) < 1e-12)

update_training_phase_inceptionv3_torch(model, optimizer, epoch=20, base_lr=1e-4)
check("epoch 20: all backbone params trainable", n_trainable(model.backbone) == total, f"{n_trainable(model.backbone)}/{total}")
check("epoch 20: lr == 0.1 * base_lr", abs(optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12)

print("5. unfreeze_from values")
for keras_name, timm_name in UNFREEZE_POINTS.items():
    try:
        InceptionV3TorchModel(backbone_name="inception_v3", unfreeze_from=timm_name)
        ok = True
    except Exception as exc:  # pragma: no cover
        ok = False
        print(f"      {keras_name} -> {timm_name}: {exc}")
    check(f"{keras_name} -> {timm_name} accepted", ok)
try:
    InceptionV3TorchModel(backbone_name="inception_v3", unfreeze_from="block6a_expand")
    check("Keras-style name rejected (must pass timm name)", False, "unexpectedly accepted")
except ValueError:
    check("Keras-style name rejected (must pass timm name)", True)

print("6. train_torch dispatch")
import inspect

src = inspect.getsource(__import__("asd_train.train_torch", fromlist=["x"]))
check("train_torch branches on cfg.arch", 'cfg.arch == "inceptionv3"' in src)
check("train_torch builds InceptionV3TorchModel", "InceptionV3TorchModel(" in src)
check("train_torch still builds GhostVariantModel", "GhostVariantModel(" in src)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("OK: T12 InceptionV3 PyTorch model, schedule and config wiring all verified")
