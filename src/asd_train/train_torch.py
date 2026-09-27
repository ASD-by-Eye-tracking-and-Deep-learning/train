"""Generic training loop shared by all 12 Ghost/CBAM PyTorch configs.

Structure (loss, phase-schedule call, scheduler step, early stopping on
val_loss, final test-set evaluation) matches the 4 original notebooks
exactly; everything that varied between them is read off ``cfg``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

from asd_train.augmentation import FMMixArgs, augment_pre_fmmix1, fmmix1
from asd_train.configs import GhostRunConfig
from asd_train.data import load_torch_dataset
from asd_train.models.ghost_variants import GhostVariantModel


@dataclass
class TorchRunResult:
    config_name: str
    best_val_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    classification_report: str
    confusion_matrix: np.ndarray
    best_model_path: str


def _make_scheduler(cfg: GhostRunConfig, optimizer):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=cfg.scheduler_eta_min
        )
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_plateau_factor, patience=cfg.scheduler_plateau_patience
        )
    return None


def run_torch_experiment(
    cfg: GhostRunConfig,
    dataset_root: str,
    output_dir: str,
    device: torch.device | None = None,
) -> TorchRunResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(dataset_root, os.path.basename(cfg.dataset_path.rstrip("/")))
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"{cfg.name}.pth")

    bundle = load_torch_dataset(
        dataset_path,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        extra_transform_augmentation=cfg.extra_transform_augmentation,
    )

    model = GhostVariantModel(
        backbone_name=cfg.backbone_name,
        variant=cfg.variant,
        use_cbam=cfg.use_cbam,
        kernel_size=cfg.kernel_size,
        ghost_ratio=cfg.ghost_ratio,
        bottleneck_expansion=cfg.bottleneck_expansion,
        bottleneck_stages=cfg.bottleneck_stages,
        adapter_reduction=cfg.adapter_reduction,
        cbam_reduction=cfg.cbam_reduction,
    ).to(device)

    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)
    fmmix_args = FMMixArgs(alpha=cfg.fmmix_alpha)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        cfg.phase_schedule(model, optimizer, epoch, cfg.lr)
        start_time = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0
        for images, labels in bundle.train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            images_fm, labels_fm = augment_pre_fmmix1(images.clone(), labels.clone())
            images_fm, computation = fmmix1(fmmix_args, images_fm, labels_fm)
            _, (p_tar, p_src), (y, y_shuf) = computation

            outputs = model(images_fm).squeeze(1)
            fmmix_kwargs = {"pos_weight": pos_weight} if cfg.use_pos_weight_in_fmmix_loss else {}
            loss_tar = F.binary_cross_entropy_with_logits(outputs, y, reduction="none", **fmmix_kwargs)
            loss_src = F.binary_cross_entropy_with_logits(outputs, y_shuf, reduction="none", **fmmix_kwargs)
            loss = (p_tar * loss_tar + p_src * loss_src).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
            optimizer.step()

            with torch.no_grad():
                preds = (outputs > 0).int()
                acc_batch = (
                    p_tar * (preds == y.int()).float() + p_src * (preds == y_shuf.int()).float()
                ).sum().item()
                train_correct += acc_batch
                train_loss += loss.item()
                train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(bundle.train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in bundle.val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze(1)
                val_loss += criterion(outputs, labels).item()
                val_correct += ((torch.sigmoid(outputs) > 0.5).int() == labels.int()).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        val_loss /= len(bundle.val_loader)

        print(
            f"[{cfg.name}] Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} | Time={time.time() - start_time:.1f}s"
        )

        if scheduler is not None:
            if cfg.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  ✓ {cfg.name} best model saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"{cfg.name} early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_all, yhat_all = [], []
    with torch.no_grad():
        for images, labels in bundle.test_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            yhat_all.extend((outputs > 0.5).int().squeeze(1).cpu().numpy())
            y_all.extend(labels.numpy())

    y_all, yhat_all = np.array(y_all), np.array(yhat_all)
    accuracy = (y_all == yhat_all).mean()
    precision = precision_score(y_all, yhat_all, average="weighted")
    recall = recall_score(y_all, yhat_all, average="weighted")
    report = classification_report(y_all, yhat_all, target_names=bundle.dataset.classes)
    conf_mat = confusion_matrix(1 - y_all, 1 - yhat_all)

    print(f"[{cfg.name}] Test Accuracy:  {accuracy:.4f}")
    print(f"[{cfg.name}] Test Precision: {precision:.4f}")
    print(f"[{cfg.name}] Test Recall:    {recall:.4f}")
    print("\n" + report)

    return TorchRunResult(
        config_name=cfg.name,
        best_val_loss=best_val_loss,
        test_accuracy=accuracy,
        test_precision=precision,
        test_recall=recall,
        classification_report=report,
        confusion_matrix=conf_mat,
        best_model_path=best_model_path,
    )
