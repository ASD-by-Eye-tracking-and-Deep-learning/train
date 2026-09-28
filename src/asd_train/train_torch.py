"""Generic training loop shared by all 12 Ghost/CBAM PyTorch configs.

Structure (loss, phase-schedule call, scheduler step, early stopping on
val_loss, final test-set evaluation) matches the 4 original notebooks
exactly; everything that varied between them is read off ``cfg``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

from asd_train.augmentation import FMMixArgs, augment_pre_fmmix1, fmmix1
from asd_train.configs import GhostRunConfig
from asd_train.data import TorchDataBundle, bundle_from_indices, build_torch_transform, load_torch_dataset, make_group_kfold_splits
from asd_train.metrics import mean_ci
from asd_train.models.ghost_variants import GhostVariantModel
from torchvision import datasets


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


@dataclass
class TorchCVResult:
    config_name: str
    fold_results: list[TorchRunResult]
    mean_accuracy: float
    ci_accuracy: tuple[float, float]
    mean_precision: float
    ci_precision: tuple[float, float]
    mean_recall: float
    ci_recall: tuple[float, float]


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
    """Single subject-independent hold-out (T5) — quick dev iteration on one
    config. For the primary reported protocol, see ``run_torch_cv_experiment``
    (T6, grouped k-fold CV)."""
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

    return _train_and_evaluate(cfg, bundle, device, best_model_path)


def _train_and_evaluate(
    cfg: GhostRunConfig,
    bundle: TorchDataBundle,
    device: torch.device,
    best_model_path: str,
) -> TorchRunResult:
    """Model build + full training loop (phase schedule, FMMix1, early
    stopping on val_loss) + final test-set evaluation, given an
    already-built ``TorchDataBundle``. Extracted from ``run_torch_experiment``
    (T6) so both the single-split path and ``run_torch_cv_experiment``'s
    per-fold loop share the exact same training/eval code."""
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


def _fold_result_to_json(result: TorchRunResult) -> dict:
    d = asdict(result)
    d["confusion_matrix"] = result.confusion_matrix.tolist()
    return d


def _save_cv_progress(config_name: str, fold_results: list[TorchRunResult], output_dir: str) -> None:
    path = os.path.join(output_dir, f"{config_name}_cv.json")
    with open(path, "w") as f:
        json.dump({"config_name": config_name, "folds": [_fold_result_to_json(r) for r in fold_results]}, f, indent=2)


def run_torch_cv_experiment(
    cfg: GhostRunConfig,
    dataset_root: str,
    output_dir: str,
    k: int = 5,
    val_size: float = 0.2,
    random_state: int = 42,
    device: torch.device | None = None,
) -> TorchCVResult:
    """Grouped k-fold CV (T6) — the primary reported protocol (Decision #6/#7,
    record.md): k=5, subject-independent per fold, every participant used as
    test exactly once across the k folds. Each fold trains and evaluates
    independently via ``_train_and_evaluate`` (same code path as
    ``run_torch_experiment``'s single hold-out). Results are written to
    ``{output_dir}/{cfg.name}_cv.json`` after every fold, not just at the end,
    so a crash partway through doesn't lose already-finished folds.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(dataset_root, os.path.basename(cfg.dataset_path.rstrip("/")))
    os.makedirs(output_dir, exist_ok=True)

    transform = build_torch_transform(cfg.img_size, cfg.extra_transform_augmentation)
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    splits = make_group_kfold_splits(dataset, k=k, val_size=val_size, random_state=random_state)

    fold_results: list[TorchRunResult] = []
    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"\n{'=' * 20} {cfg.name} — fold {fold_i + 1}/{k} {'=' * 20}")
        bundle = bundle_from_indices(dataset, cfg.batch_size, train_idx, val_idx, test_idx)
        best_model_path = os.path.join(output_dir, f"{cfg.name}_fold{fold_i}.pth")
        result = _train_and_evaluate(cfg, bundle, device, best_model_path)
        fold_results.append(result)
        _save_cv_progress(cfg.name, fold_results, output_dir)

    mean_accuracy, ci_accuracy = mean_ci([r.test_accuracy for r in fold_results])
    mean_precision, ci_precision = mean_ci([r.test_precision for r in fold_results])
    mean_recall, ci_recall = mean_ci([r.test_recall for r in fold_results])

    print(f"\n[{cfg.name}] CV accuracy:  {mean_accuracy:.4f} (95% CI {ci_accuracy[0]:.4f}-{ci_accuracy[1]:.4f})")
    print(f"[{cfg.name}] CV precision: {mean_precision:.4f} (95% CI {ci_precision[0]:.4f}-{ci_precision[1]:.4f})")
    print(f"[{cfg.name}] CV recall:    {mean_recall:.4f} (95% CI {ci_recall[0]:.4f}-{ci_recall[1]:.4f})")

    return TorchCVResult(
        config_name=cfg.name,
        fold_results=fold_results,
        mean_accuracy=mean_accuracy,
        ci_accuracy=ci_accuracy,
        mean_precision=mean_precision,
        ci_precision=ci_precision,
        mean_recall=mean_recall,
        ci_recall=ci_recall,
    )
