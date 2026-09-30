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
from asd_train.metrics import ASD_LABEL, ClinicalMetrics, compute_clinical_metrics, mean_ci
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
    clinical: ClinicalMetrics


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
    mean_sensitivity: float
    ci_sensitivity: tuple[float, float]
    mean_specificity: float
    ci_specificity: tuple[float, float]
    mean_auc_roc: float
    ci_auc_roc: tuple[float, float]


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
        lr_before_phase = optimizer.param_groups[0]["lr"]
        cfg.phase_schedule(model, optimizer, epoch, cfg.lr)
        if optimizer.param_groups[0]["lr"] != lr_before_phase:
            # A phase transition (backbone unfreeze / LR change) legitimately
            # changes the loss landscape -- don't let a stall from the
            # *previous* phase trigger early stopping before this phase gets
            # `patience` epochs of its own to find a new best. Without this,
            # logs showed most folds early-stopping mid Phase-2, never
            # reaching Phase-3 (see record.md Decision #12).
            patience_counter = 0
        start_time = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0
        for images, labels in bundle.train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()

            if cfg.use_fmmix:
                images_fm, labels_fm = augment_pre_fmmix1(images.clone(), labels.clone())
                images_fm, computation = fmmix1(fmmix_args, images_fm, labels_fm)
                _, (p_tar, p_src), (y, y_shuf) = computation

                outputs = model(images_fm).squeeze(1)
                fmmix_kwargs = {"pos_weight": pos_weight} if cfg.use_pos_weight_in_fmmix_loss else {}
                loss_tar = F.binary_cross_entropy_with_logits(outputs, y, reduction="none", **fmmix_kwargs)
                loss_src = F.binary_cross_entropy_with_logits(outputs, y_shuf, reduction="none", **fmmix_kwargs)
                loss = (p_tar * loss_tar + p_src * loss_src).mean()
                with torch.no_grad():
                    preds = (outputs > 0).int()
                    acc_batch = (
                        p_tar * (preds == y.int()).float() + p_src * (preds == y_shuf.int()).float()
                    ).sum().item()
            else:
                # T9 ablation: FMMix1 off entirely, plain BCE on the unmixed batch.
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                with torch.no_grad():
                    preds = (outputs > 0).int()
                    acc_batch = (preds == labels.int()).float().sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
            optimizer.step()

            with torch.no_grad():
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

    y_all, yhat_all, yprob_all = [], [], []
    with torch.no_grad():
        for images, labels in bundle.test_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            yhat_all.extend((outputs > 0.5).int().squeeze(1).cpu().numpy())
            yprob_all.extend(outputs.squeeze(1).cpu().numpy())
            y_all.extend(labels.numpy())

    y_all, yhat_all, yprob_all = np.array(y_all), np.array(yhat_all), np.array(yprob_all)
    accuracy = (y_all == yhat_all).mean()
    precision = precision_score(y_all, yhat_all, average="weighted")
    recall = recall_score(y_all, yhat_all, average="weighted")
    report = classification_report(y_all, yhat_all, target_names=bundle.dataset.classes)
    conf_mat = confusion_matrix(1 - y_all, 1 - yhat_all)

    # yprob_all is P(label=1) = P(non-ASD) (sigmoid output); flip to P(ASD)
    # since ASD_LABEL=0 is the positive class for clinical metrics.
    yprob_positive = 1 - yprob_all if ASD_LABEL == 0 else yprob_all
    clinical = compute_clinical_metrics(y_all, yhat_all, yprob_positive, positive_label=ASD_LABEL)

    print(f"[{cfg.name}] Test Accuracy:    {accuracy:.4f}")
    print(f"[{cfg.name}] Test Precision:   {precision:.4f}")
    print(f"[{cfg.name}] Test Recall:      {recall:.4f}")
    print(f"[{cfg.name}] Sensitivity(ASD): {clinical.sensitivity:.4f}")
    print(f"[{cfg.name}] Specificity:      {clinical.specificity:.4f}")
    print(f"[{cfg.name}] AUC-ROC:          {clinical.auc_roc:.4f}")
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
        clinical=clinical,
    )


def _fold_result_to_json(result: TorchRunResult) -> dict:
    d = asdict(result)
    d["confusion_matrix"] = result.confusion_matrix.tolist()
    return d


def _save_cv_progress(config_name: str, fold_results: list[TorchRunResult], output_dir: str) -> None:
    path = os.path.join(output_dir, f"{config_name}_cv.json")
    with open(path, "w") as f:
        json.dump({"config_name": config_name, "folds": [_fold_result_to_json(r) for r in fold_results]}, f, indent=2)


def _load_cv_progress(config_name: str, output_dir: str) -> list[TorchRunResult]:
    """Reconstruct already-completed folds from a prior run's
    ``{config_name}_cv.json``, if present -- lets ``run_torch_cv_experiment``
    resume after an interrupted session (e.g. a Colab disconnect) instead of
    re-training folds that already finished. Returns [] if no progress file
    exists yet."""
    path = os.path.join(output_dir, f"{config_name}_cv.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    results = []
    for fold in data["folds"]:
        fold = dict(fold)
        fold["confusion_matrix"] = np.array(fold["confusion_matrix"])
        fold["clinical"] = ClinicalMetrics(**fold["clinical"])
        results.append(TorchRunResult(**fold))
    return results


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

    Resumable (Decision #14, record.md): on start, any folds already recorded
    in ``{output_dir}/{cfg.name}_cv.json`` from a prior run are loaded and
    skipped rather than re-trained -- re-calling this after an interrupted
    session (e.g. a Colab disconnect) with the same ``output_dir``/``k``/
    ``random_state`` continues from the next unfinished fold instead of
    starting the config over. Relies on ``make_group_kfold_splits`` being
    deterministic (fixed ``random_state``) so fold N is the same split both
    times.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(dataset_root, os.path.basename(cfg.dataset_path.rstrip("/")))
    os.makedirs(output_dir, exist_ok=True)

    transform = build_torch_transform(cfg.img_size, cfg.extra_transform_augmentation)
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    splits = make_group_kfold_splits(dataset, k=k, val_size=val_size, random_state=random_state)

    fold_results: list[TorchRunResult] = _load_cv_progress(cfg.name, output_dir)
    if fold_results:
        print(f"[{cfg.name}] resuming: {len(fold_results)}/{k} folds already completed, loaded from disk")

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        if fold_i < len(fold_results):
            continue
        print(f"\n{'=' * 20} {cfg.name} — fold {fold_i + 1}/{k} {'=' * 20}")
        bundle = bundle_from_indices(dataset, cfg.batch_size, train_idx, val_idx, test_idx)
        best_model_path = os.path.join(output_dir, f"{cfg.name}_fold{fold_i}.pth")
        result = _train_and_evaluate(cfg, bundle, device, best_model_path)
        fold_results.append(result)
        _save_cv_progress(cfg.name, fold_results, output_dir)

    mean_accuracy, ci_accuracy = mean_ci([r.test_accuracy for r in fold_results])
    mean_precision, ci_precision = mean_ci([r.test_precision for r in fold_results])
    mean_recall, ci_recall = mean_ci([r.test_recall for r in fold_results])
    mean_sensitivity, ci_sensitivity = mean_ci([r.clinical.sensitivity for r in fold_results])
    mean_specificity, ci_specificity = mean_ci([r.clinical.specificity for r in fold_results])
    mean_auc_roc, ci_auc_roc = mean_ci([r.clinical.auc_roc for r in fold_results])

    print(f"\n[{cfg.name}] CV accuracy:    {mean_accuracy:.4f} (95% CI {ci_accuracy[0]:.4f}-{ci_accuracy[1]:.4f})")
    print(f"[{cfg.name}] CV precision:   {mean_precision:.4f} (95% CI {ci_precision[0]:.4f}-{ci_precision[1]:.4f})")
    print(f"[{cfg.name}] CV recall:      {mean_recall:.4f} (95% CI {ci_recall[0]:.4f}-{ci_recall[1]:.4f})")
    print(f"[{cfg.name}] CV sensitivity: {mean_sensitivity:.4f} (95% CI {ci_sensitivity[0]:.4f}-{ci_sensitivity[1]:.4f})")
    print(f"[{cfg.name}] CV specificity: {mean_specificity:.4f} (95% CI {ci_specificity[0]:.4f}-{ci_specificity[1]:.4f})")
    print(f"[{cfg.name}] CV AUC-ROC:     {mean_auc_roc:.4f} (95% CI {ci_auc_roc[0]:.4f}-{ci_auc_roc[1]:.4f})")

    return TorchCVResult(
        config_name=cfg.name,
        fold_results=fold_results,
        mean_accuracy=mean_accuracy,
        ci_accuracy=ci_accuracy,
        mean_precision=mean_precision,
        ci_precision=ci_precision,
        mean_recall=mean_recall,
        ci_recall=ci_recall,
        mean_sensitivity=mean_sensitivity,
        ci_sensitivity=ci_sensitivity,
        mean_specificity=mean_specificity,
        ci_specificity=ci_specificity,
        mean_auc_roc=mean_auc_roc,
        ci_auc_roc=ci_auc_roc,
    )
