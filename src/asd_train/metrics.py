"""Aggregation helpers for cross-validation results (T6) and clinical metrics
(T7 — Reviewer 2 asked for Sensitivity/Specificity/AUC-ROC, review.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import recall_score, roc_auc_score, roc_curve

# ASD is always class 0 in this project's ImageFolder label encoding
# (alphabetical: "ASD" < "non-ASD") — the condition being screened for, so
# it's the "positive" class for Sensitivity/Specificity/AUC-ROC throughout.
ASD_LABEL = 0


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, tuple[float, float]]:
    """Mean and confidence interval across a small number of values (e.g. k=5
    CV folds). Uses the t-distribution rather than a normal approximation,
    since k is small enough (df=k-1) that the difference matters.

    Returns ``(mean, (low, high))``. With only 1 value, the interval
    collapses to the mean (no variance to estimate from).
    """
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, (mean, mean)

    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = variance**0.5
    se = std / (n**0.5)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return mean, (mean - margin, mean + margin)


@dataclass
class ClinicalMetrics:
    sensitivity: float
    specificity: float
    auc_roc: float
    roc_fpr: list[float]
    roc_tpr: list[float]


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_positive: np.ndarray,
    positive_label: int = ASD_LABEL,
) -> ClinicalMetrics:
    """Sensitivity (recall of the positive/screened-for class), Specificity
    (recall of the negative class, i.e. true negative rate), and AUC-ROC,
    computed with ASD as the positive class by default (see ``ASD_LABEL``).

    ``y_prob_positive`` must already be the predicted probability of the
    *positive* class (not necessarily the model's raw sigmoid output, which
    in this codebase predicts P(label=1) = P(non-ASD) — callers pass
    ``1 - sigmoid_output`` when ``positive_label=0``).
    """
    negative_label = 1 - positive_label  # binary classification only
    sensitivity = recall_score(y_true, y_pred, pos_label=positive_label)
    specificity = recall_score(y_true, y_pred, pos_label=negative_label)

    y_true_positive = (y_true == positive_label).astype(int)
    auc = roc_auc_score(y_true_positive, y_prob_positive)
    fpr, tpr, _ = roc_curve(y_true_positive, y_prob_positive)

    return ClinicalMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        auc_roc=auc,
        roc_fpr=fpr.tolist(),
        roc_tpr=tpr.tolist(),
    )


def plot_roc_curve(metrics: ClinicalMetrics, title: str, save_path: str) -> None:
    """Render and save a single ROC curve. Not called automatically during
    training (would mean a PNG per fold per config) — call this later, e.g.
    from the notebook, on a specific fold/config's saved ClinicalMetrics."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(metrics.roc_fpr, metrics.roc_tpr, label=f"AUC = {metrics.auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
