"""Aggregation helpers for cross-validation results (T6). Also the intended
home for T7's clinical metrics (sensitivity/specificity/AUC-ROC) once added.
"""
from __future__ import annotations

from typing import Sequence

from scipy import stats


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
