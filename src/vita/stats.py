"""Paired statistics: two-sided Wilcoxon signed-rank tests with Holm correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def paired_wilcoxon(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired samples (1.0 if all differences are zero)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0):
        return 1.0
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)
