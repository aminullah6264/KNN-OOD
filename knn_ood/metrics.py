from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_curve


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_tpr = 0.95
    if np.any(tpr >= target_tpr):
        return float(np.interp(target_tpr, tpr, fpr) * 100.0)
    return 100.0


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr) * 100.0)
