import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y, p) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return 0.0
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, p))


def best_threshold(y, p) -> float:
    """Threshold that maximises F1 (to be chosen on VALIDATION data only)."""
    y = np.asarray(y)
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    f1 = f1[:-1]                      # thresholds has len n-1
    if len(f1) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def compute_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": float(thr),
    }
