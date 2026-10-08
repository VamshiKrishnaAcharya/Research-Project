"""Trust score:  T = 0.35*F1 + 0.25*PR-AUC + 0.20*Recall + 0.20*Consistency."""
import numpy as np

from . import config


def compute_trust(metrics: dict, consistency: float, w: dict = None) -> float:
    w = w or config.TRUST_W
    t = (w["f1"] * metrics["f1"]
         + w["pr_auc"] * metrics["pr_auc"]
         + w["recall"] * metrics["recall"]
         + w["consistency"] * consistency)
    return float(np.clip(t, 0.0, 1.0))


def consistency_scores(deltas: list) -> list:
    """Agreement of each client's update with the OTHER clients of its family.

    cos(delta_i, mean of the other deltas) mapped from [-1, 1] to [0, 1].
    A family with one client has nothing to compare against -> 1.0.
    """
    if len(deltas) == 1:
        return [1.0]
    D = np.stack(deltas).astype("float64")
    total = D.sum(0)
    out = []
    for i in range(len(D)):
        ref = (total - D[i]) / (len(D) - 1)
        denom = np.linalg.norm(D[i]) * np.linalg.norm(ref)
        cos = float(D[i] @ ref / denom) if denom > 1e-12 else 0.0
        out.append((cos + 1.0) / 2.0)
    return out
