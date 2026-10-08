"""Decision-level fusion of architecture-family models."""
import numpy as np

from .metrics import pr_auc


def choose_weights(y_val, probs: dict) -> dict:
    """Fusion weights chosen on VALIDATION data only (never on test)."""
    fams = sorted(probs)
    if len(fams) == 1:
        return {fams[0]: 1.0}
    if len(fams) == 2:
        a, b = fams
        best_key, best_w = -1.0, 0.5
        for w in np.linspace(0.0, 1.0, 11):
            score = pr_auc(y_val, w * probs[a] + (1 - w) * probs[b])
            key = score - 1e-6 * abs(w - 0.5)          # tie-break towards 0.5
            if key > best_key:
                best_key, best_w = key, float(w)
        return {a: best_w, b: 1.0 - best_w}
    return {f: 1.0 / len(fams) for f in fams}


def fuse(probs: dict, weights: dict) -> np.ndarray:
    return sum(weights[f] * probs[f] for f in probs)
