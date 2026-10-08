"""Final evaluation on the untouched, naturally imbalanced test set.

The decision threshold of every method is the F1-optimal threshold on the
global VALIDATION set. Test labels are used only here, only to score.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.metrics import best_threshold, compute_metrics
from atfl.utils import ensure_dirs, log, save_json

LABELS = {
    "local_only": "Local-only (mean of clients)",
    "local_ensemble": "Local ensemble (reference)",
    "fedavg_homog": "FedAvg, homogeneous BiLSTM",
    "fedavg_hetero": "FedAvg, hetero + fusion",
    "fedavg_hetero|bilstm": "  FedAvg hetero: BiLSTM family only",
    "fedavg_hetero|transformer": "  FedAvg hetero: Transformer family only",
    "atfl": "ATFL (trust, hetero + fusion)",
    "atfl|bilstm": "  ATFL: BiLSTM family only",
    "atfl|transformer": "  ATFL: Transformer family only",
    "atfl_trust_only": "Ablation: trust only (no N)",
    "atfl_no_consistency": "Ablation: no consistency term",
}
METRICS = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"]


def score(yv, pv, yt, pt):
    thr = best_threshold(yv, pv)
    return compute_metrics(yt, pt, thr)


def main():
    ensure_dirs(config.REPORT_DIR)
    rows, local_rows = {}, []
    y_test_ref = None
    for f in sorted(config.PRED_DIR.glob("*.npz")):
        name = f.stem
        z = np.load(f)
        yv, yt = z["y_val"], z["y_test"]
        y_test_ref = yt
        m = score(yv, z["p_val"], yt, z["p_test"])
        if name.startswith("local_c"):
            local_rows.append(m)
            continue
        rows[name] = m
        for key in z.files:
            if key.startswith("p_test__"):
                fam = key.split("__", 1)[1]
                rows[f"{name}|{fam}"] = score(yv, z[f"p_val__{fam}"], yt, z[key])

    if local_rows:
        avg = {k: float(np.mean([r[k] for r in local_rows])) for k in METRICS + ["tp", "fp", "tn", "fn"]}
        avg["std_f1"] = float(np.std([r["f1"] for r in local_rows]))
        avg["std_pr_auc"] = float(np.std([r["pr_auc"] for r in local_rows]))
        avg["threshold"] = float("nan")
        rows["local_only"] = avg

    order = [k for k in LABELS if k in rows]
    table = pd.DataFrame([{"method": LABELS[k], "key": k, **{m: rows[k][m] for m in METRICS},
                           "tp": rows[k]["tp"], "fp": rows[k]["fp"], "fn": rows[k]["fn"]}
                          for k in order])
    table.to_csv(config.REPORT_DIR / "test_metrics.csv", index=False)

    n_att = int(y_test_ref.sum()) if y_test_ref is not None else 0
    out = {"test_rows": int(len(y_test_ref)), "test_attacks": n_att,
           "all_normal_accuracy": 1 - n_att / max(len(y_test_ref), 1),
           "methods": {k: rows[k] for k in order}}
    save_json(out, config.REPORT_DIR / "test_metrics.json")

    fmt = table.copy()
    for m in METRICS:
        fmt[m] = fmt[m].map(lambda v: f"{v:.4f}")
    md = ["| Method | Acc | Prec | Recall | F1 | PR-AUC | ROC-AUC | FPR |", "|---|---|---|---|---|---|---|---|"]
    for _, r in fmt.iterrows():
        md.append(f"| {r['method']} | {r['accuracy']} | {r['precision']} | {r['recall']} | "
                  f"{r['f1']} | {r['pr_auc']} | {r['roc_auc']} | {r['fpr']} |")
    (config.REPORT_DIR / "test_metrics.md").write_text("\n".join(md) + "\n")

    log(f"test set: {out['test_rows']:,} rows, {n_att} attacks; "
        f"'always normal' accuracy would be {out['all_normal_accuracy']:.4%}")
    log(table[["method", "precision", "recall", "f1", "pr_auc"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
