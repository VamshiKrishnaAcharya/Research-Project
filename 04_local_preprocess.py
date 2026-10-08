"""Leakage-free, federated preprocessing.

1. Each client computes sufficient statistics (n, sum, sum of squares) of the
   signed-log1p features on ITS OWN TRAIN rows only.
2. The server combines them into a global mean/std (the only thing shared).
3. Every client scales its own train/val with that scaler; the global val and
   test sets are scaled with the same scaler (they never influence it).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.data import combine_stats, local_stats, transform
from atfl.utils import ensure_dirs, load_json, log, save_json


def main():
    ensure_dirs(config.PROC_DIR)
    feats = load_json(config.SPLIT_DIR / "split_manifest.json")["features"]
    man = load_json(config.CLIENT_DIR / "manifest.json")

    tr, va, stats = {}, {}, []
    for c in man["clients"]:
        i = c["client_id"]
        tr[i] = pd.read_csv(config.CLIENT_DIR / f"client_{i}_train.csv")
        va[i] = pd.read_csv(config.CLIENT_DIR / f"client_{i}_val.csv")
        stats.append(local_stats(tr[i][feats].to_numpy(dtype="float64")))   # client side

    mean, std, n = combine_stats(stats)                                       # server side
    save_json({"fit_on": "client_train_only", "n_rows": n, "features": feats,
               "mean": mean, "std": std}, config.PROC_DIR / "scaler.json")
    log(f"global scaler from {n:,} client-train rows ({len(feats)} features)")

    for c in man["clients"]:
        i = c["client_id"]
        np.savez(config.PROC_DIR / f"client_{i}.npz",
                 X_train=transform(tr[i][feats].to_numpy(dtype="float64"), mean, std),
                 y_train=tr[i]["label"].to_numpy().astype("int64"),
                 rid_train=tr[i]["row_id"].to_numpy(),
                 X_val=transform(va[i][feats].to_numpy(dtype="float64"), mean, std),
                 y_val=va[i]["label"].to_numpy().astype("int64"),
                 rid_val=va[i]["row_id"].to_numpy())

    gv = pd.read_csv(config.SPLIT_DIR / "val.csv")
    te = pd.read_csv(config.SPLIT_DIR / "test.csv")
    np.savez(config.PROC_DIR / "global_val.npz",
             X=transform(gv[feats].to_numpy(dtype="float64"), mean, std),
             y=gv["label"].to_numpy().astype("int64"), rid=gv["row_id"].to_numpy())
    np.savez(config.PROC_DIR / "test.npz",
             X=transform(te[feats].to_numpy(dtype="float64"), mean, std),
             y=te["label"].to_numpy().astype("int64"), rid=te["row_id"].to_numpy())
    log(f"saved processed arrays to {config.PROC_DIR}")


if __name__ == "__main__":
    main()
