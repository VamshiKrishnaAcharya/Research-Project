"""Split the global TRAIN and VAL sets into non-IID clients.

* attack rows are spread over clients per attack stage with a Dirichlet(alpha) skew
  (each client sees only some stages)
* benign rows get a quantity skew
* the test split is NOT partitioned
* client i is given architecture ARCH_CYCLE[i % 2] (BiLSTM / Transformer)
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.utils import ensure_dirs, log, save_json


def assign(df, props, K, rng):
    client = np.zeros(len(df), dtype=int)
    key = np.where(df["label"].to_numpy() == 1, df["stage"].to_numpy(), 0)
    for g in np.unique(key):
        rows = np.where(key == g)[0]
        client[rows] = rng.choice(K, size=len(rows), p=props[g])
    return client


def repair(client, y, minimum, K, rng):
    """Make sure every client holds at least `minimum` attack rows."""
    for _ in range(4 * K):
        counts = np.bincount(client[y == 1], minlength=K)
        lo, hi = int(counts.argmin()), int(counts.argmax())
        need = minimum - counts[lo]
        if need <= 0:
            return
        if counts[hi] - need < minimum:
            log(f"  warning: cannot guarantee {minimum} attacks per client (too few attacks)")
            return
        cand = np.where((client == hi) & (y == 1))[0]
        client[rng.choice(cand, size=need, replace=False)] = lo


def main():
    ensure_dirs(config.CLIENT_DIR)
    K = config.NUM_CLIENTS
    rng = np.random.default_rng(config.SEED)
    train = pd.read_csv(config.SPLIT_DIR / "train.csv")
    val = pd.read_csv(config.SPLIT_DIR / "val.csv")

    groups = sorted(set(train.loc[train["label"] == 1, "stage"]) | set(val.loc[val["label"] == 1, "stage"]))
    props = {0: rng.dirichlet([config.QTY_ALPHA] * K)}
    for g in groups:
        props[g] = rng.dirichlet([config.DIRICHLET_ALPHA] * K)

    ctr = assign(train, props, K, rng)
    cva = assign(val, props, K, rng)
    repair(ctr, train["label"].to_numpy(), config.MIN_ATTACK_TRAIN, K, rng)
    repair(cva, val["label"].to_numpy(), config.MIN_ATTACK_VAL, K, rng)

    clients = []
    for i in range(K):
        tr, va = train[ctr == i], val[cva == i]
        tr.to_csv(config.CLIENT_DIR / f"client_{i}_train.csv", index=False)
        va.to_csv(config.CLIENT_DIR / f"client_{i}_val.csv", index=False)
        stages = sorted(int(s) for s in tr.loc[tr["label"] == 1, "stage"].unique())
        clients.append({"client_id": i, "arch": config.ARCH_CYCLE[i % len(config.ARCH_CYCLE)],
                        "n_train": int(len(tr)), "n_train_attack": int(tr["label"].sum()),
                        "n_val": int(len(va)), "n_val_attack": int(va["label"].sum()),
                        "attack_stages_seen": stages})
        c = clients[-1]
        log(f"client {i} [{c['arch']:11s}] train={c['n_train']:>7,} (attacks {c['n_train_attack']:>4})  "
            f"val={c['n_val']:>6,} (attacks {c['n_val_attack']:>3})  stages={stages}")

    save_json({"num_clients": K, "dirichlet_alpha": config.DIRICHLET_ALPHA, "clients": clients},
              config.CLIENT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
