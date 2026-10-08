"""Generate a SYNTHETIC CICAPT-like flow table for pipeline smoke-testing only.

Real CICAPT-IIoT 2024 has ~0.005% attacks. This generator defaults to 1% so a
small file still contains enough attacks to train and evaluate on. Results on
this data say nothing about real APT performance.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.utils import ensure_dirs, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rows", type=int, default=60000)
    ap.add_argument("--attack_rate", type=float, default=0.01)
    ap.add_argument("--n_features", type=int, default=24)
    ap.add_argument("--out", type=str, default=str(config.RAW_DIR / "cicapt_synthetic.csv"))
    a = ap.parse_args()

    rng = np.random.default_rng(config.SEED)
    n, F = a.n_rows, a.n_features
    n_att = max(int(n * a.attack_rate), 60)

    mu = rng.normal(2.0, 0.7, F)
    sig = rng.uniform(0.5, 1.2, F)
    latent = rng.normal(size=(n, 3))
    load = rng.normal(0, 0.4, (3, F))
    Z = mu + latent @ load + rng.normal(size=(n, F)) * sig

    label = np.zeros(n, dtype=int)
    stage = np.zeros(n, dtype=int)
    att_idx = rng.choice(n, size=n_att, replace=False)
    label[att_idx] = 1
    stage[att_idx] = rng.integers(1, 5, size=n_att)          # 4 attack stages
    shift_sets = {s: rng.choice(F, size=5, replace=False) for s in range(1, 5)}
    for s in range(1, 5):
        rows = att_idx[stage[att_idx] == s]
        Z[np.ix_(rows, shift_sets[s])] += rng.uniform(1.5, 2.5, size=(len(rows), 5))

    X = np.exp(Z)
    df = pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(F)])
    df.insert(0, "Timestamp", pd.date_range("2024-01-01", periods=n, freq="s").astype(str))
    df["label"] = label
    df["stage"] = stage

    # a little dirt so the audit / cleaning steps have something to find
    bad = rng.choice(n, size=max(n // 1000, 5), replace=False)
    df.loc[bad[: len(bad) // 2], "f03"] = np.nan
    df.loc[bad[len(bad) // 2:], "f07"] = np.inf

    ensure_dirs(pathlib.Path(a.out).parent)
    df.to_csv(a.out, index=False)
    log(f"wrote {a.out}: {n} rows, {n_att} attacks ({n_att / n:.3%}), {F} features. SYNTHETIC DATA.")


if __name__ == "__main__":
    main()
