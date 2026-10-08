"""Global stratified train/val/test split, done BEFORE any fitting.

The test split is never touched again until 08_evaluate.py. Scaling and
client partitioning use train/val only.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from atfl import config
from atfl.data import (
    infer_feature_columns,
    load_and_clean,
    load_labels_chunked,
    write_splits_chunked,
)
from atfl.utils import ensure_dirs, log, save_json

CHUNKED_ROW_THRESHOLD = 2_000_000


def split_indices(y: np.ndarray):
    idx = np.arange(len(y))
    tr_idx, rest_idx = train_test_split(idx, train_size=config.TRAIN_FRAC,
                                        stratify=y, random_state=config.SEED)
    val_share = config.VAL_FRAC / (1.0 - config.TRAIN_FRAC)
    va_idx, te_idx = train_test_split(rest_idx, train_size=val_share,
                                      stratify=y[rest_idx], random_state=config.SEED)
    return tr_idx, va_idx, te_idx


def apply_train_cap(tr_idx, y):
    if config.NORMAL_TRAIN_CAP <= 0:
        return set(tr_idx)
    y_tr = y[tr_idx]
    attacks = tr_idx[y_tr == 1]
    benign = tr_idx[y_tr == 0]
    if len(benign) > config.NORMAL_TRAIN_CAP:
        rng = np.random.default_rng(config.SEED)
        benign = rng.choice(benign, size=config.NORMAL_TRAIN_CAP, replace=False)
    out = np.concatenate([attacks, benign])
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(out)
    return set(out.tolist())


def split_stats(split_dir, name):
    df = pd.read_csv(split_dir / f"{name}.csv", usecols=["label"])
    n = len(df)
    a = int(df["label"].sum())
    return {"rows": n, "attacks": a, "attack_rate": float(a / max(n, 1))}


def main():
    ensure_dirs(config.SPLIT_DIR)
    path = config.RAW_CSV
    use_chunked = path.stat().st_size > 500_000_000

    if use_chunked:
        log("large file: chunked label pass + streaming split (no in-memory load)")
        sample = pd.read_csv(path, nrows=5000, low_memory=False)
        if config.LABEL_COL not in sample.columns:
            raise ValueError(f"Label column '{config.LABEL_COL}' not found.")
        feat_cols = infer_feature_columns(sample)
        if not feat_cols:
            raise ValueError("No numeric feature columns found.")
        y = load_labels_chunked(path)
        log(f"loaded {len(y):,} labels; {int(y.sum()):,} attacks; {len(feat_cols)} features")
        tr_idx, va_idx, te_idx = split_indices(y)
        train_ids = apply_train_cap(tr_idx, y)
        if config.NORMAL_TRAIN_CAP > 0:
            log(f"benign rows in TRAIN capped to {config.NORMAL_TRAIN_CAP:,} (val/test untouched)")
        val_ids, test_ids = set(va_idx.tolist()), set(te_idx.tolist())
        write_splits_chunked(path, train_ids, val_ids, test_ids, feat_cols, config.SPLIT_DIR)
        dropped = 0
    else:
        df, feat_cols = load_and_clean(path)
        log(f"loaded {len(df):,} rows after dropping {df.attrs['dropped_duplicates']} duplicates; "
            f"{len(feat_cols)} features")
        y = df["label"].to_numpy()
        tr_idx, va_idx, te_idx = split_indices(y)
        train_ids = apply_train_cap(tr_idx, y)
        train = df.iloc[sorted(train_ids)]
        val, test = df.iloc[va_idx], df.iloc[te_idx]
        if config.NORMAL_TRAIN_CAP > 0 and len(train_ids) < len(tr_idx):
            log(f"benign rows in TRAIN capped to {config.NORMAL_TRAIN_CAP:,} (val/test untouched)")
        for name, part in (("train", train), ("val", val), ("test", test)):
            part.to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
        dropped = int(df.attrs["dropped_duplicates"])
        train_ids, val_ids, test_ids = set(train["row_id"]), set(val["row_id"]), set(test["row_id"])

    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), \
        "splits overlap"

    man = {"seed": config.SEED, "features": feat_cols,
           "splits": {n: split_stats(config.SPLIT_DIR, n) for n in ("train", "val", "test")},
           "normal_train_cap": config.NORMAL_TRAIN_CAP,
           "dropped_duplicates": dropped,
           "chunked_split": use_chunked}
    save_json(man, config.SPLIT_DIR / "split_manifest.json")
    for n, s in man["splits"].items():
        log(f"{n:5s}: {s['rows']:>9,} rows, {s['attacks']:>6,} attacks ({s['attack_rate']:.4%})")


if __name__ == "__main__":
    main()
