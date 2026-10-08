"""Data loading, cleaning and leakage-free preprocessing helpers."""
import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------ labels
def normalize_labels(series: pd.Series) -> pd.Series:
    """Return a 0/1 attack label from numeric or text labels."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) != 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    junk = {b.lower() for b in config.BENIGN_NAMES} | {
        "label", "sublabel", "sublabelcat", "nan", "",
    }
    return (~s.isin(junk)).astype(int)


def stage_codes(df: pd.DataFrame, label: pd.Series) -> pd.Series:
    """Integer attack-stage id (0 = benign). Falls back to the binary label."""
    if config.STAGE_COL not in df.columns:
        return label.copy()
    st = df[config.STAGE_COL]
    if pd.api.types.is_numeric_dtype(st):
        out = st.fillna(0).astype(int)
    else:
        s = st.astype(str).str.strip().str.lower()
        benign = {b.lower() for b in config.BENIGN_NAMES} | {"nan", ""}
        codes, _ = pd.factorize(s.where(~s.isin(benign)), sort=True)
        out = pd.Series(codes + 1, index=df.index)      # -1 (benign) -> 0
    out = out.where(label == 1, 0)
    return out.astype(int)


# ------------------------------------------------------------------ loading
def load_and_clean(csv_path):
    """Read the raw CSV and return (df, feature_cols).

    Leak-free by construction: only fixed rules are applied (drop exact
    duplicate rows, inf/NaN -> 0). Nothing is fitted here.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if config.LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{config.LABEL_COL}' not found. "
                         f"Set ATFL_LABEL_COL. Columns: {list(df.columns)[:20]}...")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # contiguous index for concat in clean_chunk

    label = normalize_labels(df[config.LABEL_COL])
    stage = stage_codes(df, label)

    feat_cols = [c for c in df.columns
                 if c not in config.NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("No numeric feature columns found.")

    X = _numeric_features(df, feat_cols)
    out = pd.concat([
        pd.Series(np.arange(len(df)), name="row_id"),
        label.rename("label"),
        stage.rename("stage"),
        X.astype("float32"),
    ], axis=1)
    out.attrs["dropped_duplicates"] = n0 - len(df)
    return out, feat_cols


def infer_feature_columns(sample: pd.DataFrame) -> list:
    """Numeric model features from a small dataframe sample (same columns as raw CSV)."""
    return [c for c in sample.columns
            if c not in config.NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(sample[c])]


def _label_usecols():
    cols = [config.LABEL_COL]
    if config.STAGE_COL in (config.LABEL_COL,):
        return cols
    return cols


def load_labels_chunked(csv_path, chunksize: int = 500_000) -> np.ndarray:
    """Attack labels for every row (memory-safe for multi-million-row CSVs)."""
    usecols = [config.LABEL_COL]
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols, low_memory=False):
        parts.append(normalize_labels(chunk[config.LABEL_COL]).to_numpy(dtype=np.int8))
    return np.concatenate(parts)


def _numeric_features(chunk: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    X = chunk[feat_cols].apply(pd.to_numeric, errors="coerce")
    return X.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_chunk(chunk: pd.DataFrame, feat_cols: list, row_id_start: int) -> pd.DataFrame:
    chunk = chunk.reset_index(drop=True)
    n = len(chunk)
    label = normalize_labels(chunk[config.LABEL_COL])
    stage = stage_codes(chunk, label)
    X = _numeric_features(chunk, feat_cols)
    return pd.concat([
        pd.Series(np.arange(row_id_start, row_id_start + n), name="row_id"),
        label.rename("label"),
        stage.rename("stage"),
        X.astype("float32"),
    ], axis=1)


def write_splits_chunked(csv_path, train_ids, val_ids, test_ids, feat_cols: list,
                         split_dir, chunksize: int = 500_000) -> None:
    """Second pass: stream raw CSV rows into train/val/test CSV files."""
    from pathlib import Path

    split_dir = Path(split_dir)
    paths = {k: split_dir / f"{k}.csv" for k in ("train", "val", "test")}
    for p in paths.values():
        if p.exists():
            p.unlink()
    header_written = {k: False for k in paths}
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if config.LABEL_COL not in chunk.columns:
            raise ValueError(f"Label column '{config.LABEL_COL}' not found.")
        clean = clean_chunk(chunk, feat_cols, offset)
        offset += len(chunk)
        for name, idset in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            part = clean[clean["row_id"].isin(idset)]
            if len(part) == 0:
                continue
            part.to_csv(paths[name], mode="a", header=not header_written[name], index=False)
            header_written[name] = True


def audit_csv_chunked(csv_path, chunksize: int = 500_000) -> dict:
    """Row/attack counts and basic quality stats without loading the full file."""
    rows = attacks = nan_cells = inf_cells = 0
    feat = None
    stage_counts = {}
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if config.LABEL_COL not in chunk.columns:
            raise ValueError(f"Label column '{config.LABEL_COL}' not found.")
        label = normalize_labels(chunk[config.LABEL_COL])
        rows += len(chunk)
        attacks += int(label.sum())
        if feat is None:
            feat = infer_feature_columns(chunk)
        num = _numeric_features(chunk, feat)
        nan_cells += int(num.isna().sum().sum())
        inf_cells += int(np.isinf(num.to_numpy(dtype="float64")).sum())
        if config.STAGE_COL in chunk.columns:
            vc = chunk.loc[label == 1, config.STAGE_COL].value_counts()
            for k, v in vc.items():
                stage_counts[str(k)] = stage_counts.get(str(k), 0) + int(v)
    return {
        "file": str(csv_path),
        "rows": rows,
        "attack_rows": attacks,
        "benign_rows": rows - attacks,
        "attack_rate": attacks / max(rows, 1),
        "all_normal_accuracy": 1 - attacks / max(rows, 1),
        "duplicate_rows": 0,
        "duplicate_note": "not computed in chunked audit (file too large for in-memory dedup)",
        "nan_cells": nan_cells,
        "inf_cells": inf_cells,
        "numeric_feature_columns": len(feat or []),
        "stage_counts": stage_counts or None,
    }


# ------------------------------------------------------------------ scaling
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def local_stats(X: np.ndarray) -> dict:
    """Sufficient statistics computed by ONE client on its own training rows."""
    Z = signed_log1p(X.astype("float64"))
    return {"n": int(Z.shape[0]), "sum": Z.sum(0), "sumsq": (Z ** 2).sum(0)}


def combine_stats(stats: list):
    """Server side: global mean/std from client sufficient statistics."""
    n = sum(s["n"] for s in stats)
    total = sum(s["sum"] for s in stats)
    totsq = sum(s["sumsq"] for s in stats)
    mean = total / n
    var = np.maximum(totsq / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return np.asarray(mean, dtype="float64"), np.asarray(std, dtype="float64"), n


def transform(X: np.ndarray, mean, std, clip: float = 5.0) -> np.ndarray:
    Z = (signed_log1p(X.astype("float64")) - mean) / std
    return np.clip(Z, -clip, clip).astype("float32")
