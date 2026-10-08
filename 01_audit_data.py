"""Audit the RAW file: size, class balance, missing/inf values, duplicates, constants."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.data import audit_csv_chunked, normalize_labels
from atfl.utils import log, save_json

CHUNKED_ROW_THRESHOLD = 2_000_000


def main():
    path = config.RAW_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found: {path}\n"
            f"Run: python run_all.py --synthetic --quick\n"
            f"Or: python run_quick_real.py\n"
            f"Or set ATFL_RAW_CSV to a file under {config.RAW_DIR}"
        )
    log(f"auditing {path}")
    use_chunked = path.stat().st_size > 500_000_000  # ~500 MiB on disk

    if use_chunked:
        rep = audit_csv_chunked(path)
        rep["chunked_audit"] = True
        feat_n = rep["numeric_feature_columns"]
        const_n = 0
    else:
        df = pd.read_csv(path, low_memory=False)
        label = normalize_labels(df[config.LABEL_COL])
        num = df.select_dtypes(include=[np.number])
        feat = [c for c in num.columns if c not in config.NON_FEATURE_COLS]
        n_att = int(label.sum())
        rep = {
            "file": str(path),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric_feature_columns": len(feat),
            "non_numeric_columns": [c for c in df.columns if c not in num.columns],
            "attack_rows": n_att,
            "benign_rows": int(len(df) - n_att),
            "attack_rate": n_att / max(len(df), 1),
            "all_normal_accuracy": 1 - n_att / max(len(df), 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "nan_cells": int(df[feat].isna().sum().sum()),
            "inf_cells": int(np.isinf(df[feat].to_numpy(dtype="float64")).sum()),
            "constant_features": [c for c in feat if df[c].nunique(dropna=True) <= 1],
            "chunked_audit": False,
        }
        if config.STAGE_COL in df.columns:
            rep["stage_counts"] = {
                str(k): int(v) for k, v in df.loc[label == 1, config.STAGE_COL].value_counts().items()
            }
        feat_n = len(feat)
        const_n = len(rep.get("constant_features", []))

    n_att = rep["attack_rows"]
    log(f"rows={rep['rows']:,}  attacks={n_att:,} ({rep['attack_rate']:.4%})")
    log(f"a model that says 'everything is normal' scores accuracy {rep['all_normal_accuracy']:.4%} "
        f"and detects 0 attacks -> accuracy is not a usable metric here")
    if use_chunked:
        log(f"chunked audit: NaN cells={rep['nan_cells']}  inf cells={rep['inf_cells']}  "
            f"features={feat_n}")
    else:
        log(f"duplicates={rep['duplicate_rows']}  NaN cells={rep['nan_cells']}  inf cells={rep['inf_cells']}  "
            f"constant features={const_n}")
    save_json(rep, config.REPORT_DIR / "data_audit.json")
    log(f"saved {config.REPORT_DIR / 'data_audit.json'}")


if __name__ == "__main__":
    main()
