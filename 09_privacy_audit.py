"""Privacy / leakage audit.

Verifies what the prototype actually does, and states what it does NOT claim.
Exit code 1 if any check FAILS.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from atfl import config
from atfl.data import combine_stats, local_stats
from atfl.utils import load_json, log, save_json

ALLOWED_UPLOAD_FIELDS = {"client_id", "state", "n_samples", "metrics"}
ALLOWED_METRIC_FIELDS = {"f1", "pr_auc", "recall", "threshold"}

CLAIMS = [
    {"mechanism": "Raw data stays on the client", "implemented": True,
     "evidence": "uploads contain only parameters, n_samples and 4 validation scalars (check 4)"},
    {"mechanism": "Leakage-free split and preprocessing", "implemented": True,
     "evidence": "checks 1-3, 5"},
    {"mechanism": "Formal differential privacy", "implemented": False,
     "evidence": "no clipping or noise is applied to updates"},
    {"mechanism": "Secure aggregation", "implemented": False,
     "evidence": "the server sees each client's plain update"},
    {"mechanism": "Byzantine robustness", "implemented": False,
     "evidence": "trust weighting is a performance heuristic, not a tested defence against malicious clients"},
]


def check(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return {"check": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def ids(path):
    return set(pd.read_csv(path, usecols=["row_id"])["row_id"])


def main():
    res = []
    feats = load_json(config.SPLIT_DIR / "split_manifest.json")["features"]
    tr, va, te = (ids(config.SPLIT_DIR / f"{n}.csv") for n in ("train", "val", "test"))

    # 1 split disjointness
    ov = len(tr & va) + len(tr & te) + len(va & te)
    res.append(check("1. train/val/test row ids are disjoint", ov == 0, f"{ov} overlapping rows"))

    # 2 client partitions
    man = load_json(config.CLIENT_DIR / "manifest.json")
    c_tr, c_va, dup = [], [], 0
    for c in man["clients"]:
        z = np.load(config.PROC_DIR / f"client_{c['client_id']}.npz")
        c_tr.append(set(z["rid_train"].tolist()))
        c_va.append(set(z["rid_val"].tolist()))
    for sets in (c_tr, c_va):
        allr = [r for s in sets for r in s]
        dup += len(allr) - len(set(allr))
    in_test = sum(len(s & te) for s in c_tr + c_va)
    outside = sum(len(s - tr) for s in c_tr) + sum(len(s - va) for s in c_va)
    res.append(check("2. clients hold disjoint rows, none from the test set",
                     dup == 0 and in_test == 0 and outside == 0,
                     f"{dup} rows shared between clients, {in_test} test rows in clients, "
                     f"{outside} rows outside their split"))

    # 3 scaler fitted on client-train only
    stats = [local_stats(pd.read_csv(config.CLIENT_DIR / f"client_{c['client_id']}_train.csv")[feats]
                         .to_numpy(dtype="float64")) for c in man["clients"]]
    mean, std, _ = combine_stats(stats)
    sc = load_json(config.PROC_DIR / "scaler.json")
    ok = np.allclose(mean, sc["mean"]) and np.allclose(std, sc["std"])
    res.append(check("3. scaler equals statistics of client TRAIN rows only", ok,
                     "recomputed mean/std match scaler.json" if ok else "scaler does not match"))

    # 4 what the server receives
    schemas = []
    for f in sorted(config.PRED_DIR.glob("*.json")):
        m = load_json(f)
        if m.get("kind") == "federated":
            schemas.append((m["method"], m["upload_schema"]))
    bad = [n for n, s in schemas
           if not set(s["fields"]) <= ALLOWED_UPLOAD_FIELDS or not set(s["metric_fields"]) <= ALLOWED_METRIC_FIELDS]
    res.append(check("4. uploads contain only parameters + scalars (no raw rows)",
                     bool(schemas) and not bad,
                     f"{len(schemas)} federated runs inspected; fields={sorted(ALLOWED_UPLOAD_FIELDS)}; "
                     f"metrics={sorted(ALLOWED_METRIC_FIELDS)}" + (f"; violations: {bad}" if bad else "")))

    # 5 test never used for selection
    flagged = [f.stem for f in config.PRED_DIR.glob("*.json")
               if load_json(f).get("test_used_for_selection") is not False]
    res.append(check("5. thresholds and fusion weights chosen on validation only", not flagged,
                     "all prediction records declare selection on global_val" if not flagged else f"flagged: {flagged}"))

    fails = sum(r["status"] == "FAIL" for r in res)
    out = {"checks": res, "claims": CLAIMS, "failed": fails,
           "notes": ["Global val set stands in for a small server-held validation set; "
                     "in a real deployment this must exist or thresholds must be chosen differently.",
                     "n_samples and validation metrics are shared with the server and reveal "
                     "client dataset size and performance.",
                     "Scaler statistics (mean/std) are shared in aggregate form."]}
    save_json(out, config.REPORT_DIR / "privacy_audit.json")

    md = ["# Privacy and leakage audit", "", "| Check | Status | Detail |", "|---|---|---|"]
    md += [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in res]
    md += ["", "## What is and is not claimed", "", "| Mechanism | Implemented | Note |", "|---|---|---|"]
    md += [f"| {c['mechanism']} | {'yes' if c['implemented'] else 'NO'} | {c['evidence']} |" for c in CLAIMS]
    md += ["", "## Caveats"] + [f"- {n}" for n in out["notes"]]
    (config.REPORT_DIR / "privacy_audit.md").write_text("\n".join(md) + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
