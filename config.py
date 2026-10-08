"""Central configuration.

Every value can be overridden with an environment variable ATFL_<NAME>,
e.g.  ATFL_ROUNDS=20 ATFL_RAW_CSV=/path/to/cicapt.csv python scripts/02_global_split.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _get(name, default, cast=str):
    v = os.environ.get(f"ATFL_{name}")
    if v is None:
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes")
    return cast(v)


# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
CLIENT_DIR = DATA_DIR / "clients"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "outputs"
PRED_DIR = OUT_DIR / "preds"
REPORT_DIR = ROOT / "reports"

# ---------------------------------------------------------------- data
SEED = _get("SEED", 42, int)
RAW_CSV = Path(_get("RAW_CSV", str(RAW_DIR / "cicapt_iiot_2024.csv")))
LABEL_COL = _get("LABEL_COL", "label")      # binary 0/1, or text with benign names below
STAGE_COL = _get("STAGE_COL", "stage")      # optional; used ONLY to build non-IID clients
BENIGN_NAMES = ("benign", "normal", "0", "none", "background")
# columns that must never be used as model features
NON_FEATURE_COLS = {
    "row_id", "label", "stage", "Label", "Stage",
    "timestamp", "Timestamp", "Flow ID", "flow_id",
    "Src IP", "Dst IP", "Source IP", "Destination IP", "src_ip", "dst_ip",
    "ts", "subLabel", "subLabelCat", "Protocol_name", "MAC", "DS status",
}
NON_FEATURE_COLS.update({LABEL_COL, STAGE_COL})

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2                               # test = 1 - TRAIN_FRAC - VAL_FRAC
# 0 = keep natural prevalence in train. >0 = cap benign rows in TRAIN only
# (val/test always keep natural prevalence).
NORMAL_TRAIN_CAP = _get("NORMAL_TRAIN_CAP", 0, int)

# ---------------------------------------------------------------- clients
NUM_CLIENTS = _get("NUM_CLIENTS", 5, int)
DIRICHLET_ALPHA = _get("DIRICHLET_ALPHA", 0.5, float)   # attack-stage skew (smaller = more non-IID)
QTY_ALPHA = _get("QTY_ALPHA", 2.0, float)               # benign quantity skew
MIN_ATTACK_TRAIN = 8
MIN_ATTACK_VAL = 3
ARCH_CYCLE = ("bilstm", "transformer")                  # client i -> ARCH_CYCLE[i % 2]

# ---------------------------------------------------------------- training
ROUNDS = _get("ROUNDS", 10, int)
LOCAL_EPOCHS = _get("LOCAL_EPOCHS", 2, int)
BATCH_SIZE = _get("BATCH_SIZE", 256, int)
LR = _get("LR", 1e-3, float)
POS_WEIGHT_CAP = 50.0
GRAD_CLIP = 1.0

# ---------------------------------------------------------------- models
SEQ_LEN = 8          # feature vector is cut into SEQ_LEN chunks -> sequence
HIDDEN = 64
D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
DROPOUT = 0.1

# ---------------------------------------------------------------- trust
#   T = 0.35*F1 + 0.25*PR-AUC + 0.20*Recall + 0.20*Consistency
TRUST_W = {"f1": 0.35, "pr_auc": 0.25, "recall": 0.20, "consistency": 0.20}
