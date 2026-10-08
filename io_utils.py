"""Loading processed data and saving predictions."""
import numpy as np

from . import config
from .utils import load_json, save_json, ensure_dirs


def load_manifest() -> dict:
    return load_json(config.CLIENT_DIR / "manifest.json")


def load_clients() -> tuple:
    """Return (clients: {cid: npz-dict}, archs: {cid: arch})."""
    man = load_manifest()
    clients, archs = {}, {}
    for c in man["clients"]:
        cid = c["client_id"]
        z = np.load(config.PROC_DIR / f"client_{cid}.npz")
        clients[cid] = {k: z[k] for k in z.files}
        archs[cid] = c["arch"]
    return clients, archs


def load_eval_sets() -> tuple:
    gv = np.load(config.PROC_DIR / "global_val.npz")
    te = np.load(config.PROC_DIR / "test.npz")
    return ({"X": gv["X"], "y": gv["y"]}, {"X": te["X"], "y": te["y"]})


def save_preds(method: str, y_val, p_val, y_test, p_test, family_val=None,
               family_test=None, meta=None) -> None:
    ensure_dirs(config.PRED_DIR)
    arrays = {"y_val": y_val, "p_val": p_val, "y_test": y_test, "p_test": p_test}
    for f, p in (family_val or {}).items():
        arrays[f"p_val__{f}"] = p
    for f, p in (family_test or {}).items():
        arrays[f"p_test__{f}"] = p
    np.savez(config.PRED_DIR / f"{method}.npz", **arrays)
    info = {"method": method,
            "selection_data": "global_val",     # threshold + fusion weights use VAL only
            "test_used_for_selection": False}
    info.update(meta or {})
    save_json(info, config.PRED_DIR / f"{method}.json")
