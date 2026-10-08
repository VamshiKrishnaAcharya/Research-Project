"""Baseline 1: every client trains alone (no collaboration).

Each client gets the same total training budget as a federated client
(ROUNDS x LOCAL_EPOCHS epochs) and is then evaluated on the global val/test.
`local_ensemble` averages the local models' probabilities as a reference only
(a real deployment would need to share inference outputs to do that).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from atfl import config
from atfl.federation import predict_proba, train_local, val_metrics
from atfl.io_utils import load_clients, load_eval_sets, save_preds
from atfl.models import build_model
from atfl.utils import log, save_json, seed_everything


def main():
    clients, archs = load_clients()
    gval, test = load_eval_sets()
    n_feat = gval["X"].shape[1]
    epochs = config.ROUNDS * config.LOCAL_EPOCHS
    dev = torch.device("cpu")

    pvs, pts, summary = [], [], []
    for cid, c in clients.items():
        seed_everything(config.SEED + 500 + cid)
        model = build_model(archs[cid], n_feat)
        train_local(model, c["X_train"], c["y_train"], epochs, dev, seed=config.SEED + 500 + cid)
        model.cpu()
        pv = predict_proba(model, gval["X"], dev)
        pt = predict_proba(model, test["X"], dev)
        own = val_metrics(predict_proba(model, c["X_val"], dev), c["y_val"])
        save_preds(f"local_c{cid}", gval["y"], pv, test["y"], pt,
                   meta={"kind": "local", "client_id": cid, "arch": archs[cid]})
        pvs.append(pv)
        pts.append(pt)
        summary.append({"client_id": cid, "arch": archs[cid], **own})
        log(f"[local] client {cid} ({archs[cid]}) trained {epochs} epochs; own-val PR-AUC={own['pr_auc']:.3f}")

    save_preds("local_ensemble", gval["y"], np.mean(pvs, 0), test["y"], np.mean(pts, 0),
               meta={"kind": "local_ensemble_reference"})
    save_json(summary, config.OUT_DIR / "local_summary.json")


if __name__ == "__main__":
    main()
