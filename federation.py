"""Federated training engine.

One engine serves every method:
  * FedAvg (homogeneous)  -> all clients BiLSTM, weight_mode="fedavg"
  * FedAvg (hetero)       -> per-family size-weighted average + decision fusion
  * ATFL                  -> per-family trust-weighted average + decision fusion
Raw client data never leaves a Client; the server only receives an *upload*:
  {client_id, state (model parameters), n_samples, metrics (f1, pr_auc, recall, threshold)}
"""
import numpy as np
import torch
import torch.nn as nn

from . import config
from .aggregation import aggregate_states, flatten_delta, mix_weights
from .fusion import choose_weights, fuse
from .metrics import best_threshold, compute_metrics, pr_auc
from .models import build_model
from .trust import compute_trust, consistency_scores
from .utils import log, seed_everything


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def predict_proba(model, X, device, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype="float32")


def train_local(model, X, y, epochs, device, seed):
    """Plain local training with a capped positive-class weight."""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pw = min(neg / max(pos, 1.0), config.POS_WEIGHT_CAP)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    gen = torch.Generator().manual_seed(int(seed))
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype("float32"))
    model.to(device).train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
    return model


def val_metrics(p, y) -> dict:
    """Metrics a client reports from ITS OWN validation set."""
    thr = best_threshold(y, p)
    m = compute_metrics(y, p, thr)
    return {"f1": m["f1"], "pr_auc": m["pr_auc"], "recall": m["recall"], "threshold": thr}


def make_upload(cid, state, n_samples, metrics) -> dict:
    return {"client_id": int(cid), "state": state,
            "n_samples": int(n_samples), "metrics": metrics}


def describe_upload(u) -> dict:
    """Schema of what the server receives (used by the privacy audit)."""
    return {
        "fields": sorted(u.keys()),
        "state_tensors": {k: list(v.shape) for k, v in u["state"].items()},
        "metric_fields": sorted(u["metrics"].keys()),
        "types": {"client_id": "int", "state": "dict[str, Tensor]",
                  "n_samples": "int", "metrics": "dict[str, float]"},
    }


# ------------------------------------------------------------------ engine
def run_federated(clients, archs, weight_mode, rounds, local_epochs, global_val, test,
                  use_consistency=True, tag="run", device=None, seed=config.SEED):
    device = device or torch.device("cpu")
    families = sorted(set(archs.values()))
    members = {f: sorted(c for c in clients if archs[c] == f) for f in families}
    n_feat = next(iter(clients.values()))["X_train"].shape[1]

    gstate = {}
    for i, f in enumerate(families):
        seed_everything(seed + i)
        gstate[f] = {k: v.detach().cpu().clone() for k, v in build_model(f, n_feat).state_dict().items()}

    history, convergence, schema = [], [], None
    for r in range(1, rounds + 1):
        for f in families:
            uploads = []
            for cid in members[f]:                         # ---- client side
                c = clients[cid]
                model = build_model(f, n_feat)
                model.load_state_dict(gstate[f])
                train_local(model, c["X_train"], c["y_train"], local_epochs, device,
                            seed=seed + 1000 * r + cid)
                model.cpu()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pv = predict_proba(model, c["X_val"], torch.device("cpu"))
                uploads.append(make_upload(cid, state, len(c["y_train"]), val_metrics(pv, c["y_val"])))
            if schema is None:
                schema = describe_upload(uploads[0])

            deltas = [flatten_delta(u["state"], gstate[f]) for u in uploads]   # ---- server side
            cons = consistency_scores(deltas) if use_consistency else [1.0] * len(uploads)
            trusts = [compute_trust(u["metrics"], cs) for u, cs in zip(uploads, cons)]
            w = mix_weights(weight_mode, [u["n_samples"] for u in uploads], trusts)
            gstate[f] = aggregate_states([u["state"] for u in uploads], w)

            for u, cs, t, wi in zip(uploads, cons, trusts, w):
                history.append({"round": r, "family": f, "client_id": u["client_id"],
                                "n_samples": u["n_samples"], **{k: u["metrics"][k] for k in ("f1", "pr_auc", "recall")},
                                "consistency": cs, "trust": t, "weight": float(wi)})

            gm = build_model(f, n_feat)
            gm.load_state_dict(gstate[f])
            pg = predict_proba(gm, global_val["X"], torch.device("cpu"))
            convergence.append({"round": r, "family": f, "val_pr_auc": pr_auc(global_val["y"], pg)})
        last = [c["val_pr_auc"] for c in convergence if c["round"] == r]
        log(f"[{tag}] round {r}/{rounds}  global-val PR-AUC by family: "
            + ", ".join(f"{f}={v:.3f}" for f, v in zip(families, last)))

    pv, pt = {}, {}
    for f in families:
        gm = build_model(f, n_feat)
        gm.load_state_dict(gstate[f])
        pv[f] = predict_proba(gm, global_val["X"], torch.device("cpu"))
        pt[f] = predict_proba(gm, test["X"], torch.device("cpu"))
    fw = choose_weights(global_val["y"], pv)
    return {"p_val": fuse(pv, fw), "p_test": fuse(pt, fw), "family_val": pv, "family_test": pt,
            "fusion_weights": fw, "history": history, "convergence": convergence,
            "upload_schema": schema, "weight_mode": weight_mode,
            "use_consistency": use_consistency, "families": families}
