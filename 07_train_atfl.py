"""ATFL: trust-weighted aggregation inside each architecture family + decision fusion.

  T = 0.35*F1 + 0.25*PR-AUC + 0.20*Recall + 0.20*Consistency
  aggregation weight = N * T, normalised within the family

Use --ablations to also run:
  atfl_trust_only      weight = T (sample count ignored)
  atfl_no_consistency  consistency term fixed to 1.0 (no update-agreement signal)
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from atfl import config
from atfl.federation import run_federated
from atfl.io_utils import load_clients, load_eval_sets, save_preds
from atfl.utils import log, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    a = ap.parse_args()

    clients, archs = load_clients()
    gval, test = load_eval_sets()
    dev = torch.device("cpu")

    runs = [("atfl", "trust_n", True)]
    if a.ablations:
        runs += [("atfl_trust_only", "trust_only", True), ("atfl_no_consistency", "trust_n", False)]

    for name, mode, use_cons in runs:
        res = run_federated(clients, archs, mode, config.ROUNDS, config.LOCAL_EPOCHS,
                            gval, test, use_consistency=use_cons, tag=name, device=dev)
        save_preds(name, gval["y"], res["p_val"], test["y"], res["p_test"],
                   family_val=res["family_val"], family_test=res["family_test"],
                   meta={"kind": "federated", "weight_mode": mode, "use_consistency": use_cons,
                         "fusion_weights": res["fusion_weights"],
                         "upload_schema": res["upload_schema"]})
        save_json({"history": res["history"], "convergence": res["convergence"]},
                  config.OUT_DIR / f"history_{name}.json")
        last = [h for h in res["history"] if h["round"] == config.ROUNDS]
        for h in last:
            log(f"[{name}] final round  client {h['client_id']} ({h['family']:11s}) "
                f"trust={h['trust']:.3f} weight={h['weight']:.3f}")
        log(f"[{name}] done; fusion weights={res['fusion_weights']}")


if __name__ == "__main__":
    main()
