"""Baseline 2: conventional FedAvg (size-weighted, no trust).

  fedavg_homog  - what standard FedAvg requires: every client uses the SAME
                  architecture (BiLSTM), parameters averaged by sample count.
  fedavg_hetero - same heterogeneous setup as ATFL (BiLSTM + Transformer
                  families, decision fusion) but size-weighted. Isolates the
                  effect of the trust weighting.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from atfl import config
from atfl.federation import run_federated
from atfl.io_utils import load_clients, load_eval_sets, save_preds
from atfl.utils import get_device, log, save_json


def main():
    clients, archs = load_clients()
    gval, test = load_eval_sets()
    dev = torch.device("cpu")

    homog = {cid: "bilstm" for cid in clients}
    runs = [("fedavg_homog", homog), ("fedavg_hetero", archs)]
    for name, arch_map in runs:
        res = run_federated(clients, arch_map, "fedavg", config.ROUNDS, config.LOCAL_EPOCHS,
                            gval, test, use_consistency=True, tag=name, device=dev)
        save_preds(name, gval["y"], res["p_val"], test["y"], res["p_test"],
                   family_val=res["family_val"] if len(res["families"]) > 1 else None,
                   family_test=res["family_test"] if len(res["families"]) > 1 else None,
                   meta={"kind": "federated", "weight_mode": "fedavg",
                         "fusion_weights": res["fusion_weights"],
                         "upload_schema": res["upload_schema"]})
        save_json({"history": res["history"], "convergence": res["convergence"]},
                  config.OUT_DIR / f"history_{name}.json")
        log(f"[{name}] done; fusion weights={res['fusion_weights']}")


if __name__ == "__main__":
    main()
