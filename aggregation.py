"""Aggregation within ONE architecture family (parameters of different
architectures are never averaged together)."""
import numpy as np
import torch


def mix_weights(mode: str, n_samples, trust) -> np.ndarray:
    """fedavg: N | trust_n: N*T | trust_only: T   (all normalised to sum 1)."""
    n = np.asarray(n_samples, dtype="float64")
    t = np.asarray(trust, dtype="float64")
    if mode == "fedavg":
        w = n
    elif mode == "trust_n":
        w = n * t
    elif mode == "trust_only":
        w = t
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.ones_like(w)
        s = w.sum()
    return w / s


def aggregate_states(states: list, weights) -> dict:
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(float(w) * s[k].float() for w, s in zip(weights, states)).to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out


def flatten_delta(state: dict, ref: dict) -> np.ndarray:
    parts = [(state[k].float() - ref[k].float()).flatten()
             for k in state if state[k].is_floating_point()]
    return torch.cat(parts).numpy()
