"""Collect results into reports/evidence.md plus three figures.

Colour follows the method (same slot everywhere); lines are 2px; one axis per
chart; a legend is always present for 2+ series.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_curve

from atfl import config
from atfl.utils import ensure_dirs, load_json, log

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
METHOD_COLOR = {"fedavg_homog": SLOT["blue"], "fedavg_hetero": SLOT["orange"],
                "atfl": SLOT["aqua"], "local_ensemble": SLOT["yellow"]}
METHOD_LABEL = {"fedavg_homog": "FedAvg (BiLSTM only)", "fedavg_hetero": "FedAvg hetero + fusion",
                "atfl": "ATFL", "local_ensemble": "Local ensemble (reference)"}
CLIENT_COLORS = [SLOT["blue"], SLOT["orange"], SLOT["aqua"]]


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def fig_pr(figdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    for m in METHOD_COLOR:
        f = config.PRED_DIR / f"{m}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p, r, _ = precision_recall_curve(z["y_test"], z["p_test"])
        ax.plot(r, p, color=METHOD_COLOR[m], linewidth=2, label=METHOD_LABEL[m])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    style(ax, "Recall", "Precision", "Precision-recall on the untouched test set")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(figdir / "pr_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_weights(figdir):
    h = load_json(config.OUT_DIR / "history_atfl.json")["history"]
    fams = sorted({r["family"] for r in h})
    fig, axes = plt.subplots(1, len(fams), figsize=(6.4 * len(fams) / 1.4, 3.8), facecolor=SURFACE, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        cids = sorted({r["client_id"] for r in h if r["family"] == fam})
        for i, cid in enumerate(cids):
            rows = [r for r in h if r["family"] == fam and r["client_id"] == cid]
            ax.plot([r["round"] for r in rows], [r["weight"] for r in rows],
                    color=CLIENT_COLORS[i % 3], linewidth=2, label=f"client {cid}")
        style(ax, "Round", "Aggregation weight", f"ATFL weights, {fam} family")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(figdir / "atfl_weights.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_convergence(figdir):
    data = {m: load_json(config.OUT_DIR / f"history_{m}.json")["convergence"] for m in ("fedavg_hetero", "atfl")
            if (config.OUT_DIR / f"history_{m}.json").exists()}
    fams = sorted({r["family"] for rows in data.values() for r in rows})
    fig, axes = plt.subplots(1, len(fams), figsize=(6.4 * len(fams) / 1.4, 3.8), facecolor=SURFACE, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for m, rows in data.items():
            rr = [r for r in rows if r["family"] == fam]
            ax.plot([r["round"] for r in rr], [r["val_pr_auc"] for r in rr],
                    color=METHOD_COLOR[m], linewidth=2, label=METHOD_LABEL[m])
        style(ax, "Round", "Global-val PR-AUC", f"Convergence, {fam} family")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1)                      # fixed scale: do not magnify noise
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(figdir / "convergence.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ensure_dirs(config.REPORT_DIR / "figures")
    figdir = config.REPORT_DIR / "figures"
    fig_pr(figdir)
    fig_weights(figdir)
    fig_convergence(figdir)

    res = load_json(config.REPORT_DIR / "test_metrics.json")
    split = load_json(config.SPLIT_DIR / "split_manifest.json")
    clients = load_json(config.CLIENT_DIR / "manifest.json")["clients"]
    audit = load_json(config.REPORT_DIR / "privacy_audit.json") if (config.REPORT_DIR / "privacy_audit.json").exists() else None
    hist = load_json(config.OUT_DIR / "history_atfl.json")["history"]
    last = [h for h in hist if h["round"] == max(x["round"] for x in hist)]

    md = ["# ATFL evidence report", "",
          f"Test set: {res['test_rows']:,} rows, {res['test_attacks']} attacks. "
          f"A detector that always says 'normal' would score {res['all_normal_accuracy']:.4%} accuracy "
          "and detect nothing, so judge methods on PR-AUC, recall and F1.", "",
          "## Data split", "", "| Split | Rows | Attacks | Attack rate |", "|---|---|---|---|"]
    md += [f"| {n} | {s['rows']:,} | {s['attacks']:,} | {s['attack_rate']:.4%} |" for n, s in split["splits"].items()]
    md += ["", "## Clients", "", "| Client | Architecture | Train rows | Train attacks | Attack stages seen |", "|---|---|---|---|---|"]
    md += [f"| {c['client_id']} | {c['arch']} | {c['n_train']:,} | {c['n_train_attack']} | {c['attack_stages_seen']} |"
           for c in clients]
    md += ["", "## Test results", "", (config.REPORT_DIR / "test_metrics.md").read_text(),
           "![PR curves](figures/pr_curves.png)", "",
           "## ATFL trust and weights, final round", "",
           "| Client | Family | N | F1 | PR-AUC | Recall | Consistency | Trust | Weight |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {h['client_id']} | {h['family']} | {h['n_samples']:,} | {h['f1']:.3f} | {h['pr_auc']:.3f} | "
           f"{h['recall']:.3f} | {h['consistency']:.3f} | {h['trust']:.3f} | {h['weight']:.3f} |" for h in last]
    md += ["", "![ATFL weights](figures/atfl_weights.png)", "", "![Convergence](figures/convergence.png)", ""]
    if audit:
        md += ["## Privacy and leakage audit", "", f"{len(audit['checks']) - audit['failed']}/{len(audit['checks'])} checks passed. "
               "Differential privacy, secure aggregation and Byzantine robustness are NOT implemented or claimed. "
               "See privacy_audit.md.", ""]
    md += ["## How to read this honestly", "",
           "- Differences between methods are only meaningful if they exceed run-to-run variance. "
           "Repeat with several seeds (ATFL_SEED) before drawing conclusions.",
           "- With very few test attacks, one or two extra detections moves recall and F1 noticeably.",
           "- Synthetic data only validates that the pipeline runs; it says nothing about real APT performance."]
    (config.REPORT_DIR / "evidence.md").write_text("\n".join(md) + "\n")
    log(f"wrote {config.REPORT_DIR / 'evidence.md'} and 3 figures")


if __name__ == "__main__":
    main()
