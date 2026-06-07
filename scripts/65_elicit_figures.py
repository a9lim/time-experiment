"""Figures for the prefilled-duration probe arm (offline; reads 62/63/64 outputs).

  (a) the readout site: R²(gt) by condition — prefill slot reads clock-elapsed at
      0.98 (vs 0.59 EOT); true=text-reading control, constant=internal; null held
      with no clock.
  (b) one duration axis transfers to natural felt: the scripted clock-elapsed
      probe applied to natural slots vs what the model felt (ρ≈0.83).
  (c) it captures the felt ordering but compresses magnitude (per variant).

    TIME_MODEL=gemma python scripts/65_elicit_figures.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.config import DATA_DIR, FIGURES_DIR, resolve_model  # noqa: E402

C_INT, C_TEXT, C_EOT, C_NULL = "#0e7490", "#7dd3fc", "#9ca3af", "#d1d5db"
VAR_C = {"neutral": "#3b6", "affect_dense": "#b36", "time_language": "#36b"}


def _sec(x, _=None):
    if x <= 0:
        return ""
    for t, d, u in ((90, 1, "s"), (5400, 60, "m"), (129600, 3600, "h"), (1e12, 86400, "d")):
        if x < t:
            v = x / d
            return f"{v:.0f}{u}" if v >= 1 else f"{v:.1f}{u}"
    return f"{x:.0f}s"


SECFMT = FuncFormatter(_sec)


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    edir = DATA_DIR / f"{base.short_name}_elicit"
    summ = json.loads((edir / "elicit_summary.json").read_text())
    vt = json.loads((edir / "verbal_target_summary.json").read_text())
    nat = list(csv.DictReader((edir / "natural_reads.csv").open()))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.7))
    fig.suptitle("Prefilled-duration probe: read elapsed at the point of use "
                 "(“It's been ▮”) — one duration axis, clock or felt",
                 fontsize=13, fontweight="bold", y=1.02)

    # (a) R² by condition
    cond = summ["conditions"]
    names = ["EOT\nstack", "ts / true\n(text)", "ts / const\n(internal)",
             "un / true\n(text)", "un / const\n(no clock)"]
    vals = [summ["eot_stack_r2"], cond["timestamped/true"]["r2_gt"],
            cond["timestamped/constant"]["r2_gt"], cond["untimestamped/true"]["r2_gt"],
            cond["untimestamped/constant"]["r2_gt"]]
    cols = [C_EOT, C_TEXT, C_INT, C_TEXT, C_NULL]
    bars = axA.bar(names, [max(v, 0) for v in vals], color=cols, width=0.7,
                   edgecolor="white")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.axhline(0, color="#999", lw=0.7)
    axA.set_ylim(-0.05, 1.12)
    axA.set_ylabel("CV $R^2$ (log elapsed)")
    axA.set_title("(a) the readout site: clock-elapsed reads at 0.98 (vs 0.59 EOT)")
    axA.tick_params(axis="x", labelsize=8)

    # (b) unified axis -> natural felt
    felt = np.array([float(r["felt_s"]) for r in nat])
    cread = np.array([float(r["clock_read_s"]) for r in nat])
    for v in sorted(set(r["variant"] for r in nat)):
        m = np.array([r["variant"] == v for r in nat])
        axB.scatter(felt[m], cread[m], s=42, color=VAR_C.get(v, "#888"),
                    edgecolor="white", linewidth=0.5, label=v)
    lim = [min(felt.min(), cread.min()) * 0.6, max(felt.max(), cread.max()) * 1.6]
    axB.plot(lim, lim, "k--", alpha=0.4, lw=1, label="y = x")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.xaxis.set_major_formatter(SECFMT); axB.yaxis.set_major_formatter(SECFMT)
    axB.set_xlabel("what the model felt (stated)")
    axB.set_ylabel("clock-elapsed axis read")
    axB.set_title(f"(b) one axis transfers to natural felt "
                  f"(ρ={vt['crossaxis_clockprobe_vs_felt_rho']:.2f})")
    axB.legend(fontsize=7.5, loc="upper left")

    # (c) per-variant felt vs slot-read (ordering vs compression)
    variants = ["neutral", "affect_dense", "time_language"]
    feltm = [np.median([float(r["felt_s"]) for r in nat if r["variant"] == v]) for v in variants]
    readm = [np.median([float(r["slot_read_s"]) for r in nat if r["variant"] == v]) for v in variants]
    x = np.arange(len(variants)); w = 0.38
    axC.bar(x - w / 2, feltm, w, color="#ea580c", label="felt (model says)")
    axC.bar(x + w / 2, readm, w, color=C_INT, label="slot-axis read")
    axC.set_yscale("log"); axC.yaxis.set_major_formatter(SECFMT)
    axC.set_xticks(x); axC.set_xticklabels(["neutral", "affect", "time-words"])
    axC.set_ylabel("duration")
    axC.set_title("(c) captures the ordering, compresses the magnitude")
    axC.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / f"{base.short_name}_elicit" / "fig_elicit.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
