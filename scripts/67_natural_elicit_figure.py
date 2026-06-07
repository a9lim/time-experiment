"""Naturalistic result, EOT probe vs prefill-slot probe (offline).

On real model-generated conversations there's no ground-truth clock, so the
reference is what the model FELT (its stated duration). The contrast on the same
turns:

  (a) prefill-slot probe — the scripted clock-elapsed axis applied to the natural
      readout slot tracks the felt estimate (ρ≈0.83), ordered by content.
  (b) EOT stack probe — the same conversations, pooled at the EOT, blow up OOD
      and don't track felt (ρ≈0.11).

    TIME_MODEL=gemma python scripts/67_natural_elicit_figure.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.analysis import apply_probe, fit_full  # noqa: E402
from time_experiment.config import DATA_DIR, FIGURES_DIR, resolve_model  # noqa: E402

VAR_C = {"neutral": "#3b6", "affect_dense": "#b36", "time_language": "#36b"}


def _sec(x, _=None):
    if x <= 0:
        return ""
    for t, d, u in ((90, 1, "s"), (5400, 60, "m"), (129600, 3600, "h"),
                    (1.3e6, 86400, "d"), (1e12, 604800, "w")):
        if x < t:
            v = x / d
            return f"{v:.0f}{u}" if v >= 1 else f"{v:.1f}{u}"
    return f"{x:.0e}s"


SECFMT = FuncFormatter(_sec)


def _slot_at(hid, source, rendering, mode, layer):
    """{(id,turn): vec(D)} at `layer` for one (source,rendering,mode). Filenames
    are source__<id>__rendering__mode.npz and the id itself can contain '__'."""
    out = {}
    for p in hid.glob(f"{source}__*__{rendering}__{mode}.npz"):
        parts = p.stem.split("__")
        idd = "__".join(parts[1:-2])      # strip source (first) + rendering,mode (last two)
        d = np.load(p)
        li = list(d["layers"]).index(layer)
        for i, t in enumerate(d["turn_idxs"]):
            out[(idd, int(t))] = d["H"][i, li, :]
    return out


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    edir = DATA_DIR / f"{base.short_name}_elicit"
    summ = json.loads((edir / "elicit_summary.json").read_text())
    Lc = int(summ["conditions"]["timestamped/constant"]["best_layer"])
    hid = edir / "hidden"
    rows = [json.loads(l) for l in (edir / "elicit_rows.jsonl").read_text().splitlines() if l.strip()]

    # Fit the clock-elapsed probe on scripted timestamped/constant slots @ Lc.
    sc = _slot_at(hid, "scripted", "timestamped", "constant", Lc)
    Xs, ys = [], []
    for r in rows:
        if r["source"] == "scripted" and r["rendering"] == "timestamped" and r["mode"] == "constant" \
                and r["gt_elapsed_s"] is not None and (r["id"], r["turn_idx"]) in sc:
            Xs.append(sc[(r["id"], r["turn_idx"])]); ys.append(math.log(max(r["gt_elapsed_s"], 1.0)))
    probe = fit_full(np.asarray(Xs, np.float32), np.asarray(ys))

    # Natural: elicit clock-read @ Lc + EOT raw read + felt, joined by (conv,turn).
    nat_slot = _slot_at(hid, "natural", "untimestamped", "constant", Lc)
    nat = {}
    for r in rows:
        if r["source"] == "natural" and (r["id"], r["turn_idx"]) in nat_slot:
            nat[(r["id"], r["turn_idx"])] = {"variant": r.get("variant")}
    natj = DATA_DIR / f"{base.short_name}_natural" / "naturalistic.jsonl"
    for l in natj.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        if d["rendering"] != "untimestamped":
            continue
        k = (d["conv_id"], d["turn_idx"])
        if k in nat and d.get("felt_s"):
            nat[k]["felt"] = float(d["felt_s"])
            nat[k]["eot"] = float(math.exp(min(d["internal_log_raw"], 40)))

    pts = [(k, v) for k, v in nat.items() if "felt" in v]
    felt = np.array([v["felt"] for _, v in pts])
    elic = np.array([float(math.exp(min(apply_probe(probe, nat_slot[k][None, :])[0], 40))) for k, _ in pts])
    eot = np.array([v["eot"] for _, v in pts])
    variants = [v["variant"] for _, v in pts]

    from scipy.stats import spearmanr
    rho_e = spearmanr(felt, elic).statistic
    rho_o = spearmanr(felt, eot).statistic

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 5), sharex=True)
    lim = (1, 3e6)
    for ax, yv, title, rho in (
        (axA, elic, f"(a) prefill-slot probe — tracks felt (ρ={rho_e:.2f})", rho_e),
        (axB, eot, f"(b) EOT stack probe — OOD, no track (ρ={rho_o:.2f})", rho_o)):
        for v in sorted(set(variants)):
            m = np.array([vv == v for vv in variants])
            ax.scatter(felt[m], np.clip(yv[m], *lim), s=46, color=VAR_C.get(v, "#888"),
                       edgecolor="white", linewidth=0.5, label=v.replace("_", " "))
        ax.plot(lim, lim, "k--", alpha=0.4, lw=1, label="y = x")
        off = int(np.sum((yv < lim[0]) | (yv > lim[1])))
        if off:
            ax.text(0.04, 0.04, f"{off} off-scale (blow-up)", transform=ax.transAxes,
                    fontsize=8, color="#b00")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.xaxis.set_major_formatter(SECFMT); ax.yaxis.set_major_formatter(SECFMT)
        ax.set_xlabel("what the model felt (stated)"); ax.set_title(title)
    axA.set_ylabel("probe's internal read")
    axA.legend(fontsize=8, loc="upper left")
    fig.suptitle("Naturalistic conversations: the prefill slot reads the felt construction "
                 "where the EOT probe can't",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = FIGURES_DIR / f"{base.short_name}_elicit" / "fig_natural_elicit.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}  (n={len(pts)}, Lc=L{Lc}, ρ_elicit={rho_e:.3f}, ρ_eot={rho_o:.3f})")


if __name__ == "__main__":
    main()
