"""fig1 / fig3, re-cast with the prefilled-duration probe (offline).

fig1_elicit_probe.png  — Aim 1 at the readout slot: per-layer R²(gt) for the
  internal (constant-prefill) vs text (true-prefill) reads — text is read early
  (L~1), the internal clock-derived quantity is computed mid-stack (~L32) and
  read at R²≈0.98; the no-clock condition stays flat (the null). Plus the
  position-confound bars at the best internal layer.

fig3_elicit_decode.png — the three-way (gt | internal coordinate | verbal) using
  the prefill-slot internal read. With a clock the internal coordinate tracks gt
  tightly; with no clock it is flat and the verbal estimate collapses to a prior.

    TIME_MODEL=gemma python scripts/66_elicit_aim_figures.py
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
from time_experiment.analysis import cv_predict, residualize  # noqa: E402
from time_experiment.config import DATA_DIR, FIGURES_DIR, resolve_model  # noqa: E402

C_REAL, C_INT, C_VERB, C_LEN, C_TEXT = "#334155", "#0e7490", "#ea580c", "#7c3aed", "#7dd3fc"


def _sec(x, _=None):
    if x <= 0:
        return ""
    for t, d, u in ((90, 1, "s"), (5400, 60, "m"), (129600, 3600, "h"), (1e12, 86400, "d")):
        if x < t:
            v = x / d
            return f"{v:.0f}{u}" if v >= 1 else f"{v:.1f}{u}"
    return f"{x:.0f}s"


SECFMT = FuncFormatter(_sec)


def assemble(rows, hid, rendering, mode):
    """X3d (N,L,D), y_gt(log), groups, tokens, [(tid,turn)], layers — scripted."""
    sel = [r for r in rows if r["source"] == "scripted" and r["rendering"] == rendering
           and r["mode"] == mode and r["gt_elapsed_s"] is not None]
    ids = sorted({r["id"] for r in sel})
    X, y, g, tok, key, layers = [], [], [], [], [], None
    for idd in ids:
        p = hid / f"scripted__{idd}__{rendering}__{mode}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        layers = [int(L) for L in d["layers"]]
        tpos = {int(t): i for i, t in enumerate(d["turn_idxs"])}
        for r in [r for r in sel if r["id"] == idd]:
            if r["turn_idx"] not in tpos:
                continue
            X.append(d["H"][tpos[r["turn_idx"]]])
            y.append(math.log(max(r["gt_elapsed_s"], 1.0)))
            g.append(idd); tok.append(r["tokens"]); key.append((idd, r["turn_idx"]))
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(g),
            np.asarray(tok, float), key, layers or [])


def verbal_map(base):
    out = {}
    for r in (json.loads(l) for l in base.turns_path.read_text().splitlines() if l.strip()):
        if r["role"] != "assistant":
            continue
        ph = "A_clock" if r["rendering"] == "timestamped" else "B_felt"
        s = (r.get("readouts", {}).get(ph) or {}).get("seconds")
        if isinstance(s, (int, float)) and math.isfinite(s) and s > 0:
            out[(r["transcript_id"], r["turn_idx"], r["rendering"])] = float(s)
    return out


def per_layer_r2(X, y, g):
    return [cv_predict(X[:, li, :], y, g)[0:3][1] for li in range(X.shape[1])]


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    edir = DATA_DIR / f"{base.short_name}_elicit"
    hid = edir / "hidden"
    rows = [json.loads(l) for l in (edir / "elicit_rows.jsonl").read_text().splitlines() if l.strip()]
    fdir = FIGURES_DIR / f"{base.short_name}_elicit"
    fdir.mkdir(parents=True, exist_ok=True)

    Xc, yc, gc, tokc, keyc, layers = assemble(rows, hid, "timestamped", "constant")
    Xt, yt, gt_, _, _, _ = assemble(rows, hid, "timestamped", "true")
    r2_const = per_layer_r2(Xc, yc, gc)
    r2_true = per_layer_r2(Xt, yt, gt_)
    Lc = int(np.argmax(r2_const))
    print(f"best internal (constant) layer: L{layers[Lc]}  R²={r2_const[Lc]:.3f}")

    # untimestamped/constant per-layer (the null) for the profile overlay
    Xu, yu, gu, toku, _, _ = assemble(rows, hid, "untimestamped", "constant")
    r2_null = per_layer_r2(Xu, yu, gu)

    # ---------------- fig1 ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    axL.axhline(0, color="#cbd5e1", lw=0.8)
    axL.plot(layers, r2_true, "-", color=C_TEXT, lw=1.6, label="true prefill (reads text)")
    axL.plot(layers, r2_const, "-o", ms=3, color=C_INT, label="constant prefill (internal)")
    axL.plot(layers, r2_null, "--", color=C_LEN, lw=1.3, label="no clock (constant)")
    axL.axvline(layers[Lc], color=C_REAL, ls=":", lw=1, alpha=0.6)
    axL.annotate(f"internal computed\nby L{layers[Lc]}  ($R^2$={r2_const[Lc]:.2f})",
                 xy=(layers[Lc], r2_const[Lc]), xytext=(layers[Lc] - 26, 0.55),
                 fontsize=8.5, color=C_REAL,
                 arrowprops=dict(arrowstyle="->", color=C_REAL, lw=0.9))
    axL.set_xlabel("layer"); axL.set_ylabel("CV $R^2$ (log elapsed)")
    axL.set_ylim(-0.1, 1.08)
    axL.set_title("(a) prefill-slot probe: text read early, internal computed mid-stack")
    axL.legend(loc="center right", fontsize=8)

    logtok = np.log(np.maximum(tokc, 1.0))
    r2_tok = cv_predict(logtok[:, None], yc, gc)[1]
    r2_par = cv_predict(Xc[:, Lc, :], residualize(yc, logtok), gc)[1]
    r2_nullbest = max(r2_null)
    names = ["internal\n$R^2$", "log-tokens\nbaseline", "partial\n(tokens out)", "no clock\n(best)"]
    vals = [r2_const[Lc], r2_tok, r2_par, r2_nullbest]
    cols = [C_INT, C_LEN, "#059669", "#d1d5db"]
    bars = axR.bar(names, [max(v, 0) for v in vals], color=cols, width=0.66)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.set_ylim(0, 1.12); axR.set_ylabel("$R^2$ (log elapsed)")
    axR.set_title(f"(b) internal, beyond length (L{layers[Lc]})")
    axR.tick_params(axis="x", labelsize=8)
    fig.suptitle("Aim 1 at the readout slot: the model represents elapsed time, computed mid-stack",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fdir / "fig1_elicit_probe.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {fdir}/fig1_elicit_probe.png")

    # ---------------- fig3 ----------------
    vmap = verbal_map(base)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)
    lim = (0.5, 3e6)
    for ax, rendering, title, vlab in (
        (axes[0], "timestamped", "(a) clock visible — internal tracks truth", "stated (clock)"),
        (axes[1], "untimestamped", "(b) no clock — internal flat, verbal a prior", "felt (no clock)")):
        X, y, g, tok, key, _ = assemble(rows, hid, rendering, "constant")
        oof = cv_predict(X[:, Lc, :], y, g)[0]
        gt_s = np.exp(y)
        internal_s = np.exp(oof)
        verbal_s = np.array([vmap.get((tid, t, rendering), np.nan) for (tid, t) in key])
        ax.plot(lim, lim, color="#94a3b8", ls="--", lw=1, label="y = x")
        ax.scatter(gt_s, internal_s, s=18, color=C_INT, alpha=0.6, edgecolor="none",
                   label="internal (prefill slot)")
        ax.scatter(gt_s, verbal_s, s=18, color=C_VERB, alpha=0.6, edgecolor="none",
                   label=f"verbal — {vlab}")
        m = np.isfinite(internal_s) & np.isfinite(gt_s)
        ig = np.corrcoef(np.log(internal_s[m]), np.log(gt_s[m]))[0, 1]
        ax.text(0.04, 0.96, f"internal·gt  r={ig:.2f}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1"))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.xaxis.set_major_formatter(SECFMT); ax.yaxis.set_major_formatter(SECFMT)
        ax.set_xlabel("true elapsed"); ax.set_title(title)
    axes[0].set_ylabel("estimated elapsed")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Three-way at the readout slot: a clean internal axis only when a clock is in context",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fdir / "fig3_elicit_decode.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {fdir}/fig3_elicit_decode.png")


if __name__ == "__main__":
    main()
