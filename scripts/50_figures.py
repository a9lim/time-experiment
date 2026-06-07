"""Processed figures for the elapsed-time study (showcase / writeup).

Reads only committed result artifacts (no model) and renders the headline
findings to ``figures/<model>/``. Sibling of attractor-study's ``40_lda_viz``
and llmoji-study's figure scripts: numbered, offline, regenerable.

Inputs (per the three pilots, all gemma-4-31b-it):
  data/gemma/fit.json            per-layer probe R^2 / Spearman + position control
  data/gemma/decode_rows.csv     per-turn gt | internal-coord | verbal, both renderings
  data/gemma/fit_oof.npz         honest out-of-fold internal coordinate (timestamped)
  data/gemma_inflation/inflation.json    length -> felt curve (Pilot 2)
  data/gemma_rates/intermittent.json     graded clock-density picture (Pilot 3)
  data/gemma/geometry.json       log-t centroid geometry (Pilot 1 secondary)

Outputs:
  figures/gemma/fig1_aim1_probe.png        the probe works, beyond position
  figures/gemma/fig2_felt_decoupled.png    "feels like hours": felt vs wall clock
  figures/gemma/fig3_threeway_decode.png   gt | internal | verbal, clock vs no-clock
  figures/gemma/fig4_clock_density.png     graded picture across clock density
  figures/gemma/fig5_geometry.png          ~1-D, ~linear time axis

Usage:
  python scripts/50_figures.py            # all figures for $TIME_MODEL (default gemma)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import sys
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- paths (no model import; keep this script pure-offline) ----------------
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGS = REPO / "figures"
MODEL = os.environ.get("TIME_MODEL", "gemma")

# --- style -----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})

# Consistent semantic palette across every figure.
C_REAL = "#334155"      # slate  — ground-truth wall clock
C_INTERNAL = "#0e7490"  # teal   — probe / internal coordinate
C_VERBAL = "#ea580c"    # orange — stated / felt duration
C_LENGTH = "#7c3aed"    # violet — context length / tokens
C_CEIL = "#059669"      # green  — full-clock ceiling
C_FLOOR = "#9ca3af"     # grey   — no-clock floor


def _sec_fmt(x, _=None):
    """Human label for a seconds value on a log axis."""
    if x <= 0:
        return ""
    for thresh, div, unit in (
        (90, 1, "s"), (5400, 60, "m"), (129600, 3600, "h"),
        (1209600, 86400, "d"), (np.inf, 604800, "w"),
    ):
        if x < thresh:
            v = x / div
            return f"{v:.0f}{unit}" if v >= 1 else f"{v:.1f}{unit}"
    return f"{x:.0f}s"


SECFMT = FuncFormatter(_sec_fmt)


def _outdir() -> Path:
    d = FIGS / MODEL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str) -> None:
    p = _outdir() / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")


def _load_json(path: Path):
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(REPO)}")
        return None
    return json.loads(path.read_text())


def _load_decode_rows(path: Path):
    if not path.exists():
        print(f"  [skip] missing {path.relative_to(REPO)}")
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "rendering": r["rendering"],
                "schedule": r["schedule"],
                "gt": float(r["gt_elapsed_s"]),
                "internal_s": float(np.exp(float(r["internal_log"]))),
                "verbal": float(r["verbal_seconds"]),
            })
    return rows


# ===========================================================================
# Fig 1 — Aim 1: the probe recovers elapsed time, beyond position
# ===========================================================================
def fig1_aim1(fit: dict) -> None:
    layers = [d["layer"] for d in fit["per_layer"]]
    r2 = np.array([d["r2"] for d in fit["per_layer"]])
    rho = np.array([d["spearman"] for d in fit["per_layer"]])
    best = fit["best_layer"]
    r2_stack = fit["r2"]                          # deployed probe = all-layer stack
    r2_single = fit.get("r2_single_best", float(r2.max()))
    best_rho = next((d["spearman"] for d in fit["per_layer"] if d["layer"] == best),
                    float(rho.max()))
    top = max(r2_stack, float(r2.max()))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})

    # (a) per-layer single-layer profile, with the deployed stack overlaid
    axL.axhline(0, color="#cbd5e1", lw=0.8, zorder=0)
    axL.plot(layers, r2, "-o", ms=3, color=C_INTERNAL, label="single-layer CV $R^2$")
    axL.axhline(r2_stack, color=C_CEIL, ls="-", lw=1.6, alpha=0.9,
                label=f"stack (all layers) $R^2$={r2_stack:.2f}")
    ax2 = axL.twinx()
    ax2.plot(layers, rho, "-", lw=1.4, color=C_VERBAL, alpha=0.85,
             label="single-layer $\\rho$")
    ax2.set_ylabel("Spearman $\\rho$", color=C_VERBAL)
    ax2.tick_params(axis="y", colors=C_VERBAL)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_VERBAL)
    ax2.grid(False)
    ax2.set_ylim(0, 0.85)

    axL.axvline(best, color=C_REAL, ls="--", lw=1, alpha=0.6)
    axL.annotate(
        f"best layer L{best}\n$R^2$={r2_single:.2f}, $\\rho$={best_rho:.2f}",
        xy=(best, r2_single), xytext=(best - 22, r2_single - 0.10),
        fontsize=9, color=C_REAL,
        arrowprops=dict(arrowstyle="->", color=C_REAL, lw=0.9))
    axL.set_xlabel("layer")
    axL.set_ylabel("CV $R^2$ (log elapsed)", color=C_INTERNAL)
    axL.tick_params(axis="y", colors=C_INTERNAL)
    axL.set_title("(a) elapsed-time probe, per layer + stack  (clock visible)")
    axL.set_ylim(-0.08, top * 1.18)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left")

    # (b) the stack beats one layer beats length, and survives token-partialling
    names = ["stack\n(all layers)", f"best layer\nL{best}",
             "log-tokens\nbaseline", "partial $R^2$\n(tokens out)"]
    vals = [r2_stack, r2_single, fit["r2_tokens"], fit["r2_partial"]]
    cols = [C_INTERNAL, "#7dd3fc", C_LENGTH, C_CEIL]
    bars = axR.bar(names, vals, color=cols, width=0.66)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.set_ylim(0, max(vals) * 1.25)
    axR.set_ylabel("$R^2$ (log elapsed)")
    axR.set_title("(b) all layers > one layer > length")
    axR.tick_params(axis="x", labelsize=8)
    axR.margins(x=0.06)

    fig.suptitle(
        "The model linearly represents elapsed time — and it isn't merely context length",
        fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig1_aim1_probe.png")


# ===========================================================================
# Fig 2 — "feels like hours": felt time is decoupled from the wall clock
# ===========================================================================
def fig2_felt(rows, inflation: dict) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) Pilot 1: per-schedule real vs felt — felt pinned ~constant
    order = ["seconds", "minutes", "hours", "days", "mixed_log"]
    unt = [r for r in rows if r["rendering"] == "untimestamped"]
    real_med, felt_med = [], []
    for s in order:
        g = [r for r in unt if r["schedule"] == s]
        real_med.append(np.median([r["gt"] for r in g]))
        felt_med.append(np.median([r["verbal"] for r in g]))
    x = np.arange(len(order))
    w = 0.38
    axL.bar(x - w / 2, real_med, w, color=C_REAL, label="real elapsed (median)")
    axL.bar(x + w / 2, felt_med, w, color=C_VERBAL, label="felt / stated (median)")
    axL.axhline(600, color=C_VERBAL, ls=":", lw=1.2, alpha=0.7)
    axL.text(len(order) - 0.4, 660, "≈10 min prior", color=C_VERBAL,
             fontsize=8.5, ha="right", va="bottom")
    axL.set_yscale("log")
    axL.yaxis.set_major_formatter(SECFMT)
    axL.set_xticks(x)
    axL.set_xticklabels(order, rotation=20, ha="right")
    axL.set_ylabel("duration")
    axL.set_title("(a) no clock: felt is flat while real spans 5 orders")
    axL.legend(loc="upper left")
    # annotate inflation/compression direction
    for xi, rm, fm in zip(x, real_med, felt_med):
        ratio = fm / rm
        lab = f"{ratio:.0f}×" if ratio >= 2 else (f"{ratio:.2f}×" if ratio < 0.5 else "≈1×")
        axL.text(xi, max(rm, fm) * 1.25, lab, ha="center", va="bottom",
                 fontsize=8, color=C_REAL)

    # (b) Pilot 2: length -> felt curve (instant vs minutes), real overlaid
    unt_i = inflation["untimestamped"]["by_schedule"]
    inst = unt_i["instant"]
    mins = unt_i["minutes"]
    tok_i = [d["med_tokens"] for d in inst]
    tok_m = [d["med_tokens"] for d in mins]
    axR.plot(tok_i, [d["med_felt_s"] for d in inst], "-o", ms=4,
             color=C_VERBAL, label="felt — instant gaps (1–8 s)")
    axR.plot(tok_m, [d["med_felt_s"] for d in mins], "s", ms=4,
             color=C_VERBAL, alpha=0.5, ls="--", label="felt — minute gaps")
    axR.plot(tok_i, [d["med_real_s"] for d in inst], "-o", ms=4,
             color=C_REAL, label="real — instant gaps")
    axR.plot(tok_m, [d["med_real_s"] for d in mins], "s", ms=4,
             color=C_REAL, alpha=0.5, ls="--", label="real — minute gaps")
    axR.axhline(7200, color=C_VERBAL, ls=":", lw=1, alpha=0.6)
    axR.text(tok_i[-1], 7700, "saturates ≈2 h", color=C_VERBAL,
             fontsize=8.5, ha="right", va="bottom")
    # 150x callout
    big = max(inst, key=lambda d: d["ratio"])
    axR.annotate(f"{big['ratio']:.0f}× inflation",
                 xy=(big["med_tokens"], big["med_felt_s"]),
                 xytext=(big["med_tokens"] - 700, big["med_felt_s"] * 2.4),
                 fontsize=9, color=C_VERBAL, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_VERBAL, lw=0.9))
    axR.set_yscale("log")
    axR.yaxis.set_major_formatter(SECFMT)
    axR.set_xlabel("conversation length (tokens)")
    axR.set_ylabel("duration")
    rl = inflation["untimestamped"]["rho_felt_vs_length"]
    rr = inflation["untimestamped"]["rho_felt_vs_real"]
    axR.set_title(f"(b) felt tracks length (ρ={rl:.2f}), not the clock (ρ={rr:.2f})")
    axR.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "\"Feels like hours\": felt duration is a context-length prior, decoupled from real time",
        fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig2_felt_decoupled.png")


# ===========================================================================
# Fig 3 — the three-way: ground truth | internal coordinate | verbal
# ===========================================================================
def fig3_threeway(rows, decode: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)

    lim = (0.5, 3e6)
    for ax, rend, title, vlabel in (
        (axes[0], "timestamped",
         "(a) clock visible — both track truth", "stated (clock arithmetic)"),
        (axes[1], "untimestamped",
         "(b) no clock — verbal collapses to a prior", "felt (no clock)"),
    ):
        g = [r for r in rows if r["rendering"] == rend]
        gt = np.array([r["gt"] for r in g])
        internal = np.array([r["internal_s"] for r in g])
        verbal = np.array([r["verbal"] for r in g])

        ax.plot(lim, lim, color="#94a3b8", ls="--", lw=1, zorder=1,
                label="y = x (perfect)")
        ax.scatter(gt, internal, s=16, color=C_INTERNAL, alpha=0.55,
                   edgecolor="none", label="internal coordinate (probe)")
        ax.scatter(gt, verbal, s=16, color=C_VERBAL, alpha=0.55,
                   edgecolor="none", label=f"verbal — {vlabel}")

        d = decode[rend]
        txt = (f"verbal·gt  r={d['corr_verbal_gt']['pearson']:.2f}\n"
               f"internal·gt  r={d['corr_internal_gt']['pearson']:.2f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                      ec="#cbd5e1", alpha=0.9))
        if rend == "untimestamped":
            ax.axhline(600, color=C_VERBAL, ls=":", lw=1.1, alpha=0.7)
            ax.text(lim[1] * 0.6, 680, "≈10 min", color=C_VERBAL,
                    fontsize=8.5, ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.xaxis.set_major_formatter(SECFMT)
        ax.yaxis.set_major_formatter(SECFMT)
        ax.set_xlabel("true elapsed")
        ax.set_title(title)
    axes[0].set_ylabel("estimated elapsed")
    axes[0].legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Three readings of elapsed time: a real internal axis only when a clock is in context",
        fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig3_threeway_decode.png")


# ===========================================================================
# Fig 4 — the graded picture across clock density (Pilot 3)
# ===========================================================================
def fig4_clock_density(inter: dict) -> None:
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) rate-sensitivity: floor / intermittent / ceiling
    conds = [
        ("no clock", inter["untimestamped"], C_FLOOR),
        ("sparse clock\n(every 4th turn)", inter["intermittent"], C_INTERNAL),
        ("full clock", inter["timestamped"], C_CEIL),
    ]
    x = np.arange(len(conds))
    rs = [c[1]["rate_sensitivity"] for c in conds]
    sp = [c[1]["spearman_true"] for c in conds]
    w = 0.38
    b1 = axL.bar(x - w / 2, rs, w, color=[c[2] for c in conds],
                 label="rate-sensitivity (fixed length)")
    b2 = axL.bar(x + w / 2, sp, w, color=[c[2] for c in conds], alpha=0.45,
                 hatch="//", label="ρ(stated, true)")
    axL.axhline(0, color="#cbd5e1", lw=0.8)
    for b, v in list(zip(b1, rs)) + list(zip(b2, sp)):
        axL.text(b.get_x() + b.get_width() / 2,
                 v + (0.03 if v >= 0 else -0.03), f"{v:.2f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    axL.set_xticks(x)
    axL.set_xticklabels([c[0] for c in conds])
    axL.set_ylim(-0.3, 1.12)
    axL.set_ylabel("correlation")
    axL.set_title("(a) clock density → how well stated time tracks rate")
    axL.legend(loc="lower right")

    # (b) sparse clock: reads last anchor, doesn't extrapolate
    im = inter["intermittent"]
    names = ["vs last\nanchor", "vs current\nturn"]
    vals = [im["ratio_vs_last_anchor"], im["ratio_vs_current"]]
    cols = [C_CEIL, C_VERBAL]
    bars = axR.bar(names, vals, color=cols, width=0.6)
    axR.axhline(1.0, color=C_CEIL, ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}×",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axR.set_ylim(0, 1.2)
    axR.set_ylabel("stated / true")
    axR.set_title("(b) latches to the\nlast stamp, no extrapolation")

    fig.suptitle(
        "A clock only as fresh as the last timestamp: sparse stamps are read, not projected forward",
        fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "fig4_clock_density.png")


# ===========================================================================
# Fig 5 — geometry of the explicit-time axis (Pilot 1 secondary)
# Computed offline from the timestamped NPZ captures (no model), sweeping
# depth so the dimensionality-vs-layer story is visible, not just one layer.
# ===========================================================================
def _geometry_layer(M, rows, cache, layer, K=15):
    """Per-layer log-t bucket-centroid geometry. Mirrors 40_geometry."""
    from time_experiment.analysis import assemble_layer
    d = assemble_layer(M, rows, layer, rendering="timestamped", cache=cache)
    X, ylog = d["X"], d["y_log"]
    t = np.exp(ylog)
    edges = np.quantile(ylog, np.linspace(0, 1, K + 1))
    edges[-1] += 1e-9
    bid = np.clip(np.digitize(ylog, edges) - 1, 0, K - 1)
    cents, t_med, y_med = [], [], []
    for b in range(K):
        m = bid == b
        if m.sum() == 0:
            continue
        cents.append(X[m].mean(0))
        t_med.append(float(np.median(t[m])))
        y_med.append(float(np.median(ylog[m])))
    cents = np.asarray(cents)
    t_med = np.asarray(t_med)
    y_med = np.asarray(y_med)
    cm = cents.mean(0)
    _, S, Vt = np.linalg.svd(cents - cm, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    pc1 = (cents - cm) @ Vt[0]
    if np.corrcoef(pc1, y_med)[0, 1] < 0:
        pc1 = -pc1
    return {
        "layer": layer,
        "pc1_share": float(ev[0]),
        "r_log": float(np.corrcoef(pc1, y_med)[0, 1]),
        "r_lin": float(np.corrcoef(pc1, t_med)[0, 1]),
        "pc1_coord": pc1, "t_med": t_med, "y_med": y_med,
    }


def compute_geometry_sweep():
    """Returns (sweep_layers, per-layer detail) or None if captures absent."""
    hidden = DATA / MODEL / "hidden"
    turns = DATA / MODEL / "turns.jsonl"
    if not hidden.exists() or not turns.exists():
        print(f"  [skip] geometry: no captures under data/{MODEL}/hidden")
        return None
    from time_experiment.analysis import (
        StatesCache, available_layers, load_rows)
    M = SimpleNamespace(turns_path=turns, hidden_dir=hidden)
    rows = load_rows(M)
    cache = StatesCache(hidden)
    avail = set(available_layers(M, rows))
    sweep = [L for L in
             [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 30, 34,
              40, 48, 52, 57, 58, 59] if L in avail]
    detail = {L: _geometry_layer(M, rows, cache, L) for L in sweep}
    return sweep, detail


def fig5_geometry(sweep, detail, geo: dict | None) -> None:
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.15, 1.25]})

    # cleanest early layer (max PC1 share, L<=8) drives the line panel
    early = [L for L in sweep if L <= 8]
    Lstar = max(early, key=lambda L: detail[L]["pc1_share"])
    g = detail[Lstar]
    logt = np.log10(g["t_med"])
    pc1 = g["pc1_coord"]

    sc = axL.scatter(logt, pc1, c=logt, cmap="viridis", s=60,
                     edgecolor="white", linewidth=0.6, zorder=3)
    axL.plot(logt, pc1, "-", color="#94a3b8", lw=1, alpha=0.6, zorder=2)
    a, b = np.polyfit(logt, pc1, 1)
    xs = np.linspace(logt.min(), logt.max(), 50)
    axL.plot(xs, a * xs + b, ":", color=C_REAL, lw=1.3, alpha=0.8,
             label="linear fit")
    axL.set_xlabel("log₁₀ elapsed seconds (bucket median)")
    axL.set_ylabel("centroid coordinate on PC1")
    axL.set_title(f"(a) one dominant axis, ordered by elapsed  (L{Lstar})")
    axL.text(0.04, 0.96,
             f"PC1 = {g['pc1_share']*100:.0f}% of centroid variance\n"
             f"PC1·t  r={g['r_lin']:.2f}   PC1·log t  r={g['r_log']:.2f}",
             transform=axL.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1"))
    axL.legend(loc="lower right")
    cb = fig.colorbar(sc, ax=axL, pad=0.02)
    cb.set_label("log₁₀ s", fontsize=8)

    # (b) dimensionality + linearity vs depth
    Ls = np.array(sweep)
    share = np.array([detail[L]["pc1_share"] for L in sweep])
    rlin = np.array([detail[L]["r_lin"] for L in sweep])
    rlog = np.array([detail[L]["r_log"] for L in sweep])
    axR.plot(Ls, share, "-o", ms=3.5, color=C_INTERNAL,
             label="PC1 share (1 = pure line)")
    axR.plot(Ls, rlin, "-", lw=1.4, color=C_REAL, label="PC1·t  (linear)")
    axR.plot(Ls, rlog, "--", lw=1.4, color="#94a3b8",
             label="PC1·log t  (Weber-Fechner)")
    axR.axvline(Lstar, color=C_CEIL, ls=":", lw=1, alpha=0.7)
    axR.set_xlabel("layer")
    axR.set_ylabel("share / correlation")
    axR.set_ylim(0, 1.05)
    axR.set_title("(b) ~1-D early, multi-dim deeper; linear beats log")
    axR.legend(loc="lower left", fontsize=8)

    if geo is not None:
        per = geo["periodicity"]
        axR.text(0.97, 0.05,
                 f"hour-of-day  R²≈{per['hour_of_day']['mean']:.02f}\n"
                 f"day-of-week  R²≈{per['day_of_week']['mean']:.02f}\n"
                 f"(no time-of-day cycle)",
                 transform=axR.transAxes, va="bottom", ha="right", fontsize=7.5,
                 color="#64748b")

    fig.suptitle(
        "The explicit-time axis: one dominant direction, roughly linear in clock magnitude",
        fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig5_geometry.png")


# ===========================================================================
def main() -> None:
    print(f"figures for model={MODEL!r} -> {(FIGS / MODEL).relative_to(REPO)}/")

    fit = _load_json(DATA / MODEL / "fit.json")
    decode = _load_json(DATA / MODEL / "decode.json")
    rows = _load_decode_rows(DATA / MODEL / "decode_rows.csv")
    geo = _load_json(DATA / MODEL / "geometry.json")
    inflation = _load_json(DATA / f"{MODEL}_inflation" / "inflation.json")
    inter = _load_json(DATA / f"{MODEL}_rates" / "intermittent.json")

    if fit:
        fig1_aim1(fit)
    if rows and inflation:
        fig2_felt(rows, inflation)
    if rows and decode:
        fig3_threeway(rows, decode)
    if inter:
        fig4_clock_density(inter)
    sweep = compute_geometry_sweep()
    if sweep:
        fig5_geometry(sweep[0], sweep[1], geo)

    print("done.")


if __name__ == "__main__":
    main()
