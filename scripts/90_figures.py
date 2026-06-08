"""Headline figures for T1-T3 (offline; reads probe_meta / felt / transfer +
decode_rows.csv / natural_reads.csv). T4's figure is emitted by 50_generation
(it owns the heavy trajectories).

  fig_t1_probe.png     — the probe: per-layer R² profile + confound bars; the
                         slot reads clock-elapsed at ~0.98 (vs the cited 0.59 EOT).
  fig_t2_felt.png      — felt is a length prior: three-way decode (clock vs no
                         clock) + the felt~length inflation curve.
  fig_t3_transfer.png  — one axis transfers: the scripted clock probe on natural
                         slots tracks felt; per-variant ordering vs magnitude.

    TIME_MODEL=gemma python scripts/90_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.config import current_model  # noqa: E402

C_REAL, C_INT, C_VERB, C_LEN, C_TEXT, C_EOT = "#334155", "#0e7490", "#ea580c", "#7c3aed", "#7dd3fc", "#9ca3af"
VAR_C = {"neutral": "#3b6", "affect_dense": "#b36", "time_language": "#36b"}
EOT_BASELINE = 0.59  # cited prior (Pilot 1b stack EOT); the slot supersedes it


def _sec(x, _=None):
    if x <= 0:
        return ""
    for t, d, u in ((90, 1, "s"), (5400, 60, "m"), (129600, 3600, "h"), (1e12, 86400, "d")):
        if x < t:
            v = x / d
            return f"{v:.0f}{u}" if v >= 1 else f"{v:.1f}{u}"
    return f"{x:.0f}s"


SECFMT = FuncFormatter(_sec)


def _load_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def fig_t1(M, fdir):
    p = M.data_dir / "probe_meta.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    layers = [r["layer"] for r in m["per_layer"]]
    r2 = [r["r2"] for r in m["per_layer"]]
    wts = [r.get("weight", 0.0) for r in m["per_layer"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
    # per-layer R² (line) + the EV weights the all-layer probe assigns (bars).
    axW = axL.twinx()
    axW.bar(layers, wts, width=0.9, color=C_INT, alpha=0.18, label="EV weight")
    axW.set_ylabel("EV weight", color=C_INT); axW.set_ylim(0, max(wts) * 1.6 + 1e-6)
    axW.tick_params(axis="y", labelcolor=C_INT)
    axL.axhline(0, color="#cbd5e1", lw=0.8)
    axL.axhline(EOT_BASELINE, color=C_EOT, ls="--", lw=1.3, label=f"EOT baseline (cited, {EOT_BASELINE})")
    axL.axhline(m["r2"], color=C_REAL, ls="-", lw=1.4, label=f"EV all-layer ($R^2$={m['r2']:.2f})")
    axL.axhline(m.get("r2_null_noclock", np.nan), color=C_LEN, ls=":", lw=1.3, label="no clock")
    axL.plot(layers, r2, "-o", ms=3, color="#0b5566", label="per-layer $R^2$")
    axL.set_xlabel("layer"); axL.set_ylabel("CV $R^2$ (log elapsed)"); axL.set_ylim(-0.1, 1.08)
    axL.set_zorder(axW.get_zorder() + 1); axL.patch.set_visible(False)
    axL.set_title("(a) EV-weighted all-layer slot probe: every layer contributes by its $R^2$")
    axL.legend(loc="center right", fontsize=8)

    names = ["EV\nall-layer", "log-tokens\nbaseline", "partial\n(tokens out)", "true\nceiling", "no clock"]
    vals = [m["r2"], m["r2_tokens"], m["r2_partial"], m.get("r2_true_ceiling", np.nan),
            m.get("r2_null_noclock", np.nan)]
    cols = [C_INT, C_LEN, "#059669", C_TEXT, "#d1d5db"]
    bars = axR.bar(names, [max(v, 0) if np.isfinite(v) else 0 for v in vals], color=cols, width=0.7)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            axR.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.set_ylim(0, 1.12); axR.set_ylabel("$R^2$ (log elapsed)")
    axR.set_title("(b) internal, beyond text & length")
    axR.tick_params(axis="x", labelsize=8)
    fig.suptitle("T1 — the model represents elapsed time, read by an EV-weighted all-layer slot probe",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(fdir / "fig_t1_probe.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"saved -> {fdir}/fig_t1_probe.png")


def fig_t2(M, fdir):
    recs = _load_csv(M.data_dir / "decode_rows.csv")
    if not recs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    lim = (0.5, 3e6)
    for ax, rendering, title in ((axes[0], "timestamped", "(a) clock visible — internal tracks truth"),
                                 (axes[1], "untimestamped", "(b) no clock — internal flat, verbal a prior")):
        rs = [r for r in recs if r["rendering"] == rendering]
        gt = np.array([float(r["gt_s"]) for r in rs])
        internal = np.array([float(r["internal_s"]) for r in rs])
        verbal = np.array([float(r["verbal_s"]) if r["verbal_s"] else np.nan for r in rs])
        ax.plot(lim, lim, color="#94a3b8", ls="--", lw=1, label="y = x")
        ax.scatter(gt, internal, s=16, color=C_INT, alpha=0.6, label="internal (slot)")
        ax.scatter(gt, verbal, s=16, color=C_VERB, alpha=0.6, label="verbal")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.xaxis.set_major_formatter(SECFMT); ax.yaxis.set_major_formatter(SECFMT)
        ax.set_xlabel("true elapsed"); ax.set_title(title)
    axes[0].set_ylabel("estimated elapsed"); axes[0].legend(loc="lower right", fontsize=8)

    # (c) verbal + probe reads vs context length (no clock): do either track length?
    un = [r for r in recs if r["rendering"] == "untimestamped"]
    if un:
        tok = np.array([float(r["tokens"]) for r in un])
        order = np.argsort(tok)
        tok = tok[order]
        internal = np.array([float(r["internal_s"]) for r in un])[order]
        verbal = np.array([float(r["verbal_s"]) if r["verbal_s"] else np.nan for r in un])[order]
        # true elapsed intentionally omitted: this panel is the model's subjective
        # internal/verbal time vs context length, where wall-clock gt (decoupled by
        # the gap schedule) is not the reference of interest.
        axes[2].scatter(tok, internal, s=16, color=C_INT, alpha=0.7, label="probe (internal)")
        axes[2].scatter(tok, verbal, s=16, color=C_VERB, alpha=0.7, label="verbal (felt)")
        for y, c in ((internal, C_INT), (verbal, C_VERB)):  # light log-log trend lines
            m = np.isfinite(y)
            if m.sum() > 3:
                b = np.polyfit(np.log(tok[m]), np.log(y[m]), 1)
                axes[2].plot(tok[m], np.exp(np.polyval(b, np.log(tok[m]))), "-", color=c, lw=1.3, alpha=0.8)
        axes[2].set_yscale("log"); axes[2].yaxis.set_major_formatter(SECFMT)
        axes[2].set_xlabel("context length (tokens)"); axes[2].set_ylabel("duration")
        axes[2].set_title("(c) no clock: verbal + probe vs context length")
        axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)
    fig.suptitle("T2 — felt time is a length-driven prior (no internal felt-elapsed beyond length)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(fdir / "fig_t2_felt.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"saved -> {fdir}/fig_t2_felt.png")


def fig_t3(M, fdir):
    nat = _load_csv(M.data_dir / "natural_reads.csv")
    tr = json.loads((M.data_dir / "transfer.json").read_text()) if (M.data_dir / "transfer.json").exists() else {}
    if not nat:
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.7))
    felt = np.array([float(r["felt_s"]) for r in nat])
    cread = np.array([float(r["clock_read_s"]) for r in nat])
    for v in sorted({r["variant"] for r in nat}):
        m = np.array([r["variant"] == v for r in nat])
        axA.scatter(felt[m], cread[m], s=42, color=VAR_C.get(v, "#888"), edgecolor="white",
                    linewidth=0.5, label=v)
    lim = [min(felt.min(), cread.min()) * 0.6, max(felt.max(), cread.max()) * 1.6]
    axA.plot(lim, lim, "k--", alpha=0.4, lw=1, label="y = x")
    axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlim(lim); axA.set_ylim(lim)
    axA.xaxis.set_major_formatter(SECFMT); axA.yaxis.set_major_formatter(SECFMT)
    rho = tr.get("crossaxis", {}).get("rho_clockprobe_vs_felt", float("nan"))
    axA.set_xlabel("what the model felt (stated)"); axA.set_ylabel("scripted clock-elapsed axis read")
    axA.set_title(f"(a) one axis transfers to natural felt (ρ={rho:.2f})")
    axA.legend(fontsize=7.5, loc="upper left")

    variants = ["neutral", "affect_dense", "time_language"]
    feltm = [np.median([float(r["felt_s"]) for r in nat if r["variant"] == v]) or np.nan for v in variants]
    readm = [np.median([float(r["slot_read_s"]) for r in nat if r["variant"] == v]) or np.nan for v in variants]
    x = np.arange(len(variants)); w = 0.38
    axB.bar(x - w / 2, feltm, w, color=C_VERB, label="felt (model says)")
    axB.bar(x + w / 2, readm, w, color=C_INT, label="slot-axis read")
    axB.set_yscale("log"); axB.yaxis.set_major_formatter(SECFMT)
    axB.set_xticks(x); axB.set_xticklabels(["neutral", "affect", "time-words"])
    axB.set_ylabel("duration"); axB.set_title("(b) captures the ordering, compresses the magnitude")
    axB.legend(fontsize=8)
    fig.suptitle("T3 — one duration axis serves clock-reading and felt-construction, and it transfers",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(fdir / "fig_t3_transfer.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"saved -> {fdir}/fig_t3_transfer.png")


def fig_probe_linear(M, fdir):
    """Probe-read felt time vs context length on LINEAR axes (no-clock slots).
    The probe's internal elapsed read lives in log coordinates; a linear-axis view
    tests whether it grows linearly (or saturates) in raw seconds with length.
    Coloured by schedule — under no clock the read should not separate by the
    (invisible) narrated schedule, only by length."""
    from scipy.stats import pearsonr
    recs = _load_csv(M.data_dir / "decode_rows.csv")
    un = [r for r in recs if r["rendering"] == "untimestamped"]
    if not un:
        return
    tok = np.array([float(r["tokens"]) for r in un])
    internal = np.array([float(r["internal_s"]) for r in un])
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for sch in sorted({r["schedule"] for r in un}):
        m = np.array([r["schedule"] == sch for r in un])
        ax.scatter(tok[m], internal[m], s=20, alpha=0.6, label=sch)
    b = np.polyfit(tok, internal, 1)
    r = float(pearsonr(tok, internal)[0])
    xs = np.array([tok.min(), tok.max()])
    ax.plot(xs, np.polyval(b, xs), "-", color=C_REAL, lw=1.8,
            label=f"linear fit: {b[0]:.3f} s/tok, r={r:.2f}")
    ax.set_ylim(0, float(np.percentile(internal, 98)) * 1.1)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("probe-read felt time (s, linear axis)")
    ax.set_title("Probe felt time vs context length (no clock, linear axes)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fdir / "fig_probe_vs_length_linear.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"saved -> {fdir}/fig_probe_vs_length_linear.png")


def main() -> None:
    M = current_model()
    fdir = M.figures_dir
    fdir.mkdir(parents=True, exist_ok=True)
    fig_t1(M, fdir)
    fig_t2(M, fdir)
    fig_t3(M, fdir)
    fig_probe_linear(M, fdir)
    print(f"\nfigures -> {fdir}/  (T4: run 50_generation.py)")


if __name__ == "__main__":
    main()
