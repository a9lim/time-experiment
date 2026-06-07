"""Mahalanobis-whitened probe read on the naturalistic captures (offline).

The raw linear probe blows up on natural conversation activations — they sit
off the scripted manifold and the probe extrapolates without bound (reads from
sub-second to weeks). This script:

  1. Builds a per-layer Mahalanobis whitener (saklas ``LayerWhitener``) from the
     SCRIPTED timestamped activations — the in-distribution reference.
  2. For each natural turn, computes the per-layer Mahalanobis distance
     d_M(L) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) — an explicit OFF-MANIFOLD score — and the
     scripted in-distribution radius τ_L (a high quantile of scripted d_M).
  3. Produces a WHITENED read: cap each layer's activation at the manifold
     radius (Mahalanobis shrinkage, x' = μ + min(1, τ_L/d_M)·(x-μ)), which
     preserves direction but bounds the off-manifold magnitude, then applies the
     stack probe to the projected activation.

Reads: raw stack (reference / should reproduce the driver), whitened stack,
single-layer L59 (raw + whitened). If the whitened read stabilizes AND tracks
conversation length, a linear read survives OOD; if it goes flat, the scripted
time-axis genuinely isn't present in natural text (only the verbal readout is).

    TIME_MODEL=gemma python scripts/61_whiten_natural.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas.core.mahalanobis import LayerWhitener  # noqa: E402

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_stacked, assemble_all_layers, load_rows, load_stacked_probe,
)
from time_experiment.config import DATA_DIR, FIGURES_DIR, resolve_model  # noqa: E402

RIDGE_SCALE = 1.0
TAU_QUANTILE = 0.99   # scripted in-distribution Mahalanobis radius per layer


def build_whitener(X3d_s: np.ndarray, layers: list[int]):
    """LayerWhitener over scripted timestamped activations + per-layer means."""
    neutral, means = {}, {}
    for li, L in enumerate(layers):
        neutral[L] = torch.from_numpy(np.ascontiguousarray(X3d_s[:, li, :])).float()
        means[L] = neutral[L].mean(dim=0)
    w = LayerWhitener.from_neutral_activations(neutral, means, ridge_scale=RIDGE_SCALE)
    if not w.covers_all(layers):
        missing = [L for L in layers if not w.covers(L)]
        print(f"  WARN: whitener missing layers {missing} (degenerate); they pass through")
    return w, means


def layer_distances(w, means, layers, X_LD: np.ndarray) -> np.ndarray:
    """Per-layer Mahalanobis distance of one (L, D) activation from the scripted
    distribution. Uncovered layers -> nan."""
    d = np.full(len(layers), np.nan)
    for li, L in enumerate(layers):
        if not w.covers(L):
            continue
        v = torch.from_numpy(X_LD[li] - means[L].numpy()).float()
        d[li] = float(w.mahalanobis_norm(L, v))
    return d


def scripted_radii(w, means, layers, X3d_s: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer (tau_L = q-quantile, median) of the scripted in-distribution
    Mahalanobis distances — the manifold radius + a reference scale."""
    n = X3d_s.shape[0]
    D = np.full((n, len(layers)), np.nan)
    for li, L in enumerate(layers):
        if not w.covers(L):
            continue
        mu = means[L].numpy()
        for i in range(n):
            v = torch.from_numpy(X3d_s[i, li, :] - mu).float()
            D[i, li] = float(w.mahalanobis_norm(L, v))
    tau = np.nanquantile(D, q, axis=0)
    med = np.nanmedian(D, axis=0)
    return tau, med


def shrink(X_LD: np.ndarray, means, layers, tau: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Mahalanobis shrinkage onto the manifold: cap each layer's centered
    activation at radius tau_L (distance scales linearly along the ray from μ)."""
    out = np.array(X_LD, dtype=np.float64, copy=True)
    for li, L in enumerate(layers):
        if not np.isfinite(d[li]) or not np.isfinite(tau[li]) or d[li] <= tau[li]:
            continue
        mu = means[L].numpy()
        out[li] = mu + (tau[li] / d[li]) * (X_LD[li] - mu)
    return out


def l59_read(probe, layers, X3d_TLD: np.ndarray, Lstar: int) -> np.ndarray:
    """Single-layer base read (log-seconds) at Lstar from the stacked probe's
    per-layer base (which IS a single-layer probe fit on all scripted data)."""
    li = layers.index(Lstar)
    Xs = (X3d_TLD[:, li, :] - probe["base_mean"][li]) / probe["base_scale"][li]
    return Xs @ probe["base_coef"][li] + probe["base_intercept"][li]


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    probe, fit_meta = load_stacked_probe(base.data_dir / "probe.npz")
    layers = [int(L) for L in probe["layers"]]
    Lstar = int(fit_meta["best_layer"])

    # Scripted timestamped reference (in-distribution).
    rows = load_rows(base)
    cache = StatesCache(base.hidden_dir)
    d_s = assemble_all_layers(base, rows, rendering="timestamped", cache=cache)
    X3d_s = d_s["X3d"]
    print(f"scripted reference: {X3d_s.shape}  (building whitener, ridge_scale={RIDGE_SCALE})")
    w, means = build_whitener(X3d_s, layers)
    tau, med_s = scripted_radii(w, means, layers, X3d_s, TAU_QUANTILE)
    print(f"scripted Mahalanobis radius (median over layers): tau≈{np.nanmedian(tau):.1f}, "
          f"med≈{np.nanmedian(med_s):.1f}")

    nat_dir = DATA_DIR / f"{base.short_name}_natural"
    sidecars = sorted(glob.glob(str(nat_dir / "hidden" / "*.npz")))
    if not sidecars:
        raise SystemExit(f"no natural sidecars under {nat_dir}/hidden; run 60_naturalistic.py first")

    out_rows: list[dict] = []
    for path in sidecars:
        name = Path(path).stem                      # <conv>__<rendering>
        conv_id, rendering = name.rsplit("__", 1)
        npz = np.load(path)
        H, tns, toks, gts = npz["H"], npz["turn_idxs"], npz["tokens"], npz["gt_elapsed_s"]

        raw_read = apply_stacked(probe, H.astype(np.float64))        # (T,) blows up
        l59_raw = l59_read(probe, layers, H.astype(np.float64), Lstar)
        H_proj = np.empty_like(H, dtype=np.float64)
        ood_med, ood_max = [], []
        for t in range(H.shape[0]):
            d = layer_distances(w, means, layers, H[t])
            H_proj[t] = shrink(H[t], means, layers, tau, d)
            ratio = d / np.where(med_s > 0, med_s, np.nan)
            ood_med.append(float(np.nanmedian(ratio)))
            ood_max.append(float(np.nanmax(ratio)))
        whit_read = apply_stacked(probe, H_proj)                     # (T,) bounded
        l59_whit = l59_read(probe, layers, H_proj, Lstar)

        for i in range(H.shape[0]):
            out_rows.append({
                "conv_id": conv_id, "rendering": rendering, "turn_idx": int(tns[i]),
                "prompt_tokens": float(toks[i]),
                "gt_elapsed_s": (None if not np.isfinite(gts[i]) else float(gts[i])),
                "raw_log": float(raw_read[i]),
                "whit_log": float(whit_read[i]),
                "l59_raw_log": float(l59_raw[i]),
                "l59_whit_log": float(l59_whit[i]),
                "ood_ratio_med": ood_med[i], "ood_ratio_max": ood_max[i],
            })

    (nat_dir / "whitened.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n")
    summary = summarize(out_rows)
    (nat_dir / "whitened_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== whitened summary ===")
    print(json.dumps(summary, indent=2))
    make_plot(out_rows, FIGURES_DIR / f"{base.short_name}_natural" / "fig_whitened.png")
    print(f"\nrows -> {nat_dir}/whitened.jsonl")


def _spear(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan
    return float(spearmanr(a[m], b[m]).statistic)


def _rng(a) -> list:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(a.min()), float(a.max())] if len(a) else [math.nan, math.nan]


def summarize(rows: list[dict]) -> dict:
    un = [r for r in rows if r["rendering"] == "untimestamped"]
    ts = [r for r in rows if r["rendering"] == "timestamped"]
    out: dict[str, object] = {"n_untimestamped": len(un), "n_timestamped": len(ts)}
    for tag, key in (("raw", "raw_log"), ("whitened", "whit_log"),
                     ("l59_raw", "l59_raw_log"), ("l59_whitened", "l59_whit_log")):
        out[tag] = {
            "untimestamped_log_range": _rng([r[key] for r in un]),
            "rho_vs_tokens_untimestamped": _spear([r["prompt_tokens"] for r in un],
                                                  [r[key] for r in un]),
            "rho_vs_gt_control": _spear([math.log(max(r["gt_elapsed_s"], 1.0)) for r in ts],
                                        [r[key] for r in ts]),
        }
    out["ood_ratio_median_over_turns"] = float(np.nanmedian([r["ood_ratio_med"] for r in rows]))
    out["ood_ratio_max_over_turns"] = float(np.nanmax([r["ood_ratio_max"] for r in rows]))
    return out


def make_plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    un = [r for r in rows if r["rendering"] == "untimestamped"]
    ts = [r for r in rows if r["rendering"] == "timestamped"]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))

    # (A) untimestamped: raw vs whitened read vs length
    xs = [r["prompt_tokens"] for r in un]
    axA.scatter(xs, [math.exp(min(r["raw_log"], 30)) for r in un], c="#bbb", s=28,
                label="raw stack (OOD)")
    axA.scatter(xs, [math.exp(min(r["whit_log"], 30)) for r in un], c="#0e7490", s=34,
                label="whitened stack")
    axA.scatter(xs, [math.exp(min(r["l59_whit_log"], 30)) for r in un], c="#7c3aed",
                s=22, marker="^", label="whitened L59")
    axA.set_yscale("log"); axA.set_xlabel("conversation length (tokens)")
    axA.set_ylabel("probe read (s)"); axA.set_title("(a) no clock: raw blows up, whitened bounded")
    axA.legend(fontsize=7); axA.grid(True, alpha=0.3)

    # (B) control: whitened read vs injected clock
    gt = [max(r["gt_elapsed_s"], 1.0) for r in ts]
    axB.scatter(gt, [math.exp(min(r["whit_log"], 30)) for r in ts], c="#0e7490", s=34,
                label="whitened stack")
    if gt:
        lim = [min(gt) * 0.5, max(gt) * 2]
        axB.plot(lim, lim, "k--", alpha=0.4, label="y=x")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("injected elapsed (s)"); axB.set_ylabel("whitened read (s)")
    rho = _spear([math.log(max(r["gt_elapsed_s"], 1.0)) for r in ts], [r["whit_log"] for r in ts])
    axB.set_title(f"(b) control: whitened vs injected clock (ρ={rho:.2f})")
    axB.legend(fontsize=7); axB.grid(True, alpha=0.3)

    # (C) OOD distance vs length (how far off-manifold natural text sits)
    axC.scatter([r["prompt_tokens"] for r in un], [r["ood_ratio_med"] for r in un],
                c="#0e7490", s=30, label="untimestamped")
    axC.scatter([r["prompt_tokens"] for r in ts], [r["ood_ratio_med"] for r in ts],
                c="#ea580c", s=30, marker="s", label="timestamped")
    axC.axhline(1.0, color="k", ls="--", alpha=0.4, label="scripted median (=1)")
    axC.set_xlabel("conversation length (tokens)")
    axC.set_ylabel("Mahalanobis dist / scripted median")
    axC.set_title("(c) how off-manifold natural text is")
    axC.legend(fontsize=7); axC.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved figure -> {path}")


if __name__ == "__main__":
    main()
