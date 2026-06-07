"""Analyze the prefilled-duration probe (offline).

Reads the slot activations from 62 and asks:

  1. Does the time-token slot read elapsed BETTER than the conversation EOT?
     Per-layer + best probe R²(log gt) on scripted, vs the EOT baselines in
     fit.json (single L59 ≈ 0.52, stack ≈ 0.59).

  2. The internal-vs-text control. `true`-prefill probes high partly by reading
     the injected phrase. `constant`-prefill holds the text fixed, so its probe
     R²(gt) — and the partial after residualizing log-tokens — is the model's
     INTERNAL elapsed surfaced at the readout token, beyond text and beyond
     length. The true-vs-constant gap is text-reading vs internal coordinate.

  3. Transfer / OOD. Does the anchored slot avoid the EOT OOD blowup? Mahalanobis
     distance of natural constant-prefill slots vs scripted constant-prefill
     slots (compare to the EOT site's 3.2× median).

    TIME_MODEL=gemma python scripts/63_elicit_analyze.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas.core.mahalanobis import LayerWhitener  # noqa: E402

from time_experiment.analysis import cv_predict, residualize  # noqa: E402
from time_experiment.config import DATA_DIR, resolve_model  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def assemble(rows, hid: Path, source, rendering, mode, *, need_gt=True):
    """X3d (N,L,D), y_log_gt, groups, tokens, layers for one (source,rendering,mode)."""
    sel = [r for r in rows if r["source"] == source and r["rendering"] == rendering
           and r["mode"] == mode]
    ids = sorted({r["id"] for r in sel})
    X, y, g, tok, layers = [], [], [], [], None
    for idd in ids:
        p = hid / f"{source}__{idd}__{rendering}__{mode}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        layers = [int(L) for L in d["layers"]]
        tpos = {int(t): i for i, t in enumerate(d["turn_idxs"])}
        for r in [r for r in sel if r["id"] == idd]:
            if need_gt and r["gt_elapsed_s"] is None:
                continue
            ti = tpos.get(r["turn_idx"])
            if ti is None:
                continue
            X.append(d["H"][ti])
            y.append(math.log(max(r["gt_elapsed_s"], 1.0)) if r["gt_elapsed_s"] else math.nan)
            g.append(idd); tok.append(r["tokens"])
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(g),
            np.asarray(tok, float), layers or [])


def best_probe(X3d, y, groups):
    """Per-layer sweep -> (best_layer, best_r2, all_r2)."""
    n, L, _ = X3d.shape
    r2s = []
    for li in range(L):
        _, r2, _ = cv_predict(X3d[:, li, :], y, groups)
        r2s.append(r2)
    bi = int(np.argmax(r2s))
    return bi, r2s[bi], r2s


def maha_ood(scripted_X3d, layers, natural_X3d) -> tuple[float, float]:
    """Median/max (over natural turns) of median-over-layers Mahalanobis ratio
    (natural distance / scripted-median distance)."""
    neutral = {L: torch.from_numpy(np.ascontiguousarray(scripted_X3d[:, i, :])).float()
               for i, L in enumerate(layers)}
    means = {L: neutral[L].mean(0) for L in layers}
    w = LayerWhitener.from_neutral_activations(neutral, means, ridge_scale=1.0)
    med_s = {}
    for i, L in enumerate(layers):
        if not w.covers(L):
            continue
        ds = [float(w.mahalanobis_norm(L, neutral[L][j] - means[L])) for j in range(scripted_X3d.shape[0])]
        med_s[L] = float(np.median(ds))
    ratios = []
    for t in range(natural_X3d.shape[0]):
        rr = []
        for i, L in enumerate(layers):
            if L not in med_s or med_s[L] <= 0:
                continue
            v = torch.from_numpy(natural_X3d[t, i, :] - means[L].numpy()).float()
            rr.append(float(w.mahalanobis_norm(L, v)) / med_s[L])
        if rr:
            ratios.append(float(np.median(rr)))
    return (float(np.median(ratios)) if ratios else math.nan,
            float(np.max(ratios)) if ratios else math.nan)


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    edir = DATA_DIR / f"{base.short_name}_elicit"
    hid = edir / "hidden"
    rows = load_rows(edir / "elicit_rows.jsonl")
    eot = json.loads((base.data_dir / "fit.json").read_text())
    print(f"EOT baselines (fit.json): single L{eot['best_layer']} R²={eot['r2_single_best']:.3f}, "
          f"stack R²={eot['r2']:.3f}, partial={eot['r2_partial']:.3f}")

    summary: dict = {"eot_single_r2": eot["r2_single_best"], "eot_stack_r2": eot["r2"],
                     "eot_partial_r2": eot["r2_partial"], "conditions": {}}

    for rendering in ("timestamped", "untimestamped"):
        for mode in ("true", "constant"):
            X, y, g, tok, layers = assemble(rows, hid, "scripted", rendering, mode)
            if len(y) < 8:
                print(f"[{rendering}/{mode}] only {len(y)} samples, skipping")
                continue
            bi, br2, _ = best_probe(X, y, g)
            logtok = np.log(np.maximum(tok, 1.0))
            _, r2_tok, _ = cv_predict(logtok[:, None], y, g)
            _, r2_par, _ = cv_predict(X[:, bi, :], residualize(y, logtok), g)
            summary["conditions"][f"{rendering}/{mode}"] = {
                "n": len(y), "best_layer": layers[bi], "r2_gt": br2,
                "r2_tokens": r2_tok, "partial_r2": r2_par,
            }
            print(f"[{rendering}/{mode}]  n={len(y)}  best L{layers[bi]}  "
                  f"R²(gt)={br2:+.3f}  tok={r2_tok:+.3f}  partial={r2_par:+.3f}")

    # Transfer / OOD: natural constant slots vs scripted constant slots.
    Xs, ys, gs, toks, layers = assemble(rows, hid, "scripted", "untimestamped", "constant")
    Xn, yn, gn, tokn, _ = assemble(rows, hid, "natural", "untimestamped", "constant", need_gt=False)
    if len(Xs) and len(Xn):
        med, mx = maha_ood(Xs, layers, Xn)
        summary["natural_ood_ratio_median"] = med
        summary["natural_ood_ratio_max"] = mx
        print(f"\nnatural constant-slot OOD vs scripted: median {med:.2f}×, max {mx:.2f}×  "
              f"(EOT site was 3.2× / 18.8×)")

    (edir / "elicit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {edir}/elicit_summary.json")


if __name__ == "__main__":
    main()
