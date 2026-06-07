"""Geometry of the elapsed-time representation (Aim 1, secondary).

Runs on existing timestamped captures — no re-emit. Three questions:

1. Weber-Fechner: is the time axis linear in log(t) or in t? Bin into log-t
   buckets, take per-bucket centroids, find the dominant time axis (PC1 of the
   centroids — unbiased by the log/linear choice), and test whether the PC1
   coordinate is more linear against log(t) or against t.
2. Curvature / intrinsic dim: PCA on the time-bucket centroids — does elapsed
   trace a straight line (flat) or a curved arc, and in how many dims?
3. Periodicity: can we cyclically decode hour-of-day and day-of-week (the
   sin/cos phases) from the activations? Absolute wall-clock = BASE_DATETIME +
   elapsed, so the timestamps the model read carry these cycles.

    TIME_MODEL=gemma python scripts/40_geometry.py [--layer L] [--buckets K]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, assemble_layer, cv_predict, load_rows,
)
from time_experiment.config import BASE_DATETIME, current_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=-1, help="-1 = fit.json best layer")
    ap.add_argument("--buckets", type=int, default=15)
    args = ap.parse_args()

    M = current_model()
    rows = load_rows(M)
    cache = StatesCache(M.hidden_dir)
    L = args.layer
    if L < 0:
        # geometry is inherently single-layer: use the interpretable best layer.
        L = int(json.loads((M.data_dir / "fit.json").read_text())["best_layer"])
    print(f"model: {M.short_name}  layer: L{L}  (timestamped rendering)")

    d = assemble_layer(M, rows, L, rendering="timestamped", cache=cache)
    X, y_log = d["X"], d["y_log"]
    t = np.exp(y_log)
    n = len(t)
    print(f"n={n} samples")

    # --- log-t bucket centroids ---
    K = args.buckets
    edges = np.quantile(y_log, np.linspace(0, 1, K + 1))
    edges[-1] += 1e-9
    bin_id = np.clip(np.digitize(y_log, edges) - 1, 0, K - 1)
    cents, t_med, ylog_med = [], [], []
    for b in range(K):
        m = bin_id == b
        if m.sum() == 0:
            continue
        cents.append(X[m].mean(0))
        t_med.append(float(np.median(t[m])))
        ylog_med.append(float(np.median(y_log[m])))
    cents = np.asarray(cents)
    t_med = np.asarray(t_med)
    ylog_med = np.asarray(ylog_med)

    # --- (2) curvature / intrinsic dim via PCA on centroids ---
    cmean = cents.mean(0)
    _U, S, Vt = np.linalg.svd(cents - cmean, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    pc = (cents - cmean) @ Vt.T  # (K, r) centroid coords
    print("\n[curvature] PCA explained-variance ratio of time-bucket centroids:")
    print("  " + "  ".join(f"PC{i+1}={ev[i]:.3f}" for i in range(min(5, len(ev)))))
    print(f"  PC1 share = {ev[0]:.3f}  (->1.0 = flat/linear; <1 = curved/multi-dim)")

    # --- (1) Weber-Fechner: PC1 coord vs log(t) vs t ---
    c1 = pc[:, 0]
    # orient so c1 increases with time
    if np.corrcoef(c1, ylog_med)[0, 1] < 0:
        c1 = -c1
    r_log = float(np.corrcoef(c1, ylog_med)[0, 1])
    r_lin = float(np.corrcoef(c1, t_med)[0, 1])
    print("\n[Weber-Fechner] dominant time axis (PC1) linearity:")
    print(f"  Pearson(PC1, log t) = {r_log:+.3f}")
    print(f"  Pearson(PC1, t)     = {r_lin:+.3f}")
    print(f"  -> {'LOG-linear (Weber-Fechner)' if abs(r_log) > abs(r_lin) + 0.02 else ('linear in t' if abs(r_lin) > abs(r_log) + 0.02 else 'ambiguous')}")

    # --- (3) periodicity: cyclic decode of hour-of-day & day-of-week ---
    secs = t
    hours = np.array([(BASE_DATETIME + timedelta(seconds=float(s))).hour
                      + (BASE_DATETIME + timedelta(seconds=float(s))).minute / 60.0
                      for s in secs])
    dows = np.array([(BASE_DATETIME + timedelta(seconds=float(s))).weekday()
                     + hours[i] / 24.0 for i, s in enumerate(secs)])
    groups = d["groups"]

    def cyclic_r2(phase_frac: np.ndarray, period_label: str) -> dict:
        th = 2 * np.pi * phase_frac
        _, r2c, _ = cv_predict(X, np.cos(th), groups)
        _, r2s, _ = cv_predict(X, np.sin(th), groups)
        print(f"  {period_label}: R2(cos)={r2c:+.3f}  R2(sin)={r2s:+.3f}  mean={0.5*(r2c+r2s):+.3f}")
        return {"r2_cos": r2c, "r2_sin": r2s, "mean": 0.5 * (r2c + r2s)}

    print("\n[periodicity] cyclic decode (CV R^2; >0 = cycle is encoded):")
    per_hour = cyclic_r2(hours / 24.0, "hour-of-day")
    per_dow = cyclic_r2(dows / 7.0, "day-of-week ")

    out = {
        "layer": L, "n": n, "buckets": K,
        "pca_ev_ratio": ev[:5].tolist(),
        "pc1_share": float(ev[0]),
        "weber_fechner": {"r_log": r_log, "r_lin": r_lin},
        "periodicity": {"hour_of_day": per_hour, "day_of_week": per_dow},
        "centroid_pc": pc[:, :3].tolist(), "t_med": t_med.tolist(),
    }
    (M.data_dir / "geometry.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved geometry.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
