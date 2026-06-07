"""T1 — the probe: elapsed time is linearly represented, read at the elicitation
slot (offline; reads 10_capture's slot sidecars).

Per-layer grouped-CV ridge of the constant-prefill slot activations -> log(elapsed)
on the timestamped rendering. Reports:

  - best layer (the representational locus, selected on gt R² — non-circular)
  - the position-confound controls at that layer: token baseline, partial R²
    after residualizing out log-tokens
  - the true-vs-constant gap: `true` prefill reads the injected phrase (the text
    ceiling); `constant` holds the text fixed, so its R² is the model's internal
    coordinate beyond text and beyond length
  - the no-clock null (untimestamped/constant per-layer)
  - secondary geometry of the locus: dimensionality (PC1 of log-t centroids) and
    whether the axis is more linear in raw vs log elapsed

Saves the canonical single-layer probe (`probe.npz`) + its out-of-fold internal
coordinate (`fit_oof.npz`) for T2/T3, and `probe_meta.json`.

    TIME_MODEL=gemma python scripts/20_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, assemble, cv_predict, ev_combined_oof, fit_ev_probe, load_rows,
    residualize, save_ev_probe,
)
from time_experiment.config import current_model  # noqa: E402


def geometry(X_NL_D: np.ndarray, gt_log: np.ndarray, n_bins: int = 10) -> dict:
    """Compact geometry of the elapsed locus: bucket by log-elapsed, take bin
    centroids, and report PC1 explained variance (dimensionality) + whether the
    dominant axis is more linear in raw vs log elapsed."""
    order = np.argsort(gt_log)
    bins = np.array_split(order, n_bins)
    cents, b_loge = [], []
    for b in bins:
        if len(b) == 0:
            continue
        cents.append(X_NL_D[b].mean(0))
        b_loge.append(float(gt_log[b].mean()))
    C = np.asarray(cents)
    b_loge = np.asarray(b_loge)
    Cc = C - C.mean(0)
    _U, Svals, Vt = np.linalg.svd(Cc, full_matrices=False)
    pc1_var = float((Svals[0] ** 2) / (Svals ** 2).sum())
    pc1 = Cc @ Vt[0]
    raw = np.exp(b_loge)
    r_lin = abs(float(np.corrcoef(pc1, raw)[0, 1]))
    r_log = abs(float(np.corrcoef(pc1, b_loge)[0, 1]))
    return {"pc1_explained_var": pc1_var, "r_pc1_vs_raw": r_lin, "r_pc1_vs_log": r_log}


def main() -> None:
    M = current_model()
    rows = load_rows(M.rows_path)
    cache = StatesCache(M.hidden_dir)

    # constant-prefill, timestamped: the clean clock-elapsed axis.
    d = assemble(rows, cache, source="scripted", rendering="timestamped", mode="constant")
    if len(d["gt_log"]) < 8:
        raise SystemExit(f"only {len(d['gt_log'])} timestamped/constant samples — run 10_capture first")
    X3d, y, groups, tokens, layers = d["X3d"], d["gt_log"], d["groups"], d["tokens"], d["layers"]
    print(f"model: {M.short_name}  n={len(y)}  layers={len(layers)}")

    # EV-weighted all-layer read (saklas idiom): every layer contributes,
    # weighted by the variance it explains (its grouped-CV R²).
    ev = ev_combined_oof(X3d, y, groups)
    r2_ev, rho_ev, w, r2pl = ev["r2"], ev["spearman"], ev["weights"], ev["r2_per_layer"]
    bi = int(np.argmax(r2pl)); Lstar = int(layers[bi])   # locus for geometry only
    for li, L in enumerate(layers):
        mark = "  <- locus" if li == bi else ""
        print(f"  L{L:>3}  R2={r2pl[li]:+.3f}  w={w[li]:.3f}{mark}")

    # position-confound controls on the EV-combined read.
    log_tok = np.log(np.maximum(tokens, 1.0))
    _, r2_tok, _ = cv_predict(log_tok[:, None], y, groups)
    r2_partial = ev_combined_oof(X3d, residualize(y, log_tok), groups)["r2"]

    # true-prefill (text ceiling) + no-clock null, EV-combined for consistency.
    dt = assemble(rows, cache, source="scripted", rendering="timestamped", mode="true")
    true_r2 = ev_combined_oof(dt["X3d"], dt["gt_log"], dt["groups"])["r2"] if len(dt["gt_log"]) >= 8 else float("nan")
    du = assemble(rows, cache, source="scripted", rendering="untimestamped", mode="constant")
    null_r2 = ev_combined_oof(du["X3d"], du["gt_log"], du["groups"])["r2"] if len(du["gt_log"]) >= 8 else float("nan")

    print(f"\nEV-weighted all-layer probe (locus L{Lstar}, {int((w > 0.01).sum())} layers w>0.01):")
    print(f"  constant prefill (internal):  R2={r2_ev:+.3f}  rho={rho_ev:+.3f}")
    print(f"  true prefill (text ceiling):  R2={true_r2:+.3f}")
    print(f"  no clock (constant):          R2={null_r2:+.3f}   <- the null")
    print(f"  token baseline:               R2={r2_tok:+.3f}")
    print(f"  partial (internal | tokens):  R2={r2_partial:+.3f}")
    verdict = ("internal coordinate beyond text and length"
               if r2_partial > 0.3 else "weak — re-check capture")
    print(f"  -> {verdict}")

    geo = geometry(X3d[:, bi, :], y)
    print(f"  geometry @locus L{Lstar}: PC1 var={geo['pc1_explained_var']:.2f}  "
          f"r(raw)={geo['r_pc1_vs_raw']:.2f}  r(log)={geo['r_pc1_vs_log']:.2f}")

    # deploy: EV-weighted all-layer probe + its out-of-fold internal coordinate.
    probe = fit_ev_probe(X3d, y, groups, layers)
    meta = {
        "probe_kind": "ev", "rendering": "timestamped", "mode": "constant",
        "locus_layer": Lstar, "n": int(len(y)), "r2": float(r2_ev), "spearman": float(rho_ev),
        "r2_true_ceiling": float(true_r2), "r2_null_noclock": float(null_r2),
        "r2_tokens": float(r2_tok), "r2_partial": float(r2_partial),
        "per_layer": [{"layer": int(L), "r2": float(r2pl[i]), "weight": float(w[i])}
                      for i, L in enumerate(layers)],
        "geometry": geo,
    }
    save_ev_probe(M.probe_path, probe, meta=meta)
    np.savez(M.data_dir / "fit_oof.npz",
             id=d["groups"], turn_idx=d["turn_idx"], oof_pred_log=ev["oof"], y_log=y)
    (M.data_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved probe.npz (EV all-layer) + fit_oof.npz + probe_meta.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
