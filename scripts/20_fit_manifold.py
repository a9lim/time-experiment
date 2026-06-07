"""Aim 1: fit the elapsed-time probe.

The deployable probe is the all-layer STACK (per-layer ridge base learners + a
meta-ridge over their out-of-fold predictions). It beat the single best layer
and brute concat in the architecture bake-off (`21_layer_probe_compare`): it
pulls in complementary early-layer signal that any single layer misses, while
staying low-capacity at the meta level (robust where concat overfits p>>n).

Still computes the per-layer sweep — it yields the interpretable best single
layer (the representational *locus*, used by `40_geometry`) and the
single-layer baseline for the confound table. Reports, at the stack:

  - probe R^2 / Spearman (grouped, out-of-fold, nested)
  - token baseline R^2 (how much of log-elapsed is just context length)
  - partial R^2 after residualizing out log-tokens (carries time beyond position)

Saves the stacked probe (`probe.npz`) for the transfer test + the stacked
out-of-fold predictions (`fit_oof.npz`, the honest timestamped internal coord).

    TIME_MODEL=gemma python scripts/20_fit_manifold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_stacked, assemble_all_layers, assemble_layer,
    available_layers, cv_predict, fit_stacked_full, load_rows, residualize,
    save_stacked_probe, stacked_cv, stacked_layer_weights,
)
from time_experiment.config import current_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendering", default="timestamped",
                    help="rendering to fit on (default: timestamped — clean label)")
    ap.add_argument("--roles", default="all", help="'all' or comma-separated role filter")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    M = current_model()
    rows = load_rows(M)
    if not rows:
        raise SystemExit(f"no rows at {M.turns_path}; run 10_emit.py first")
    roles = None if args.roles == "all" else tuple(args.roles.split(","))
    cache = StatesCache(M.hidden_dir)
    layers = available_layers(M, rows)

    print(f"model: {M.short_name}  rendering: {args.rendering}  roles: {args.roles}")
    print(f"layers: {layers}")

    # Per-layer sweep -> interpretable best single layer (locus) + baseline.
    per_layer: list[dict] = []
    for L in layers:
        d1 = assemble_layer(M, rows, L, rendering=args.rendering, roles=roles, cache=cache)
        if len(d1["y_log"]) < 8:
            raise SystemExit(f"only {len(d1['y_log'])} samples — generate a larger corpus first")
        _, r2, rho = cv_predict(d1["X"], d1["y_log"], d1["groups"], n_splits=args.n_splits)
        per_layer.append({"layer": L, "r2": r2, "spearman": rho})
        print(f"  L{L:>3}  R2={r2:+.3f}  rho={rho:+.3f}")
    best = max(per_layer, key=lambda r: r["r2"])
    Lstar = int(best["layer"])

    # All-layer assembly (shared by the stack fit + confound controls).
    d = assemble_all_layers(M, rows, rendering=args.rendering, roles=roles, cache=cache)
    X3d, y, groups = d["X3d"], d["y_log"], d["groups"]
    n = len(y)
    log_tokens = np.log(np.maximum(d["tokens"], 1.0))
    resid = residualize(y, log_tokens)

    # Stacked probe: out-of-fold predictions (raw) + partial-on-tokens.
    oof_stack, r2_stack, rho_stack = stacked_cv(X3d, y, groups, n_splits=args.n_splits)
    _, r2_tok, _ = cv_predict(log_tokens[:, None], y, groups, n_splits=args.n_splits)
    _, r2_partial, _ = stacked_cv(X3d, resid, groups, n_splits=args.n_splits)

    li = layers.index(Lstar)
    _, r2_single, _ = cv_predict(X3d[:, li, :], y, groups, n_splits=args.n_splits)

    print(f"\nbest single layer: L{Lstar}  (R2={best['r2']:+.3f})")
    print(f"position-confound controls @ STACK (all layers):")
    print(f"  stacked probe (acts -> log_elapsed):        R2={r2_stack:+.3f}  rho={rho_stack:+.3f}")
    print(f"  single-layer L{Lstar} baseline:              R2={r2_single:+.3f}")
    print(f"  token baseline (log_tokens -> log_elapsed):  R2={r2_tok:+.3f}")
    print(f"  partial (stack -> elapsed | tokens):         R2={r2_partial:+.3f}")
    verdict = ("representation carries time BEYOND position"
               if r2_partial > 0.1 else "time signal may be largely position/length")
    print(f"  -> {verdict}")

    # Stack depth profile (which layers the meta leans on).
    w = stacked_layer_weights(X3d, y, groups, n_splits=args.n_splits)
    order = np.argsort(-np.abs(w))[:8]
    top = [(int(layers[i]), round(float(w[i]), 3)) for i in order]
    print(f"  stack top-weight layers (|w|): {top}")

    # Deployable stacked probe + honest out-of-fold internal coordinate.
    probe = fit_stacked_full(X3d, y, groups, layers, n_splits=args.n_splits)
    fit_meta = {
        "probe_kind": "stacked", "rendering": args.rendering, "roles": args.roles,
        "n": int(n), "best_layer": Lstar,
        "r2": float(r2_stack), "spearman": float(rho_stack),
        "r2_single_best": float(r2_single),
        "r2_tokens": float(r2_tok), "r2_partial": float(r2_partial),
        "stack_top_layers": top, "per_layer": per_layer,
    }
    save_stacked_probe(M.data_dir / "probe.npz", probe, meta=fit_meta)
    np.savez(
        M.data_dir / "fit_oof.npz",
        transcript_id=d["groups"], turn_idx=d["turn_idx"],
        oof_pred_log=oof_stack, y_log=y, layer=np.int64(-1),  # -1 = stacked (all layers)
    )
    (M.data_dir / "fit.json").write_text(json.dumps(fit_meta, indent=2))

    insample = apply_stacked(probe, X3d)
    print(f"\nin-sample stacked probe corr (sanity): "
          f"pearson={np.corrcoef(insample, y)[0, 1]:+.3f}")
    print(f"saved stacked probe + oof + fit.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
