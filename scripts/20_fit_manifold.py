"""Aim 1: fit the elapsed-time probe.

Per-layer grouped-CV ridge regression of the timestamped-rendering EOT
activation onto log(elapsed seconds). Reports the R^2 / Spearman layer profile,
picks the best layer, and runs the position-confound controls:

  - token baseline: how much of log-elapsed is just raw context length?
  - partial probe:  does the activation predict elapsed *beyond* token count?
    (residualize log-elapsed on log-tokens, then probe the residual)

Saves the best-layer probe (full fit on timestamped data, for the transfer
test in 30_decode) and the out-of-fold predictions (honest internal coordinate
for the timestamped decode).

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
    StatesCache, apply_probe, assemble_layer, available_layers, cv_predict,
    fit_full, load_rows, residualize, save_probe,
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

    # Per-layer probe sweep.
    per_layer: list[dict] = []
    n = 0
    for L in layers:
        d = assemble_layer(M, rows, L, rendering=args.rendering, roles=roles, cache=cache)
        n = len(d["y_log"])
        if n < 8:
            raise SystemExit(f"only {n} samples — generate a larger corpus first")
        _, r2, rho = cv_predict(d["X"], d["y_log"], d["groups"], n_splits=args.n_splits)
        per_layer.append({"layer": L, "r2": r2, "spearman": rho})
        print(f"  L{L:>3}  R2={r2:+.3f}  rho={rho:+.3f}")

    best = max(per_layer, key=lambda r: r["r2"])
    Lstar = best["layer"]
    print(f"\nbest layer: L{Lstar}  R2={best['r2']:+.3f}  rho={best['spearman']:+.3f}  (n={n})")

    # Position-confound controls at the best layer.
    d = assemble_layer(M, rows, Lstar, rendering=args.rendering, roles=roles, cache=cache)
    y, groups, tokens = d["y_log"], d["groups"], d["tokens"]
    log_tokens = np.log(np.maximum(tokens, 1.0))

    _, r2_tok, _ = cv_predict(log_tokens[:, None], y, groups, n_splits=args.n_splits)
    resid = residualize(y, log_tokens)
    _, r2_partial, _ = cv_predict(d["X"], resid, groups, n_splits=args.n_splits)
    oof_full, r2_full, rho_full = cv_predict(d["X"], y, groups, n_splits=args.n_splits)

    print(f"\nposition-confound controls @ L{Lstar}:")
    print(f"  probe (activation -> log_elapsed):            R2={r2_full:+.3f}")
    print(f"  token baseline (log_tokens -> log_elapsed):   R2={r2_tok:+.3f}")
    print(f"  partial (activation -> elapsed | tokens):     R2={r2_partial:+.3f}")
    verdict = ("representation carries time BEYOND position"
               if r2_partial > 0.1 else
               "time signal may be largely position/length")
    print(f"  -> {verdict}")

    # Save the full-fit probe (transfer) + out-of-fold preds (honest decode).
    probe = fit_full(d["X"], y)
    fit_meta = {
        "rendering": args.rendering, "roles": args.roles, "n": int(n),
        "best_layer": int(Lstar), "r2": float(r2_full), "spearman": float(rho_full),
        "r2_tokens": float(r2_tok), "r2_partial": float(r2_partial),
        "per_layer": per_layer,
    }
    save_probe(M.data_dir / "probe.npz", probe, layer=Lstar, meta=fit_meta)
    np.savez(
        M.data_dir / "fit_oof.npz",
        transcript_id=d["groups"], turn_idx=d["turn_idx"],
        oof_pred_log=oof_full, y_log=y, layer=np.int64(Lstar),
    )
    (M.data_dir / "fit.json").write_text(json.dumps(fit_meta, indent=2))

    # Sanity: in-sample probe should track gt (transfer code path check).
    insample = apply_probe(probe, d["X"])
    print(f"\nin-sample probe corr (sanity): "
          f"pearson={np.corrcoef(insample, y)[0, 1]:+.3f}")
    print(f"saved probe + oof + fit.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
