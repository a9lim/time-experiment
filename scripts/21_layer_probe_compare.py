"""Compare elapsed-time probe architectures: single best layer vs all-layer
CONCAT vs all-layer STACK. Fully offline — re-fits on the existing per-layer
sidecars (no model, no re-emit).

Two questions:
  1. Decode power (timestamped): does pooling all layers beat the single best
     layer, and does the disciplined stack beat brute concat or overfit?
  2. The null power-check (untimestamped/felt): the headline negative is
     "no felt-elapsed beyond position, partial R²≈0 at every *single* layer."
     A signal distributed across layers would be invisible to single-layer
     probing. If concat/stack partial-R² stays ≈0 the null is robust to
     capacity; if it jumps, that's a distributed felt-time signal the
     per-layer sweep missed.

For each rendering we report, per architecture: raw CV R² (activation ->
log-elapsed), and the partial CV R² after residualizing out log-tokens (the
position control). Same samples + grouped folds across architectures.

    TIME_MODEL=gemma python scripts/21_layer_probe_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, assemble_all_layers, concat_cv, cv_predict, load_rows,
    residualize, stacked_cv, stacked_layer_weights,
)
from time_experiment.config import current_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderings", default="timestamped,untimestamped")
    ap.add_argument("--roles", default="all", help="'all' or comma-separated roles")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    M = current_model()
    rows = load_rows(M)
    if not rows:
        raise SystemExit(f"no rows at {M.turns_path}; run 10_emit.py first")
    roles = None if args.roles == "all" else tuple(args.roles.split(","))
    cache = StatesCache(M.hidden_dir)

    # Best single layer from the existing Aim-1 fit (chosen on timestamped R²);
    # reused for both renderings so the single-layer row matches the findings.
    fit_path = M.data_dir / "fit.json"
    if not fit_path.exists():
        raise SystemExit("run 20_fit_manifold.py first (need best_layer from fit.json)")
    Lstar = int(json.loads(fit_path.read_text())["best_layer"])

    renderings = [r.strip() for r in args.renderings.split(",") if r.strip()]
    out: dict = {"best_single_layer": Lstar, "renderings": {}}

    for rendering in renderings:
        d = assemble_all_layers(M, rows, rendering=rendering, roles=roles, cache=cache)
        X3d, y, groups = d["X3d"], d["y_log"], d["groups"]
        layers = d["layers"]
        n, L, D = X3d.shape
        log_tok = np.log(np.maximum(d["tokens"], 1.0))
        resid = residualize(y, log_tok)          # log-elapsed with length removed
        li = layers.index(Lstar)

        print(f"\n=== {rendering} ===  n={n}  L={L}  D={D}  groups={len(np.unique(groups))}")
        _, r2_tok, _ = cv_predict(log_tok[:, None], y, groups, n_splits=args.n_splits)
        print(f"  token baseline (log_tokens -> log_elapsed):  R²={r2_tok:+.3f}")

        archs: dict[str, dict] = {}

        def _run(name: str, fn) -> None:
            t0 = time.time()
            _, raw, rho = fn(X3d if name != "single" else X3d[:, li, :], y, groups,
                             n_splits=args.n_splits)
            _, partial, _ = fn(X3d if name != "single" else X3d[:, li, :], resid, groups,
                               n_splits=args.n_splits)
            archs[name] = {"raw_r2": raw, "partial_r2": partial, "spearman": rho}
            print(f"  {name:18s} raw R²={raw:+.3f}   partial R²(|tokens)={partial:+.3f}"
                  f"   ρ={rho:+.3f}   ({time.time()-t0:.0f}s)")

        _run("single", cv_predict)            # single best layer (label below)
        archs[f"single_L{Lstar}"] = archs.pop("single")
        _run("concat", concat_cv)             # all layers flattened
        _run("stack", stacked_cv)             # nested learned layer weights

        # Stack depth profile (interpretation only): which layers the meta leans on.
        w = stacked_layer_weights(X3d, y, groups, n_splits=args.n_splits)
        order = np.argsort(-np.abs(w))[:8]
        top = [(int(layers[i]), round(float(w[i]), 3)) for i in order]
        print(f"  stack top-weight layers (|w|): {top}")

        out["renderings"][rendering] = {
            "n": n, "r2_tokens": r2_tok, "arch": archs,
            "stack_top_layers": top,
        }

    out_path = M.data_dir / "layer_probe_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
