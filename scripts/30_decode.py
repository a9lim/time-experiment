"""Aim 2: decode + adjudicate H1/H2/H3.

The 3-way per assistant turn: ground-truth elapsed | internal coordinate
(the probe's read) | verbal estimate (the model's stated duration).

Internal coordinate:
  - timestamped:   out-of-fold probe predictions (honest; from 20_fit).
  - untimestamped: the timestamped-trained probe applied to implicit-time
    activations — the explicit->implicit TRANSFER test (the money experiment).

For each rendering we report corr(internal, gt), corr(verbal, gt),
corr(verbal, internal), the verbal overshoot factor (the 4-7x phenomenon), and
a suggested H1/H2/H3 reading:

  H1 pure output confabulation : internal tracks gt, verbal decoupled from internal
  H2 represented elapsed inflated: verbal tracks internal, internal itself > gt
  H3 calibrated-but-misapplied  : internal tracks available signal, verbal overshoot
                                  is the token->seconds gap, not representational

    TIME_MODEL=gemma python scripts/30_decode.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_stacked, classify_hypothesis, load_rows,
    load_stacked_probe,
)
from time_experiment.config import (  # noqa: E402
    DEFAULT_READOUT_BY_RENDERING, MIN_ELAPSED_S, RENDERINGS, current_model,
)


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """(pearson, spearman, n) over finite pairs."""
    from scipy.stats import pearsonr, spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan, math.nan, int(m.sum())
    return (float(pearsonr(a[m], b[m])[0]),
            float(spearmanr(a[m], b[m])[0]), int(m.sum()))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x over finite pairs (1.0 = perfect log-scale tracking)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return math.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def main() -> None:
    M = current_model()
    rows = load_rows(M)
    probe, fit_meta = load_stacked_probe(M.data_dir / "probe.npz")
    oof = np.load(M.data_dir / "fit_oof.npz", allow_pickle=False)
    oof_lookup = {
        (str(t), int(k)): float(p)
        for t, k, p in zip(oof["transcript_id"], oof["turn_idx"], oof["oof_pred_log"])
    }
    cache = StatesCache(M.hidden_dir)
    print(f"model: {M.short_name}  probe: STACK (all layers)  (fit R2={fit_meta['r2']:+.3f})")

    summary: dict[str, dict] = {}
    merged: list[dict] = []

    for rendering in RENDERINGS:
        phrasing = DEFAULT_READOUT_BY_RENDERING[rendering]
        gt_log, internal_log, verbal_log, verbal_s, gt_s = [], [], [], [], []
        refusals = 0
        for r in rows:
            if r["rendering"] != rendering or r["role"] != "assistant":
                continue
            if r["gt_elapsed_s"] < MIN_ELAPSED_S:
                continue
            key = (r["transcript_id"], r["turn_idx"])
            g = math.log(r["gt_elapsed_s"])

            # internal coordinate
            if rendering == "timestamped":
                if key not in oof_lookup:
                    continue
                icoord = oof_lookup[key]
            else:  # transfer: timestamped-trained STACK probe on implicit-time acts
                ts = cache.get(r["transcript_id"], rendering)
                icoord = float(apply_stacked(probe, ts.turn_all_layers(r["turn_idx"])[None, :, :])[0])

            # verbal estimate
            rd = r["readouts"].get(phrasing) or {}
            vs = rd.get("seconds", math.nan)
            if vs is None or not (isinstance(vs, (int, float)) and math.isfinite(vs) and vs > 0):
                refusals += 1
                vlog = math.nan
                vs = math.nan
            else:
                vlog = math.log(vs)

            gt_log.append(g); internal_log.append(icoord); verbal_log.append(vlog)
            verbal_s.append(vs); gt_s.append(r["gt_elapsed_s"])
            merged.append({
                "rendering": rendering, "transcript_id": r["transcript_id"],
                "turn_idx": r["turn_idx"], "schedule": r["schedule"],
                "gt_elapsed_s": r["gt_elapsed_s"], "internal_log": icoord,
                "verbal_seconds": vs, "phrasing": phrasing,
            })

        gt_log = np.array(gt_log); internal_log = np.array(internal_log)
        verbal_log = np.array(verbal_log)
        verbal_s = np.array(verbal_s); gt_s = np.array(gt_s)
        n = len(gt_log)
        if n == 0:
            print(f"\n[{rendering}] no assistant-turn samples")
            continue

        ig_p, ig_s, _ = _corr(internal_log, gt_log)
        vg_p, vg_s, n_v = _corr(verbal_log, gt_log)
        vi_p, vi_s, _ = _corr(verbal_log, internal_log)
        # overshoot factors (ratio space)
        fin_v = np.isfinite(verbal_s)
        overshoot_verbal = float(np.median(verbal_s[fin_v] / gt_s[fin_v])) if fin_v.any() else math.nan
        overshoot_internal = float(np.median(np.exp(internal_log) / gt_s))
        slope_ig = _slope(gt_log, internal_log)
        slope_vg = _slope(gt_log, verbal_log)

        summary[rendering] = {
            "n": n, "n_verbal": n_v, "refusals": refusals,
            "phrasing": phrasing,
            "corr_internal_gt": {"pearson": ig_p, "spearman": ig_s},
            "corr_verbal_gt": {"pearson": vg_p, "spearman": vg_s},
            "corr_verbal_internal": {"pearson": vi_p, "spearman": vi_s},
            "overshoot_verbal_median": overshoot_verbal,
            "overshoot_internal_median": overshoot_internal,
            "slope_internal_gt": slope_ig, "slope_verbal_gt": slope_vg,
        }

        print(f"\n[{rendering}]  n={n}  verbal_parsed={n_v}  refusals={refusals}  "
              f"phrasing={phrasing}")
        print(f"  corr internal~gt:        r={ig_p:+.3f}  rho={ig_s:+.3f}"
              + ("   <- TRANSFER (explicit->implicit)" if rendering == "untimestamped" else "  (out-of-fold)"))
        print(f"  corr verbal~gt:          r={vg_p:+.3f}  rho={vg_s:+.3f}")
        print(f"  corr verbal~internal:    r={vi_p:+.3f}  rho={vi_s:+.3f}")
        print(f"  overshoot (median ratio): verbal x{overshoot_verbal:.2f}   "
              f"internal x{overshoot_internal:.2f}")
        print(f"  slope internal~gt={slope_ig:+.2f}  verbal~gt={slope_vg:+.2f}  (1.0 = perfect)")

    # --- H1/H2/H3 suggested reading (interpretive heuristic) -------------
    print("\n" + "=" * 64)
    print("suggested reading (heuristic — inspect the numbers above):")
    u = summary.get("untimestamped")
    if u:
        verdict = classify_hypothesis(
            corr_verbal_internal=u["corr_verbal_internal"]["spearman"],
            corr_internal_gt=u["corr_internal_gt"]["spearman"],
            overshoot_internal=u["overshoot_internal_median"],
            overshoot_verbal=u["overshoot_verbal_median"],
        )
        summary["verdict"] = verdict
        print(f"  {verdict}")
    else:
        print("  (no untimestamped data — run 10_emit with both renderings)")

    out_json = M.data_dir / "decode.json"
    out_json.write_text(json.dumps(summary, indent=2))
    out_csv = M.data_dir / "decode_rows.csv"
    if merged:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0]))
            w.writeheader()
            w.writerows(merged)
    print(f"\nsaved {out_json.name} + {out_csv.name} -> {M.data_dir}/")


if __name__ == "__main__":
    main()
