"""Intermittent timestamps: with a clock only every Nth turn (uniform rate),
does the model EXTRAPOLATE the rate to un-timestamped turns, or fall back to
conversation length?

Design dissociates rate from length: uniform-rate transcripts at several rates
(rate_5min..rate_1d). At a fixed checkpoint turn the length is ~constant across
rates, so any dependence of the stated duration on the rate is rate-tracking,
not length. Readouts are asked on un-timestamped turns (no current clock).

Reads the rate corpus run (TIME_VARIANT=rates). Compares three renderings:
  timestamped  (ceiling: clock on every turn)
  intermittent (the test: clock every Nth turn)
  untimestamped(floor: no clock -> length fallback)

    TIME_MODEL=gemma TIME_VARIANT=rates python scripts/42_intermittent.py [--timestamp-stride 4]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import load_rows  # noqa: E402
from time_experiment.config import DEFAULT_READOUT_BY_RENDERING, current_model  # noqa: E402


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _collect(rows, rendering):
    ph = DEFAULT_READOUT_BY_RENDERING[rendering]
    pts = []  # (rate, turn, tokens, true_elapsed, stated)
    for r in rows:
        if r["rendering"] != rendering or r["role"] != "assistant":
            continue
        s = (r["readouts"].get(ph) or {}).get("seconds")
        if not _finite(s) or not _finite(r["gt_elapsed_s"]):
            continue
        pts.append((r["schedule"], r["turn_idx"], r["prompt_tokens"],
                    float(r["gt_elapsed_s"]), float(s)))
    return ph, pts


def _loglog_slope(true, stated):
    lt, ls = np.log(true), np.log(stated)
    return float(np.polyfit(lt, ls, 1)[0])


def main() -> None:
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp-stride", type=int, default=4)
    args = ap.parse_args()
    stride = args.timestamp_stride

    M = current_model()
    rows = load_rows(M)
    out: dict = {"model": M.short_name, "timestamp_stride": stride}

    for rendering in ("timestamped", "intermittent", "untimestamped"):
        ph, pts = _collect(rows, rendering)
        if not pts:
            continue
        true = np.array([p[3] for p in pts])
        stated = np.array([p[4] for p in pts])
        slope = _loglog_slope(true, stated)
        rho_true = float(spearmanr(true, stated)[0])

        # rate-sensitivity controlling for length: within each checkpoint turn
        # (length ~fixed), correlate stated with true across rates.
        per_turn_rho = []
        for k in sorted({p[1] for p in pts}):
            grp = [(p[3], p[4]) for p in pts if p[1] == k]
            if len({g[0] for g in grp}) >= 3:  # >=3 distinct true (i.e. rates)
                per_turn_rho.append(spearmanr([g[0] for g in grp],
                                              [g[1] for g in grp])[0])
        rate_sens = float(np.nanmean(per_turn_rho)) if per_turn_rho else float("nan")

        print(f"\n===== {rendering} ({ph}) — n={len(pts)} =====")
        print(f"  log-log slope stated~true: {slope:+.2f}   (1=tracks time, 0=ignores)")
        print(f"  spearman stated~true:      {rho_true:+.3f}")
        print(f"  rate-sensitivity @fixed length (mean rho over turns): {rate_sens:+.3f}")
        print("    (>0.5 = extrapolates the rate; ~0 = length fallback)")

        entry = {"phrasing": ph, "n": len(pts), "loglog_slope": slope,
                 "spearman_true": rho_true, "rate_sensitivity": rate_sens}

        # For intermittent: does stated match current-turn elapsed (extrapolate)
        # or the last-anchor elapsed (just read the most recent timestamp)?
        if rendering == "intermittent":
            cur_ratios, anch_ratios = [], []
            for (_rate, k, _tok, true_cur, stated_s) in pts:
                gap = true_cur / k if k else float("nan")   # uniform rate
                anchor_turn = (k // stride) * stride
                true_anchor = anchor_turn * gap
                if true_cur > 0:
                    cur_ratios.append(stated_s / true_cur)
                if true_anchor > 0:
                    anch_ratios.append(stated_s / true_anchor)
            mc = float(st.median(cur_ratios)) if cur_ratios else float("nan")
            ma = float(st.median(anch_ratios)) if anch_ratios else float("nan")
            print(f"  stated/true(current-turn): median {mc:.2f}   "
                  f"stated/true(last-anchor): median {ma:.2f}")
            print("    (~1 current = extrapolates to now; ~1 anchor = reads last stamp)")
            entry["ratio_vs_current"] = mc
            entry["ratio_vs_last_anchor"] = ma

        out[rendering] = entry

    (M.data_dir / "intermittent.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved intermittent.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
