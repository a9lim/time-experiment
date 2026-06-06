"""Inflation arm: does the felt estimate grow with conversation LENGTH while
real elapsed stays tiny — i.e. the "feels like hours" regime?

Reads the inflation variant (TIME_VARIANT=inflation -> data/<model>_inflation/).
For each assistant checkpoint it has: turn_idx, prompt_tokens (length), real
elapsed, and the verbal estimate. The questions:

  - does felt track conversation LENGTH (tokens/turns) rather than real time?
  - what's the inflation ratio felt/real, and does it climb with length in the
    tiny-real-elapsed ("instant") schedule?

    TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/41_inflation.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import load_rows  # noqa: E402
from time_experiment.config import DEFAULT_READOUT_BY_RENDERING, current_model  # noqa: E402


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _collect(rows, rendering):
    phrasing = DEFAULT_READOUT_BY_RENDERING[rendering]
    pts = []  # (schedule, turn_idx, tokens, gt, felt)
    for r in rows:
        if r["rendering"] != rendering or r["role"] != "assistant":
            continue
        felt = (r["readouts"].get(phrasing) or {}).get("seconds")
        if not _finite(felt):
            continue
        pts.append((r["schedule"], r["turn_idx"], r["prompt_tokens"],
                    r["gt_elapsed_s"], float(felt)))
    return phrasing, pts


def main() -> None:
    from scipy.stats import spearmanr
    M = current_model()
    rows = load_rows(M)
    out: dict = {"model": M.short_name}

    for rendering in ("untimestamped", "timestamped"):
        phrasing, pts = _collect(rows, rendering)
        if not pts:
            continue
        schedules = sorted({p[0] for p in pts})
        toks = [p[2] for p in pts]
        felts = [p[4] for p in pts]
        gts = [p[3] for p in pts]
        rho_len = float(spearmanr(toks, felts)[0])
        rho_gt = float(spearmanr(gts, felts)[0])
        print(f"\n===== {rendering} ({phrasing}) — n={len(pts)} =====")
        print(f"  felt vs conversation LENGTH (tokens): rho={rho_len:+.3f}")
        print(f"  felt vs REAL elapsed:                 rho={rho_gt:+.3f}")
        print(f"  {'schedule':10} {'turn':>4} {'med tok':>8} {'med real':>12} "
              f"{'med felt':>10} {'ratio':>8}")
        sched_summ: dict = {}
        for sch in schedules:
            rows_s = sorted([p for p in pts if p[0] == sch], key=lambda x: x[1])
            # bucket by turn_idx (each checkpoint is its own length rung)
            by_turn: dict = {}
            for _, k, tk, gt, fl in rows_s:
                by_turn.setdefault(k, {"tok": [], "gt": [], "felt": []})
                by_turn[k]["tok"].append(tk); by_turn[k]["gt"].append(gt)
                by_turn[k]["felt"].append(fl)
            rung = []
            for k in sorted(by_turn):
                d = by_turn[k]
                mt, mg, mf = st.median(d["tok"]), st.median(d["gt"]), st.median(d["felt"])
                ratio = mf / mg if mg > 0 else float("nan")
                print(f"  {sch:10} {k:>4} {mt:>8.0f} {mg:>12.0f} {mf:>10.0f} {ratio:>8.2f}")
                rung.append({"turn": k, "med_tokens": mt, "med_real_s": mg,
                             "med_felt_s": mf, "ratio": ratio})
            sched_summ[sch] = rung
        out[rendering] = {
            "phrasing": phrasing, "n": len(pts),
            "rho_felt_vs_length": rho_len, "rho_felt_vs_real": rho_gt,
            "by_schedule": sched_summ,
        }

    # headline: deepest captured rung in the tiny-elapsed schedule
    u = out.get("untimestamped", {})
    inst = u.get("by_schedule", {}).get("instant")
    if inst:
        deep = inst[-1]
        print(f"\nHEADLINE (instant, deepest rung @ turn {deep['turn']}): "
              f"real ~{deep['med_real_s']:.0f}s, felt ~{deep['med_felt_s']:.0f}s "
              f"-> {deep['ratio']:.1f}x inflation")

    (M.data_dir / "inflation.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved inflation.json -> {M.data_dir}/")


if __name__ == "__main__":
    main()
