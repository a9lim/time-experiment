"""T2 — felt time is a length-driven prior (offline; reads probe + rows).

Subsumes the decode + inflation + intermittent analyses:

  1. Three-way decode (gt | internal coordinate | verbal estimate), per rendering.
     internal: timestamped = out-of-fold (fit_oof); untimestamped = the
     timestamped probe applied to no-clock slots (the explicit->implicit transfer).
     -> corr(internal,gt), corr(verbal,gt), corr(verbal,internal), overshoots,
        and an H1/H2/H3 reading.
  2. The Aim-2 null: untimestamped/constant slot encodes nothing about elapsed
     beyond length (best-layer R² + partial|tokens ≈ 0).
  3. felt ~ length: the verbal estimate tracks conversation LENGTH, not the clock;
     the inflation ratio felt/real per schedule x length rung.
  4. Clock-density gradient (if the intermittent rendering is present): does a
     sparse clock get extrapolated, or does the model read the last anchor?

Runs on whatever corpus the model dir holds, so TIME_VARIANT=inflation / rates
point it at the dense / uniform-rate corpora.

    TIME_MODEL=gemma python scripts/30_felt.py
    TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/30_felt.py
    TIME_MODEL=gemma TIME_VARIANT=rates python scripts/30_felt.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_ev_probe, assemble, best_layer_sweep, classify_hypothesis,
    cv_predict, load_ev_probe, load_rows, residualize,
)
from time_experiment.config import current_model  # noqa: E402


def _corr(a, b):
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan, math.nan, int(m.sum())
    return float(pearsonr(a[m], b[m])[0]), float(spearmanr(a[m], b[m])[0]), int(m.sum())


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def constant_rows(rows, rendering):
    """assistant constant-mode rows for a rendering, sorted, with verbal."""
    return [r for r in rows if r["source"] == "scripted" and r["rendering"] == rendering
            and r["mode"] == "constant" and r["role"] == "assistant"]


def decode(rows, cache, probe, oof_lookup, rendering):
    """Three-way for one rendering."""
    d = assemble(rows, cache, source="scripted", rendering=rendering, mode="constant",
                 need_gt=True)
    if len(d["gt_log"]) == 0:
        return None
    gt_log = d["gt_log"]
    if rendering == "timestamped":
        internal = np.array([oof_lookup.get((str(i), int(t)), math.nan)
                             for i, t in zip(d["groups"], d["turn_idx"])])
    else:  # transfer: timestamped-trained EV probe on no-clock slots
        internal = apply_ev_probe(probe, d["X3d"])
    verbal_s = d["verbal_s"]
    verbal_log = np.where(np.isfinite(verbal_s) & (verbal_s > 0), np.log(np.where(verbal_s > 0, verbal_s, 1)), math.nan)

    ig = _corr(internal, gt_log)
    vg = _corr(verbal_log, gt_log)
    vi = _corr(verbal_log, internal)
    fin_v = np.isfinite(verbal_s) & (verbal_s > 0)
    ov_verb = float(np.median(verbal_s[fin_v] / np.exp(gt_log[fin_v]))) if fin_v.any() else math.nan
    ov_int = float(np.median(np.exp(internal) / np.exp(gt_log)))
    refusals = int((~fin_v).sum())
    recs = [{"rendering": rendering, "id": str(i), "turn_idx": int(t), "schedule": str(s),
             "gt_s": float(math.exp(g)), "internal_s": float(math.exp(ic)),
             "verbal_s": (float(v) if _finite(v) else "")}
            for i, t, s, g, ic, v in zip(d["groups"], d["turn_idx"], d["schedule"],
                                         gt_log, internal, verbal_s)]
    return {
        "n": int(len(gt_log)), "refusals": refusals,
        "corr_internal_gt": {"pearson": ig[0], "spearman": ig[1]},
        "corr_verbal_gt": {"pearson": vg[0], "spearman": vg[1]},
        "corr_verbal_internal": {"pearson": vi[0], "spearman": vi[1]},
        "overshoot_internal_median": ov_int, "overshoot_verbal_median": ov_verb,
    }, recs


def felt_length(rows, rendering):
    """felt ~ conversation length vs real elapsed, per schedule x turn rung."""
    from scipy.stats import spearmanr
    pts = [(r["schedule"], r["turn_idx"], r["tokens"], r["gt_elapsed_s"], r["verbal_seconds"])
           for r in constant_rows(rows, rendering)
           if _finite(r["verbal_seconds"]) and _finite(r["gt_elapsed_s"])]
    if not pts:
        return None
    toks = [p[2] for p in pts]; felts = [p[4] for p in pts]; gts = [p[3] for p in pts]
    out = {"n": len(pts),
           "rho_felt_vs_length": float(spearmanr(toks, felts)[0]),
           "rho_felt_vs_real": float(spearmanr(gts, felts)[0]), "by_schedule": {}}
    for sch in sorted({p[0] for p in pts}):
        by_turn: dict = {}
        for _, k, tk, gt, fl in (p for p in pts if p[0] == sch):
            by_turn.setdefault(k, {"tok": [], "gt": [], "felt": []})
            by_turn[k]["tok"].append(tk); by_turn[k]["gt"].append(gt); by_turn[k]["felt"].append(fl)
        out["by_schedule"][sch] = [
            {"turn": k, "med_tokens": st.median(v["tok"]), "med_real_s": st.median(v["gt"]),
             "med_felt_s": st.median(v["felt"]),
             "ratio": (st.median(v["felt"]) / st.median(v["gt"]) if st.median(v["gt"]) > 0 else math.nan)}
            for k, v in sorted(by_turn.items())]
    return out


def gradient(rows, stride=4):
    """Clock-density gradient: rate-sensitivity at fixed length + (intermittent)
    last-anchor vs current-turn ratio."""
    from scipy.stats import spearmanr
    out = {}
    for rendering in ("timestamped", "intermittent", "untimestamped"):
        pts = [(r["schedule"], r["turn_idx"], r["tokens"], r["gt_elapsed_s"], r["verbal_seconds"])
               for r in constant_rows(rows, rendering)
               if _finite(r["verbal_seconds"]) and _finite(r["gt_elapsed_s"])]
        if not pts:
            continue
        true = np.array([p[3] for p in pts]); stated = np.array([p[4] for p in pts])
        slope = float(np.polyfit(np.log(true), np.log(stated), 1)[0])
        per_turn = []
        for k in sorted({p[1] for p in pts}):
            grp = [(p[3], p[4]) for p in pts if p[1] == k]
            if len({g[0] for g in grp}) >= 3:
                per_turn.append(spearmanr([g[0] for g in grp], [g[1] for g in grp])[0])
        entry = {"n": len(pts), "loglog_slope": slope,
                 "spearman_true": float(spearmanr(true, stated)[0]),
                 "rate_sensitivity": float(np.nanmean(per_turn)) if per_turn else math.nan}
        if rendering == "intermittent":
            cur, anch = [], []
            for (_r, k, _t, true_cur, stated_s) in pts:
                gap = true_cur / k if k else math.nan
                true_anchor = ((k // stride) * stride) * gap
                if true_cur > 0:
                    cur.append(stated_s / true_cur)
                if true_anchor > 0:
                    anch.append(stated_s / true_anchor)
            entry["ratio_vs_current"] = float(st.median(cur)) if cur else math.nan
            entry["ratio_vs_last_anchor"] = float(st.median(anch)) if anch else math.nan
        out[rendering] = entry
    return out


def main() -> None:
    M = current_model()
    rows = load_rows(M.rows_path)
    cache = StatesCache(M.hidden_dir)
    probe, pmeta = load_ev_probe(M.probe_path)
    oof = np.load(M.data_dir / "fit_oof.npz", allow_pickle=False)
    oof_lookup = {(str(i), int(t)): float(p)
                  for i, t, p in zip(oof["id"], oof["turn_idx"], oof["oof_pred_log"])}
    print(f"model: {M.short_name}  EV all-layer probe (R2={pmeta['r2']:.3f})")

    summary: dict = {"probe_kind": "ev", "probe_r2": pmeta["r2"]}
    decode_recs: list[dict] = []

    # 1. three-way decode
    for rendering in ("timestamped", "untimestamped"):
        got = decode(rows, cache, probe, oof_lookup, rendering)
        if not got:
            continue
        dec, recs = got
        summary[f"decode_{rendering}"] = dec
        decode_recs.extend(recs)
        tag = "  <- TRANSFER" if rendering == "untimestamped" else "  (out-of-fold)"
        print(f"\n[{rendering}] n={dec['n']} refusals={dec['refusals']}")
        print(f"  internal~gt   r={dec['corr_internal_gt']['pearson']:+.3f}{tag}")
        print(f"  verbal~gt     r={dec['corr_verbal_gt']['pearson']:+.3f}")
        print(f"  verbal~intern r={dec['corr_verbal_internal']['pearson']:+.3f}")
        print(f"  overshoot: verbal x{dec['overshoot_verbal_median']:.2f}  internal x{dec['overshoot_internal_median']:.2f}")

    u = summary.get("decode_untimestamped")
    if u:
        verdict = classify_hypothesis(
            corr_verbal_internal=u["corr_verbal_internal"]["spearman"],
            corr_internal_gt=u["corr_internal_gt"]["spearman"],
            overshoot_internal=u["overshoot_internal_median"],
            overshoot_verbal=u["overshoot_verbal_median"])
        summary["verdict"] = verdict
        print(f"\nreading: {verdict}")

    # 2. Aim-2 null: untimestamped/constant slot beyond length?
    du = assemble(rows, cache, source="scripted", rendering="untimestamped", mode="constant")
    if len(du["gt_log"]) >= 8:
        bi, br2, _ = best_layer_sweep(du["X3d"], du["gt_log"], du["groups"])
        log_tok = np.log(np.maximum(du["tokens"], 1.0))
        _, par, _ = cv_predict(du["X3d"][:, bi, :], residualize(du["gt_log"], log_tok), du["groups"])
        summary["noclock_null"] = {"best_r2": float(br2), "partial_r2": float(par)}
        print(f"\nno-clock null: best R2(gt)={br2:+.3f}  partial|tokens={par:+.3f}  (≈0 = no felt beyond length)")

    # 3. felt ~ length
    fl = felt_length(rows, "untimestamped")
    if fl:
        summary["felt_length"] = fl
        print(f"\nfelt~length rho={fl['rho_felt_vs_length']:+.3f}  felt~real rho={fl['rho_felt_vs_real']:+.3f}")
        inst = fl["by_schedule"].get("instant") or next(iter(fl["by_schedule"].values()), None)
        if inst:
            deep = inst[-1]
            print(f"  deepest rung: real ~{deep['med_real_s']:.0f}s  felt ~{deep['med_felt_s']:.0f}s "
                  f"-> {deep['ratio']:.1f}x")

    # 4. clock-density gradient (only meaningful if intermittent present)
    if any(r["rendering"] == "intermittent" for r in rows):
        summary["gradient"] = gradient(rows)
        for rendering, e in summary["gradient"].items():
            extra = ""
            if rendering == "intermittent":
                extra = f"  cur={e['ratio_vs_current']:.2f} anchor={e['ratio_vs_last_anchor']:.2f}"
            print(f"  gradient {rendering}: slope={e['loglog_slope']:+.2f} "
                  f"rate-sens={e['rate_sensitivity']:+.3f}{extra}")

    (M.data_dir / "felt.json").write_text(json.dumps(summary, indent=2))
    if decode_recs:
        import csv
        with (M.data_dir / "decode_rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(decode_recs[0]))
            w.writeheader(); w.writerows(decode_recs)
    print(f"\nsaved felt.json (+ decode_rows.csv) -> {M.data_dir}/")


if __name__ == "__main__":
    main()
