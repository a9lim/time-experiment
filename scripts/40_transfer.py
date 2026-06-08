"""T3 — one axis, and it transfers: scripted clock-elapsed -> natural felt
(offline; reads probe + rows).

The scripted corpus has no felt variance (neutral content, ~constant felt); the
variance lives in the natural conversations (neutral ~min / affect ~min+ /
time-language ~hours). So:

  Test A — within natural: does the constant-prefill slot linearly encode
           log(felt), and beyond conversation length? (per-layer grouped-CV by
           conversation + partial|tokens)
  Test B — cross-axis: apply the saved scripted clock-elapsed probe (Lstar,
           gt-selected, non-circular) to natural slots — does its read track
           natural FELT, or only length? One axis for clock-reading and felt.
  OOD    — how far natural slots sit off the scripted manifold (Mahalanobis
           ratio). The slot sits ~6× off but *tightly* (median≈max); the EOT
           site's 3.2×/18.8× heavy tail made its probe explode, the slot's
           bounded offset doesn't — so the raw EV read stays usable unwhitened.
  Control— injected clock on natural prose: does the verbal recover it, and does
           the slot-probe read recover it? (the behavioral-vs-probe dissociation)
  Content— per-variant felt + slot-read medians (ordering vs magnitude).

    TIME_MODEL=gemma python scripts/40_transfer.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_ev_probe, assemble, cv_predict, load_ev_probe, load_rows,
    maha_scorer, residualize,
)
from time_experiment.config import current_model  # noqa: E402


def _rho(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else math.nan


def maha_ratio(scripted_X3d, layers, natural_X3d):
    """Median/max over natural turns of the median-over-layers Mahalanobis ratio
    (natural distance / scripted-median distance). ~1 = on-manifold."""
    scorer = maha_scorer(scripted_X3d, layers)
    if scorer is None:
        return math.nan, math.nan
    ratios = scorer(natural_X3d)
    return (float(np.median(ratios)) if len(ratios) else math.nan,
            float(np.max(ratios)) if len(ratios) else math.nan)


def main() -> None:
    M = current_model()
    rows = load_rows(M.rows_path)
    cache = StatesCache(M.hidden_dir)
    probe, pmeta = load_ev_probe(M.probe_path)
    layers = [int(L) for L in probe["layers"]]
    out: dict = {"probe_kind": "ev", "probe_r2": pmeta["r2"]}

    # natural felt slots (constant, no gt) — felt = verbal estimate.
    dn = assemble(rows, cache, source="natural", rendering="untimestamped",
                  mode="constant", need_gt=False)
    felt_s = dn["verbal_s"]
    keep = np.isfinite(felt_s) & (felt_s > 0)
    if keep.sum() < 8:
        raise SystemExit(f"only {int(keep.sum())} natural felt turns — run 01_natural + 10_capture")
    Xn = dn["X3d"][keep]; felt_log = np.log(felt_s[keep]); gn = dn["groups"][keep]
    tn = dn["tokens"][keep]; vn = dn["variant"][keep]
    print(f"model: {M.short_name}  EV all-layer probe  natural felt turns={int(keep.sum())}  "
          f"conversations={len(set(gn))}")

    # Test A — within natural: felt readable from slot, beyond length?
    log_tok = np.log(np.maximum(tn, 1.0))
    _, r2_len, _ = cv_predict(log_tok[:, None], felt_log, gn)
    best = (-1, -1e9)
    for li in range(Xn.shape[1]):
        _, r2, _ = cv_predict(Xn[:, li, :], felt_log, gn)
        if r2 > best[1]:
            best = (li, r2)
    abi, ar2 = best
    _, a_par, _ = cv_predict(Xn[:, abi, :], residualize(felt_log, log_tok), gn)
    out["within_natural"] = {"length_baseline_r2": float(r2_len),
                             "best_layer": int(layers[abi]), "best_r2": float(ar2),
                             "partial_r2": float(a_par)}
    print(f"\n[A] within-natural felt: length R2={r2_len:+.3f}  best L{layers[abi]} R2={ar2:+.3f}  "
          f"partial|len={a_par:+.3f}")

    # Test B — cross-axis: saved scripted EV clock probe on natural slots.
    clock_read = apply_ev_probe(probe, Xn)
    out["crossaxis"] = {"rho_clockprobe_vs_felt": _rho(clock_read, felt_log),
                        "rho_clockprobe_vs_length": _rho(clock_read, tn)}
    print(f"[B] cross-axis (EV clock probe -> natural):  "
          f"felt rho={out['crossaxis']['rho_clockprobe_vs_felt']:+.3f}  "
          f"length rho={out['crossaxis']['rho_clockprobe_vs_length']:+.3f}")

    # OOD — natural vs scripted (timestamped/constant) manifold distance.
    ds = assemble(rows, cache, source="scripted", rendering="timestamped", mode="constant")
    if len(ds["gt_log"]) >= 8:
        med, mx = maha_ratio(ds["X3d"], layers, Xn)
        out["ood_ratio_median"], out["ood_ratio_max"] = med, mx
        print(f"[OOD] natural slot off-manifold: median {med:.2f}x, max {mx:.2f}x  "
              f"(bounded: median≈max; EOT was 3.2x/18.8x heavy-tailed -> its probe blew up)")

    # Control — injected clock on natural prose (timestamped natural).
    di = assemble(rows, cache, source="natural", rendering="timestamped",
                  mode="constant", need_gt=True)
    if len(di["gt_log"]) >= 5:
        inj_clock_read = apply_ev_probe(probe, di["X3d"])
        out["injected_control"] = {
            "rho_verbal_vs_injected": _rho(np.log(np.where(di["verbal_s"] > 0, di["verbal_s"], np.nan)), di["gt_log"]),
            "rho_probe_vs_injected": _rho(inj_clock_read, di["gt_log"])}
        print(f"[control] injected clock: verbal recovers rho={out['injected_control']['rho_verbal_vs_injected']:+.3f}  "
              f"probe recovers rho={out['injected_control']['rho_probe_vs_injected']:+.3f}")

    # Content — per-variant felt + slot read.
    slot_read = cv_predict(Xn[:, abi, :], felt_log, gn)[0]
    out["per_variant"] = {}
    print("\n[content] per variant (felt vs slot-read, median seconds):")
    for v in sorted(set(vn)):
        m = vn == v
        fm = float(np.exp(np.median(felt_log[m]))); sm = float(np.exp(np.median(slot_read[m])))
        out["per_variant"][str(v)] = {"n": int(m.sum()), "felt_s": fm, "slot_read_s": sm}
        print(f"  {str(v):14s} n={int(m.sum())}  felt~{fm:7.0f}s  slot_read~{sm:7.0f}s")

    (M.data_dir / "transfer.json").write_text(json.dumps(out, indent=2))
    import csv
    with (M.data_dir / "natural_reads.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "variant", "tokens", "felt_s", "clock_read_s", "slot_read_s"])
        for i in range(len(felt_log)):
            w.writerow([gn[i], vn[i], tn[i], round(math.exp(felt_log[i]), 1),
                        round(math.exp(clock_read[i]), 1), round(math.exp(slot_read[i]), 1)])
    print(f"\nsaved transfer.json (+ natural_reads.csv) -> {M.data_dir}/")


if __name__ == "__main__":
    main()
