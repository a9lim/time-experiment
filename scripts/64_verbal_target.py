"""Can the prefill-slot read the model's STATED duration (the felt construction),
including on natural conversations where there's no clock? (offline)

The felt variance lives in the natural data (neutral ~5 min / affect ~10 min /
time-language ~2 h), not the neutral scripted corpus. So:

  Test A — within natural: does the constant-prefill slot linearly encode log(felt)
           — and beyond conversation length (the content effect)? Per-layer
           grouped-CV (by conversation), + partial after residualizing log-tokens.
  Test B — cross-axis: apply the scripted timestamped/constant *clock-elapsed*
           probe to the natural slots; does its output track natural felt, or only
           length? (Is felt on the clock axis, or a separate construction?)

Verbal estimates: scripted from turns.jsonl (A_clock/B_felt seconds); natural
felt_s from naturalistic.jsonl. Slot activations from 62 (constant mode).

n is small (≈25 natural turns, ~3 felt levels) — read as directional.

    TIME_MODEL=gemma python scripts/64_verbal_target.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    apply_probe, cv_predict, fit_full, residualize,
)
from time_experiment.config import DATA_DIR, resolve_model  # noqa: E402


def _rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _scripted_verbal(base) -> dict:
    """(tid, turn, rendering) -> stated seconds from the EOT readout fork."""
    out = {}
    for r in _rows(base.turns_path):
        if r["role"] != "assistant":
            continue
        ph = "A_clock" if r["rendering"] == "timestamped" else "B_felt"
        sec = (r.get("readouts", {}).get(ph) or {}).get("seconds")
        if isinstance(sec, (int, float)) and math.isfinite(sec) and sec > 0:
            out[(r["transcript_id"], r["turn_idx"], r["rendering"])] = float(sec)
    return out


def _natural_felt(base) -> dict:
    """(conv_id, turn) -> felt seconds (untimestamped B_felt)."""
    out = {}
    p = DATA_DIR / f"{base.short_name}_natural" / "naturalistic.jsonl"
    for r in _rows(p):
        if r["rendering"] != "untimestamped":
            continue
        s = r.get("felt_s")
        if isinstance(s, (int, float)) and math.isfinite(s) and s > 0:
            out[(r["conv_id"], r["turn_idx"])] = float(s)
    return out


def _load_slots(hid: Path, source, rendering, mode, ids):
    """{id: (H (T,L,D), layers, {turn: idx})}."""
    out = {}
    for idd in ids:
        p = hid / f"{source}__{idd}__{rendering}__{mode}.npz"
        if p.exists():
            d = np.load(p)
            out[idd] = (d["H"], [int(L) for L in d["layers"]],
                        {int(t): i for i, t in enumerate(d["turn_idxs"])})
    return out


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    edir = DATA_DIR / f"{base.short_name}_elicit"
    hid = edir / "hidden"
    rows = _rows(edir / "elicit_rows.jsonl")
    sver, nfelt = _scripted_verbal(base), _natural_felt(base)

    # --- assemble natural constant slots + felt + tokens ---
    nat_ids = sorted({r["id"] for r in rows if r["source"] == "natural"})
    nat = _load_slots(hid, "natural", "untimestamped", "constant", nat_ids)
    Xn, yn, gn, tn, vn, layers = [], [], [], [], [], None
    for r in rows:
        if r["source"] != "natural":
            continue
        key = (r["id"], r["turn_idx"])
        if key not in nfelt or r["id"] not in nat:
            continue
        H, layers, tpos = nat[r["id"]]
        if r["turn_idx"] not in tpos:
            continue
        Xn.append(H[tpos[r["turn_idx"]]]); yn.append(math.log(nfelt[key]))
        gn.append(r["id"]); tn.append(r["tokens"]); vn.append(r.get("variant"))
    Xn = np.asarray(Xn, np.float32); yn = np.asarray(yn); gn = np.asarray(gn)
    tn = np.asarray(tn, float); vn = np.asarray(vn)
    if len(yn) < 8 or layers is None:
        raise SystemExit(f"only {len(yn)} natural felt samples — need 62 natural capture + naturalistic.jsonl")
    print(f"natural: n={len(yn)}  conversations={len(set(gn))}  "
          f"felt levels={sorted(set(round(math.exp(v)) for v in yn))}")

    # --- Test A: within-natural felt readability (per-layer grouped CV) ---
    print("\n[Test A] natural constant-slot -> log(felt), grouped CV by conversation")
    logtok = np.log(np.maximum(tn, 1.0))
    _, r2_tok, _ = cv_predict(logtok[:, None], yn, gn, n_splits=5)
    best = (-1, -1e9)
    per_layer = []
    for li in range(Xn.shape[1]):
        _, r2, _ = cv_predict(Xn[:, li, :], yn, gn, n_splits=5)
        per_layer.append(r2)
        if r2 > best[1]:
            best = (li, r2)
    bi, br2 = best
    _, r2_par, _ = cv_predict(Xn[:, bi, :], residualize(yn, logtok), gn, n_splits=5)
    print(f"  length baseline R²(felt)= {r2_tok:+.3f}")
    print(f"  best layer L{layers[bi]}  R²(felt)= {br2:+.3f}   partial(|len)= {r2_par:+.3f}")
    # felt vs length descriptively
    from scipy.stats import spearmanr
    print(f"  felt~length rho= {spearmanr(tn, yn).statistic:+.3f}")

    # --- Test B: cross-axis. scripted timestamped/constant clock probe -> natural ---
    sc_ids = sorted({r["id"] for r in rows if r["source"] == "scripted"})
    scc = _load_slots(hid, "scripted", "timestamped", "constant", sc_ids)
    Xs, ys = [], []
    for r in rows:
        if r["source"] != "scripted" or r["rendering"] != "timestamped" or r["mode"] != "constant":
            continue
        if r["id"] not in scc or r["gt_elapsed_s"] is None:
            continue
        H, _, tpos = scc[r["id"]]
        if r["turn_idx"] in tpos:
            Xs.append(H[tpos[r["turn_idx"]]]); ys.append(math.log(max(r["gt_elapsed_s"], 1.0)))
    Xs = np.asarray(Xs, np.float32); ys = np.asarray(ys)
    probe = fit_full(Xs[:, bi, :], ys)               # clock-elapsed axis @ best felt layer
    clock_read = apply_probe(probe, Xn[:, bi, :])    # apply to natural slots
    rho_felt = spearmanr(clock_read, yn).statistic
    rho_len = spearmanr(clock_read, tn).statistic
    print(f"\n[Test B] scripted clock-elapsed probe (L{layers[bi]}) applied to natural slots:")
    print(f"  clock-read ~ natural felt:   rho= {rho_felt:+.3f}")
    print(f"  clock-read ~ natural length: rho= {rho_len:+.3f}")

    # --- descriptive: per-variant felt + best-layer slot read ---
    slot_read = cv_predict(Xn[:, bi, :], yn, gn, n_splits=5)[0]
    print("\n[per variant] natural (felt vs slot-read, median seconds):")
    for v in sorted(set(vn)):
        m = vn == v
        print(f"  {v:14s} n={m.sum()}  felt~{np.exp(np.median(yn[m])):7.0f}s   "
              f"slot_read~{np.exp(np.median(slot_read[m])):7.0f}s")

    out = {
        "natural_n": int(len(yn)), "felt_length_baseline_r2": float(r2_tok),
        "felt_best_layer": int(layers[bi]), "felt_best_r2": float(br2),
        "felt_partial_r2": float(r2_par),
        "crossaxis_clockprobe_vs_felt_rho": float(rho_felt),
        "crossaxis_clockprobe_vs_length_rho": float(rho_len),
    }
    (edir / "verbal_target_summary.json").write_text(json.dumps(out, indent=2))

    # Per-row natural reads for plotting.
    import csv
    with (edir / "natural_reads.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "variant", "tokens", "felt_s", "clock_read_s", "slot_read_s"])
        for i in range(len(yn)):
            w.writerow([gn[i], vn[i], tn[i], round(math.exp(yn[i]), 1),
                        round(math.exp(clock_read[i]), 1), round(math.exp(slot_read[i]), 1)])
    print(f"\nsaved -> {edir}/verbal_target_summary.json + natural_reads.csv")


if __name__ == "__main__":
    main()
