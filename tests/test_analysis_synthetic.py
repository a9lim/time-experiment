"""End-to-end analysis test on fabricated data (no model).

Builds a fake model directory whose slot activations linearly encode log(elapsed)
along one direction (on ONE of three layers), plus an ORTHOGONAL token-count
component, and whose verbal readouts overshoot ground truth ~5x while tracking
it. Then runs the real (single-layer slot) analysis pipeline and asserts:

  - the per-layer sweep picks the signal layer and recovers log-elapsed
  - the probe beats the token-only baseline AND survives token-partialling
    (the position-confound control)
  - the timestamped-trained probe transfers to untimestamped activations
  - the H1/H2/H3 classifier reads this construction as H3
  - assemble keeps natural (no-gt) rows only under need_gt=False, never leaking
    them into a gt fit (the unified-schema guard)

    python3 tests/test_analysis_synthetic.py
"""
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_ev_probe, apply_probe, assemble, best_layer_sweep,
    classify_hypothesis, cv_predict, ev_combined_oof, fit_ev_probe, fit_full,
    load_ev_probe, load_rows, residualize, save_ev_probe,
)
from time_experiment.storage import save_states, sidecar_path  # noqa: E402

RENDERINGS = ("timestamped", "untimestamped")
LAYERS = [2, 5, 9]      # signal lives on layer 5; 2 and 9 are noise
SIGNAL_LAYER = 5
fails = 0


def check(cond: bool, msg: str) -> None:
    global fails
    print(f"{'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails += 1


def _write_rows(out, rows):
    for r in rows:
        out.write(json.dumps(r) + "\n")


def build_fake(model, *, n_tx=30, T=8, D=32, alpha=1.2, sigma=1.5, sigma_noise=3.0,
               beta=2.0, overshoot=5.0, seed=0):
    rng = np.random.default_rng(seed)
    d_time = rng.standard_normal(D); d_time /= np.linalg.norm(d_time)
    d_tok = rng.standard_normal(D)
    d_tok -= (d_tok @ d_time) * d_time            # orthogonalize vs time dir
    d_tok /= np.linalg.norm(d_tok)

    model.hidden_dir.mkdir(parents=True, exist_ok=True)
    model.rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with model.rows_path.open("w") as out:
        # --- scripted: both renderings, mode=constant ---
        for i in range(n_tx):
            rate = math.exp(rng.uniform(math.log(10), math.log(1e6)))  # s/turn
            for rendering in RENDERINGS:
                states, elapsed_by_turn = {}, {}
                for k in range(T):
                    elapsed = (k + 1) * rate
                    tokens = 50 * (k + 1)
                    sig = (alpha * math.log(elapsed)) * d_time + (beta * math.log(tokens)) * d_tok
                    states[k] = {
                        2: (sigma_noise * rng.standard_normal(D)).astype(np.float32),
                        5: (sig + sigma * rng.standard_normal(D)).astype(np.float32),
                        9: (sigma_noise * rng.standard_normal(D)).astype(np.float32),
                    }
                    elapsed_by_turn[k] = elapsed
                    role = "user" if k % 2 == 0 else "assistant"
                    verbal_s = None
                    if role == "assistant" and rng.random() >= 0.12:  # 1/8 refuse
                        verbal_s = float(elapsed * overshoot * math.exp(rng.normal(0, 0.2)))
                    rows.append({
                        "source": "scripted", "id": f"tx{i:03d}", "rendering": rendering,
                        "mode": "constant", "turn_idx": k, "role": role,
                        "gt_elapsed_s": elapsed, "tokens": tokens, "schedule": "synthetic",
                        "variant": None, "verbal_dist": None, "verbal_seconds": verbal_s,
                    })
                save_states(
                    sidecar_path(model.hidden_dir, "scripted", f"tx{i:03d}", rendering, "constant"),
                    states=states, elapsed_by_turn=elapsed_by_turn,
                )
        _write_rows(out, rows)

        # --- natural: untimestamped, constant, NO gt (the no-gt guard) ---
        nat_rows = []
        for c in range(6):
            states = {}
            for k in range(T):
                states[k] = {L: (sigma_noise * rng.standard_normal(D)).astype(np.float32) for L in LAYERS}
                role = "user" if k % 2 == 0 else "assistant"
                if role != "assistant":
                    continue
                nat_rows.append({
                    "source": "natural", "id": f"conv{c}", "rendering": "untimestamped",
                    "mode": "constant", "turn_idx": k, "role": role,
                    "gt_elapsed_s": None, "tokens": 50 * (k + 1), "schedule": None,
                    "variant": ("affect" if c % 2 else "neutral"),
                    "verbal_dist": None, "verbal_seconds": float(300 * (c + 1)),
                })
            save_states(
                sidecar_path(model.hidden_dir, "natural", f"conv{c}", "untimestamped", "constant"),
                states=states, elapsed_by_turn=None,
            )
        _write_rows(out, nat_rows)
    return overshoot


def main() -> int:
    with tempfile.TemporaryDirectory() as dd:
        ddp = Path(dd)
        model = SimpleNamespace(
            rows_path=ddp / "rows.jsonl", hidden_dir=ddp / "hidden", data_dir=ddp,
        )
        build_fake(model)
        rows = load_rows(model.rows_path)
        cache = StatesCache(model.hidden_dir)

        # --- T1: per-layer sweep picks the signal layer, recovers log-elapsed ---
        d = assemble(rows, cache, source="scripted", rendering="timestamped", mode="constant")
        bi, br2, _ = best_layer_sweep(d["X3d"], d["gt_log"], d["groups"])
        check(d["layers"][bi] == SIGNAL_LAYER, f"sweep picks layer {SIGNAL_LAYER} (got {d['layers'][bi]})")
        check(br2 > 0.7, f"slot probe recovers log-elapsed (CV R2={br2:.3f} > 0.7)")

        # --- position-confound controls ---
        Xb = d["X3d"][:, bi, :]
        log_tok = np.log(d["tokens"])
        _, r2_tok, _ = cv_predict(log_tok[:, None], d["gt_log"], d["groups"])
        resid = residualize(d["gt_log"], log_tok)
        _, r2_partial, _ = cv_predict(Xb, resid, d["groups"])
        check(br2 > r2_tok + 0.2, f"probe beats token baseline (R2 {br2:.3f} vs {r2_tok:.3f})")
        check(r2_partial > 0.5, f"survives token-partialling (partial R2={r2_partial:.3f} > 0.5)")

        # --- transfer: single-layer timestamped probe -> untimestamped acts ---
        probe = fit_full(Xb, d["gt_log"])
        du = assemble(rows, cache, source="scripted", rendering="untimestamped", mode="constant")
        pred_u = apply_probe(probe, du["X3d"][:, bi, :])
        r_transfer = float(np.corrcoef(pred_u, du["gt_log"])[0, 1])
        check(r_transfer > 0.7, f"single-layer transfer holds (r={r_transfer:.3f} > 0.7)")

        # --- EV-weighted all-layer probe (saklas idiom) ---
        ev = ev_combined_oof(d["X3d"], d["gt_log"], d["groups"])
        check(abs(ev["weights"].sum() - 1.0) < 1e-9, "EV weights sum to 1")
        check(int(np.argmax(ev["weights"])) == bi,
              f"EV weight concentrates on the signal layer (argmax w == L{SIGNAL_LAYER})")
        check(ev["r2"] >= br2 - 0.05, f"EV combined R²={ev['r2']:.3f} ≳ best single {br2:.3f}")
        ev_probe = fit_ev_probe(d["X3d"], d["gt_log"], d["groups"], d["layers"])
        pred_u_ev = apply_ev_probe(ev_probe, du["X3d"])
        r_ev = float(np.corrcoef(pred_u_ev, du["gt_log"])[0, 1])
        check(r_ev > 0.7, f"EV probe transfers explicit->implicit (r={r_ev:.3f} > 0.7)")
        # save/load round-trip is bit-faithful
        evp_path = ddp / "probe.npz"
        save_ev_probe(evp_path, ev_probe, meta={"probe_kind": "ev"})
        ev_loaded, _meta = load_ev_probe(evp_path)
        check(np.allclose(apply_ev_probe(ev_loaded, du["X3d"]), pred_u_ev),
              "EV probe save/load round-trips")

        # --- T2 decode: internal | gt | verbal on untimestamped assistant turns ---
        from scipy.stats import spearmanr
        internal = pred_u
        gt_log = du["gt_log"]
        verbal_s = du["verbal_s"]
        fin = np.isfinite(verbal_s)
        verbal_log = np.where(fin, np.log(np.where(fin, verbal_s, 1.0)), np.nan)
        vi = float(spearmanr(verbal_log[fin], internal[fin])[0])
        ig = float(spearmanr(internal, gt_log)[0])
        ov_int = float(np.median(np.exp(internal) / np.exp(gt_log)))
        ov_verb = float(np.median(verbal_s[fin] / np.exp(gt_log[fin])))
        check(abs(ov_int - 1.0) < 0.6, f"internal ~ calibrated to gt (overshoot x{ov_int:.2f} ~ 1)")
        check(2.0 < ov_verb < 12.0, f"verbal overshoots ~5x (overshoot x{ov_verb:.2f})")
        verdict = classify_hypothesis(
            corr_verbal_internal=vi, corr_internal_gt=ig,
            overshoot_internal=ov_int, overshoot_verbal=ov_verb,
        )
        print(f"     verdict: {verdict}")
        check(verdict.startswith("H3"), "classifier reads construction as H3")

        # --- unified-schema guard: natural (no gt) only under need_gt=False ---
        dn = assemble(rows, cache, source="natural", rendering="untimestamped",
                      mode="constant", need_gt=False)
        check(len(dn["gt_log"]) > 0, f"natural rows assembled with need_gt=False (n={len(dn['gt_log'])})")
        check(bool(np.all(np.isnan(dn["gt_log"]))), "natural gt_log is all-NaN")
        check(set(dn["variant"]) == {"neutral", "affect"}, "natural variants carried through")
        check(np.isfinite(dn["verbal_s"]).all(), "natural verbal_s carried through")
        dn_gt = assemble(rows, cache, source="natural", rendering="untimestamped",
                         mode="constant", need_gt=True)
        check(len(dn_gt["gt_log"]) == 0, "no-gt natural rows never leak into a gt fit")

    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
