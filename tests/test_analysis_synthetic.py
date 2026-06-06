"""End-to-end analysis test on fabricated data (no model).

Builds a fake model directory whose EOT activations linearly encode
log(elapsed) along one direction, plus an ORTHOGONAL token-count component, and
whose verbal readouts overshoot ground truth ~5x while tracking it. Then runs
the real analysis pipeline and asserts:

  - the per-layer probe recovers log-elapsed (high grouped-CV R^2)
  - the probe beats the token-only baseline AND survives token-partialling
    (the position-confound control)
  - the timestamped-trained probe transfers to untimestamped activations
  - the H1/H2/H3 classifier reads this construction as H3

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
    apply_probe, assemble_layer, classify_hypothesis, cv_predict, fit_full,
    load_rows, residualize,
)
from time_experiment.storage import save_transcript_states, sidecar_path  # noqa: E402

RENDERINGS = ("timestamped", "untimestamped")
fails = 0


def check(cond: bool, msg: str) -> None:
    global fails
    print(f"{'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails += 1


def build_fake(model, *, n_tx=30, T=8, D=32, layer=5,
               alpha=1.2, sigma=1.5, beta=2.0, overshoot=5.0, seed=0):
    rng = np.random.default_rng(seed)
    d_time = rng.standard_normal(D); d_time /= np.linalg.norm(d_time)
    d_tok = rng.standard_normal(D)
    d_tok -= (d_tok @ d_time) * d_time            # orthogonalize vs time dir
    d_tok /= np.linalg.norm(d_tok)

    model.hidden_dir.mkdir(parents=True, exist_ok=True)
    model.turns_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with model.turns_path.open("w") as out:
        for i in range(n_tx):
            rate = math.exp(rng.uniform(math.log(10), math.log(1e6)))  # s/turn
            for rendering in RENDERINGS:
                states, elapsed_by_turn = {}, {}
                for k in range(T):
                    elapsed = (k + 1) * rate
                    log_e = math.log(elapsed)
                    tokens = 50 * (k + 1)
                    log_t = math.log(tokens)
                    v = (alpha * log_e) * d_time + (beta * log_t) * d_tok
                    v = v + sigma * rng.standard_normal(D)
                    states[k] = {layer: v.astype(np.float32)}
                    elapsed_by_turn[k] = elapsed
                    role = "user" if k % 2 == 0 else "assistant"
                    readouts = {}
                    if role == "assistant":
                        # ~5x overshoot, tracking gt; one in eight refuses.
                        if rng.random() < 0.12:
                            secs = float("nan")
                        else:
                            secs = elapsed * overshoot * math.exp(rng.normal(0, 0.2))
                        readouts["A_clock" if rendering == "timestamped" else "B_felt"] = {
                            "raw": f"~{secs:.0f}s", "seconds": secs,
                        }
                    rows.append({
                        "transcript_id": f"tx{i:03d}", "schedule": "synthetic",
                        "turn_count": T, "target_words": 20, "rendering": rendering,
                        "turn_idx": k, "role": role, "gt_elapsed_s": elapsed,
                        "prompt_tokens": tokens, "readouts": readouts,
                    })
                    out.write(json.dumps(rows[-1]) + "\n")
                save_transcript_states(
                    sidecar_path(model.hidden_dir, f"tx{i:03d}", rendering),
                    states=states, elapsed_by_turn=elapsed_by_turn,
                )
    return layer, overshoot


def main() -> int:
    with tempfile.TemporaryDirectory() as dd:
        ddp = Path(dd)
        model = SimpleNamespace(
            turns_path=ddp / "turns.jsonl", hidden_dir=ddp / "hidden",
            data_dir=ddp,
        )
        layer, _ = build_fake(model)
        rows = load_rows(model)

        # --- Aim 1: probe recovers log-elapsed (timestamped) ---
        d = assemble_layer(model, rows, layer, rendering="timestamped")
        _, r2, _ = cv_predict(d["X"], d["y_log"], d["groups"])
        check(r2 > 0.7, f"probe recovers log-elapsed (CV R2={r2:.3f} > 0.7)")

        # --- position-confound controls ---
        log_tok = np.log(d["tokens"])
        _, r2_tok, _ = cv_predict(log_tok[:, None], d["y_log"], d["groups"])
        resid = residualize(d["y_log"], log_tok)
        _, r2_partial, _ = cv_predict(d["X"], resid, d["groups"])
        check(r2 > r2_tok + 0.2, f"probe beats token baseline (R2 {r2:.3f} vs {r2_tok:.3f})")
        check(r2_partial > 0.5, f"survives token-partialling (partial R2={r2_partial:.3f} > 0.5)")

        # --- transfer: timestamped probe -> untimestamped acts ---
        probe = fit_full(d["X"], d["y_log"])
        du = assemble_layer(model, rows, layer, rendering="untimestamped")
        pred_u = apply_probe(probe, du["X"])
        r_transfer = float(np.corrcoef(pred_u, du["y_log"])[0, 1])
        check(r_transfer > 0.7, f"explicit->implicit transfer holds (r={r_transfer:.3f} > 0.7)")

        # --- Aim 2: decode quantities on untimestamped assistant turns ---
        from scipy.stats import spearmanr
        internal, gt_log, verbal_log, ratios_int, ratios_verb = [], [], [], [], []
        for r in rows:
            if r["rendering"] != "untimestamped" or r["role"] != "assistant":
                continue
            ts_path = sidecar_path(model.hidden_dir, r["transcript_id"], "untimestamped")
            from time_experiment.storage import load_transcript_states
            v = load_transcript_states(ts_path).vec(r["turn_idx"], layer)
            icoord = float(apply_probe(probe, v[None, :])[0])
            g = math.log(r["gt_elapsed_s"])
            vs = r["readouts"].get("B_felt", {}).get("seconds", float("nan"))
            internal.append(icoord); gt_log.append(g)
            ratios_int.append(math.exp(icoord) / r["gt_elapsed_s"])
            if isinstance(vs, (int, float)) and math.isfinite(vs) and vs > 0:
                verbal_log.append(math.log(vs)); ratios_verb.append(vs / r["gt_elapsed_s"])
            else:
                verbal_log.append(math.nan)

        internal = np.array(internal); gt_log = np.array(gt_log)
        verbal_log = np.array(verbal_log)
        mask = np.isfinite(verbal_log)
        vi = float(spearmanr(verbal_log[mask], internal[mask])[0])
        ig = float(spearmanr(internal, gt_log)[0])
        ov_int = float(np.median(ratios_int))
        ov_verb = float(np.median(ratios_verb))
        check(abs(ov_int - 1.0) < 0.6, f"internal ~ calibrated to gt (overshoot x{ov_int:.2f} ~ 1)")
        check(2.0 < ov_verb < 12.0, f"verbal overshoots ~5x (overshoot x{ov_verb:.2f})")
        verdict = classify_hypothesis(
            corr_verbal_internal=vi, corr_internal_gt=ig,
            overshoot_internal=ov_int, overshoot_verbal=ov_verb,
        )
        print(f"     verdict: {verdict}")
        check(verdict.startswith("H3"), "classifier reads construction as H3")

    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
