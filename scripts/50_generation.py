"""T4 — generation-side time is a separate, flat axis (offline; reads the EV
slot probe + 11_gen_capture trajectories).

The reading line (T1-T3) probes time read from a finished context. This probes
time as *experienced during production*. The reading-elapsed axis is the
canonical EV-weighted all-layer slot probe — the axis that actually carries felt
time — so A3 asks the sharp question: does producing tokens move THAT axis?

  A1  drift — apply the EV probe to each generated token; does the elapsed
      coordinate drift with generation position s? (Spearman per generation)
  A2  position decodability — decode token-fraction from the trajectory per layer
      (grouped-CV). High = position is strongly encoded.
  A3  shared vs separate — per layer, |cosine| between the generation-progress
      direction and the reading-elapsed direction, **EV-weighted across layers**
      by the probe's own weights. High = a shared time axis.
  A4  behavioral — felt-production duration vs tokens generated.

    TIME_MODEL=gemma python scripts/50_generation.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.analysis import (  # noqa: E402
    apply_ev_probe, cv_predict, ev_layer_direction, fit_full, load_ev_probe,
    probe_direction,
)
from time_experiment.config import current_model  # noqa: E402

C_INT = "#0e7490"


def main() -> None:
    M = current_model()
    probe, _pmeta = load_ev_probe(M.probe_path)
    plays = [int(L) for L in probe["layers"]]
    w_ev = probe["weights"]

    trajs = {}
    for p in sorted(glob.glob(str(M.gen_dir / "hidden" / "*.npz"))):
        d = np.load(p)
        trajs[Path(p).stem] = (d["H"], [int(L) for L in d["layers"]])
    if not trajs:
        raise SystemExit(f"no trajectories under {M.gen_dir}/hidden — run 11_gen_capture.py")
    tlayers = trajs[next(iter(trajs))][1]
    # align trajectory columns to the probe's layer order (both full, sorted).
    shared = [L for L in plays if L in tlayers]
    pidx = [plays.index(L) for L in shared]          # probe column per shared layer
    tidx = [tlayers.index(L) for L in shared]        # traj column per shared layer
    print(f"model: {M.short_name}  EV probe ({len(plays)} layers, {len(shared)} in trajectory)  "
          f"generations: {list(trajs)}")

    def aligned(H):
        """(T, len(shared), D) reordered to the probe's shared-layer order."""
        return H[:, tidx, :]

    # A1 — EV-weighted reading coordinate vs generation position.
    # apply_ev_probe needs a probe restricted to the shared layers.
    sub = {k: (probe[k][pidx] if k in ("base_mean", "base_scale", "base_coef", "base_intercept")
               else probe[k]) for k in probe}
    sub["weights"] = w_ev[pidx] / max(w_ev[pidx].sum(), 1e-12)   # renormalize on shared
    sub["layers"] = np.array(shared)
    a1 = []
    for gid, (H, _) in trajs.items():
        coord = apply_ev_probe(sub, aligned(H))
        a1.append(float(spearmanr(np.arange(len(coord)), coord).statistic))
    print(f"A1 drift (EV reading coord ~ position): per gen {[round(r,2) for r in a1]}  mean={np.mean(a1):+.2f}")

    # A2 — decode generation position (fraction) per trajectory layer.
    a2_profile = []
    for lj in range(len(tlayers)):
        X, y, g = [], [], []
        for gid, (H, _) in trajs.items():
            T = H.shape[0]
            X.append(H[:, lj, :]); y.append(np.arange(T) / max(T - 1, 1)); g.append([gid] * T)
        X = np.concatenate(X); y = np.concatenate(y); g = np.concatenate(g)
        a2_profile.append(cv_predict(X, y, g, n_splits=min(5, len(trajs)))[1])
    a2_max = max(a2_profile)
    print(f"A2 decode position: max R²={a2_max:+.3f} @L{tlayers[int(np.argmax(a2_profile))]}")

    # A3 — |cos(gen-progress, reading-elapsed)| per shared layer, EV-weighted.
    cos_by_layer, wts = [], []
    for _L, pj, tj in zip(shared, pidx, tidx):
        X, y = [], []
        for gid, (H, _) in trajs.items():
            T = H.shape[0]
            X.append(H[:, tj, :]); y.append(np.arange(T) / max(T - 1, 1))
        X = np.concatenate(X); y = np.concatenate(y)
        w_gen = probe_direction(fit_full(X, y))
        w_read = ev_layer_direction(probe, pj)
        cos_by_layer.append(abs(float(w_gen @ w_read))); wts.append(float(w_ev[pj]))
    cos_by_layer = np.array(cos_by_layer); wts = np.array(wts)
    a3_weighted = float((cos_by_layer * wts).sum() / max(wts.sum(), 1e-12))
    a3_max = float(cos_by_layer.max())
    print(f"A3 |cos(gen-progress, reading-elapsed)|: EV-weighted {a3_weighted:.3f}, max {a3_max:.3f}  "
          f"(low = separate axes)")

    # A4 — felt-production duration vs tokens (+ topic variation).
    felt_rows = []
    grp = M.gen_dir / "gen_rows.jsonl"
    if grp.exists():
        for line in grp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                for c in r.get("felt", []):
                    if c.get("felt_s"):
                        felt_rows.append((r["gen_id"], c["s"], float(c["felt_s"])))
    a4_rho = float("nan"); a4_spread = float("nan"); a4_by_topic = {}
    if felt_rows:
        ss = np.array([s for _, s, _ in felt_rows], float)
        ff = np.array([f for _, _, f in felt_rows], float)
        a4_rho = float(spearmanr(ss, ff).statistic)
        smax = max(ss)
        a4_by_topic = {g: f for g, s, f in felt_rows if s == smax}
        if len(a4_by_topic) > 1:
            a4_spread = float(max(a4_by_topic.values()) / max(min(a4_by_topic.values()), 1e-9))
    print(f"A4 felt~tokens rho={a4_rho:+.2f}  topic spread={a4_spread:.1f}x  "
          f"(felt-writing grows with context, varies by topic)")

    # ---- figure: (a) probe drift = the probe-time view; (b) position decode;
    # (c) verbal felt logits (grows + topic-varies) ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))
    for gid, (H, _) in trajs.items():
        coord = apply_ev_probe(sub, aligned(H))
        f = np.arange(len(coord)) / max(len(coord) - 1, 1)
        axA.plot(f, (coord - coord.mean()) / (coord.std() + 1e-9), "-", alpha=0.7, label=gid)
    axA.set_xlabel("generation position (fraction)"); axA.set_ylabel("EV reading-elapsed coord (z)")
    axA.set_title(f"(a) probe: producing tokens does NOT drive the elapsed axis (ρ≈{np.mean(a1):+.2f})")
    axA.legend(fontsize=7); axA.grid(True, alpha=0.3)

    axB.plot(tlayers, a2_profile, "-o", ms=3, color=C_INT)
    axB.set_ylim(-0.1, 1); axB.set_xlabel("layer"); axB.set_ylabel("decode position R²")
    axB.set_title(f"(b) position IS encoded (max R²={a2_max:.2f}); EV-weighted |cos w/ elapsed|={a3_weighted:.2f}")
    axB.grid(True, alpha=0.3)

    if felt_rows:
        for gid in sorted({r[0] for r in felt_rows}):
            pts = sorted([(s, f) for g, s, f in felt_rows if g == gid])
            axC.plot([s for s, _ in pts], [f for _, f in pts], "-o", ms=4, alpha=0.8, label=gid)
        axC.set_yscale("log"); axC.set_xlabel("tokens generated"); axC.set_ylabel("felt-writing duration (s)")
        axC.set_title(f"(c) verbal (felt logits): grows w/ tokens (ρ={a4_rho:+.2f}), varies by topic ({a4_spread:.1f}×)")
        axC.legend(fontsize=7); axC.grid(True, alpha=0.3)
    else:
        axC.text(0.5, 0.5, "no felt readouts", ha="center", transform=axC.transAxes)

    fig.suptitle("T4 — generation-side time: the internal elapsed axis stays flat & orthogonal to output "
                 "position,\nwhile felt-writing time grows with tokens and varies by topic (seconds regime)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = M.figures_dir / "fig_t4_generation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")

    summary = {
        "probe_kind": "ev",
        "a1_drift_mean_rho": float(np.mean(a1)), "a1_per_gen": a1,
        "a2_decode_position_r2_max": float(a2_max),
        "a3_cosine_ev_weighted": a3_weighted, "a3_cosine_max": a3_max,
        "a4_felt_vs_tokens_rho": a4_rho, "a4_topic_spread_ratio": a4_spread,
        "a4_felt_by_topic_at_max": a4_by_topic,
        "n_felt": len(felt_rows),
    }
    (M.gen_dir / "generation.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}  +  {M.gen_dir}/generation.json")


if __name__ == "__main__":
    main()
