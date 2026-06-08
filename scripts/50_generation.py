"""T4 — generation-side time is a separate, flat axis (offline; reads the EV
slot probe + 11_gen_capture trajectories).

The reading line (T1-T3) probes time read from a finished context. This probes
time as *experienced during production*. The reading-elapsed axis is the
canonical EV-weighted all-layer slot probe — the axis that actually carries felt
time — so A3 asks the sharp question: does producing tokens move THAT axis?

  A1  drift — apply the EV probe to each generated token; does the elapsed
      coordinate drift with generation position s? (Spearman per generation,
      reported as mean ± 95% CI over all trajectories — needs --n-seeds>1)
  A2  position decodability — decode token-fraction from the trajectory per layer
      (grouped-CV by topic so seeds of one prompt never straddle the split).
  A3  shared vs separate — per layer, |cosine| between the generation-progress
      direction and the reading-elapsed direction, **EV-weighted across layers**
      by the probe's own weights. High = a shared time axis.
  A4  behavioral — felt-production duration vs tokens; topic spread reported
      against within-topic (seed) dispersion so the effect is signal, not n=1.
  OOD mid-generation tokens vs the scripted slot manifold (Mahalanobis ratio).
      The EV probe is fit at the elicitation slot; applied per generated token it
      is off-slot. Bounded ratio => A1's null is a genuine read; a blown-up ratio
      => the probe reads off-manifold noise and A1≈0 is an artifact. (T3's
      natural-slot reference is ≈6× bounded.)

    TIME_MODEL=gemma python scripts/50_generation.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.analysis import (  # noqa: E402
    StatesCache, apply_ev_probe, assemble, cv_predict, ev_layer_direction,
    fit_full, load_ev_probe, load_rows, maha_scorer, probe_direction,
)
from time_experiment.config import current_model  # noqa: E402

C_INT = "#0e7490"


def _topic(stem: str) -> str:
    """Trajectory stem 'pyproj__s2' -> topic 'pyproj' (back-compat: bare 'pyproj')."""
    return stem.split("__")[0]


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
    stems = list(trajs)
    topic_of = {s: _topic(s) for s in stems}
    topics = sorted(set(topic_of.values()))
    n_seeds = {t: sum(topic_of[s] == t for s in stems) for t in topics}
    tlayers = trajs[stems[0]][1]
    # align trajectory columns to the probe's layer order (both full, sorted).
    shared = [L for L in plays if L in tlayers]
    pidx = [plays.index(L) for L in shared]          # probe column per shared layer
    tidx = [tlayers.index(L) for L in shared]        # traj column per shared layer
    print(f"model: {M.short_name}  EV probe ({len(plays)} layers, {len(shared)} in trajectory)  "
          f"{len(stems)} trajectories over {len(topics)} topics (seeds/topic: {n_seeds})")

    def aligned(H):
        """(T, len(shared), D) reordered to the probe's shared-layer order."""
        return H[:, tidx, :]

    # A1 — EV-weighted reading coordinate vs generation position.
    # apply_ev_probe needs a probe restricted to the shared layers.
    sub = {k: (probe[k][pidx] if k in ("base_mean", "base_scale", "base_coef", "base_intercept")
               else probe[k]) for k in probe}
    sub["weights"] = w_ev[pidx] / max(w_ev[pidx].sum(), 1e-12)   # renormalize on shared
    sub["layers"] = np.array(shared)
    a1, a1_by_topic = [], {}
    for stem in stems:
        coord = apply_ev_probe(sub, aligned(trajs[stem][0]))
        r = float(spearmanr(np.arange(len(coord)), coord).statistic)
        a1.append(r); a1_by_topic.setdefault(topic_of[stem], []).append(r)
    a1 = np.array(a1)
    a1_ci = float(1.96 * a1.std(ddof=1) / np.sqrt(len(a1))) if len(a1) > 1 else float("nan")
    a1_topic_mean = {t: float(np.mean(v)) for t, v in a1_by_topic.items()}
    print(f"A1 drift (EV reading coord ~ position): mean={a1.mean():+.3f} ± {a1_ci:.3f} (95% CI, "
          f"n={len(a1)} traj)  per-topic {[round(a1_topic_mean[t], 2) for t in topics]}")

    # A2/A3 each fit one ridge per layer over an (n_tokens·n_traj, hidden) matrix
    # via RidgeCV's GCV-SVD — the run's real wall-clock cost. Three controls keep
    # it bounded at long context (it ballooned to ~90 min single-thread at 768 tok):
    #   • outer thread pool over the independent layers (LAPACK SVD drops the GIL);
    #   • single-threaded BLAS in the env (VECLIB/OMP/OPENBLAS=1) so the 8 workers
    #     parallelize cleanly instead of Accelerate over-subscribing 25 threads
    #     onto 6 perf cores — the actual bottleneck the first threaded run hit;
    #   • A23_STRIDE token subsample + float32: A2 (position decode) and A3 (gen-
    #     direction orthogonality) are overdetermined diagnostics, NOT inputs to
    #     the curvature/slope result (that reads the cheap spliced slots, no SVD),
    #     so a strided sample of thousands of tokens leaves R²/cos unmoved at 3 dp.
    nw = max(1, min(8, (os.cpu_count() or 4)))
    A23_STRIDE = 2

    # A2 — decode generation position (fraction) per trajectory layer, grouped by
    # topic (seeds of one prompt are correlated — never split them across folds).
    def _a2_layer(lj: int) -> float:
        X, y, g = [], [], []
        for stem in stems:
            H = trajs[stem][0]; T = H.shape[0]; idx = np.arange(0, T, A23_STRIDE)
            X.append(H[idx, lj, :]); y.append(idx / max(T - 1, 1)); g.append([topic_of[stem]] * len(idx))
        X = np.concatenate(X).astype(np.float32, copy=False); y = np.concatenate(y); g = np.concatenate(g)
        return cv_predict(X, y, g, n_splits=min(5, len(topics)))[1]
    with ThreadPoolExecutor(max_workers=nw) as ex:
        a2_profile = list(ex.map(_a2_layer, range(len(tlayers))))
    a2_max = max(a2_profile)
    print(f"A2 decode position: max R²={a2_max:+.3f} @L{tlayers[int(np.argmax(a2_profile))]}")

    # A3 — |cos(gen-progress, reading-elapsed)| per shared layer, EV-weighted.
    def _a3_layer(pjtj: tuple[int, int]) -> tuple[float, float]:
        pj, tj = pjtj
        X, y = [], []
        for stem in stems:
            H = trajs[stem][0]; T = H.shape[0]; idx = np.arange(0, T, A23_STRIDE)
            X.append(H[idx, tj, :]); y.append(idx / max(T - 1, 1))
        X = np.concatenate(X).astype(np.float32, copy=False); y = np.concatenate(y)
        w_gen = probe_direction(fit_full(X, y))
        w_read = ev_layer_direction(probe, pj)
        return abs(float(w_gen @ w_read)), float(w_ev[pj])
    with ThreadPoolExecutor(max_workers=nw) as ex:
        _a3 = list(ex.map(_a3_layer, zip(pidx, tidx)))
    cos_by_layer = np.array([c for c, _ in _a3]); wts = np.array([w for _, w in _a3])
    a3_weighted = float((cos_by_layer * wts).sum() / max(wts.sum(), 1e-12))
    a3_max = float(cos_by_layer.max())
    print(f"A3 |cos(gen-progress, reading-elapsed)|: EV-weighted {a3_weighted:.3f}, max {a3_max:.3f}  "
          f"(low = separate axes)")

    # OOD — are mid-generation tokens on the scripted slot manifold? Whiten on the
    # scripted timestamped/constant slots (restricted to the shared layers) and
    # score every trajectory token. Pooled median/max + drift-with-position: does
    # the read wander further off-manifold as the rollout grows?
    ood_med = ood_max = ood_drift = float("nan")
    scorer = None
    rows = load_rows(M.rows_path)
    ds = assemble(rows, StatesCache(M.hidden_dir), source="scripted",
                  rendering="timestamped", mode="constant")
    if len(ds["gt_log"]) >= 8:
        scorer = maha_scorer(ds["X3d"][:, pidx, :], shared)
    if scorer is None:
        print("OOD: saklas Mahalanobis unavailable (or no scripted slots) — skipping")
    else:
        pooled, drifts = [], []
        for stem in stems:
            r = scorer(aligned(trajs[stem][0]))
            if len(r):
                pooled.append(r)
                if len(r) >= 3:
                    drifts.append(float(spearmanr(np.arange(len(r)), r).statistic))
        if pooled:
            allr = np.concatenate(pooled)
            ood_med = float(np.median(allr)); ood_max = float(np.max(allr))
            ood_drift = float(np.mean(drifts)) if drifts else float("nan")
        print(f"OOD raw gen-token vs scripted slot: median {ood_med:.1f}×, max {ood_max:.1f}×  "
              f"drift-with-position ρ={ood_drift:+.2f}  (T3 natural-slot ref ≈6× bounded)")

    # A1' SPLICED (in-domain) — the canonical fix (11_gen_capture writes gen/sliced/):
    # at each stride, cut the generation, fork, and read the EV probe at the forked
    # elicitation slot — the SAME instrument T1/T2 use, so the read is on-manifold.
    # Question shifts from "does the act move the axis" (raw A1, flat) to "is the
    # model's own accumulated output counted by the length->time code" (expected to
    # GROW ~V*tokens). Plus: does self-context get the same V≈0.3 s/tok as scripted?
    a1s_mean = a1s_ci = sp_slope = sp_ood = float("nan")
    a1s_by_topic = {}
    spliced = {Path(p).stem: dict(np.load(p))
               for p in sorted(glob.glob(str(M.gen_dir / "sliced" / "*.npz")))}
    if not spliced:
        print("A1' SPLICED: no gen/sliced/*.npz yet — re-run 11_gen_capture for the in-domain read")
    else:
        slayers = [int(L) for L in spliced[next(iter(spliced))]["layers"]]
        if not all(L in slayers for L in shared):
            print("A1' SPLICED: sliced-slot layers don't cover the probe — skipping")
        else:
            ssidx = [slayers.index(L) for L in shared]   # sliced col per shared layer
            a1s_list, reads, ctxs = [], [], []
            for stem, d in spliced.items():
                read = apply_ev_probe(sub, d["slots"][:, ssidx, :])   # (S,) log-seconds
                reads.append(read); ctxs.append(d["ctx_tokens"].astype(float))
                if len(read) >= 3:
                    r = float(spearmanr(d["strides"], read).statistic)
                    a1s_list.append(r); a1s_by_topic.setdefault(_topic(stem), []).append(r)
            if a1s_list:
                a1s_mean = float(np.mean(a1s_list))
                a1s_ci = (float(1.96 * np.std(a1s_list, ddof=1) / np.sqrt(len(a1s_list)))
                          if len(a1s_list) > 1 else float("nan"))
            cc = np.concatenate(ctxs); sec = np.exp(np.concatenate(reads))
            if np.ptp(cc) > 0:
                sp_slope = float(np.polyfit(cc, sec, 1)[0])   # s/token, vs T1 V≈0.29
            if scorer is not None:
                oo = [scorer(d["slots"][:, ssidx, :]) for d in spliced.values()]
                oo = [x for x in oo if len(x)]
                if oo:
                    sp_ood = float(np.median(np.concatenate(oo)))
            print(f"A1' SPLICED (in-domain elicitation slot): ρ(elapsed,pos)={a1s_mean:+.3f} ± {a1s_ci:.3f}"
                  f"  slope≈{sp_slope:.3f} s/tok (T1 V≈0.29)  slot OOD {sp_ood:.1f}× (raw was {ood_med:.1f}×)")

    # A4 — felt-production duration vs tokens, with replicates: topic spread vs
    # within-topic (seed) dispersion. (topic, seed, s, felt_s) per checkpoint.
    felt_rows = []
    grp = M.gen_dir / "gen_rows.jsonl"
    if grp.exists():
        for line in grp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                for c in r.get("felt", []):
                    if c.get("felt_s"):
                        felt_rows.append((r["gen_id"], int(r.get("seed_idx", 0)),
                                          c["s"], float(c["felt_s"])))
    a4_rho = a4_spread = a4_within = float("nan")
    topic_med = {}
    if felt_rows:
        ss = np.array([s for _, _, s, _ in felt_rows], float)
        ff = np.array([f for _, _, _, f in felt_rows], float)
        a4_rho = float(spearmanr(ss, ff).statistic)
        smax = max(ss)
        at_max = {}
        for tp, _k, s, f in felt_rows:
            if s == smax:
                at_max.setdefault(tp, []).append(f)
        topic_med = {tp: float(np.median(v)) for tp, v in at_max.items()}
        if len(topic_med) > 1:
            a4_spread = max(topic_med.values()) / max(min(topic_med.values()), 1e-9)
        within = [max(v) / max(min(v), 1e-9) for v in at_max.values() if len(v) > 1]
        if within:
            a4_within = float(np.median(within))
    if np.isfinite(a4_within):
        verdict = "topic effect > seed noise" if a4_spread > a4_within else "NOT separable from seed noise"
        print(f"A4 felt~tokens ρ={a4_rho:+.2f}  topic spread={a4_spread:.1f}×  "
              f"within-topic (seed) dispersion={a4_within:.1f}×  ({verdict})")
    else:
        print(f"A4 felt~tokens ρ={a4_rho:+.2f}  topic spread={a4_spread:.1f}× (n=1/topic — re-run with --n-seeds>1)")

    # ---- figure: (a) probe drift + OOD caveat; (b) position decode; (c) felt ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))
    labeled = set()
    for stem in stems:
        coord = apply_ev_probe(sub, aligned(trajs[stem][0]))
        f = np.arange(len(coord)) / max(len(coord) - 1, 1)
        tp = topic_of[stem]
        axA.plot(f, (coord - coord.mean()) / (coord.std() + 1e-9), "-", alpha=0.5,
                 label=tp if tp not in labeled else None)
        labeled.add(tp)
    axA.set_xlabel("generation position (fraction)"); axA.set_ylabel("EV reading-elapsed coord (z)")
    sp_note = (f"in-domain splice: ρ={a1s_mean:+.2f}, slot {sp_ood:.0f}× (raw {ood_med:.0f}×)"
               if np.isfinite(a1s_mean) else f"raw gen tokens {ood_med:.0f}× off-slot (splice pending)")
    axA.set_title(f"(a) generative ACT doesn't move elapsed: raw ρ={a1.mean():+.2f}±{a1_ci:.2f}\n{sp_note}")
    axA.legend(fontsize=7, ncol=2); axA.grid(True, alpha=0.3)

    axB.plot(tlayers, a2_profile, "-o", ms=3, color=C_INT)
    axB.set_ylim(-0.1, 1); axB.set_xlabel("layer"); axB.set_ylabel("decode position R²")
    axB.set_title(f"(b) position IS encoded (max R²={a2_max:.2f}); EV-weighted |cos w/ elapsed|={a3_weighted:.2f}")
    axB.grid(True, alpha=0.3)

    if felt_rows:
        for tp in topics:
            by_s = {}
            for t2, _k, s, f in felt_rows:
                if t2 == tp:
                    by_s.setdefault(s, []).append(f)
            pts = sorted((s, float(np.mean(v))) for s, v in by_s.items())
            if pts:
                axC.plot([s for s, _ in pts], [f for _, f in pts], "-o", ms=4, alpha=0.8, label=tp)
        axC.set_yscale("log"); axC.set_xlabel("tokens generated"); axC.set_ylabel("felt-writing duration (s)")
        axC.set_title(f"(c) verbal (felt logits): grows w/ tokens (ρ={a4_rho:+.2f}), "
                      f"topic {a4_spread:.1f}× vs seed {a4_within:.1f}×")
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
        "n_trajectories": len(stems), "n_topics": len(topics), "seeds_per_topic": n_seeds,
        "a1_drift_mean_rho": float(a1.mean()), "a1_drift_ci95": a1_ci,
        "a1_per_traj": dict(zip(stems, [float(x) for x in a1])),
        "a1_by_topic_mean": a1_topic_mean,
        "a2_decode_position_r2_max": float(a2_max),
        "a3_cosine_ev_weighted": a3_weighted, "a3_cosine_max": a3_max,
        "ood_slot_ratio_median": ood_med, "ood_slot_ratio_max": ood_max,
        "ood_drift_with_position_rho": ood_drift,
        "a1_spliced_rho": a1s_mean, "a1_spliced_ci95": a1s_ci,
        "a1_spliced_by_topic": {t: float(np.mean(v)) for t, v in a1s_by_topic.items()},
        "spliced_slope_s_per_tok": sp_slope,
        "spliced_slot_ood_median": sp_ood,
        "a4_felt_vs_tokens_rho": a4_rho,
        "a4_topic_spread_ratio": a4_spread,
        "a4_within_topic_dispersion": a4_within,
        "a4_felt_by_topic_at_max": topic_med,
        "n_felt": len(felt_rows),
    }
    (M.gen_dir / "generation.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}  +  {M.gen_dir}/generation.json")


if __name__ == "__main__":
    main()
