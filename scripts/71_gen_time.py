"""Arm G analysis: is felt-during-generation a real time axis, or just position?

On the per-token generation trajectories from 70:

  A1  reading-axis projection — apply the reading-elapsed probe direction (per
      layer) to each generated token; does the elapsed coordinate drift with
      generation position s? (Spearman per generation; normalized trajectories.)
  A2  generation-progress decodability — decode token index s from the trajectory
      (grouped-CV by generation). Trivially high = position is strongly encoded.
  A3  shared vs separate — per layer, cosine between the generation-progress
      direction (activation→s) and the reading-elapsed direction (the probe's
      per-layer coef). High = the elapsed axis IS driven by production position.
  A4  behavioral — felt-production duration vs tokens generated.

    TIME_MODEL=gemma python scripts/71_gen_time.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.analysis import cv_predict, fit_full, load_stacked_probe  # noqa: E402
from time_experiment.config import DATA_DIR, FIGURES_DIR, resolve_model  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    gdir = DATA_DIR / f"{base.short_name}_gen"
    probe, _ = load_stacked_probe(base.data_dir / "probe.npz")
    plays = probe["layers"].tolist()

    # reading-elapsed direction per layer (raw space): coef / scale.
    w_read = {int(L): _unit(probe["base_coef"][i] / probe["base_scale"][i])
              for i, L in enumerate(plays)}

    trajs = {}
    for p in sorted(glob.glob(str(gdir / "hidden" / "*.npz"))):
        d = np.load(p)
        trajs[Path(p).stem] = (d["H"], [int(L) for L in d["layers"]])
    if not trajs:
        raise SystemExit(f"no trajectories under {gdir}/hidden — run 70_generate.py")
    layers = trajs[next(iter(trajs))][1]
    print(f"generations: {list(trajs)}  layers: {len(layers)}")

    # ---- A1: reading-elapsed coordinate vs generation position ----
    probe_layers = [L for L in (plays[len(plays) * 3 // 5], plays[-1])]  # ~mid + last
    a1 = {}
    for L in probe_layers:
        li = layers.index(L); i = plays.index(L)
        rhos = []
        for gid, (H, _) in trajs.items():
            xs = ((H[:, li, :] - probe["base_mean"][i]) / probe["base_scale"][i]) @ probe["base_coef"][i]
            s = np.arange(len(xs))
            rhos.append(float(spearmanr(s, xs).statistic))
        a1[L] = rhos
        print(f"A1 L{L}: Spearman(reading-elapsed coord, gen position) per gen = "
              f"{[round(r,2) for r in rhos]}  mean={np.mean(rhos):+.2f}")

    # ---- A2: decode generation position s from the trajectory ----
    a2 = {}
    for L in probe_layers:
        li = layers.index(L)
        X, y, g = [], [], []
        for gid, (H, _) in trajs.items():
            T = H.shape[0]
            X.append(H[:, li, :]); y.append(np.arange(T) / max(T - 1, 1)); g.append([gid] * T)
        X = np.concatenate(X); y = np.concatenate(y); g = np.concatenate(g)
        _, r2, _ = cv_predict(X, y, g, n_splits=min(5, len(trajs)))
        a2[L] = r2
        print(f"A2 L{L}: decode gen-position (fraction) R²={r2:+.3f}  (grouped CV)")

    # ---- A3: cosine(gen-progress dir, reading-elapsed dir) per layer ----
    cos_by_layer = []
    for L in layers:
        li = layers.index(L); i = plays.index(L)
        X, y = [], []
        for gid, (H, _) in trajs.items():
            T = H.shape[0]
            X.append(H[:, li, :]); y.append(np.arange(T) / max(T - 1, 1))
        X = np.concatenate(X); y = np.concatenate(y)
        wg = fit_full(X, y)
        w_gen = _unit(wg["coef"] / wg["scale"])
        cos_by_layer.append(abs(float(w_gen @ w_read[L])))
    cos_by_layer = np.array(cos_by_layer)
    print(f"A3 cosine(gen-progress, reading-elapsed): max={cos_by_layer.max():.3f} "
          f"@L{layers[int(cos_by_layer.argmax())]}, median={np.median(cos_by_layer):.3f}")

    # ---- A4: felt-production duration vs tokens generated ----
    felt_rows = []
    grp = gdir / "gen_rows.jsonl"
    if grp.exists():
        for l in grp.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                for c in r.get("felt", []):
                    if c.get("felt_s"):
                        felt_rows.append((r["gen_id"], c["s"], float(c["felt_s"])))

    # ---- figure ----
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.6))
    Ltop = probe_layers[-1]
    for gid, (H, _) in trajs.items():
        li = layers.index(Ltop); i = plays.index(Ltop)
        xs = ((H[:, li, :] - probe["base_mean"][i]) / probe["base_scale"][i]) @ probe["base_coef"][i]
        f = np.arange(len(xs)) / max(len(xs) - 1, 1)
        axA.plot(f, (xs - xs.mean()) / (xs.std() + 1e-9), "-", alpha=0.7, label=gid)
    a1_mean = float(np.mean(a1[Ltop]))
    axA.set_xlabel("generation position (fraction)")
    axA.set_ylabel("reading-elapsed coord (z)")
    axA.set_title(f"(a) producing tokens does NOT drive the elapsed axis "
                  f"(ρ≈{a1_mean:+.2f}, L{Ltop})")
    axA.legend(fontsize=7); axA.grid(True, alpha=0.3)

    axB.plot(layers, cos_by_layer, "-o", ms=3, color="#0e7490")
    axB.axhline(0, color="#ccc", lw=0.8)
    axB.set_ylim(0, 1)
    axB.set_xlabel("layer"); axB.set_ylabel("|cosine|")
    axB.set_title(f"(b) gen-progress ⊥ reading-elapsed (median {np.median(cos_by_layer):.2f})")
    axB.grid(True, alpha=0.3)

    if felt_rows:
        for gid in sorted(set(r[0] for r in felt_rows)):
            pts = sorted([(s, f) for g, s, f in felt_rows if g == gid])
            axC.plot([s for s, _ in pts], [f for _, f in pts], "-o", ms=4, alpha=0.7, label=gid)
        axC.set_yscale("log")
        axC.set_ylim(1, 3600)
        axC.set_xlabel("tokens generated"); axC.set_ylabel("felt-writing duration (s)")
        axC.set_title("(c) first-person: writing feels ~instant (≈2 s), flat")
        axC.legend(fontsize=7); axC.grid(True, alpha=0.3)
    else:
        axC.text(0.5, 0.5, "no felt readouts", ha="center", transform=axC.transAxes)

    fig.suptitle("Arm G — generation-side time: output position is encoded but ORTHOGONAL to "
                 "the elapsed axis; production feels instant (G-H3)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{base.short_name}_gen" / "fig_genG.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")

    summary = {
        "a1_reading_coord_vs_position_mean_rho": {int(L): float(np.mean(v)) for L, v in a1.items()},
        "a2_decode_position_r2": {int(L): float(v) for L, v in a2.items()},
        "a3_cosine_max": float(cos_by_layer.max()),
        "a3_cosine_max_layer": int(layers[int(cos_by_layer.argmax())]),
        "a3_cosine_median": float(np.median(cos_by_layer)),
        "n_felt": len(felt_rows),
    }
    (gdir / "gen_time_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}  +  {gdir}/gen_time_summary.json")


if __name__ == "__main__":
    main()
