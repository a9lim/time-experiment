"""T5 — the causal arm: is the elapsed axis load-bearing?

T1-T4 are decoding: the probe *reads* an elapsed coordinate off the residual
stream, and the verbal estimate tracks it (or, in Qwen, doesn't). That is
correlational. This script intervenes — it adds an elapsed direction into the
residual stream during the elicitation forward and asks whether the model's
*stated* duration moves with it. If it does, specifically (a matched-norm random
direction does nothing), the representation is load-bearing.

Two things make the intervention valid rather than an off-manifold artifact, both
learned the hard way (an early all-layer log-s-scaled push knocked Qwen off the
manifold — its verbal responded to *generic* perturbation, random inflated it as
much as the real axis):

1. **Direction = contrastive diff-of-means** (``meandiff``), not the ridge probe
   normal. The ridge normal (``time``) is ~orthogonal to the data axis (cos≈0.2)
   and is causally inert — it moves its own linear readout but not the model. The
   diff-of-means is the literature-standard causal steering vector. Both are
   reported; the contrast is itself a result (the probe direction is NOT the
   causal direction).

2. **Dose = a fraction of the residual-stream norm** (``normfrac``, the default),
   not a log-seconds target. ``delta_l = dose * |h_l| * unit_l`` keeps the push a
   controlled fraction of the activation at every layer — matched across
   directions (same ``|h_l|``) and **across models** (so the same dose is the same
   off-manifold risk on a 3B and a 27B), on-manifold for small dose. The legacy
   ``logs`` mode (dose = intended Δlog-s of the layer's read) is kept for the
   single-layer locus site where it stays on-manifold and is interpretable.

Sites (which layers to steer):
  - ``locus``: the single representational-locus layer — cleanest/most on-manifold.
  - ``band``: the ``--band`` highest-EV-weight layers — the effective middle ground.
  - ``all``: every EV-weight>1e-2 layer — strongest, compounds, off-manifold risk.

Three readouts per (context, dose, direction), all under steering:
  - **verbal** (the causal test): the soft-distribution point estimate from the
    slot logits — the model's own W_U readout, a *different* readout than the probe.
  - **probe re-read** (the manipulation check): the EV probe on the steered slot —
    confirms the coordinate moved, and reports the *achieved* dose per model.
  - **entropy**: spread of the spoken estimate.

The Qwen adjudication: on a faithful model the verbal moves with meandiff
(specifically); on Qwen the probe re-read moves but the verbal should not — the
causal form of its confabulation. Always read the RANDOM control first: if random
is not ~flat, the dose is too large (off-manifold) and the run is uninformative —
lower ``--doses`` or use a smaller site.

Memory: in-place add_, so the per-forward peak ≈ a normal 10_capture run. Validate
on TIME_MODEL=llama32_3b first; watch Activity Monitor on the first 31B run.

    TIME_MODEL=llama32_3b python scripts/60_steer.py --site locus --contexts 6  # smoke
    TIME_MODEL=gemma      python scripts/60_steer.py --site band --band 6        # pilot
    TIME_MODEL=qwen       python scripts/60_steer.py --site band --band 6        # the contrast
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from saklas import SaklasSession  # noqa: E402

from time_experiment.analysis import apply_ev_probe, load_ev_probe  # noqa: E402
from time_experiment.capture import (  # noqa: E402
    capture_slot, content_position, dist_entropy, elicit_render, release_memory,
    verbal_distribution,
)
from time_experiment.config import (  # noqa: E402
    CONSTANT_PHRASE, ELICIT_PROMPT, MAX_CONTEXT_TOKENS, TRANSCRIPTS_DIR, current_model,
)
from time_experiment.transcripts import build_messages, load_corpus  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False


# --- steering axes (units + residual norm, from probe + capture data) -----
def build_axes(M, probe: dict, *, frac: float = 1 / 3, rand_seed: int = 1) -> dict:
    """Per-layer UNIT steering directions + the residual-norm reference.

    Loads the captured timestamped/constant slot activations (the clean
    clock-elapsed data the probe was fit on) once and returns, per layer:

      - ``meandiff`` unit: ``v_l / |v_l|`` where ``v_l = mean(X|hi) - mean(X|lo)``
        over the top/bottom ``frac`` terciles of log-elapsed — the contrastive
        data axis (the causal direction).
      - ``time`` unit: ``g_l / |g_l|`` where ``g_l = coef_l/scale_l`` — the ridge
        probe normal (cautionary control; usually off the data axis).
      - ``random`` unit: a fixed random unit vector (matched-norm control; in
        normfrac mode it shares the same ``|h_l|`` scaling as the others).
      - ``hnorm``: median ``|h_l|`` over the capture rows — the dose reference.
      - ``logs_step``: ``v_l / Δlog`` (meandiff) and ``g_l/|g_l|^2`` (time) for the
        legacy logs dose mode; ``cos`` of meandiff vs ridge per layer.
    """
    from time_experiment.analysis import StatesCache, assemble, load_rows
    rows = load_rows(M.rows_path)
    cache = StatesCache(M.hidden_dir)
    d = assemble(rows, cache, source="scripted", rendering="timestamped", mode="constant")
    X3d, y = d["X3d"], d["gt_log"]                                  # (N, L, D), (N,)
    layers = [int(L) for L in probe["layers"]]
    if X3d.shape[1] != len(layers):
        raise SystemExit("build_axes: capture layer count != probe layer count")
    g = probe["base_coef"] / probe["base_scale"]                   # ridge read-gradient (L, D)
    gnorm = np.linalg.norm(g, axis=1)
    lo_t, hi_t = np.quantile(y, [frac, 1 - frac])
    lo, hi = y <= lo_t, y >= hi_t
    dlog = float(y[hi].mean() - y[lo].mean())

    rng = np.random.default_rng(rand_seed)
    unit = {"meandiff": {}, "time": {}, "random": {}}
    logs_step = {"meandiff": {}, "time": {}, "random": {}}
    hnorm, cos = {}, {}
    for li, L in enumerate(layers):
        v = X3d[hi, li, :].mean(0) - X3d[lo, li, :].mean(0)        # (D,)
        nv, gn = max(np.linalg.norm(v), 1e-12), max(gnorm[li], 1e-12)
        unit["meandiff"][L] = v / nv
        unit["time"][L] = g[li] / gn
        r = rng.standard_normal(g.shape[1]); r /= max(np.linalg.norm(r), 1e-12)
        unit["random"][L] = r
        hnorm[L] = float(np.median(np.linalg.norm(X3d[:, li, :], axis=1)))
        cos[L] = float(v @ g[li] / (nv * gn))
        # legacy logs-mode steps (read-shift == dose per layer)
        logs_step["time"][L] = g[li] / (gn ** 2)
        logs_step["meandiff"][L] = v / max(dlog, 1e-6)
        logs_step["random"][L] = r * float(np.linalg.norm(v / max(dlog, 1e-6)))
    return {"unit": unit, "hnorm": hnorm, "logs_step": logs_step,
            "cos": cos, "dlog": dlog, "layers": layers}


def deltas_for(axes: dict, direction: str, dose_mode: str, layers_to_steer) -> dict:
    """The per-layer steering vector at dose=1 for a (direction, mode), so the hook
    adds ``dose * delta_l``. normfrac: ``|h_l| * unit_l`` (a unit fraction of the
    residual norm). logs: the read-shift-calibrated step."""
    out = {}
    for L in layers_to_steer:
        if dose_mode == "normfrac":
            out[L] = axes["hnorm"][L] * axes["unit"][direction][L]
        else:  # logs
            out[L] = axes["logs_step"][direction][L]
    return out


@contextlib.contextmanager
def steer(session, delta_by_layer: dict, dose: float):
    """Register additive forward hooks (dose * delta_l) on the given layers for the
    block. Hooks fire BEFORE any capture hook registered later, so a slot capture
    taken inside sees the steered activations. dose=0 is a true no-op."""
    handles = []
    if dose != 0.0:
        dev = session.device
        for L, delta in delta_by_layer.items():
            vec = torch.tensor(dose * delta, device=dev, dtype=torch.float32)

            def mk(v):
                def hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    hs.add_(v.to(hs.dtype))
                    return out
                return hook
            handles.append(session.layers[int(L)].register_forward_hook(mk(vec)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --- one steered read -----------------------------------------------------
def steered_reads(session, probe, msgs_q, rendered_constant, locus_li):
    """Verbal point/entropy + EV probe re-read + locus-layer read, under whatever
    steering hooks are currently registered."""
    v_sec, v_dist = verbal_distribution(session, msgs_q)
    states, _ = capture_slot(session, rendered_constant)
    X3d = np.stack([states[int(L)] for L in probe["layers"]])[None]   # (1, L, D)
    probe_read = float(apply_ev_probe(probe, X3d)[0])
    Xs = (states[int(probe["layers"][locus_li])] - probe["base_mean"][locus_li]) \
        / probe["base_scale"][locus_li]
    locus_read = float(Xs @ probe["base_coef"][locus_li] + probe["base_intercept"][locus_li])
    return {
        "verbal_log": math.log(max(v_sec, 1e-6)),
        "verbal_entropy": dist_entropy(v_dist),
        "probe_read_log": probe_read,
        "locus_read_log": locus_read,
    }


# --- context selection ----------------------------------------------------
def select_contexts(session, corpus, *, rendering, n_contexts, cap):
    """Elicitation contexts (msgs_q ending in ELICIT_PROMPT) for assistant turns,
    spread across conversations and turn depth. Skips over-cap contexts."""
    with_ts = rendering == "timestamped"
    pool = []
    for tx in corpus:
        for turn in tx.turns:
            if turn.role != "assistant":
                continue
            msgs_q = build_messages(tx, turn.idx, with_timestamps=with_ts,
                                    extra_user=ELICIT_PROMPT)
            rendered = elicit_render(session, msgs_q, CONSTANT_PHRASE)
            _, ntok = content_position(session, rendered)
            if ntok > cap:
                continue
            pool.append({"conv_id": tx.id, "turn_idx": turn.idx, "schedule": tx.schedule,
                         "gt": turn.elapsed_s, "tokens": ntok, "msgs_q": msgs_q,
                         "rendered_constant": rendered})
    pool.sort(key=lambda c: (c["turn_idx"], c["conv_id"]))
    if n_contexts and len(pool) > n_contexts:
        idx = np.linspace(0, len(pool) - 1, n_contexts).round().astype(int)
        pool = [pool[i] for i in dict.fromkeys(idx)]
    return pool


# --- dose-response statistics ---------------------------------------------
def grouped_boot_slope(triples, *, n_boot=2000, seed=0):
    """OLS slope of y on dose + a conversation-grouped bootstrap 95% CI."""
    if len(triples) < 3:
        return float("nan"), (float("nan"), float("nan"))
    a = np.array([t[1] for t in triples], float)
    y = np.array([t[2] for t in triples], float)
    point = float(np.polyfit(a, y, 1)[0])
    by_conv = defaultdict(list)
    for c, al, yv in triples:
        by_conv[c].append((al, yv))
    convs = list(by_conv)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(convs), len(convs), replace=True)
        A, Y = [], []
        for j in pick:
            for al, yv in by_conv[convs[j]]:
                A.append(al); Y.append(yv)
        if len(set(A)) >= 2:
            boots.append(np.polyfit(A, Y, 1)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (math.nan, math.nan))
    return point, (float(lo), float(hi))


def monotonicity(triples):
    """Spearman of y vs dose (clean dose-response is monotone; ~0 or sign-flipping
    means off-manifold). A separate flag from the slope — a big slope over a
    non-monotone curve is the artifact signature."""
    from scipy.stats import spearmanr
    if len(triples) < 4:
        return float("nan")
    a = [t[1] for t in triples]; y = [t[2] for t in triples]
    return float(spearmanr(a, y).statistic)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for site in sorted({r["site"] for r in rows}):
        for direction in sorted({r["direction"] for r in rows}):
            sel = [r for r in rows if r["site"] == site and r["direction"] == direction]
            if not sel:
                continue
            block = {}
            for key in ("verbal_log", "probe_read_log", "locus_read_log", "verbal_entropy"):
                trip = [(r["conv_id"], r["dose"], r[key]) for r in sel
                        if r[key] is not None and math.isfinite(r[key])]
                pt, (lo, hi) = grouped_boot_slope(trip)
                block[key] = {"slope": pt, "ci": [lo, hi], "rho": monotonicity(trip)}
            out[f"{site}/{direction}"] = block
    return out


# --- figure ---------------------------------------------------------------
def make_figure(rows, summary, M, *, site, dose_mode):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (skipping figure: {e})")
        return
    sel = [r for r in rows if r["site"] == site]
    if not sel:
        return
    dirs = sorted({r["direction"] for r in sel})
    panels = [("verbal_log", "verbal estimate  log(s)  — the causal test"),
              ("probe_read_log", "probe re-read  log(s)  — manipulation check")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.4), sharex=True)
    colors = {"meandiff": "#c1432f", "time": "#2563eb", "random": "#6b7280"}
    xlabel = ("steering dose  (|δ|/|h| per layer)" if dose_mode == "normfrac"
              else "steering α  (intended Δlog-s per layer)")
    for ax, (key, title) in zip(axes, panels):
        for d in dirs:
            pts = defaultdict(list)
            for r in sel:
                if r["direction"] == d and r[key] is not None and math.isfinite(r[key]):
                    pts[r["dose"]].append(r[key])
            xs = sorted(pts)
            mean = [float(np.mean(pts[x])) for x in xs]
            se = [float(np.std(pts[x]) / math.sqrt(len(pts[x]))) if len(pts[x]) > 1 else 0.0
                  for x in xs]
            ax.errorbar(xs, mean, yerr=se, marker="o", capsize=3,
                        color=colors.get(d, None), label=d)
            blk = summary.get(f"{site}/{d}", {}).get(key, {})
            if blk.get("slope") is not None and math.isfinite(blk["slope"]):
                ax.annotate(f"{d}: β={blk['slope']:+.2f} ρ={blk.get('rho', float('nan')):+.2f}",
                            xy=(0.04, 0.93 - 0.08 * dirs.index(d)),
                            xycoords="axes fraction", color=colors.get(d, "k"), fontsize=8.5)
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("log seconds")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"{M.short_name} — causal steering of the elapsed axis "
                 f"({site} site, {dose_mode})", fontsize=12)
    fig.tight_layout()
    M.figures_dir.mkdir(parents=True, exist_ok=True)
    path = M.figures_dir / f"steer_{site}_{dose_mode}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot")
    ap.add_argument("--rendering", default="untimestamped",
                    choices=["untimestamped", "timestamped"])
    ap.add_argument("--site", default="band", choices=["locus", "band", "all"])
    ap.add_argument("--band", type=int, default=6,
                    help="site=band: number of highest-EV-weight layers to steer")
    ap.add_argument("--dose-mode", default="normfrac", choices=["normfrac", "logs"])
    ap.add_argument("--doses", default="",
                    help="comma list of doses; default depends on dose-mode "
                         "(normfrac: -0.3..0.3; logs: -2..2)")
    ap.add_argument("--directions", default="meandiff,time,random")
    ap.add_argument("--contexts", type=int, default=20)
    ap.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    ap.add_argument("--weight-floor", type=float, default=1e-2)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if args.doses.strip():
        doses = [float(a) for a in args.doses.split(",") if a.strip() != ""]
    else:
        doses = [-0.3, -0.15, 0.0, 0.15, 0.3] if args.dose_mode == "normfrac" \
            else [-2.0, -1.0, 0.0, 1.0, 2.0]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    M = current_model()
    if not M.probe_path.exists():
        raise SystemExit(f"no probe at {M.probe_path}; run 20_probe.py first")
    probe, meta = load_ev_probe(M.probe_path)
    layers = [int(L) for L in probe["layers"]]
    weights = probe["weights"]
    locus_layer = int(meta.get("locus_layer", layers[int(np.argmax(weights))]))
    locus_li = layers.index(locus_layer)

    axes = build_axes(M, probe)
    cos_locus = axes["cos"][locus_layer]
    cos_all = float(np.mean([axes["cos"][L] for L in layers]))
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"probe: EV all-layer, locus L{locus_layer}; "
          f"cos(meandiff,ridge) locus={cos_locus:+.2f} mean={cos_all:+.2f} "
          f"(low => ridge normal is off the data axis)")

    order = list(np.argsort(weights)[::-1])
    if args.site == "locus":
        steer_layers = [locus_layer]
    elif args.site == "band":
        steer_layers = sorted(layers[li] for li in order[:args.band])
    else:
        steer_layers = [layers[li] for li in range(len(layers)) if weights[li] > args.weight_floor]
    print(f"site={args.site} -> steering {len(steer_layers)} layer(s): {steer_layers}")
    print(f"dose-mode={args.dose_mode}  doses={doses}  directions={directions}")

    delta1 = {d: deltas_for(axes, d, args.dose_mode, steer_layers) for d in directions}

    corpus_path = TRANSCRIPTS_DIR / f"{args.corpus}.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"no corpus at {corpus_path}; run 00_corpus.py first")
    corpus = load_corpus(corpus_path)

    print(f"loading {M.model_id} ...")
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)

        contexts = select_contexts(session, corpus, rendering=args.rendering,
                                   n_contexts=args.contexts, cap=args.max_context_tokens)
        n_cond = len(contexts) * len(directions) * len(doses)
        print(f"contexts: {len(contexts)}  -> {n_cond} conditions\n")

        rows: list[dict] = []
        done = 0
        for ci, ctx in enumerate(contexts):
            for direction in directions:
                for dose in doses:
                    with steer(session, delta1[direction], dose):
                        reads = steered_reads(session, probe, ctx["msgs_q"],
                                              ctx["rendered_constant"], locus_li)
                    release_memory(session.device)
                    rows.append({
                        "conv_id": ctx["conv_id"], "turn_idx": ctx["turn_idx"],
                        "schedule": ctx["schedule"], "tokens": ctx["tokens"],
                        "gt_elapsed_s": ctx["gt"], "rendering": args.rendering,
                        "site": args.site, "dose_mode": args.dose_mode,
                        "direction": direction, "dose": dose, **reads,
                    })
                    done += 1
            if (ci + 1) % 4 == 0 or ci == len(contexts) - 1:
                print(f"  [{done}/{n_cond}] ctx {ci + 1}/{len(contexts)} "
                      f"(conv {ctx['conv_id']} t{ctx['turn_idx']})")

        summary = summarize(rows)

    out_dir = M.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.site}_{args.dose_mode}"
    (out_dir / f"steer_{tag}_rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    payload = {"model": M.short_name, "rendering": args.rendering, "site": args.site,
               "dose_mode": args.dose_mode, "doses": doses, "directions": directions,
               "locus_layer": locus_layer, "steer_layers": steer_layers,
               "cos_meandiff_ridge": {"locus": cos_locus, "mean": cos_all},
               "n_contexts": len(contexts), "summary": summary}
    (out_dir / f"steer_{tag}.json").write_text(json.dumps(payload, indent=2))

    print("\n=== dose-response (slope [95% CI], ρ=monotonicity; READ RANDOM FIRST) ===")
    for k in sorted(summary):
        b = summary[k]
        def fmt(key):
            s = b[key]["slope"]; lo, hi = b[key]["ci"]; rho = b[key]["rho"]
            return f"{s:+.3f}[{lo:+.2f},{hi:+.2f}]ρ{rho:+.2f}"
        print(f"  {k:>16}:  verbal {fmt('verbal_log')}   probe {fmt('probe_read_log')}")
    print("\n  valid iff RANDOM verbal slope ~0 AND |ρ|<~0.5 (else dose too large,")
    print("  off-manifold — lower --doses or shrink the site). Then: MEANDIFF verbal")
    print("  slope>0 monotone = load-bearing; TIME(ridge) ~0 w/ probe>0 = inert normal;")
    print("  meandiff probe>0 but verbal~0 = the causal form of Qwen's confabulation.")
    print(f"\nsaved steer_{tag}_rows.jsonl + steer_{tag}.json -> {out_dir}/")

    if not args.no_fig:
        make_figure(rows, summary, M, site=args.site, dose_mode=args.dose_mode)


if __name__ == "__main__":
    main()
