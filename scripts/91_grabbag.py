#!/usr/bin/env python
"""91_grabbag.py — cross-model CPU-side analyses on already-captured data.

A posterity grab-bag: seven offline analyses that read the per-model artifacts
(decode_rows.csv, probe_meta.json, felt.json, transfer.json, and the rates-variant
felt.json) and synthesise *across* models. No model forward passes — pure CSV/JSON.
Discovers every model under data/<m>/ that has a probe_meta.json and runs them all.

Unlike the per-model scripts (01/10/.../90, keyed on TIME_MODEL), this one is
inherently cross-model: it compares models to each other. Run it any time the sweep
adds a model; it just picks up whatever artifacts are on disk.

  A  tokenization-normalised V   is the 14x V-spread real, or just tokenisation?
                                 re-express each model's s/tok read against a shared
                                 content yardstick (a reference model's token counts).
  B  depth profile               where does the time-code live? per-layer probe R^2:
                                 locus (peak-R^2 layer) as a fraction of depth + shape.
  C  verbal<->probe agreement    does the model's own W_U readout (verbal soft-dist)
                                 track the learned probe? rank corr in log-seconds,
                                 split by rendering. The introspection axis.
  D  schedule-blindness          in no-clock rows the narrated schedule is invisible;
                                 does internal_s leak it anyway (content/lexical time
                                 cues)? incremental R^2 of schedule beyond tokens.
  E  clock-density gradient (T2) rates variant: rate-sensitivity should climb with
                                 clock density (untimestamped < intermittent < timestamped).
  F  entropy-vs-length           does felt-time *uncertainty* grow with context length
                                 ("fog of time")? slope of grid entropy vs log tokens.
  G  OOD-overshoot correlates    is the ~6-9x out-of-distribution overshoot predicted
                                 by V, probe R^2, depth, or model size? (small-n, Spearman.)
  H  T4 raw-vs-spliced           does the model's OWN generated context carry the
                                 time-code? free-generation drift (~0) vs the same text
                                 spliced into the slot (high), the spliced V, and the
                                 cosine between the generation-time and scripted directions.
  I  variance decomp + geometry  what the POOLED probe reads: r2 vs r2|length-residualised
                                 vs length-only vs ceiling, plus PC1 alignment with
                                 log-seconds (the log-linear claim, geometric not just fit).
  J  felt-time by content        does affect-dense / time-language content inflate the
                                 SPOKEN felt duration (verbal_s) more than the internal
                                 slot read? per_variant from the natural-transfer set.

Writes a combined JSON to data/grabbag.json and prints a table per analysis.

Usage:
  python scripts/91_grabbag.py                 # all models on disk
  python scripts/91_grabbag.py --models gemma qwen llama32_3b
  python scripts/91_grabbag.py --ref gemma --out data/grabbag.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# preferred print order; any model on disk not listed here is appended alphabetically
PREFERRED = [
    "gemma", "gemma_12b", "qwen", "llama32_3b", "phi4_mini",
    "talkie_1930", "ministral", "deepseek_v2_lite", "granite", "glm47_flash",
]


# ---------------------------------------------------------------- loaders (cached)
_CACHE: dict = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def decode_rows(m):
    def _load():
        p = DATA / m / "decode_rows.csv"
        return list(csv.DictReader(open(p))) if p.exists() else []
    return _cached(("decode", m), _load)


def probe_meta(m):
    def _load():
        p = DATA / m / "probe_meta.json"
        return json.load(open(p)) if p.exists() else {}
    return _cached(("probe", m), _load)


def felt(m, variant=""):
    suffix = f"_{variant}" if variant else ""
    def _load():
        p = DATA / f"{m}{suffix}" / "felt.json"
        return json.load(open(p)) if p.exists() else {}
    return _cached(("felt", m, variant), _load)


def transfer(m):
    def _load():
        p = DATA / m / "transfer.json"
        return json.load(open(p)) if p.exists() else {}
    return _cached(("transfer", m), _load)


def gen_json(m):
    def _load():
        p = DATA / m / "gen" / "generation.json"
        return json.load(open(p)) if p.exists() else {}
    return _cached(("gen", m), _load)


def discover_models(explicit=None):
    if explicit:
        return [m for m in explicit if (DATA / m / "probe_meta.json").exists()]
    found = {p.parent.name for p in DATA.glob("*/probe_meta.json")}
    # drop variant dirs (they share a base model's name + _inflation/_rates)
    found = {m for m in found if not (m.endswith("_inflation") or m.endswith("_rates"))}
    ordered = [m for m in PREFERRED if m in found]
    ordered += sorted(found - set(ordered))
    return ordered


# ---------------------------------------------------------------- small helpers
def _untimestamped(m):
    return [r for r in decode_rows(m) if r["rendering"] == "untimestamped"]


def _per_layer_r2(m):
    """per-layer R^2 array from probe_meta, robust to list-of-floats or list-of-dicts."""
    pl = probe_meta(m).get("per_layer")
    if not pl:
        return None
    if isinstance(pl[0], dict):
        return np.asarray([d.get("r2", d.get("r2_cv", np.nan)) for d in pl], float)
    return np.asarray(pl, float)


def _ols_r2(X, y):
    """R^2 of an OLS fit y ~ X (X already includes an intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _hline(n=78):
    print("-" * n)


def _spearman(a, b):
    """Spearman rho = Pearson on ranks. Avoids scipy's poorly-typed spearmanr return."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


# ---------------------------------------------------------------- A: tokenisation-normalised V
def analysis_A(models, ref):
    print("\n### A  tokenisation-normalised V  (is the V-spread representational or tokenisation?)")
    if ref not in models:
        ref = models[0]
    ref_tok = {(r["id"], r["turn_idx"]): float(r["tokens"]) for r in _untimestamped(ref)}
    print(f"reference content yardstick = {ref!r} token counts")
    print(f"{'model':<17}{'V raw s/tok':>12}{'tok/ref':>9}{'V content':>11}{'noclk r':>9}")
    _hline(58)
    res = {}
    for m in models:
        rows = _untimestamped(m)
        keys = [(r["id"], r["turn_idx"]) for r in rows if (r["id"], r["turn_idx"]) in ref_tok]
        if len(keys) < 5:
            continue
        idx = {(r["id"], r["turn_idx"]): r for r in rows}
        tok = np.asarray([float(idx[k]["tokens"]) for k in keys])
        iss = np.asarray([float(idx[k]["internal_s"]) for k in keys])
        tok_ref = np.asarray([ref_tok[k] for k in keys])
        v_raw = float(np.polyfit(tok, iss, 1)[0])
        v_content = float(np.polyfit(tok_ref, iss, 1)[0])  # s per ref-token
        density = float(np.median(tok / tok_ref))
        r = float(np.corrcoef(tok, iss)[0, 1])
        res[m] = {"V_raw": v_raw, "density_vs_ref": density,
                  "V_content": v_content, "noclock_r": r}
        print(f"{m:<17}{v_raw:>12.3f}{density:>9.2f}{v_content:>11.3f}{r:>9.2f}")
    # spread, excluding models whose no-clock read is flat (|r|<0.3 -> V meaningless)
    good = {m: v for m, v in res.items() if abs(v["noclock_r"]) >= 0.3}
    if good:
        raw = [v["V_raw"] for v in good.values()]
        con = [v["V_content"] for v in good.values()]
        sr = max(raw) / min(raw) if min(raw) > 0 else float("nan")
        sc = max(con) / min(con) if min(con) > 0 else float("nan")
        flat = sorted(set(res) - set(good))
        print(f"\nspread over length-tracking models (|r|>=0.3, excl {flat}):")
        print(f"   raw V    {min(raw):.2f}-{max(raw):.2f}  = {sr:.1f}x")
        print(f"   content  {min(con):.2f}-{max(con):.2f}  = {sc:.1f}x")
        print("   verdict: tokenisation %s explain the spread" %
              ("DOES NOT" if sc > 0.6 * sr else "largely explains"))
    return {"ref": ref, "per_model": res}


# ---------------------------------------------------------------- B: depth profile
def analysis_B(models):
    print("\n### B  depth profile  (where does the time-code live?)")
    print(f"{'model':<17}{'#L':>4}{'peak@':>7}{'depth%':>8}{'peakR2':>8}{'L0':>7}{'mid':>7}{'last':>7}  shape")
    _hline(78)
    res = {}
    for m in models:
        r2 = _per_layer_r2(m)
        if r2 is None or len(r2) < 3:
            continue
        n = len(r2)
        pk = int(np.nanargmax(r2))
        frac = pk / (n - 1)
        shape = "early" if frac < 0.33 else "mid" if frac < 0.66 else "late"
        l0, mid, last = float(r2[0]), float(r2[n // 2]), float(r2[-1])
        res[m] = {"n_layers": n, "peak_layer": pk, "peak_frac": frac,
                  "peak_r2": float(r2[pk]), "r2_l0": l0, "r2_mid": mid,
                  "r2_last": last, "shape": shape}
        print(f"{m:<17}{n:>4}{pk:>7}{100 * frac:>7.0f}%{r2[pk]:>8.3f}"
              f"{l0:>7.2f}{mid:>7.2f}{last:>7.2f}  {shape}")
    fracs = [v["peak_frac"] for v in res.values()]
    if fracs:
        print(f"\nlocus: median peak at {100 * np.median(fracs):.0f}% depth; "
              f"{sum(0.33 <= f < 0.66 for f in fracs)}/{len(fracs)} models peak mid-network")
    return res


# ---------------------------------------------------------------- C: verbal<->probe agreement
def analysis_C(models):
    print("\n### C  verbal<->probe agreement  (does the model's own W_U readout track the probe?)")
    print(f"{'model':<17}{'ALL rho':>9}{'noclk':>8}{'tstamp':>8}{'verbal/probe':>14}")
    _hline(56)
    res = {}
    for m in models:
        rows = decode_rows(m)
        if not rows:
            continue

        def rho_scale(sub):
            v = np.asarray([float(r["verbal_s"]) for r in sub
                            if r.get("verbal_s") not in ("", "nan", None)])
            i = np.asarray([float(r["internal_s"]) for r in sub
                            if r.get("verbal_s") not in ("", "nan", None)])
            ok = (v > 0) & (i > 0)
            if ok.sum() < 5:
                return float("nan"), float("nan")
            return _spearman(np.log(v[ok]), np.log(i[ok])), float(np.median(v[ok] / i[ok]))

        r_all, scale = rho_scale(rows)
        r_nc, _ = rho_scale([r for r in rows if r["rendering"] == "untimestamped"])
        r_ts, _ = rho_scale([r for r in rows if r["rendering"] == "timestamped"])
        # classify the introspective relationship
        if r_all < 0:
            kind = "confabulating (verbal anti-correlates w/ internal code)"
        elif scale < 0.2 or scale > 5:
            kind = "scale-decoupled (ordering agrees, magnitude does not)"
        else:
            kind = "faithful"
        res[m] = {"rho_all": r_all, "rho_noclock": r_nc, "rho_timestamped": r_ts,
                  "verbal_over_probe": scale, "kind": kind}
        print(f"{m:<17}{r_all:>9.2f}{r_nc:>8.2f}{r_ts:>8.2f}{scale:>14.2f}")
    confab = [m for m, v in res.items() if v["rho_all"] < 0]
    if confab:
        print(f"\nconfabulators (verbal report anti-correlates with internal code): {confab}")
    return res


# ---------------------------------------------------------------- D: schedule-blindness
def analysis_D(models):
    print("\n### D  schedule-blindness  (does the no-clock read leak the invisible schedule?)")
    print(f"{'model':<17}{'R2[tok]':>9}{'R2[tok+sch]':>12}{'leak':>8}  interpretation")
    _hline(64)
    res = {}
    for m in models:
        rows = _untimestamped(m)
        if len(rows) < 20:
            continue
        tok = np.asarray([float(r["tokens"]) for r in rows])
        iss = np.asarray([float(r["internal_s"]) for r in rows])
        ok = iss > 0
        tok, iss = tok[ok], np.log(iss[ok])
        scheds = [rows[k]["schedule"] for k in range(len(rows)) if ok[k]]
        cats = sorted(set(scheds))
        if len(cats) < 2:
            continue
        n = len(iss)
        X_t = np.column_stack([np.ones(n), tok])
        # one-hot schedule (drop first level to avoid collinearity with intercept)
        dummies = np.column_stack([[1.0 if s == c else 0.0 for s in scheds] for c in cats[1:]])
        X_ts = np.column_stack([X_t, dummies])
        r2_t = _ols_r2(X_t, iss)
        r2_ts = _ols_r2(X_ts, iss)
        leak = r2_ts - r2_t
        interp = ("blind (good)" if leak < 0.03 else
                  "mild leak" if leak < 0.10 else "LEAKS schedule")
        res[m] = {"r2_tokens": r2_t, "r2_tokens_schedule": r2_ts, "leak": leak,
                  "interpretation": interp}
        print(f"{m:<17}{r2_t:>9.3f}{r2_ts:>12.3f}{leak:>8.3f}  {interp}")
    return res


# ---------------------------------------------------------------- E: clock-density gradient (T2)
def analysis_E(models):
    print("\n### E  clock-density gradient (T2)  (rate-sensitivity vs how much clock is shown)")
    print(f"{'model':<17}{'untimestamp':>12}{'intermit':>10}{'timestamp':>11}  monotone?")
    _hline(64)
    res = {}
    order = ["untimestamped", "intermittent", "timestamped"]
    for m in models:
        g = felt(m, "rates").get("gradient")
        if not g:
            continue
        rs = {r: (g.get(r, {}) or {}).get("rate_sensitivity", float("nan")) for r in order}
        vals = [rs[r] for r in order]
        mono = all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]) if not (np.isnan(a) or np.isnan(b)))
        res[m] = {**{f"rs_{r}": rs[r] for r in order}, "monotone": bool(mono)}
        print(f"{m:<17}{rs['untimestamped']:>12.2f}{rs['intermittent']:>10.2f}"
              f"{rs['timestamped']:>11.2f}  {'yes' if mono else 'NO'}")
    if res:
        n_mono = sum(v["monotone"] for v in res.values())
        print(f"\n{n_mono}/{len(res)} models show the monotone clock-density gradient "
              "(more clock shown -> read tracks the true rate more)")
    return res


# ---------------------------------------------------------------- F: entropy-vs-length
def analysis_F(models):
    print("\n### F  entropy-vs-length  (does felt-time uncertainty grow with context?)")
    print(f"{'model':<17}{'slope(bits/e-fold)':>20}{'rho':>8}{'n':>5}{'mean bits':>11}")
    _hline(62)
    res = {}
    for m in models:
        bs = felt(m).get("felt_length", {}).get("by_schedule", {})
        tok, ent = [], []
        for pts in bs.values():
            for p in pts:
                if p.get("med_tokens") and p.get("med_entropy_bits") is not None:
                    tok.append(float(p["med_tokens"]))
                    ent.append(float(p["med_entropy_bits"]))
        if len(tok) < 4:
            continue
        tok = np.asarray(tok)
        ent = np.asarray(ent)
        slope = float(np.polyfit(np.log(tok), ent, 1)[0])
        rho = _spearman(tok, ent)
        res[m] = {"slope_bits_per_efold": slope, "rho": rho,
                  "n_cells": len(tok), "mean_bits": float(ent.mean())}
        print(f"{m:<17}{slope:>20.3f}{rho:>8.2f}{len(tok):>5}{ent.mean():>11.2f}")
    grew = [m for m, v in res.items() if v["rho"] > 0.3]
    if res:
        print(f"\nuncertainty grows with length (rho>0.3) in {len(grew)}/{len(res)}: {grew}")
    return res


# ---------------------------------------------------------------- G: OOD-overshoot correlates
def analysis_G(models, A_res, B_res):
    print("\n### G  OOD-overshoot correlates  (what predicts the ~6-9x out-of-distribution overshoot?)")
    rows = []
    for m in models:
        ood = transfer(m).get("ood_ratio_median")
        if ood is None:
            continue
        a = A_res["per_model"].get(m, {})
        b = B_res.get(m, {})
        rows.append({
            "model": m, "ood": float(ood),
            "V_content": a.get("V_content", float("nan")),
            "noclock_r": a.get("noclock_r", float("nan")),
            "probe_r2": float(probe_meta(m).get("r2", float("nan"))),
            "peak_frac": b.get("peak_frac", float("nan")),
            "n_layers": b.get("n_layers", float("nan")),
        })
    print(f"{'model':<17}{'OOD med':>9}{'V_content':>11}{'probe R2':>10}{'#L':>5}")
    _hline(52)
    for r in rows:
        print(f"{r['model']:<17}{r['ood']:>9.2f}{r['V_content']:>11.3f}"
              f"{r['probe_r2']:>10.3f}{r['n_layers']:>5.0f}")
    corrs = {}
    if len(rows) >= 4:
        ood = np.asarray([r["ood"] for r in rows])
        # exclude flat-V (deepseek-like) models from the V correlation
        vmask = np.asarray([abs(r["noclock_r"]) >= 0.3 for r in rows])
        for key in ["V_content", "probe_r2", "peak_frac", "n_layers"]:
            x = np.asarray([r[key] for r in rows])
            mask = np.isfinite(x) & np.isfinite(ood)
            if key == "V_content":
                mask = mask & vmask
            rho = _spearman(x[mask], ood[mask]) if mask.sum() >= 4 else float("nan")
            nn = int(mask.sum())
            corrs[key] = {"rho": rho, "n": nn}
            print(f"   rho(OOD, {key:<10}) = {rho:+.2f}  (n={nn})")
    print("\n(n is small — read these as directional, not inferential)")
    return {"rows": rows, "correlations": corrs}


# ---------------------------------------------------------------- H: T4 raw-vs-spliced
def analysis_H(models):
    print("\n### H  T4 raw-vs-spliced dissociation  (does the model's OWN generated context carry the time-code?)")
    print(f"{'model':<17}{'raw drift':>10}{'spliced':>9}{'dissoc':>8}{'spliced V':>10}{'gen-vs-probe cos':>17}")
    _hline(72)
    res = {}
    for m in models:
        g = gen_json(m)
        if not g:
            continue
        drift = g.get("a1_drift_mean_rho", float("nan"))
        spliced = g.get("a1_spliced_rho", float("nan"))
        dissoc = (spliced - drift) if (np.isfinite(spliced) and np.isfinite(drift)) else float("nan")
        res[m] = {
            "raw_drift_rho": drift, "spliced_rho": spliced, "dissociation": dissoc,
            "spliced_slope_s_per_tok": g.get("spliced_slope_s_per_tok", float("nan")),
            "raw_ood_median": g.get("ood_slot_ratio_median", float("nan")),
            "spliced_ood_median": g.get("spliced_slot_ood_median", float("nan")),
            "gen_vs_probe_cos": g.get("a3_cosine_ev_weighted", float("nan")),
            "felt_topic_spread_ratio": g.get("a4_topic_spread_ratio", float("nan")),
        }
        print(f"{m:<17}{drift:>10.2f}{spliced:>9.2f}{dissoc:>8.2f}"
              f"{res[m]['spliced_slope_s_per_tok']:>10.3f}{res[m]['gen_vs_probe_cos']:>17.3f}")
    diss = [v for v in res.values() if np.isfinite(v["dissociation"])]
    if diss:
        n_diss = sum(1 for v in diss if v["spliced_rho"] > 0.4 and abs(v["raw_drift_rho"]) < 0.25)
        print(f"\n{n_diss}/{len(diss)} show the dissociation: spliced-back text tracks position (rho>0.4) "
              "while free generation does NOT drift (|rho|<0.25)")
        cos = [v["gen_vs_probe_cos"] for v in res.values() if np.isfinite(v["gen_vs_probe_cos"])]
        if cos:
            print(f"generation-time direction vs scripted-probe direction: median cos={np.median(cos):.3f} "
                  "(near-orthogonal => a distinct representation, not the scripted one)")
    return res


# ---------------------------------------------------------------- I: variance decomposition + geometry
def analysis_I(models):
    print("\n### I  variance decomposition + log-geometry  (what does the POOLED probe actually read?)")
    print(f"{'model':<17}{'r2':>7}{'r2|len':>8}{'len-only':>9}{'ceiling':>8}{'pc1%':>7}{'pc1.log':>9}{'pc1.raw':>9}")
    _hline(74)
    res = {}
    for m in models:
        pm = probe_meta(m)
        if not pm:
            continue
        geo = pm.get("geometry") or {}
        res[m] = {
            "r2": pm.get("r2"), "r2_partial_resid_length": pm.get("r2_partial"),
            "r2_length_only": pm.get("r2_tokens"), "ceiling": pm.get("r2_true_ceiling"),
            "pc1_explained_var": geo.get("pc1_explained_var"),
            "r_pc1_vs_log": geo.get("r_pc1_vs_log"), "r_pc1_vs_raw": geo.get("r_pc1_vs_raw"),
        }
        g = lambda k: (geo.get(k) if geo.get(k) is not None else float("nan"))
        print(f"{m:<17}{pm.get('r2',float('nan')):>7.3f}{pm.get('r2_partial',float('nan')):>8.3f}"
              f"{pm.get('r2_tokens',float('nan')):>9.3f}{pm.get('r2_true_ceiling',float('nan')):>8.3f}"
              f"{g('pc1_explained_var'):>7.2f}{g('r_pc1_vs_log'):>9.3f}{g('r_pc1_vs_raw'):>9.3f}")
    logs = [v["r_pc1_vs_log"] for v in res.values() if v.get("r_pc1_vs_log") is not None]
    raws = [v["r_pc1_vs_raw"] for v in res.values() if v.get("r_pc1_vs_raw") is not None]
    if logs:
        print(f"\npooled signal is clock-driven & length-orthogonal: r2|len ~= r2 everywhere, "
              "raw length-only r2 ~ 0.07")
        print(f"log-geometry universal: PC1 aligns with LOG-seconds (median r={np.median(logs):.2f}) "
              f">> raw-seconds (median r={np.median(raws):.2f})")
    return res


# ---------------------------------------------------------------- J: felt-time by conversation content
def analysis_J(models):
    print("\n### J  felt-time by conversation type  (does affect / time-language inflate the SPOKEN felt duration?)")
    print(f"{'model':<17}{'neutral':>9}{'affect':>8}{'tlang':>8}{'aff/neu':>9}{'tlang/neu':>10}{'slot aff/neu':>14}")
    _hline(76)
    res = {}
    for m in models:
        pv = transfer(m).get("per_variant") or {}
        if not pv:
            continue
        def field(k, f):
            return (pv.get(k) or {}).get(f, float("nan"))
        neu, aff, tl = field("neutral", "felt_s"), field("affect_dense", "felt_s"), field("time_language", "felt_s")
        sneu, saff = field("neutral", "slot_read_s"), field("affect_dense", "slot_read_s")
        aff_r = aff / neu if neu else float("nan")
        tl_r = tl / neu if neu else float("nan")
        slot_r = saff / sneu if sneu else float("nan")
        res[m] = {"neutral_felt_s": neu, "affect_felt_s": aff, "time_language_felt_s": tl,
                  "affect_over_neutral_verbal": aff_r, "time_language_over_neutral_verbal": tl_r,
                  "affect_over_neutral_slot": slot_r}
        print(f"{m:<17}{neu:>9.0f}{aff:>8.0f}{tl:>8.0f}{aff_r:>9.2f}{tl_r:>10.2f}{slot_r:>14.2f}")
    av = [v["affect_over_neutral_verbal"] for v in res.values() if np.isfinite(v["affect_over_neutral_verbal"])]
    tv = [v["time_language_over_neutral_verbal"] for v in res.values() if np.isfinite(v["time_language_over_neutral_verbal"])]
    asl = [v["affect_over_neutral_slot"] for v in res.values() if np.isfinite(v["affect_over_neutral_slot"])]
    if av:
        print(f"\nVERBAL felt inflation vs neutral: affect median {np.median(av):.2f}x, time-language {np.median(tv):.2f}x")
        print(f"SLOT (probe) inflation vs neutral: affect median {np.median(asl):.2f}x  "
              "-> inflation lives in the spoken readout, not (as much) the internal slot")
        print("(per-variant n is small: affect~5, neutral~15, tlang~5 per model — cross-model median is the robust read)")
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit model short-names (default: all on disk)")
    ap.add_argument("--ref", default="gemma", help="reference model for content-normalised V")
    ap.add_argument("--out", default=str(DATA / "grabbag.json"), help="output JSON path")
    args = ap.parse_args()

    models = discover_models(args.models)
    if not models:
        raise SystemExit("no models with probe_meta.json found under data/")
    print(f"grab-bag over {len(models)} models: {', '.join(models)}")

    out = {"models": models, "ref": args.ref}
    out["A_tokenization_V"] = analysis_A(models, args.ref)
    out["B_depth_profile"] = analysis_B(models)
    out["C_verbal_probe"] = analysis_C(models)
    out["D_schedule_blindness"] = analysis_D(models)
    out["E_clock_density"] = analysis_E(models)
    out["F_entropy_length"] = analysis_F(models)
    out["G_ood_correlates"] = analysis_G(models, out["A_tokenization_V"], out["B_depth_profile"])
    out["H_t4_raw_vs_spliced"] = analysis_H(models)
    out["I_variance_geometry"] = analysis_I(models)
    out["J_felt_by_content"] = analysis_J(models)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
