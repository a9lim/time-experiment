#!/usr/bin/env python
"""analysis_k_standalone.py — value-vs-readout test (§1 fingerprint + §2 joint-calibration).

Self-contained: needs ONLY numpy + scipy and the per-model decode_rows.csv files.
No repo install, no saklas/torch/llmoji. Computes V_content inline (no need to run
analysis_A first). CPU, seconds.

decode_rows.csv columns used: rendering, id, turn_idx, tokens, internal_s, verbal_s
(the file 30_felt.py writes; gt_s is present but not needed here).

Usage
-----
  # point at a folder laid out like the repo's data/ :  <dir>/<model>/decode_rows.csv
  python analysis_k_standalone.py --data-dir ./data

  # or pass explicit files (model name = parent folder, or use model:path)
  python analysis_k_standalone.py gemma:./gemma_decode_rows.csv qwen:./qwen_decode_rows.csv

  # choose the reference model for the shared-token V yardstick (default: gemma)
  python analysis_k_standalone.py --data-dir ./data --ref gemma --out k_result.json
"""
from __future__ import annotations
import argparse, csv, glob, json, os
import numpy as np
from scipy.stats import rankdata


# ----------------------------------------------------------------- io
def load_decode(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def discover(data_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "*", "decode_rows.csv"))):
        m = os.path.basename(os.path.dirname(p))
        if m.endswith("_inflation") or m.endswith("_rates"):
            continue  # variant corpora, not separate models (mirror grabbag.discover_models)
        out[m] = p
    return out

def untimestamped(rows):
    return [r for r in rows if r.get("rendering") == "untimestamped"]

def col(rows, key, cast=float):
    vals = []
    for r in rows:
        v = r.get(key)
        if v in ("", "nan", None):
            vals.append(np.nan)
        else:
            try: vals.append(cast(v))
            except Exception: vals.append(np.nan)
    return np.asarray(vals, float)


# ----------------------------------------------------------------- stats
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: return float("nan")
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])

def boot_ci_by_id(stat_fn, ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); uids = np.unique(ids); out = []
    for _ in range(n_boot):
        pick = rng.choice(uids, size=len(uids), replace=True)
        idx = np.concatenate([np.where(ids == u)[0] for u in pick])
        try:
            v = stat_fn(idx)
            if np.isfinite(v): out.append(v)
        except Exception:
            pass
    if not out: return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

def saturating_fit(tokens, verbal_s):
    t, v = np.asarray(tokens, float), np.asarray(verbal_s, float)
    ok = np.isfinite(t) & np.isfinite(v) & (v > 0); t, v = t[ok], v[ok]
    if len(t) < 8: return float("nan"), float("nan")
    try:
        from scipy.optimize import curve_fit
        f = lambda tt, F, tau: F * (1.0 - np.exp(-tt / np.maximum(tau, 1e-6)))
        p, _ = curve_fit(f, t, v, p0=[np.percentile(v, 90), np.median(t)],
                         maxfev=10000, bounds=([0, 1.0], [np.inf, np.inf]))
        return float(p[0]), float(p[1])
    except Exception:
        best = (float("nan"), float("nan"), np.inf)
        for tau in np.logspace(0, np.log10(max(t.max(), 2.0)), 40):
            g = 1.0 - np.exp(-t / tau); d = float(g @ g)
            if d <= 0: continue
            F = float((v @ g) / d); sse = float(((v - F * g) ** 2).sum())
            if sse < best[2]: best = (F, float(tau), sse)
        return best[0], best[1]


# ----------------------------------------------------------------- the analysis
def run(model_paths, ref="gemma", n_boot=1000):
    raw = {m: load_decode(p) for m, p in model_paths.items()}

    # shared-token yardstick for V_content (same corpus across models -> same ids/turns)
    if ref not in raw: ref = next(iter(raw))
    ref_tok = {(r["id"], r["turn_idx"]): float(r["tokens"])
               for r in untimestamped(raw[ref]) if r.get("tokens") not in ("", None)}

    res = {}
    hdr = (f"{'model':<17}{'b(noclk)':>9}{'b CI':>15}{'var_v/x':>9}{'scale':>9}"
           f"{'priorC s':>10}{'Fmax':>9}{'tau':>8}{'V_cont':>8}{'V*tau':>9}  class")
    print(hdr); print("-" * len(hdr))

    for m, rows in raw.items():
        uts = untimestamped(rows)
        v_all = col(uts, "verbal_s"); x_all = col(uts, "internal_s")
        tok_all = col(uts, "tokens"); turn_all = col(uts, "turn_idx")
        ids_all = np.asarray([r.get("id") for r in uts])
        ok = np.isfinite(v_all) & np.isfinite(x_all) & (v_all > 0) & (x_all > 0)
        if ok.sum() < 20:
            print(f"{m:<17}  (only {int(ok.sum())} usable no-clock rows — skipped)")
            continue
        v, x, tok, ids = v_all[ok], x_all[ok], tok_all[ok], ids_all[ok]
        y, xl = np.log(v), np.log(x)

        # §1 fingerprint
        b, a = np.polyfit(xl, y, 1)
        def _slope(idx):
            xi, vi = x_all[idx], v_all[idx]
            mm = np.isfinite(xi) & np.isfinite(vi) & (xi > 0) & (vi > 0)
            return np.polyfit(np.log(xi[mm]), np.log(vi[mm]), 1)[0] if mm.sum() >= 5 else np.nan
        b_lo, b_hi = boot_ci_by_id(_slope, ids_all, n_boot=n_boot)
        var_ratio = float(np.var(y) / np.var(xl)) if np.var(xl) > 0 else float("nan")
        scale = float(np.median(v / x))
        # implied prior center is only meaningful under genuine shrinkage (0<b<~0.9);
        # at b~1 (echo) it explodes and is "not applicable" -> NaN
        prior_center = float(np.exp(a / (1 - b))) if 0.0 < b < 0.9 else float("nan")

        # V from no-clock internal_s vs tokens  (V_raw on own tokens; V_content on ref tokens)
        noclock_r = float(np.corrcoef(tok, x)[0, 1])
        V_raw = float(np.polyfit(tok, x, 1)[0])  # NB: slope of internal_s (seconds) on tokens
        keys = [(r["id"], r["turn_idx"]) for r in uts]
        aligned = np.array([ref_tok.get(k, np.nan) for k in keys])[ok]
        am = np.isfinite(aligned)
        V_content = float(np.polyfit(aligned[am], x[am], 1)[0]) if am.sum() >= 5 else V_raw

        # §2 saturating bridge
        Fmax, tau = saturating_fit(tok, v)
        Vtau = float(V_content * tau) if np.isfinite(V_content) and np.isfinite(tau) else float("nan")

        flat_V = abs(noclock_r) < 0.3
        if b <= 0:                      cls = "confabulating (b<=0)"
        elif var_ratio < 0.25:          cls = "prior-dominated (verbal near-constant)"
        elif scale > 5 or scale < 0.2:  cls = "scale-decoupled"
        elif flat_V:                    cls = "flat-V (no length signal; V meaningless)"
        else:                           cls = "faithful (echo OR mild shrink -> needs IV §3)"

        res[m] = {"b_noclock": float(b), "b_ci": [b_lo, b_hi],
                  "var_verbal_over_var_internal": var_ratio, "scale_verbal_over_internal": scale,
                  "prior_center_s": prior_center, "Fmax_s": Fmax, "tau_tokens": tau,
                  "V_raw": V_raw, "V_content": V_content, "noclock_r": noclock_r,
                  "V_times_tau_s": Vtau, "n_rows": int(ok.sum()), "class": cls}
        print(f"{m:<17}{b:>9.2f}{f'[{b_lo:+.2f},{b_hi:+.2f}]':>15}{var_ratio:>9.2f}{scale:>9.2f}"
              f"{prior_center:>10.0f}{Fmax:>9.0f}{tau:>8.0f}{V_content:>8.2f}{Vtau:>9.0f}  {cls}")

    # §2 joint-calibration: length-tracking, non-confabulating only
    elig = [r for r in res.values()
            if np.isfinite(r["Fmax_s"]) and np.isfinite(r["V_times_tau_s"])
            and abs(r["noclock_r"]) >= 0.3
            and r["class"].split()[0] not in ("confabulating", "scale-decoupled", "flat-V")]
    if len(elig) >= 4:
        F = [r["Fmax_s"] for r in elig]; Vt = [r["V_times_tau_s"] for r in elig]
        rho = spearman(F, Vt)
        print(f"\njoint-calibration  rho(Fmax, V*tau) = {rho:+.2f}  "
              f"(n={len(elig)}, length-tracking non-confabulating)")
        print("  >0 => verbal ceiling scales with the value's rate (value-calibrated prior);"
              "  ~0/<0 => prior independent of value")
    else:
        print(f"\njoint-calibration: only {len(elig)} eligible models (need >=4) — add models or "
              "check noclock_r >= 0.3")

    # quick reading
    confab = [m for m, r in res.items() if r["b_noclock"] <= 0]
    prior_dom = [m for m, r in res.items() if r["class"].startswith("prior-dominated")]
    print("\nreading:")
    print(f"  confabulators (b<=0, dilution-proof): {confab or 'none'}")
    print(f"  prior-dominated (verbal saturates vs climbing internal): {prior_dom or 'none'}")
    print("  -> any non-empty set above is evidence AGAINST pure 'one law, lossy echo'.")
    print("  -> if faithful models cluster in 'needs IV', stand up §3 (split-half) to settle echo-vs-shrink.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder with <model>/decode_rows.csv subdirs")
    ap.add_argument("paths", nargs="*", help="explicit model:path pairs (or bare paths)")
    ap.add_argument("--ref", default="gemma", help="reference model for shared-token V (default gemma)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="k_result.json")
    a = ap.parse_args()

    model_paths = {}
    if a.data_dir:
        model_paths.update(discover(a.data_dir))
    for tok in a.paths:
        if ":" in tok and not tok.startswith(("/", ".")):
            m, p = tok.split(":", 1); model_paths[m] = p
        else:
            model_paths[os.path.basename(os.path.dirname(os.path.abspath(tok)))] = tok
    if not model_paths:
        ap.error("no decode_rows.csv found — pass --data-dir or model:path pairs")

    print(f"models: {list(model_paths)}\n")
    res = run(model_paths, ref=a.ref, n_boot=a.n_boot)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
