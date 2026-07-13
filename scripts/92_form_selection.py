#!/usr/bin/env python
"""form_selection.py — which functional form does felt-time follow?

Adjudicates "one law / echo" vs "two laws" vs "ceiling-prior" by fitting the
tokens->internal and tokens->verbal mappings (no-clock condition) to competing
psychophysical forms and selecting by grouped cross-validation.

Forms (all fit with the SAME lognormal noise model — minimise Σ(log y − log f)² —
so AIC/BIC/CV are comparable across forms):

  power      f = a·t^β            (Stevens; β=1 linear/proportional, β<1 compressive,
                                    β>1 expansive; UNBOUNDED — keeps climbing)
  log        f = a·ln(t) + b      (Fechner; limiting compression, UNBOUNDED)
  saturating f = F·(1 − e^(−t/τ)) (ceiling/shrinkage prior; BOUNDED → asymptote F)
  affine     f = V·t + c          (linear with offset)

β from the power fit is the headline scalar (cluster-bootstrap CI by conversation).

The in-distribution fit is WEAKLY IDENTIFIED among power-β<1 / log / saturating
over a bounded token range — they all bend. The decisive tiebreak is OOD
extrapolation (§ ood_tiebreak): a saturating ceiling predicts the OOD verbal
PLATEAUS near F; an unbounded law predicts it KEEPS CLIMBING. The doc's OOD
overshoot (G) is therefore evidence the in-dist "plateau" is an unbounded
compressive law sampled over a narrow range, not a true asymptote — this fit tests
that per model.

Reads decode_rows.csv (cols: rendering,id,turn_idx,tokens,internal_s,verbal_s).
numpy + scipy only. CPU, seconds.
"""
from __future__ import annotations
import argparse, csv, glob, json, os
import numpy as np
from scipy.optimize import least_squares


# ----------------------------------------------------------------- io
def load(path):
    with open(path) as f: return list(csv.DictReader(f))
def discover(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*", "decode_rows.csv"))):
        m = os.path.basename(os.path.dirname(p))
        if not (m.endswith("_inflation") or m.endswith("_rates")): out[m] = p
    return out
def noclock(rows): return [r for r in rows if r.get("rendering") == "untimestamped"]
def colf(rows, k):
    out = []
    for r in rows:
        v = r.get(k)
        out.append(np.nan if v in ("", "nan", None) else float(v))
    return np.asarray(out, float)


# ----------------------------------------------------------------- forms (log-residual fits)
def _nlls(x, y, f, p0, bounds):
    ly = np.log(y)
    def resid(p):
        fx = f(x, *p)
        return np.log(np.where(fx > 0, fx, 1e-12)) - ly
    try:
        sol = least_squares(resid, p0, bounds=bounds, max_nfev=20000)
        return np.asarray(sol.x), float((sol.fun ** 2).sum())
    except Exception:
        return np.asarray(p0), float("inf")

def fit_power(x, y):  # log-log OLS gives β directly
    lx, ly = np.log(x), np.log(y)
    b, loga = np.polyfit(lx, ly, 1)
    sse = float(((ly - (loga + b * lx)) ** 2).sum())
    return {"name": "power", "k": 2, "sse": sse, "beta": float(b),
            "params": {"a": float(np.exp(loga)), "beta": float(b)},
            "logpred": (lambda xx, loga=loga, b=b: loga + b * np.log(xx))}

def fit_log(x, y):
    f = lambda xx, a, b: a * np.log(xx) + b
    p, sse = _nlls(x, y, f, [max(np.std(np.log(y)), 1.0), np.log(np.median(y))],
                   ([1e-9, -np.inf], [np.inf, np.inf]))
    return {"name": "log", "k": 2, "sse": sse, "beta": float("nan"),
            "params": {"a": float(p[0]), "b": float(p[1])},
            "logpred": (lambda xx, p=p: np.log(np.maximum(p[0] * np.log(xx) + p[1], 1e-12)))}

def fit_sat(x, y):
    f = lambda xx, F, tau: F * (1.0 - np.exp(-xx / np.maximum(tau, 1e-6)))
    p, sse = _nlls(x, y, f, [np.percentile(y, 90), np.median(x)],
                   ([1e-6, 1.0], [np.inf, np.inf]))
    return {"name": "saturating", "k": 2, "sse": sse, "beta": float("nan"),
            "params": {"F_max": float(p[0]), "tau": float(p[1])},
            "logpred": (lambda xx, p=p: np.log(np.maximum(p[0] * (1 - np.exp(-xx / max(p[1], 1e-6))), 1e-12)))}

def fit_affine(x, y):
    f = lambda xx, V, c: V * xx + c
    p, sse = _nlls(x, y, f, [np.median(y) / np.median(x), 1.0],
                   ([1e-12, -np.inf], [np.inf, np.inf]))
    return {"name": "affine", "k": 2, "sse": sse, "beta": float("nan"),
            "params": {"V": float(p[0]), "c": float(p[1])},
            "logpred": (lambda xx, p=p: np.log(np.maximum(p[0] * xx + p[1], 1e-12)))}

FITTERS = [fit_power, fit_log, fit_sat, fit_affine]


def _ic(sse, n, k):
    if not np.isfinite(sse) or sse <= 0 or n <= k + 1: return float("inf"), float("inf")
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return float(aic), float(bic)


def grouped_cv(x, y, ids, fitter, k_folds=5, seed=0):
    """Out-of-fold mean log-squared-error, folds split by conversation id."""
    rng = np.random.default_rng(seed); uids = np.unique(ids); rng.shuffle(uids)
    folds = np.array_split(uids, min(k_folds, len(uids)))
    errs = []
    for te in folds:
        te = set(te.tolist()); tr_m = np.array([i not in te for i in ids]); te_m = ~tr_m
        if tr_m.sum() < 8 or te_m.sum() < 1: continue
        fit = fitter(x[tr_m], y[tr_m])
        pred = fit["logpred"](x[te_m]); err = (np.log(y[te_m]) - pred) ** 2
        errs.append(np.mean(err[np.isfinite(err)]))
    return float(np.mean(errs)) if errs else float("inf")


def beta_ci_by_id(x, y, ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); uids = np.unique(ids); out = []
    for _ in range(n_boot):
        pick = rng.choice(uids, size=len(uids), replace=True)
        idx = np.concatenate([np.where(ids == u)[0] for u in pick])
        try: out.append(np.polyfit(np.log(x[idx]), np.log(y[idx]), 1)[0])
        except Exception: pass
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def fit_target(x, y, ids, n_boot=1000):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y, ids = x[m], y[m], ids[m]
    if len(x) < 12: return None
    n = len(x)
    rows = []
    for fitter in FITTERS:
        fit = fitter(x, y); aic, bic = _ic(fit["sse"], n, fit["k"])
        cv = grouped_cv(x, y, ids, fitter)
        rows.append({"form": fit["name"], "aic": aic, "bic": bic, "cv_logmse": cv,
                     "params": fit["params"], "beta": fit["beta"]})
    winner = min(rows, key=lambda r: r["cv_logmse"])
    pw = next(r for r in rows if r["form"] == "power")
    blo, bhi = beta_ci_by_id(x, y, ids, n_boot=n_boot)
    return {"n": n, "beta": pw["beta"], "beta_ci": [blo, bhi],
            "winner": winner["form"], "cv": {r["form"]: r["cv_logmse"] for r in rows},
            "aic": {r["form"]: r["aic"] for r in rows}, "all": rows}


def ood_tiebreak(in_x, in_y, ood_x, ood_y_median):
    """For each in-dist verbal form, predict at the OOD token range; which matches the
    observed OOD median? saturating -> plateau (rejected if OOD overshoots); power/log -> climb."""
    m = np.isfinite(in_x) & np.isfinite(in_y) & (in_x > 0) & (in_y > 0)
    x, y = in_x[m], in_y[m]; out = {}
    xq = float(np.median(ood_x))
    for fitter in FITTERS:
        fit = fitter(x, y); pred = float(np.exp(fit["logpred"](np.array([xq]))[0]))
        out[fit["name"]] = {"pred_at_ood": pred,
                            "ratio_pred_over_obs": pred / ood_y_median if ood_y_median else float("nan")}
    out["observed_ood_median"] = float(ood_y_median)
    out["ood_token_median"] = xq
    return out


# ----------------------------------------------------------------- driver
def run(model_paths, n_boot=1000):
    res = {}
    hdr = (f"{'model':<17}{'beta_int':>9}{'  int CI':>16}{'beta_verb':>10}{'  verb CI':>16}"
           f"{'verb_form':>12}{'b_int-b_v':>10}  note")
    print(hdr); print("-" * len(hdr))
    for m, p in model_paths.items():
        rows = noclock(load(p))
        x = colf(rows, "tokens"); ids = np.asarray([r.get("id") for r in rows])
        fi = fit_target(x, colf(rows, "internal_s"), ids, n_boot)
        fv = fit_target(x, colf(rows, "verbal_s"), ids, n_boot)
        if fi is None or fv is None:
            print(f"{m:<17}  (insufficient rows)"); continue
        bi, bv = fi["beta"], fv["beta"]
        gap = bi - bv
        note = []
        if fi["beta_ci"][0] <= 0 <= fi["beta_ci"][1]: note.append("internal FLAT (no length law)")
        if fv["beta_ci"][0] <= 0 <= fv["beta_ci"][1]: note.append("verbal≈const (pure prior)")
        # saturating wins spuriously when tau >> range (then it IS linear in-range, not a ceiling).
        # Only call it a true ceiling if tau sits inside the observed token range.
        max_tok = np.nanmax(x)
        if fv["winner"] == "saturating":
            tau = next(r["params"].get("tau", np.inf) for r in fv["all"] if r["form"] == "saturating")
            if tau > max_tok:
                note.append(f"sat τ={tau:.0f}>range → linear-equiv NOT ceiling (trust β)")
            else:
                note.append(f"true ceiling (τ={tau:.0f}<range; needs OOD confirm)")
        if abs(gap) < 0.15 and "internal FLAT" not in " ".join(note): note.append("β match → echo")
        elif gap > 0.15 and "verbal≈const" not in " ".join(note): note.append("verbal more compressive")
        res[m] = {"internal": fi, "verbal": fv, "beta_gap": gap}
        ic_lo, ic_hi = fi["beta_ci"]; vc_lo, vc_hi = fv["beta_ci"]
        int_ci = f"[{ic_lo:+.2f},{ic_hi:+.2f}]"; verb_ci = f"[{vc_lo:+.2f},{vc_hi:+.2f}]"
        print(f"{m:<17}{bi:>9.2f}{int_ci:>16}{bv:>10.2f}{verb_ci:>16}"
              f"{fv['winner']:>12}{gap:>10.2f}  {'; '.join(note)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir"); ap.add_argument("paths", nargs="*")
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--out", default="form_selection.json")
    a = ap.parse_args()
    mp = {}
    if a.data_dir: mp.update(discover(a.data_dir))
    for t in a.paths:
        if ":" in t and not t.startswith(("/", ".")): k, v = t.split(":", 1); mp[k] = v
        else: mp[os.path.basename(os.path.dirname(os.path.abspath(t)))] = t
    if not mp: ap.error("no decode_rows.csv found")
    print(f"models: {list(mp)}\n")
    res = run(mp, n_boot=a.n_boot)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
