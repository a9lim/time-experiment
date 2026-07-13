# Value-vs-readout test (analysis K)

**Question.** Is the verbal estimate *one law read lossily* (the repo's "saturating
echo") or *a second process* — a learned readout prior in tension with the internal
value? Echo and two-process make opposite predictions about the **slope and shape**
of verbal-vs-internal.

> **Method note, read first.** The obvious CPU-only test — "regress verbal on
> internal, look for residual structure in tokens" — is **degenerate** and was cut.
> In the no-clock condition the internal value *is* `V·tokens`, so tokens and the
> value are the same axis; the probe read is a *noisy* version of that value and
> tokens is a *cleaner* one, so conditioning on the noisy probe leaves
> value-variance that tokens mops up. On synthetic **pure-echo** data this produces
> `ΔR²_tokens = 0.13` — a false positive for two-process. The discriminators below
> are the ones that survive this regression-dilution (errors-in-variables) trap.
> The trap itself is a methodological result worth stating in the writeup.

CPU-only screen runs on existing `decode_rows.csv`; the per-model adjudicator and
the OOD arm need activations. Slots into `scripts/91_grabbag.py` beside
`analysis_C`/`analysis_D`, and turns C's three discrete buckets into a continuous
discrepancy magnitude with a mechanism.

---

## 0. Data

`data/<m>/decode_rows.csv` (from `30_felt.py`):
`rendering, id, turn_idx, schedule, tokens, gt_s, internal_s, verbal_s`.

- internal coordinate `x = log(internal_s)` — EV-probe OOF read (log-s).
- verbal estimate `y = log(verbal_s)` — soft-dist point (log-s). Drop
  `verbal_s ∈ {"", nan, ≤0}` as `analysis_C` does.
- group `id`; split by `rendering`. **No-clock (`untimestamped`) is primary.**

---

## 1. The fingerprint (CPU-only, dilution-robust) — the headline screen

Three numbers per model × rendering. None can be faked by measurement noise on `x`:
dilution attenuates a slope *toward* 0 but cannot flip its sign, and cannot collapse
`var(y)/var(x)` to ~0.1 or inflate the scale to ~10×.

| quantity | definition | echo | two-process / confabulation |
|---|---|---|---|
| **slope sign** | `b` in `y = a + b·x` | `b > 0` | Qwen-type: `b ≤ 0` (verbal anti-tracks the value) |
| **variance ratio** | `var(y) / var(x)` | `~1` | prior-dominated: `≪ 1` (verbal near-constant) |
| **scale** | `median(verbal_s / internal_s)` | `~1` | scale-decoupled: `≫1` or `≪1` (right shape, wrong gain) |

Synthetic separation (validated):

```
faithful         b=+0.83  var_v/var_x=1.09  scale=0.97
scale-decoupled  b=+0.80  var_v/var_x=1.06  scale=11.8
qwen             b=-0.26  var_v/var_x=0.08  scale=1584
```

`scale` overlaps `analysis_C`'s `verbal_over_probe`; reuse or recompute. The new
content is **`b` and `var_v/var_x` jointly**: C's hard `ρ<0` cutoff calls one
confabulator (Qwen); the fingerprint asks **how many models are *mild*
confabulators** — small but nonzero negative `b`, or a depressed variance ratio —
that the binary cutoff missed. Run on all 10 tonight.

**Concrete prediction (from the doc's own T2).** The doc reports gemma's verbal
*saturates* (~210 s plateau by turn 3) while the internal coordinate *keeps climbing
linearly*. That is exactly `var(y) ≪ var(x)` with `b > 0` — so gemma should land in
**`prior-dominated`, not the neutral `faithful` band.** A depressed `var_v/var_x`
against a climbing internal read *is* the ceiling/shrinkage signature; the doc's
"saturating echo" and a "value-calibrated ceiling prior" are the same observation,
and §3's IV separates them (echo of a saturating value → `b_deatt≈1`; ceiling prior →
`b_deatt<1`). If most faithful models come back `prior-dominated`, that is a result,
not noise: it says the saturation is a property of the *readout*, which §2/§3 then
test for value-calibration.

CIs: cluster-bootstrap `b` and `var_v/var_x` by `id` (within-conversation turns are
correlated; per-row CIs lie). Helper in §7.

---

## 2. Joint-calibration (CPU-only) — the cross-model test echo can't make

This is where K meets last round's β-fit and yields a prediction the echo view has
no way to produce. On no-clock rows, fit verbal-in-seconds against length:

```
verbal_s ≈ F_max · (1 − exp(−tokens / τ))          # saturating readout
```

(`curve_fit`; coarse `τ`-grid + closed-form `F_max` fallback if SciPy optimise is
unavailable). Under **two-process-with-joint-calibration** — the verbal readout is a
prior calibrated to agree with the linear value over the *training* range — three
numbers coincide:

```
implied prior center  exp(a/(1−b))   ≈   F_max   ≈   V · τ
```

`V` = the model's no-clock rate (`analysis_A`'s `V_content`, length-tracking models
only; the OOD overshoot is the *same* parameter, so this also unifies with G).

**Cross-model prediction:** higher-V models have a proportionally higher verbal
ceiling / earlier knee → `ρ(F_max, V·τ) > 0` over the 8 length-tracking models.
Synthetic check returned `ρ = +1.00`.

- Holds → the verbal channel is a value-calibrated prior; "two laws in tension" made
  quantitative.
- Fails → the prior is independent of the value; "tension" is the wrong frame (two
  *unrelated* signals). Either way, a finding.

Form-selection bonus: a clean saturating fit (real finite `τ`) ⇒ a **prior/ceiling**;
a better `a·ln(t)` fit (compare AIC) ⇒ a genuine **second log-law**. Same fit
adjudicates "two laws" vs "value-vs-prior".

---

## 3. The per-model adjudicator (needs activations) — echo vs genuine shrinkage

For the **faithful majority**, the fingerprint shows `b ≈ 0.8`, `var≈1`, `scale≈1` —
consistent with *both* clean echo (slope diluted from 1) and *mild shrinkage* toward
a prior. Only de-attenuation tells them apart. The naive slope cannot.

Fit the EV probe twice on **disjoint layer halves** (even vs odd sidecar layers) →
two independent noisy reads `x₁, x₂` of the same slot. The two-sample / IV estimator
cancels the shared-signal dilution:

```
b_deatt = cov(y, x₂) / cov(x₁, x₂)            # symmetrise over (x₁,x₂)
```

Validated: synthetic pure-echo `b_naive = 0.83 → b_deatt = 1.03`.

- `b_deatt ≈ 1` → the compression was **dilution**; verbal is a clean noisy echo.
  Two-process then rests entirely on §4 (OOD).
- `b_deatt < 1` (CI excludes 1) → **genuine shrinkage** toward a prior; with §2's
  `F_max ≈ V·τ`, the prior is value-calibrated.

Wire from `assemble()` + `fit_ev_probe()` on a sliced `X3d[:, even, :]` / `[:, odd, :]`;
each half is a valid all-layer EV probe over its layer subset. Loading mirrors
`30_felt.py` (it already builds the OOF internal read this re-derives per half).

---

## 4. The OOD divergence arm (needs activations + saklas) — the cleanest adjudicator

In-distribution the prior and the value agree over the training range — *that is why
faithful models read as faithful*. The two-process signature is that they **diverge
where the prior loses its grip: off-manifold.** Measured as a **divergence**, not a
partial-R² (so it sidesteps the §-note trap entirely).

Per natural-transfer slot (the 5–9× off-manifold set, T3):

- `internal = apply_ev_probe(probe, natural_X3d)`            *(analysis.py)*
- `verbal`   = the natural verbal readout (captured for T3; in natural
  `rows.jsonl` / `transfer` artifacts)
- `ood`      = `maha_scorer(scripted_X3d, layers)(natural_X3d)`   *(analysis.py;
  same whitener T3 uses → matches `ood_ratio_median`)*

Bin by `ood` (terciles or continuous); per bin report **gap = median(`log verbal −
log internal`)**.

- **Two-process (faithful):** gap ≈ 0 in the low-OOD bin, **rising** with OOD. This
  is the per-slot resolution of the G-finding overshoot: the overshoot *is* the
  prior detaching from the value as you leave the manifold, and `ρ(OOD,V)=0.75`
  becomes within-model and mechanistic.
- **Echo:** gap flat across OOD bins (no second process to lose grip).

---

## 5. Per-class predictions (the table that confirms or kills it)

| class (models) | `b` no-clock | var_v/var_x | scale | `b_deatt` (§3) | OOD gap trend (§4) | reading |
|---|---|---|---|---|---|---|
| **faithful** (gemma×2, llama, phi, ministral, granite, GLM) | ~0.8 | ~1 | ~1 | **echo →1** or **shrink <1** | **rising** | prior ≈ value in-dist; detaches OOD |
| **confabulating** (Qwen) | **≤ 0** | **≪1** (~0.08) | ~2750× | n/a (already detached) | large everywhere | prior fully detached & anti-aligned |
| **scale-decoupled** (DeepSeek) | >0 | ~1 | **~12×** | ~1 | offset, weak trend | right shape, wrong gain — miscalibrated prior |

**Qwen is decisive and CPU-only** (no OOD arm needed): `b ≤ 0` and `var_v/var_x ≪ 1`
are both dilution-proof. The echo view *cannot* produce them — a lossy copy of `x`
cannot anti-correlate with `x`, nor be driven *down* when `x` is forced *up* (T5's
ρ=−0.98). The fingerprint is the cheap, captures-only detector of the same
dissociation T5 shows causally — so the real yield of running it across all 10 is the
**count of mild confabulators** between faithful Qwen-the-extreme and the clean cases.

---

## 6. Drop-in code (`scripts/91_grabbag.py`, CPU-only: §1 + §2)

```python
# ---------------------------------------------------------------- K: value-vs-readout
def _boot_ci_by_id(fn, ids, n_boot=1000, seed=0):
    """Cluster-by-id bootstrap CI for a scalar statistic fn(idx_mask)."""
    rng = np.random.default_rng(seed); uids = np.unique(ids); out = []
    for _ in range(n_boot):
        pick = rng.choice(uids, size=len(uids), replace=True)
        idx = np.concatenate([np.where(ids == u)[0] for u in pick])
        try: out.append(fn(idx))
        except Exception: pass
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),)*2


def _saturating_fit(tokens, verbal_s):
    """verbal_s ≈ F_max·(1 − exp(−tokens/τ)); returns (F_max, tau) or (nan, nan)."""
    t = np.asarray(tokens, float); v = np.asarray(verbal_s, float)
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
        for tau in np.logspace(0, np.log10(max(t.max(), 2)), 40):
            g = 1.0 - np.exp(-t / tau); d = float(g @ g)
            if d <= 0: continue
            F = float((v @ g) / d); sse = float(((v - F * g) ** 2).sum())
            if sse < best[2]: best = (F, float(tau), sse)
        return best[0], best[1]


def analysis_K(models, A_res):
    print("\n### K  value-vs-readout  (echo of the internal value, or a separate prior?)")
    print(f"{'model':<17}{'b(noclk)':>9}{'var_v/x':>9}{'scale':>9}{'priorC':>9}"
          f"{'Fmax':>9}{'tau':>8}{'V*tau':>9}  class")
    _hline(96)
    res = {}
    for m in models:
        rows = _untimestamped(m)
        keep = [r for r in rows if r.get("verbal_s") not in ("", "nan", None)]
        v = np.asarray([float(r["verbal_s"]) for r in keep])
        x = np.asarray([float(r["internal_s"]) for r in keep])
        tok = np.asarray([float(r["tokens"]) for r in keep])
        ids = np.asarray([r["id"] for r in keep])
        ok = (v > 0) & (x > 0)
        if ok.sum() < 20: continue
        y, xl, ids_ok = np.log(v[ok]), np.log(x[ok]), ids[ok]
        b, a = np.polyfit(xl, y, 1)
        b_lo, b_hi = _boot_ci_by_id(lambda i: np.polyfit(np.log(x[i][x[i]>0]), np.log(v[i][x[i]>0]), 1)[0]
                                    if (v[i]>0).all() and (x[i]>0).all() else float("nan"), ids_ok)
        var_ratio = float(np.var(y) / np.var(xl)) if np.var(xl) > 0 else float("nan")
        scale = float(np.median(v[ok] / x[ok]))
        prior_center = float(np.exp(a / (1 - b))) if abs(1 - b) > 1e-3 else float("nan")
        Fmax, tau = _saturating_fit(tok[ok], v[ok])
        V = (A_res["per_model"].get(m, {}) or {}).get("V_content", float("nan"))
        Vtau = float(V * tau) if np.isfinite(V) and np.isfinite(tau) else float("nan")
        if b <= 0:                       cls = "confabulating (b<=0)"
        elif var_ratio < 0.25:           cls = "prior-dominated (verbal near-constant)"
        elif scale > 5 or scale < 0.2:   cls = "scale-decoupled"
        else:                            cls = "faithful (echo OR mild shrink -> needs §3 IV)"
        res[m] = {"b_noclock": float(b), "b_ci": [b_lo, b_hi],
                  "var_verbal_over_var_internal": var_ratio, "scale_verbal_over_internal": scale,
                  "prior_center_s": prior_center, "Fmax_s": Fmax, "tau_tokens": tau,
                  "V_content": float(V), "V_times_tau_s": Vtau, "class": cls}
        print(f"{m:<17}{b:>9.2f}{var_ratio:>9.2f}{scale:>9.2f}{prior_center:>9.0f}"
              f"{Fmax:>9.0f}{tau:>8.0f}{Vtau:>9.0f}  {cls}")
    # joint-calibration: ONLY length-tracking, non-confabulating models (mirror A/G's
    # |noclock_r|>=0.3 gate). Including Qwen (tau collapses) or DeepSeek (V flat) inverts
    # the correlation — they are the models whose readout/value link is the thing in
    # question, so they cannot test whether the link is value-calibrated.
    pts = [(r["Fmax_s"], r["V_times_tau_s"]) for r in res.values()
           if np.isfinite(r["Fmax_s"]) and np.isfinite(r["V_times_tau_s"]) and r["V_content"] > 0
           and r["class"] not in ("confabulating", "scale-decoupled")]
    if len(pts) >= 4:
        F, Vt = np.asarray(pts).T
        print(f"\njoint-calibration  rho(Fmax, V*tau) = {_spearman(F, Vt):+.2f}  "
              f"(n={len(pts)}, length-tracking faithful only)")
        print("  >0 => verbal ceiling is set by the value's rate (value-calibrated prior)")
    return res
```

Register in `main()` after C/D (pass `A_res`, already built for A/G):

```python
    out["K_value_vs_readout"] = analysis_K(models, out["A_toknorm_V"])  # match A's key
```

---

## 7. What each outcome buys

- **Fingerprint all-faithful + §3 `b_deatt≈1` + §4 flat** → repo framing is right:
  one law, lossy readout. Clean negative; drop two-process language.
- **Fingerprint shows Qwen `b≤0`/`var≪1` (replicated) + §2 `ρ(Fmax,V·τ)>0` + §4
  rising gap** → the new spine: a linear internal value and a value-calibrated
  readout prior, agreeing in-distribution, diverging off-manifold by the same `V`
  that drives the OOD overshoot. T5's causal inversion is the extreme of a continuum
  K measures correlationally across all 10.
- **Mixed** (faithful models with small negative `b` or depressed `var`) → most
  likely, most interesting: confabulation is **graded**, not a Qwen singleton, and K
  ranks it — reframing C's three buckets as one axis = discrepancy magnitude.

Run §1+§2 tonight (CPU, all 10). Stand up §3 (IV) and §4 (OOD) **iff** the
fingerprint shows any sub-faithful structure or Qwen's `b≤0` replicates — they're the
adjudicators, not the screen.
