# findings

Citable results on **gemma-4-31b-it** (2026-06-07), unified slot pipeline; pilot
scale. T1–T4 below are the **gemma-4-31b-it reference deep-dive**; the encoding now
**replicates across 3 models / 2 families** (gemma-4-12B-it unified, Qwen3.6-27B) —
see [Cross-model replication](#cross-model-replication-2-families-3-models). The
headline (linear, V≈0.3 s/tok) holds everywhere; the *behavioral* readout does not
(Qwen confabulates), so treat T2/T3/T4 as model-specific until the third family lands.

## Headline

**Elapsed conversational time is linearly encoded in context length, on the
residual stream.** With no clock in the transcript, the slot probe's elapsed read
grows as a clean linear function of token count:

> **internal elapsed ≈ V · tokens,  V ≈ 0.29 s/token,  r = 0.88, intercept ≈ 0**
> (n=432 no-clock turns; equally a log–log law, r=0.85 — the slot encodes
> log-tokens on a log-elapsed axis, which exponentiates to a ~linear
> token→seconds rate).

This is the **token-time hypothesis** (`T_wall = T_tok · V`, *Discrete Minds in a
Continuous World*, EMNLP-F 2025) **made representational and measured**: they
*assumed* a constant per-token rate and calibrated it from output token counts;
we read it **off the activations** and find it is genuinely linear, through the
origin, with a measured V. It extends Gurnee & Tegmark (linear-probe time from the
residual stream) from **absolute/calendar** time to **elapsed conversational**
time. The model's *stated* duration confirms the direction (felt rises with
length) but as a **noisier, saturating echo** of the clean internal code (below).

**Provenance.** Probe = EV-weighted all-layer **prefilled elicitation slot**
(saklas explained-variance aggregation, one prompt). Verbal estimate = soft
duration distribution from the slot logits; point = **log-interpolated median**
(`capture.dist_point`, robust to multimodal tails) + **entropy** co-reported
(`capture.dist_entropy`). No sampling, **0 refusals**. EOT-site numbers (Pilots
1–4) survive only as cited history.

---

## T1 — the encoding: elapsed time linearly probed at the slot

EV all-layer probe on the timestamped/constant slot; target log(elapsed s),
grouped-CV by conversation (n=432):

| metric | value | reading |
|---|---:|---|
| EV all-layer R² | **0.984** | the deployed probe |
| best single layer (L32) | 0.995 | representational locus |
| true-prefill ceiling | 0.9997 | text-reading ceiling |
| log-tokens baseline | 0.066 | length alone ≈ nothing **with a clock** |
| partial R² (tokens out) | **0.983** | elapsed *beyond* length |
| no-clock null (vs gt) | −0.21 | can't read *true* elapsed without a clock |

With a clock, the slot reads elapsed at R²=0.98 **beyond** length (partial 0.98)
and beyond text (the true−constant gap) — i.e. when the clock is present the model
reads it, not just length. **Remove the clock and the same axis falls back to a
linear function of length** (the Headline): the probe can no longer predict *true*
elapsed (null −0.21, because the gap schedule decouples true time from length) but
its read is now ≈ 0.29 s/token × context. So the elapsed axis is real and
clock-driven when a clock exists, and **defaults to a linear length→time code when
one doesn't** — exactly the token-time substrate.

**Geometry (slot locus).** PC1 of the log-t centroids explains 0.70 of variance
and is **log-linear** (r(PC1, log-t)=0.95 vs raw 0.56): the axis lives in
Weber–Fechner / log-duration coordinates — consistent with the headline (log-tokens
on a log-elapsed axis → linear seconds-per-token). Per-layer R² climbs 0.49 (L0) →
0.99 plateau from L24; EV weights near-uniform. (EOT baseline, cited/superseded:
L59 R²=0.52, stack 0.59.)

---

## T2 — context length drives felt time; behavior confirms it, saturating

**The behavioral read confirms the linear direction.** The verbal soft estimate
**rises with context length** — per-turn median felt 41 → 213 → 224 → 210 → 266 s
as context grows; ρ(felt, length) = 0.23 over all turns, **0.52 excluding the t11
depth artifact** (below). So the model's *stated* duration is **not** independent
of length: it tracks it, confirming token-time behaviorally.

**But the behavioral code is a degraded, saturating echo of the internal one.**
Side by side on the same no-clock turns:

| read | shape | vs length | probe↔read |
|---|---|---:|---:|
| **probe** (activation) | clean **linear**, through origin, V=0.29 s/tok | r=0.88 | — |
| **verbal** (W_U logits) | **saturating** (jumps to ~210 s by turn 3, plateaus) | r=0.21 | r=0.23 |

The internal code keeps climbing linearly (67 → 98 → 146 → 204 s) while the stated
estimate saturates (~210–266 s) and only weakly agrees with the probe turn-by-turn
(r=0.23). So the representation encodes a **cleaner, more linear** length→time rate
than the model's words reflect — a soft dissociation (internal precise, behavioral
lossy/saturating), **not** a decoupling (the earlier "flat prior" reading was an
artifact of the t11 collapse + multimodal noise depressing ρ; withdrawn).

**The no-clock null, positively.** The probe predicts *nothing* about true elapsed
without a clock (partial R² −0.14) — but that null's *positive content* is the
Headline: what the no-clock slot encodes along the elapsed axis is **length**,
linearly, and nothing beyond it.

**Depth multimodality (surfaced as entropy).** At deep turns the no-clock felt
distribution goes multimodal — turn 9 is a trimodal 30s/5min/6h vote (entropy
1.65 bits, peak; "6 hours" mass schedule-independent at ~0.36). The geometric mean
amplified this into a fake ~900 s spike; the log-interp-median point + entropy
co-stat fixes it (point lands on the central mode, multimodality reads off
`med_entropy_bits`). The t11 collapse (felt → ~23 s at the deepest turn) is the
other half of the same depth instability; it is **not** a final-turn effect
(`idx=7` final vs non-final are identical).

**Clock-density gradient (robust).** Rate-sensitivity at fixed length: **full
clock 0.93 / sparse-intermittent 0.74 / no clock −0.09**. Sparse reads the last
anchor but **undercounts** it (ratio_vs_last_anchor 0.71, vs_current 0.37) — uses
the anchor, doesn't extrapolate. Graded: full → accurate; sparse → reads last
anchor, undercounts since; none → linear length code.

**Reading (H3, confirmed and quantified).** The internal coordinate faithfully
tracks the **only available signal** (tokens) on a calibrated scale; the
wall-clock error is exactly the missing token→seconds mapping — and **V≈0.3 s/token
is that mapping, measured.** H2 (genuinely represented *more* time) is rejected;
the H1 flavor (behavior diverges from the internal coordinate) survives only in the
soft form above (saturating echo, not decoupling).

---

## T3 — the length→time axis transfers to natural felt

The scripted clock-elapsed EV probe, applied to **natural** conversation slots,
tracks natural **felt** at **ρ=0.42** — but tracks **length at ρ=0.61**: on natural
prose the same axis is entangled with length at least as much as with felt, as the
Headline predicts (the axis *is* a length→time code off-clock). Within-natural:
felt readable from the slot (best L34 R²=0.45) but not beyond length (partial|len
−0.17).

**Off-manifold but bounded.** Natural slots sit **5.97× median / 6.31× max** off
the scripted manifold — *not* ≈1×, but **tight** (median≈max), unlike the EOT
site's heavy tail (3.2×/18.8×) that made its probe explode. Bounded → the raw EV
read stays usable unwhitened.

**The probe reads an injected clock (the EOT dissociation is gone).** On
injected-clock natural prose the **probe** recovers the clock at **ρ=0.785**,
better than verbal **0.676**. At the EOT site the probe direction couldn't (−0.12)
while verbal could; at the slot the activation direction genuinely carries
clock-reading.

**Content moves felt, modestly.** Per-variant felt: neutral 42 s < affect 226 s ≈
time-language 248 s; slot read tracks the ordering. Content drives felt ~5×, no 2 h
ceiling (the EOT-era extreme was prompt-driven).

---

## T4 — generation-side: the elapsed axis is read at query time, not written during production

T1–T3 read time from a finished context; T4 asks whether *producing* tokens writes
the same axis. Two reads of one rollout (5 prompts × 3 seeds, 768 tok) dissociate:

- **During production the residual stream doesn't carry the elapsed axis — but the
  null is off-manifold.** Apply the EV reading probe to each generated token and its
  coordinate doesn't drift with position (A1 ρ=−0.03). Yet mid-stream tokens sit
  **18.9× off** the scripted slot manifold (max 42×), so the probe is extrapolating —
  A1≈0 alone can't carry an orthogonality claim. Position itself is richly encoded
  (A2 R²=0.86) but ~orthogonal to the elapsed direction (A3 |cos|=0.04).
- **Fork to the slot and the axis appears.** Cut each partial generation, re-render
  `ELICIT_PROMPT` + constant prefill, read the same probe at that slot. The fork is
  **in-domain** — OOD collapses 18.9×→**5.98×** (the T3 natural band) — and the read
  is a strong, monotone elapsed-vs-position relationship: **ρ=+0.875 ± 0.037, every
  topic +0.82 to +0.91** (the n=4-checkpoint bridge outlier of −0.13 resolved to +0.91
  at n=12). The raw-vs-spliced split *is* the evidence: felt time is a property of the
  accumulated context **read at query time**, not of the generative act.
- **Self-context is counted, but discounted ~5×.** The recovered slope is **0.06
  s/tok**, and it stays ~flat (0.047→0.059) when the rollout span quadruples (256→768
  tok) — a real rate, not range restriction. Against scripted **V≈0.29 s/tok**: an
  uninterrupted self-generated monologue accrues felt-elapsed at ~**1/5** the rate of
  externally-timestamped conversation.
- **Behavioral felt-writing** still grows with tokens (A4 ρ=0.49) and is **topic-
  driven, not seed-driven** — topic spread **2.46×** (pyproj ≈ 2.5× bridge) ≫
  within-topic seed dispersion **1.07×** across 3 seeds. "Instant" survives only as a
  seconds-regime magnitude; "seconds" is behavior, not claimed phenomenology.

---

## Cross-model replication (2 families, 3 models)

Same corpus, same pipeline, run on **gemma-4-12B-it** (`gemma4_unified` — the
encoder-free omni arch, 48L) and **Qwen3.6-27B** (`qwen3_5`, 64L), 2026-06-09/10.
The result splits cleanly into a **universal representation** and a **model-specific
readout**.

**T1 — the encoding is universal across size *and* family.** EV all-layer probe,
timestamped/constant slot, grouped-CV, n=432 each:

| model | arch (layers) | EV R² | partial \| len | r(PC1,log-t) | **V (s/tok)** | V-fit r | locus |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma-4-31b-it | gemma4 (60) | 0.984 | 0.983 | 0.95 | **0.294** | 0.88 | L32 |
| gemma-4-12B-it | gemma4_unified (48) | 0.981 | 0.979 | 0.94 | **0.292** | 0.76 | L34 |
| Qwen3.6-27B | qwen3_5 (64) | 0.989 | 0.988 | 0.98 | **0.324** | 0.88 | L32 |

All three encode log-elapsed at **R²≈0.98–0.99 beyond length**, log-linear geometry,
**V≈0.29–0.32 s/tok**. The log-tokens baseline is **0.066 byte-identical** across all
three (same corpus → tokenization stable across the transformers 5.6.2→5.10.2 bump
the 12B/Qwen required; see caveats). This is the first cross-*family* confirmation:
V≈0.3 s/tok is not a gemma artifact.

**T2/T3/T4 — the behavioral readout is model-specific, and Qwen dissociates.**

| model | verdict | felt~real ρ | entropy (bits) | verbal set-point (no clock) | T3 probe→felt ρ | →length ρ | partial\|len | T4 spliced ρ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 31b | mixed (saturating echo) | 0.10 | 1.03 | grows 42→266 s, t11 collapse | 0.42 | 0.61 | −0.17 | 0.875 |
| 12B | mixed | 0.13 | 1.13 | ~45 s, flat | **0.80** | 0.60 | **+0.32** | 0.747 |
| Qwen | **H1: confabulated at output** | 0.18 | **0.60** | **~4.4 days, flat** | **−0.03** | 0.86 | −0.23 | 0.267 |

**Qwen is the textbook dissociation the experiment was built to find.** Its internal
probe reads true elapsed at R²=0.99, but its verbal soft-readout returns a fixed
**~3.8×10⁵ s (~4.4 days) regardless of schedule or turn** (1 token or 750, instant
schedule or days), at 0.60 bits — confidently. At the instant/turn-1 cell (2.6 s
elapsed) that is a **146,000× overshoot**; it is "right" only on the days schedule by
coincidence. The probe transfers to **length** (ρ=0.86) but **not** to Qwen's
natural felt (ρ=−0.03), and the injected-clock control confirms the split: the
**probe** recovers an injected clock (ρ=0.78) while the **verbal** estimate does not
(ρ=−0.11). `felt.json` verdict, verbatim: *"H1 — internal coordinate tracks reality
but the verbal estimate is decoupled from it: confabulated at output."* The gemmas
sit on the softer "saturating echo" end (the 31b deep-dive in T2); the 12B actually
transfers to natural felt **beyond length** (partial|len +0.32), the strongest H2-ish
signal of the three — but on n=25 natural turns, so held loosely.

*Phenomenology caveat (repo ethics line).* Qwen's "4 days" is parsimoniously a
peaked verbal **prior**, not evidence it *experiences* 4 days having passed. The soft
readout keeps this as a low-entropy distribution rather than forcing a point — which
is exactly why the confabulation is legible instead of collapsing to a refusal NaN.

**Caveats.** (1) Pilot scale, n=432 turns/model, 2 families — V-universality wants a
third (ministral). (2) The 31B reference was captured on transformers 5.6.2; the 12B
(`gemma4_unified` needs ≥5.10.1) and Qwen ran on 5.10.2 — cross-version, but the
identical token baseline says tokenization/rendering didn't shift. (3) `gemma4_unified`
is the encoder-free omni wrapper; saklas extracts the text decoder
(`language_model.*`) — a broken extraction can't yield R²=0.98 with matched geometry,
so the path is validated empirically (and now in saklas's `_TESTED_ARCHS`). (4) The
12B's V-fit r=0.76 is noisier on the linear axis than its log-axis R²=0.98.

---

## Relation to prior work

| | quantity | level | linearity | rate V |
|---|---|---|---|---|
| Gurnee & Tegmark 2310.02207 | **absolute** time | representational (probe) | — | — |
| Discrete Minds 2506.05790 | elapsed/wall-clock | **behavioral** | *assumed* `∝` | calibrated from output rate |
| **this work** | **elapsed** conversational | **representational** (probe) | **measured** linear, r=0.88 | **V≈0.3 s/tok off activations** |

We **confirm** token-time (both probe and behavior increase with length) and
contribute the piece they lacked: the rate **measured on the residual stream**, for
**elapsed** (not absolute) time, with the internal code shown to be a **cleaner
linear law than the behavioral readout** expresses.

## Estimator (settled)

Verbal point = log-interpolated median (`capture.dist_point`); entropy co-reported
(`capture.dist_entropy`). The distribution (`verbal_dist`) is the source of truth;
the scalar is a robust summary, not the object.

## What would make it a paper

Multi-model replication (≥3 families) — **2/3 done**: V≈0.3 and linearity hold on
gemma (2 sizes) + Qwen, **ministral** is the natural third family (and the
size-vs-family confound is now partly broken — 12B vs 31B same family, Qwen
cross-family). The cross-model dissociation (universal encoding, model-specific
readout; Qwen confabulates) is arguably the **stronger framing** than bare
replication. Remaining: the "probe isn't just a length detector" control foregrounded
(timestamped partial R²=0.98 beyond length, schedule-independence); a **causal
steering** confirmation along the length→time axis; robustness of the readout
behavior across elicitation prompts (esp. whether Qwen's flat ~4-day prior survives
rephrasing / its thinking mode); and a 31B re-capture on transformers 5.10.2 to make
the cross-model comparison fully within-version.
