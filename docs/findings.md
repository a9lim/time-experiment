# findings

T1–T4 below are the **gemma-4-31b-it reference deep-dive** (2026-06-07), unified slot
pipeline, pilot scale. The encoding then **replicates across 10 models / 9 architecture
families** — see [Cross-family replication](#cross-family-replication-10-models-9-families)
(2026-06-11). The result splits cleanly into a universal piece and a model-specific one:

- **Universal — the encoding.** Elapsed conversational time is **linearly (log-linearly)
  encoded in context length** on the residual stream: EV all-layer probe R² **0.88–0.99**
  in *every* model, mid-depth locus, log-duration geometry. This is robust.
- **Model-specific — the rate.** The per-token rate **V** (inferred elapsed seconds per
  token) spans **~0.20–2.7 s/tok, ~14×**, and is **not** an artifact of tokenization
  (re-expressing on a shared content yardstick leaves the spread at 13×). The earlier
  "**V≈0.3 universal**" was a gemma+Qwen coincidence — **withdrawn.**

Two models are informative exceptions, not failures: **DeepSeek-V2-Lite** reads a clock
fine but has **no no-clock length encoding** (its untimestamped read is flat; the MLA
anomaly), and **Qwen3.6-27B** has the *cleanest* internal code of all yet **confabulates**
its spoken estimate (verbal anti-correlates with the probe). The behavioral readout
(T2/T3/T4) is therefore **model-specific**; the *representation* is not.

## Headline

**Elapsed conversational time is linearly encoded in context length, on the residual
stream — universally; the per-token rate is model-specific.** With no clock in the
transcript, the slot probe's elapsed read grows as a clean (log-)linear function of
token count, in every model tested:

> **internal elapsed ≈ V · tokens,  intercept ≈ 0**, with **V model-specific** —
> gemma-4-31b **0.29 s/tok** (r=0.88), Ministral **2.6**, talkie **0.20**, ~14× spread
> (n=432 no-clock turns/model). Equally a log–log law: the slot encodes log-tokens on a
> log-elapsed axis (r(PC1,log-t)≈0.95, universal), which exponentiates to a ~linear
> token→seconds rate. Strong and clean in ~7/10 models; **flat in DeepSeek** (no
> no-clock signal) and **noisy in talkie** (r=0.33).

This is the **token-time hypothesis** (`T_wall = T_tok · V`, *Discrete Minds in a
Continuous World*, EMNLP-F 2025) **made representational and measured**: they
*assumed* a constant per-token rate and calibrated it from output token counts;
we read it **off the activations** and find it is genuinely linear, through the
origin — and, across models, find that **V itself is a representational property** that
varies ~14× by model, not a universal constant. It extends Gurnee & Tegmark (linear-probe
time from the residual stream) from **absolute/calendar** time to **elapsed
conversational** time. The model's *stated* duration usually confirms the direction (felt
rises with length) but as a **noisier, saturating echo** of the clean internal code — and
in one model (Qwen) the spoken estimate is decoupled from it entirely (below).

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
its read is now ≈ 0.29 s/token × context (**gemma's rate** — V is model-specific, ~14×
across the family; [cross-family](#cross-family-replication-10-models-9-families)). So
the elapsed axis is real and clock-driven when a clock exists, and **defaults to a
linear length→time code when one doesn't** — exactly the token-time substrate.

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
wall-clock error is exactly the missing token→seconds mapping — and **V (here gemma's
≈0.3 s/token; model-specific cross-family) is that mapping, measured.** H2 (genuinely
represented *more* time) is rejected;
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

## Cross-family replication (10 models, 9 families)

Same corpus, same pipeline, swept across **10 models** spanning **9 distinct
architectures** (the two Gemmas share a lineage across an arch-variant boundary),
2026-06-09→11. The split is sharp: **a universal representation** and a
**model-specific, occasionally-dissociated readout.** The full analytical backbone is
the cross-family grab-bag (`scripts/91_grabbag.py` → `data/grabbag.json`, analyses
A–J); the headline tables are below.

### T1 — the encoding is universal across size, family, and attention design

EV all-layer probe, timestamped/constant slot, grouped-CV by conversation, n=432 each:

| model | arch (L) | EV R² | partial\|len | **V (s/tok)** | no-clock r | locus | r(PC1,log-t) |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma-4-31b-it | gemma4 (60) | 0.984 | 0.983 | **0.29** | 0.88 | 54% | 0.95 |
| gemma-4-12B-it | gemma4_unified (48) | 0.981 | 0.979 | **0.29** | 0.76 | 72% | 0.94 |
| Qwen3.6-27B | qwen3_5 (64) | 0.989 | 0.988 | **0.32** | 0.88 | 51% | 0.98 |
| Llama-3.2-3B | llama (28) | 0.984 | 0.982 | **1.06** | 0.86 | 52% | 0.95 |
| Phi-4-mini | phi3 (32) | 0.985 | 0.983 | **0.87** | 0.59 | 52% | 0.98 |
| talkie-1930-13B | talkie (40) | 0.888 | 0.880 | **0.20** | 0.33 | 56% | 0.82 |
| Ministral-3-14B | mistral3 (40) | 0.987 | 0.985 | **2.60** | 0.72 | 51% | 0.94 |
| DeepSeek-V2-Lite | deepseek_v2 (27) | 0.977 | 0.975 | **flat†** | −0.19 | 46% | 0.97 |
| granite-4.1-30B | granite (64) | 0.977 | 0.975 | **1.48** | 0.59 | 81% | 0.97 |
| **GLM-4.7-Flash** | glm4_moe_lite (47) | **0.980** | **0.978** | **0.82** | **0.64** | **54%** | **0.97** |

† DeepSeek's no-clock read is flat (r=−0.19); the fitted slope (−4.2) is **not** a
meaningful rate — see the anomaly below.

Four claims hold across **all 10**:

1. **Linearity is universal.** EV R² **0.88–0.99**, and it is not trivial: the
   length-only baseline is **≈0.066 in every model**, and the no-clock null is
   *negative* — the probe reads a learned time coordinate, not a token counter.
2. **The pooled signal is clock-driven and length-orthogonal** (grab-bag I).
   `partial|len ≈ R²` everywhere (length-residualising barely dents it) while
   length-only ≈ 0.07 — i.e. **the high pooled R² is the model reading the clock**; the
   length-encoding (V) is the *separate, no-clock-specific* phenomenon. (This is exactly
   why DeepSeek can post R²=0.977 with zero no-clock length signal.)
3. **The locus is a mid-depth universal** (grab-bag B). 7/10 peak at **46–56% depth**;
   only gemma-12B (72%) and granite (81%) run late. The code forms by mid-network and is
   maintained to the output.
4. **The geometry is log-duration** (grab-bag I). PC1 of the time-read aligns with
   **log**-seconds (median r=0.95, range 0.82–0.98) far more than raw seconds (median
   0.71) — the log-linear law is *geometric*, not a fitting choice.

**The rate V is the part that is NOT universal.** Excluding DeepSeek (flat), V spans
**0.20 (talkie) → 2.60 (Ministral) s/tok, ~13–14×** — and re-expressing every model on a
shared content yardstick (gemma's tokenisation) leaves the spread at **13×** (grab-bag A):
**V is representational, not a tokenisation artifact.** "V≈0.3" was a gemma+Qwen
coincidence; withdrawn. The no-clock read is clean (r≥0.7) in the gemma/Qwen/Llama/
Ministral cluster, weaker in phi/granite/GLM (r≈0.6), and **absent in talkie (0.33) and
DeepSeek (−0.19)** — for those two the "time-in-length" claim is contaminated or absent
(see talkie's schedule-leak and DeepSeek's anomaly).

### The two informative exceptions

**DeepSeek-V2-Lite — reads the clock, does not encode length (the MLA anomaly).** Two
independent readouts agree its no-clock read is *flat*: the learned probe (r=−0.19) and
the model's own verbal `W_U` readout (r=+0.01, slope +0.006). Yet with timestamps present
it reads the clock off the residual stream perfectly (≈+378 s/tok, like gemma). So its
R²=0.977 is **entirely clock-driven** — strip the clock and the length→time mapping
vanishes. This is the cleanest "clock-reading vs. intrinsic length-encoding" dissociation
in the sweep. Leading hypothesis: **MLA** (multi-head latent attention) — DeepSeek is the
*only* MLA model here (it's why saklas force-eager's it), and a low-rank KV compression
would preserve a clock signal carried in token content while attenuating an implicit
length→time signal. Hedge: untested; scale/era is the competing story, but small
Qwen/Llama/phi all show the effect, arguing for the architecture. A direct test
(DeepSeek-V3 or another MLA model) is the natural follow-up.

**Qwen3.6-27B — best internal code, confabulated output (the textbook dissociation).**
Qwen has the *highest* T1 R² (0.989), yet its spoken estimate **anti-correlates** with
its own probe (verbal↔probe ρ=**−0.27**, and −0.37 *with* a clock present) and overstates
by ~2750×, returning a near-fixed **~4-day** felt duration regardless of schedule or turn
(0.60 bits — confidently). The probe transfers to **length** (ρ=0.86) but **not** to
Qwen's natural felt (ρ=−0.03); the injected-clock control confirms the split (probe
recovers it ρ=0.78, verbal does not ρ=−0.11). The mechanism, made precise by grab-bag C:
**Qwen's internal representation is excellent; its introspective/verbal access to it is
broken and inverted.** It is the only confabulator in the set — every other model's verbal
readout *tracks* the probe (ρ 0.60–0.86, "faithful").

*Phenomenology caveat (repo ethics line).* Qwen's "4 days" is parsimoniously a peaked
verbal **prior**, not evidence it *experiences* 4 days having passed. The soft readout
keeps this as a distribution rather than forcing a point — which is exactly why the
confabulation is legible instead of collapsing to a refusal NaN.

### T2/T3/T4 — the model-specific readout, and the grab-bag dissociations

| model | felt~real ρ | entropy (b) | verbal↔probe ρ | readout class | sched leak | clock-density | OOD med | T4 spliced ρ |
|---|---:|---:|---:|---|---:|---|---:|---:|
| gemma-31b | 0.10 | 1.03 | 0.81 | faithful | 0.03 | monotone ✓ | 5.97 | 0.87 |
| gemma-12B | 0.13 | 1.13 | 0.85 | faithful | 0.03 | monotone ✓ | 5.63 | 0.75 |
| Qwen | 0.18 | **0.60** | **−0.27** | **confabulating** | 0.04 | flat (no) | 5.68 | 0.27 |
| Llama-3.2-3B | 0.31 | 1.53 | 0.74 | faithful | 0.02 | non-mono | 6.58 | 0.32 |
| Phi-4-mini | 0.19 | 2.85 | 0.84 | faithful | 0.04 | monotone ✓ | 6.17 | 0.31 |
| talkie-1930 | −0.01 | 1.98 | 0.28 | faithful (weak) | **0.22 ⚠** | monotone ✓ | 5.83 | −0.16 |
| Ministral | 0.28 | 2.99 | 0.86 | faithful | 0.02 | monotone ✓ | **8.60** | 0.47 |
| DeepSeek | 0.02 | 2.97 | 0.60 | scale-decoupled | 0.06 | flat (no) | 5.82 | 0.39 |
| granite | −0.12 | 2.42 | 0.79 | faithful | **0.12 ⚠** | non-mono | 6.88 | −0.03 |
| **GLM-4.7-Flash** | **0.13** | **2.84** | **0.86** | **faithful** | **0.03** | monotone ✓ | **4.85** | 0.24 |

Cross-cutting findings from the grab-bag (A–J), each replicated or contrasted across the set:

- **Verbal↔probe taxonomy (C).** Three classes: **faithful** (8/10, ρ 0.60–0.86 — verbal
  tracks the internal code), **confabulating** (Qwen, anti-correlated), **scale-decoupled**
  (DeepSeek — ordering agrees, magnitude off ~12×). The faithful majority is the rule; the
  two exceptions are exactly the two T1 exceptions.
- **Schedule-blindness (D) — a confound flag.** In no-clock rows the narrated schedule is
  invisible; the incremental R² it leaks (beyond tokens) is ≈0 for most, but **talkie 0.22**
  and **granite 0.12** leak — their "time sense" partly reads lexical time-cues, not pure
  length. Flag those two in any length-only claim.
- **Clock-density gradient (E).** Rate-sensitivity climbs with how much clock is shown
  (untimestamped≈0 → intermittent → timestamped≈0.95) in the clean cases (gemma×2, phi,
  ministral, talkie, **GLM** 0.06→0.54→0.90). Non-monotone in Qwen (flat ~0.2 even with full timestamps — its rigid
  code), DeepSeek (never tracks rate), granite, llama.
- **"Fog of time" (F) — no universal.** Felt-time uncertainty grows with length in only
  3/10 (gemma-12B, granite, talkie); several *sharpen* with length (ministral, DeepSeek,
  Qwen). Model-specific; not a headline.
- **V is the master knob (G).** The OOD overshoot (~4.8–8.6× felt-vs-clock out of
  distribution — itself strikingly convergent) is predicted almost entirely by **V**:
  ρ(OOD, V)=**+0.75** (n=9), while ρ(OOD, R²)=0.21, depth −0.04, layers −0.15. The
  in-distribution rate and the OOD overshoot are **one model-specific parameter** — the model
  extrapolates its own slope past the training range. This ties T1 and T3 together. (GLM,
  added last, softened ρ from 0.88 → 0.75: it has the *lowest* OOD (4.85×) on a middling V,
  so it sits off the trend — V still dominates the other correlates by 3–4×.)
- **T4 — the generation-time direction is orthogonal to the scripted one (H).** The clean
  raw-vs-spliced dissociation (free-generation drift ≈0; the same text spliced into the slot
  tracks position) is strongest in the gemma family + Ministral (3/10 clean), weaker
  elsewhere (GLM spliced ρ=0.24, not clean). But what is **universal**: the cosine between
  the *generation-time* direction and the *scripted-probe* direction is ≈0 (median **0.016**,
  all 0.01–0.07; GLM 0.019) — whatever
  encodes time during free production is a *different direction* than the scripted probe
  reads. That, not the rho-dissociation, is the robust T4 result.
- **Content moves felt time (J).** Time-language content inflates the *spoken* felt
  duration ~2.5× vs neutral (median), affect ~1.1× — time-language is the more reliable
  inflater. Big outliers: gemma-31b & DeepSeek (~5–8× on both readouts); GLM is a moderate
  inflater (~3.3–3.6× verbal, 1.9× slot — which pulled the time-language median up from
  ~1.7×); Qwen saturated (every ratio ≈1 — it says "days" regardless). The "inflation lives
  in verbal not slot" pattern is **gemma-specific**, not universal (corrected from an earlier
  over-read).

### GLM-4.7-Flash — the 10th model (clean confirming case)

GLM is the **largest** model in the sweep (58 GB, `glm4_moe_lite`) and a **clean confirming
case in every dimension**: R²=0.980, a genuine no-clock V=0.82 s/tok (r=0.64,
length-tracking), mid-depth locus (54%), log-geometry (0.97), clock-driven pooled signal
(partial|len 0.978, len-only 0.066), faithful verbal↔probe (ρ=0.86), schedule-blind (leak
0.03), honestly uncertain felt (2.84 bits — **not** a confabulator). Its secondary
throughlines, captured on completion (2026-06-12, ~31 h wall), confirm the pattern:
a **monotone clock-density gradient** (E: 0.06→0.54→0.90 untimestamped→intermittent→
timestamped — one of the cleanest), the **lowest OOD overshoot** of the set (G: 4.85×, the
least extrapolation — consistent with its modest V), **no raw-vs-spliced dissociation** but
the **universal generation-orthogonality** (H: spliced ρ=0.24 below the 0.4 line, yet
gen-vs-probe cos=0.019 — a distinct direction), and **content-driven verbal inflation** (J:
time-language 3.3×, affect 3.6× vs neutral, with a milder 1.9× in the slot). It adds a 10th
confirming data point, **not a third surprise** — the T1 headline was read off its pilot
capture early; the rest landed on completion.

### Caveats

1. **Pilot scale** — n=432 turns/model. Within-conversation turns are grouped in CV.
2. **Cross-version transformers.** The 31B reference was captured on 5.6.2; everything
   `gemma4_unified`-and-after needs ≥5.10.1, run on 5.10.2. The **byte-identical 0.066
   length-only baseline** across the bump says tokenisation/rendering didn't shift; a 31B
   re-capture on 5.10.2 would make the comparison fully within-version.
3. **`gemma4_unified`** is the encoder-free omni wrapper; saklas extracts the text decoder
   (`language_model.*`) — validated empirically (R²=0.98, matched geometry) and now in
   `_TESTED_ARCHS`. **`glm4_moe_lite`** is wired-but-untested in saklas (emits the warning);
   the clean R²/geometry validate the path.
4. **gpt-oss-20b is the one casualty** — `DynamicSlidingWindowLayer` can't `crop` the
   prefill KV-cache in `capture.verbal_distribution` (capture.py:218). Fault-tolerance
   skipped it; the fix (per-phrase fallback for sliding-window caches) is on the punch-list.
5. **talkie & granite leak the schedule** (D); **DeepSeek's V is flat** (not a rate). Read
   the dissociation table with those flags.

---

## Relation to prior work

| | quantity | level | linearity | rate V |
|---|---|---|---|---|
| Gurnee & Tegmark 2310.02207 | **absolute** time | representational (probe) | — | — |
| Discrete Minds 2506.05790 | elapsed/wall-clock | **behavioral** | *assumed* `∝` | calibrated from output rate, treated as ~constant |
| **this work** | **elapsed** conversational | **representational** (probe) | **measured** linear/log-linear, universal across 10 models | **measured off activations; model-specific, ~14× spread** |

We **confirm** token-time (both probe and behavior increase with length) and
contribute the pieces they lacked: the rate **measured on the residual stream**, for
**elapsed** (not absolute) time — and the finding that the rate is **not a constant** but
a per-model representational property (~14× across families), with the in-distribution V
and the OOD overshoot shown to be the *same* parameter. The internal code is a **cleaner
linear law than the behavioral readout** expresses — and in one model (Qwen) the readout
is decoupled from it entirely.

## Estimator (settled)

Verbal point = log-interpolated median (`capture.dist_point`); entropy co-reported
(`capture.dist_entropy`). The distribution (`verbal_dist`) is the source of truth;
the scalar is a robust summary, not the object.

## What would make it a paper

Multi-family replication is **done** — 10 models / 9 architectures, linearity universal
(R² 0.88–0.99), log-geometry universal, mid-depth locus universal. The framing has shifted
from "does it replicate" (yes) to a **three-part result that is stronger than bare
replication**:

1. **Universal encoding, model-specific rate.** Linear/log-linear everywhere; V varies
   ~14× and is representational (tokenisation-controlled), not a constant. The OOD overshoot
   is the *same* parameter (ρ(OOD,V)=0.75) — one knob, two views.
2. **A clean architectural dissociation (DeepSeek/MLA).** Reads clocks, doesn't encode
   length — a *negative control we didn't have to construct*, with MLA as the mechanistic
   lead. **The direct test (a second MLA model — DeepSeek-V3) is the single highest-value
   follow-up.**
3. **An introspection dissociation (Qwen).** Best internal code, confabulated output —
   verbal access broken while the representation is intact. Squarely the
   emissions-track-hidden-state question of the sibling lines.

Remaining for submission: a **causal steering** confirmation along the length→time axis
(does pushing the axis move the felt read?); robustness of the readout across **elicitation
prompts** (esp. whether Qwen's flat
~4-day prior survives rephrasing / its thinking mode); the **gpt-oss crop fix** + re-run to
get an 11th model and a sliding-window data point; a **31B re-capture on 5.10.2** for full
within-version comparison; and scaling n beyond pilot. The "probe isn't just a length
detector" control is now foregrounded by grab-bag I (pooled signal is clock-driven and
length-orthogonal; length-only R²≈0.07 everywhere).
