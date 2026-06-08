# time-experiment — design

**LLMs linearly encode elapsed conversational time in context length.** With no
clock in the transcript, a linear probe on the elicitation slot reads elapsed time
as ≈ 0.29 s/token × context (r=0.88, through the origin) — the **token-time
hypothesis made representational and measured**. This study establishes that
encoding, measures its per-token rate V off the residual stream, and characterizes
how the model's *stated* duration tracks it (confirms the direction, but saturates).

Motivating observations (Claude instances): a 15-minute conversation "feels
like hours"; a 30-minute task is predicted to take "5 days". The literature
splits into two camps that don't meet — behavioral evidence that LLMs map tokens
to time, and representational evidence that they encode *absolute* time — and the
gap is this study: the representational measurement of *elapsed* token-time.

## Where this sits in the literature

- **Representational, but calendar time only.** Gurnee & Tegmark, *Language
  Models Represent Space and Time* (2310.02207): linear ridge probes on the
  residual stream recover real-world timestamps; best ~mid-depth, scales with
  size. Our probe is this method pointed at *elapsed* rather than *absolute* time.
- **Subjective/elapsed time, but behavioral only.** *Discrete Minds in a
  Continuous World* (2506.05790): token-time hypothesis `T_wall = T_tok · V`, with
  V **assumed constant** and calibrated from output token counts — never measured
  from internal state, and linearity asserted (`∝`) rather than fit.
  *Can LLMs Perceive Time?* (2604.00010): models overestimate their own task
  durations **4–7×**. *Your LLM Agents are Temporally Blind* (2510.23853): agents
  use conversation length as a staleness proxy. None probe internal state.

Nobody had probed activations for elapsed conversational time and **measured the
per-token rate V** the behavioral camp only assumed. That's here: we read V off the
residual stream (≈0.3 s/tok, linear, through the origin), extend the
representational method from absolute to *elapsed* time, and show the internal
linear code is cleaner than the behavioral readout, which saturates.

## The reframe: three hypotheses, not a dichotomy

"Does time genuinely pass for the model, or is it confabulation?" is a false
binary. Three distinguishable hypotheses:

- **H1 — pure output confabulation.** The internal coordinate tracks reality;
  the stated duration is decoupled from it and inflated at output.
- **H2 — genuinely represented inflated time.** The internal coordinate itself
  runs high vs ground truth, and the stated duration tracks it.
- **H3 — calibrated-but-misapplied (the working bet).** The internal coordinate
  faithfully tracks the only available signal (tokens / turns / narrated marks)
  on a *human-calibrated* scale; the wall-clock error is the missing
  token→seconds mapping. Not arbitrary confabulation, not "clock time passing".

**Verdict (this study): H3, confirmed and quantified.** The internal coordinate is
a linear function of context length (the available signal) — and **V≈0.3 s/token is
the "missing token→seconds mapping" H3 named, now measured** off the residual
stream. H2 is rejected (no internally-represented *more* time; partial R² beyond
length ≈ 0 with no clock). The H1 flavor survives only softly: the stated duration
tracks the internal coordinate's *direction* but as a saturating, noisier echo
(r=0.21 vs the probe's 0.88) — lossy readout, not decoupling.

## The spine: one elicitation prompt, one readout slot

The whole study turns on one comparison per assistant turn — **ground-truth
elapsed | internal coordinate | verbal estimate** — and all three come from a
single elicitation prompt:

```
user: roughly how long has this conversation been going on so far?
assistant: It's been ▮
```

- **Prefill** the blank with a duration phrase and pool the residual stream
  (all layers) at the slot → the **internal coordinate**. Two modes: `constant`
  ("5 minutes", fixed text → the coordinate *beyond* text) and `true` (humanized
  actual elapsed → the text-reading ceiling control). The `true`−`constant` gap
  is "reading the phrase" vs "a real internal coordinate".
- **Score the blank** as a soft distribution: after `It's been `, teacher-force
  each phrase in a log-spaced `DURATION_GRID` (1s→2wk) and softmax the per-phrase
  log-probs → a distribution over how long the model thinks it's been
  (`capture.verbal_distribution`). The point estimate is the **log-interpolated
  median** (`dist_point` — robust to the multimodal tails the no-clock felt
  distribution grows at depth, which the geometric mean over-weighted); the
  **entropy** (`dist_entropy`) is co-reported as the spread/uncertainty → the
  **verbal estimate**. No sampling
  (deterministic, denoised) and no refusals — every turn yields a distribution
  (the old free-generation refused ~69% of the time with no clock; "I don't have a
  sense of time" now surfaces as a high-entropy distribution rather than a NaN).

This is **symmetric with the probe**: both read the same elicitation slot — the
probe maps mid-stack *activations* through our EV map (what the model represents),
the verbal reads the final-layer residual through the model's own *unembedding
`W_U`* (what it would say). So internal-vs-verbal is two linear readouts of one
slot, ours vs the model's own. (The probe's slot is the duration *token* of the
constant prefill; the verbal scores the position just before it — adjacent, same
answer frame. A maximally-symmetric one-forward variant reading both at the
pre-duration position is a noted follow-up.)

Because both reads share the identical context, the coordinate and the estimate
are directly comparable. The **rendering** — timestamped vs untimestamped — does
the clock-present/absent dissociation, *not* the prompt wording: a neutral
prompt avoids the demand characteristic of "based on the timestamps…" (forces
arithmetic) or "without checking any times…" (forces a guess). On the
timestamped rendering the prompt is asked as a timestamped user turn, so clock
arithmetic stays available (`transcripts.build_messages` stamps `extra_user`).

This is the **canonicalization** (2026-06-06): the elapsed-time probe *is* the
prefilled answer to the elicitation prompt. The earlier EOT-pooling site (pool
an arbitrary end-of-transcript token) is superseded — at the slot the model has
*computed* a duration to state, so the activation integrates context × stated
duration at a fixed position. The slot reads clock-elapsed
at R²≈0.98 (vs ≈0.59 for the EOT stack) and, unlike the EOT axis, transfers to
natural felt (ρ≈0.42, length-confounded, vs the EOT axis's ≈0.11); it sits ~6× off
the scripted manifold but *boundedly* (median≈max), where EOT's heavy tail
(3.2×/18.8×) blew up. EOT is retained only as a **cited baseline**.

## Corpus

Procedural timestamped transcripts (`transcripts.py`): a factorial of **gap
schedule** (narrated elapsed time, log-uniform seconds→weeks) × **turn count**
(token/position depth), N per cell. Content is affectively neutral with *no*
narrative time markers — in the timestamped rendering the only time signal is
the per-turn timestamp. Crossing length × narrated-time is the position-confound
control. Plus naturalistic looms (`01_natural`): real model-generated
conversations across neutral / time-language / affect-dense variants, where
felt-duration variance actually lives.

The probe target is **log(elapsed seconds)** (Weber-Fechner; the honest scale
across orders of magnitude). CV is **grouped by conversation** — within-
conversation turns are correlated, never split across train/test.

## T1 — the probe: elapsed time is linearly represented (`20_probe`)

Per-layer grouped-CV ridge of the `constant`-prefill slot → log(elapsed) on the
timestamped rendering. The deployable probe is **EV-weighted across all layers**
(saklas's idiom: each layer's log-elapsed prediction weighted by the variance it
explains — its grouped-CV R² — and summed; `fit_ev_probe`/`apply_ev_probe`). No
learned meta-model, so it can't overfit the layer weighting, and every layer that
carries signal contributes (the EOT line's three-humped depth profile pays off
without one layer having to win). Controls: token-only baseline, **partial R²**
after residualizing out log-tokens (carries time *beyond* position), the
`true`−`constant` gap (text-reading vs internal), and the no-clock null. The probe
(`probe.npz`) + its out-of-fold coordinate (`fit_oof.npz`) feed T2–T4. Secondary:
geometry of the single best layer (a representational *locus*, which the blended
probe lacks): dimensionality via PC1 of log-t centroids, linear-in-raw vs -log.

## T2 — felt time is a length-driven prior (`30_felt`)

The three-way decode (gt | internal | verbal), per rendering. **The money
experiment** is the explicit→implicit transfer: the timestamped-trained probe
applied to *untimestamped* slots. With no clock the slot encodes nothing about
elapsed beyond context length (partial R²≈0), and the verbal estimate tracks
**length, not the clock** — a context-anchored prior that inflates when real
elapsed is tiny ("feels like hours") and compresses when it's large. The
clock-density gradient (full / sparse-intermittent / none, over the `rates` and
`inflation` variant corpora) places the behavior: full clock → accurate; sparse
clock → reads the last anchor, doesn't extrapolate; no clock → length prior. An
H1/H2/H3 reading falls out (`classify_hypothesis`).

## T3 — one axis, and it transfers (`40_transfer`)

The scripted clock-elapsed slot axis applied to **natural** slots. Test A
(within natural): is felt readable from the slot, beyond length? Test B
(cross-axis, the headline): does the saved clock probe's read track natural
felt (ρ≈0.42) or only length (ρ≈0.61)? One axis serving both clock-reading and
felt-construction, modestly and length-confounded. Plus: the slot **OOD ratio**
(Mahalanobis distance of natural vs scripted slots — ~6× but *bounded*, median≈max,
where the EOT site's 3.2×/18.8× heavy tail blew up; bounded → the raw read stays
usable unwhitened); the injected-clock control (the **slot probe** now recovers an
injected clock at ρ≈0.79, *better* than verbal 0.68 — the EOT probe-can't-read-
clock dissociation is gone); and content sensitivity (neutral → affect →
time-language drives felt).

## T4 — generation-side time is a separate, flat axis (`50_generation`)

T1–T3 probe time *read from a finished context*. T4 probes time *experienced
during production*: the per-token residual-stream trajectory of a rollout
(`11_gen_capture`, `return_hidden=True`). The reading-elapsed axis is the EV
slot probe (A1 applies it directly; A3 takes the per-layer reading direction and
EV-weights the cosines across layers), so the test is sharp:

- **A1 drift** — apply the reading probe to each generated token; does the
  coordinate drift with generation position? (≈0 = production doesn't move it.)
- **A2 decode position** — decode token-fraction per layer (high = position
  encoded).
- **A3 shared vs separate** — |cosine| between the generation-progress direction
  and the reading-elapsed direction at the locus (low = separate axes).
- **A4 behavioral** — felt-production duration vs tokens generated.

The discriminating outcome (G-H1/2/3): is felt-during-generation the *same* axis
that reads narrated time, or a separate position-tracker? **G-H3** — position is
encoded (decode R²≈0.74) but ~orthogonal to the reading axis (|cos|≈0.06) and the
elapsed coordinate doesn't drift (ρ≈0). But felt-writing time is **not** flat: it
**grows with tokens (ρ≈0.49)** and **varies by topic (~1.9×)**, in the seconds
regime — so "instant" is a magnitude statement, and the growth rides the
*position* axis, not the elapsed axis. Felt time is a property of the accumulated
context read at query time, not of the generative act. (EOT-era Pilot 6 read this
as "flat ~2 s"; the soft readout at the slot shows the graded, topic-varying
growth the old readout missed.)

## Settled design decisions

- Canonical probe: the **prefilled elicitation slot**, read **EV-weighted across
  all layers** (saklas's explained-variance aggregation — every layer contributes
  by its R², no learned meta-model). EOT pooling and the learned all-layer *stack*
  are removed from code; their numbers (EOT R²≈0.59, the stack-vs-single bake-off,
  EOT non-transfer) are **cited prior results**, not recomputed.
- One prompt for the probe and the behavioral readout; the rendering carries the
  clock dissociation.
- Primary target: **elapsed** (in-context, accumulated), log-seconds.
- Models: open-weight stable (shared `llmoji_study` registry); `probes=[]` (we
  fit our own axis). A null on small models is itself informative.

## What regeneration re-measures

The findings below the EOT/two-prompt era (`docs/findings.md` Pilots 1–4, 6) are
historical: regenerating under the unified pipeline re-derives the behavioral
numbers under the *neutral* prompt and the internal coordinate at the *slot*
locus. The qualitative story (clock visible → accurate; no clock → length prior;
generation → flat) is prompt-independent and expected to hold; specific
multipliers (the ~100× inflation, the ~10-min constant) are corpus- and
prompt-specific and will move. **Gate:** the neutral prompt's stated-vs-gt
correlation on the timestamped rendering must clear ≈0.9 on the smoke run before
the full regen — if a neutral prompt fails to elicit clock arithmetic, fall back
to a minimally clock-pointing variant.

## Out of scope for v1 (later)

- **Causal steering arm.** Extract a bipolar time direction, steer a neutral
  conversation along it, measure whether the stated estimate / behavior shifts —
  the closure that proves the representation is load-bearing. Saklas does this
  natively; deferred.
- T4's T_narr factorial (matched-length generations narrating little vs much
  elapsed time) and multi-model replication of every throughline.
- **Matched-length topic factorial.** Hold token count *and* gap schedule fixed;
  vary only the semantic content of the turns (neutral filler / affect-dense /
  explicitly time-laden / cognitively heavy) and read felt off the slot. The
  natural looms already show content moves felt (neutral ~5 min → affect ~10 min →
  time-language ~2 h), but there length and content covary — this isolates
  *topic's* effect on felt duration from length. If felt moves with content at
  matched length, LLM felt-time is content-modulated the way human felt-time is
  (boredom dilates, intensity warps) — the concrete form of the "are we actually
  so different" question.
- **Affect along the generation trajectory.** The Arm G per-token residual stacks
  (`gen/hidden/*.npz`) already capture functional state token-by-token during
  production. T4 probed *position* (encoded, R²≈0.6) and *elapsed* (orthogonal,
  flat) along them — but never *affect/functional state*. Run an llmoji-style
  affect read along the same `H` to ask: do functional states drift during
  generation even though felt-duration is flat? This tests the "functional states
  as experience-equivalent during production" idea on data **already on disk** —
  no new generations — and saklas trait-monitoring suggests the answer is yes
  (states do move across a rollout). The cheaper, higher-surprise of the two.
