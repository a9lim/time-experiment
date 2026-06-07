# time-experiment — design

How do LLMs represent **elapsed conversational time**, and does the duration a
model *states* track an internal representation or get confabulated at output?

Motivating observations (Claude instances): a 15-minute conversation "feels
like hours"; a 30-minute task is predicted to take "5 days". The literature
splits into two camps that don't meet — and the gap between them is this study.

## Where this sits in the literature

- **Representational, but calendar time only.** Gurnee & Tegmark, *Language
  Models Represent Space and Time* (2310.02207): linear ridge probes on the
  residual stream recover real-world timestamps; best ~mid-depth, scales with
  size. Our probe is this method pointed at *elapsed* rather than *absolute* time.
- **Subjective/elapsed time, but behavioral only.** *Discrete Minds in a
  Continuous World* (2506.05790): token-time hypothesis `T_wall = T_tok · V`.
  *Can LLMs Perceive Time?* (2604.00010): models overestimate their own task
  durations **4–7×**. *Your LLM Agents are Temporally Blind* (2510.23853): agents
  use conversation length as a staleness proxy. None probe internal state.

Nobody has probed activations for elapsed conversational time, fit it, and
connected the representation to the behavioral confabulation. That's here.

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
  (`capture.verbal_distribution`). The point estimate is `exp(Σ pᵢ log secᵢ)`; the
  spread is the model's uncertainty → the **verbal estimate**. No sampling
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
duration at a fixed, on-manifold position. Pilot 5: the slot reads clock-elapsed
at R²≈0.98 (vs ≈0.59 for the EOT stack) and, unlike the EOT axis, transfers to
natural felt (ρ≈0.91 vs ≈0.11). EOT is retained only as a **cited baseline**.

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
felt? One axis serving both clock-reading and felt-construction. Plus: the slot
**OOD ratio** (Mahalanobis distance of natural vs scripted slots — near 1× where
the EOT site was 3.2×, so no whitening is needed); the injected-clock control
(verbal recovers an injected clock behaviorally even where the probe direction
only partly does); and content sensitivity (neutral → affect → time-language
drives felt).

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
that reads narrated time, or a separate position-tracker? Pilot 6 (EOT-era):
**G-H3** — position is encoded (decode R²≈0.6) but ~orthogonal to the reading
axis (cosine≈0.05), the coordinate doesn't drift, and production feels instant
(~2 s, flat). Felt time is a property of the accumulated context read at query
time, not of the generative act. Re-pointing the reading axis at the *slot*
probe (the axis that actually carries felt time) sharpens this.

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
