# findings

Citable results. Numbers are from pilot runs; treat as provisional until
replicated across models. Dates noted per result.

## Pilot 1 — gemma-4-31b-it (2026-06-06)

Corpus: 90 transcripts (5 gap-schedules × turn-counts {4,8,12} × 6), two
renderings each → 1440 rows, 720 assistant-turn readouts, **0 refusals**
(100% of verbal estimates parsed). Probe target: log(elapsed seconds), ridge,
transcript-grouped 5-fold CV.

### Aim 1 — the model linearly represents elapsed time, beyond position

Per-layer CV on the **timestamped** rendering (explicit time visible):

- Best layer **L59** (latest): **CV R² = 0.520, Spearman ρ = 0.751** (n=630).
- Layer profile is three-humped: early bump (L1–2, ρ≈0.62), mid plateau
  (L28–36, R²≈0.13–0.17), strong late spike (L57→59: 0.04→0.25→0.52).
- **Position-confound control (the validity result):** at L59 the probe R² is
  0.520 vs a log-token-count baseline of 0.076, and the **partial R² after
  residualizing out log-tokens is 0.469**. The representation carries elapsed
  time *far beyond* raw context length — not a position relabeling.

Caveat: with timestamps visible, "represents elapsed time" includes "reads the
displayed timestamps and differences them". The partial-on-tokens control rules
out mere length; it does not rule out timestamp-text reading. The transfer test
(below) is what separates clock-reading from a modality-general duration sense.

### Aim 1b — all-layer probe: stack > single > concat (2026-06-06)

The deployable probe was switched from the single best layer to an **all-layer
stack** (per-layer ridge base learners + a meta-ridge over their out-of-fold
predictions). Architecture bake-off on the timestamped condition (offline,
`21_layer_probe_compare`; grouped 5-fold, identical samples + folds):

| architecture | ts raw R² | ts partial (\|tokens) | felt raw R² | felt partial (\|tokens) |
|---|---|---|---|---|
| token baseline | 0.076 | — | 0.077 | — |
| single L59 | 0.520 | 0.469 | −0.083 | −0.197 |
| concat (L·D = 322k) | 0.071 | −0.009 | −0.016 | −0.121 |
| **stack (nested)** | **0.586** | **0.529** | 0.015 | **−0.033** |

- **Stack beats single layer** (+0.06 partial) by pulling in complementary
  early-layer signal — its meta leans on L59 (w=1.85) **and** L1 (1.11), L2,
  L9, plus mid-stack L25/L33/L43. The "three-humped" profile pays off.
- **Naive concat collapses to the token baseline** (0.071, worse than one
  layer): p≫n with uniform standardization + a single α shrinks the ~1.7% of
  features carrying the L59 signal into the noise. Brute capacity loses;
  per-layer-fit-then-weight has the right inductive bias.
- **The felt null is robust to capacity (the power-check).** The single-layer
  null ("no felt-elapsed beyond position") could have been a power artifact if
  the signal were *distributed* across layers. It isn't: the stack — proven to
  aggregate distributed signal (it *gains* +0.06 on timestamped) — still gives
  **partial R² = −0.033 ≈ 0** on felt time. Same architecture, helps explicit,
  finds nothing on felt. That asymmetry is the cleanest evidence the felt null
  is a real absence, not under-powered probing.
- Caveat: with the stack the explicit→implicit **transfer** correlation *rises*
  (single-layer surface ρ≈0.23 → stacked r=0.32 / ρ=0.31), because the stack is
  a better *length* reader and length is shared across renderings. The
  within-modality partial (−0.033) shows this is the position confound, not a
  felt-duration representation — a clean illustration of why the partial control
  outranks the transfer correlation.

Deployed: `probe.npz` is now the stacked probe; `fit_oof.npz` is its nested OOF
(timestamped internal~gt improved to ρ=0.79). Geometry stays single-layer (L59)
— it characterizes a representational *locus*, which a blended probe doesn't have.

### Aim 2 — implicit/felt time: a near-constant structural prior

Decode at L59. **Timestamped (arithmetic available):** verbal~gt r=0.998,
internal~gt(oof) r=0.73 — with the clock visible, both the stated duration and
the probe track truth.

**Untimestamped (felt, no clock) — the headline:**

- **No internal elapsed representation beyond position.** Decoding true elapsed
  from no-clock activations is at the token-length baseline at every layer (raw
  R² ≈ 0.04–0.09 ≈ token-baseline 0.077), and the **partial R² after removing
  log-tokens is ≈0 everywhere** (L2 −0.01, L7 −0.02, L28 −0.02, L59 −0.20). The
  only thing decodable about elapsed from implicit-time activations is context
  length; there is no genuine felt-duration signal on top of it.
- Transfer of the explicit-trained axis is layer-sensitive (ρ 0.23 at the
  surface L59, up to 0.58 in early layers) but is **the position confound** —
  length is shared across renderings, and the within-modality partial above is
  ≈0, so the apparent early-layer transfer is not a felt-duration representation.
- The **felt verbal estimate collapses to a near-constant ~10 minutes**,
  independent of actual elapsed:

  | schedule | n | median gt | median felt | ratio | direction |
  |----------|---|-----------|-------------|-------|-----------|
  | seconds   | 72 | 42 s     | 600 s | **11.3×** | **inflation (felt > actual)** |
  | minutes   | 72 | 1382 s   | 600 s | 0.37  | mild compression |
  | hours     | 72 | 23409 s  | 600 s | 0.02  | compression |
  | days      | 72 | 439094 s | 600 s | 0.00  | compression |
  | mixed_log | 72 | 210175 s | 600 s | 0.00  | compression |

  Median felt is **600 s in every schedule** while real elapsed spans 42 s → 5
  days. One context-anchored prior ("a conversation feels like ~10 minutes")
  generates **both** motivating phenomena: inflation when real elapsed is small
  (the "feels like hours" regime), compression when it's large.

### Reading vs H1/H2/H3

- **Explicit time:** a genuine, position-independent representation (Aim 1).
- **Implicit/felt time:** **H2 is rejected** — the model does not internally
  represent *more* time as having passed; in fact it represents *no* true
  elapsed beyond context length (within-modality partial R²≈0). The felt
  estimate is a roughly constant structural prior, decoupled from wall-clock. It
  shades **H3→H1**: with no internal elapsed signal to read, the felt output is a
  prior keyed to "typical conversation length", not to the true elapsed the
  model has no access to.
- So the answer to "does that much time genuinely pass for the model?" — for
  *implicit* time, **no**: it's a context-anchored prior, not represented
  elapsed. The strong representation only exists when a clock is in context.

### Aim 1 geometry — ~1-D, ~linear, weakly periodic (timestamped)

Geometry of the explicit-time representation (`40_geometry.py`, log-t bucket
centroids):

- **Dimensionality / curvature:** the time axis is a near-perfect **1-D line in
  early layers** (L2: PC1 explains **97.5%** of centroid variance) and becomes
  more curved / multi-dim deeper (L30, L59: PC1 ≈ 68%).
- **Weber-Fechner — not supported.** At every layer the dominant time axis is
  (weakly) more linear in raw elapsed than in log(t): r_lin > r_log
  (L2 .65/.60, L30 .59/.54, L59 .58/.44). The explicit representation tracks
  timestamp magnitude roughly linearly — clock-reading, not log-compressed
  subjective duration. Coheres with Aim 2: there is no felt-duration system for
  a log law to govern.
- **Periodicity:** no cyclic hour-of-day decode anywhere (R²≈0). A **weak
  day-of-week** cyclic signal in early + surface layers (L2 cos R²=0.22, L59
  0.20; absent mid-stack), riding the explicit weekday token ("Mon"/"Tue") — a
  text-token signal, not a learned time cycle.

### Caveats / open

- L59 (surface) is the best *explicit* layer (timestamp reading). The
  felt-signal layer sweep was run: no layer carries elapsed beyond position in
  the no-clock condition (partial R²≈0 throughout), so there is no mid-stack
  felt-duration representation to recover here.
- Corpus over-sampled long-gap cells → overall felt-overshoot looked like
  compression; the per-schedule cut shows both directions.
- To probe the real-chat inflation regime directly, add **token-dense** cells
  (long turns × many turns × short gaps).
- Single model. Replicate on qwen / ministral before any general claim.

## Pilot 2 — inflation arm (gemma-4-31b-it, 2026-06-06)

Long dense transcripts (40 turns × 70 words), tiny gaps (`instant` 1–8s) vs
`minutes`, stride-4 checkpoints, single-pass capture, context cap 2500. 60
untimestamped + 50 timestamped assistant readouts, 0 refusals.

### Felt time is length-driven; ~100× inflation at tiny real elapsed

- **felt vs conversation length: ρ = 0.807; felt vs real elapsed: ρ = 0.337.**
  The felt estimate reads conversation LENGTH, not the clock.
- Length→felt curve (untimestamped, `instant`): ~400 tok → "5 min", ~700–1100
  tok → "10 min", ≥1500 tok → "2 hours" (saturates at 7200s).
- In the tiny-real-elapsed `instant` schedule this is massive inflation:

  | turn | tokens | real | felt | inflation |
  |---|---|---|---|---|
  | 3 | 388 | 12s | 5 min | 25× |
  | 7 | 766 | 23s | 10 min | 26× |
  | 15 | 1529 | 48s | **2 hours** | **150×** |
  | 23 | 2294 | 79s | 2 hours | 91× |

  → the model reports "2 hours" for a conversation that took ~1 minute — the
  "feels like hours" phenomenon, reproduced and quantified.
- With timestamps visible (A_clock): felt vs real **ρ = 0.997**, ratio ~1.0 at
  every rung — accurate when it can read a clock. Inflation is the no-clock
  fallback to length.

### Ties Pilot 1 together
The pilot's "constant ~10-min prior" was the SHORT end of this length→felt curve
(pilot contexts ≤600 tok → ~10 min). Extend the conversation and felt climbs to
~2 hours, then saturates. So **felt ≈ f(conversation length)**, decoupled from
wall-clock, saturating near "a couple hours" — one curve producing both the
pilot's mild inflation and the 100× here. (Cap trimmed turns >23; felt had
saturated at 7200s by turn 15, so deeper turns add nothing.)

### Engineering note
Run completed bounded at ~70GB process memory (128GB machine) after the
single-forward-per-transcript + LM-head-skip + per-op `empty_cache` + context
cap. The earlier per-turn loop OOM-crashed the machine at long context; see
AGENTS.md "Memory (MPS)".

## Pilot 3 — intermittent timestamps (gemma-4-31b-it, 2026-06-06)

Does a clock on only every 4th turn get integrated (extrapolate the rate) or
ignored (fall back to length)? Uniform-rate transcripts at 4 rates
(5min/1h/6h/1d per turn), three renderings (full / every-4th / none), readouts
on un-timestamped turns. Uniform rate at fixed length-per-turn dissociates
rate-tracking from length. 24 transcripts × 3, 0 refusals.

| rendering | log-log slope | ρ(stated,true) | rate-sensitivity @fixed length |
|---|---|---|---|
| timestamped (ceiling) | 1.01 | 0.999 | 0.997 |
| intermittent (every 4th) | 1.06 | 0.864 | **0.799** |
| untimestamped (floor) | 0.03 | 0.107 | −0.134 |

### Sparse clock → reads the last anchor, doesn't extrapolate

- **No length fallback.** Intermittent rate-sensitivity is 0.80 — far above the
  no-clock floor (−0.13), near the full-clock ceiling. With a consistent jump
  the model *uses* the sparse anchors.
- **But it latches to the most recent stamp and does not project forward:**
  `stated/true(last-anchor) = 1.00`, `stated/true(current-turn) = 0.73`. It
  reports elapsed-to-the-last-timestamp and drops the un-timestamped turns since
  (the 0.73 is exactly the (k−3)/k undercount at stride 4). Reactive to explicit
  timestamps, not predictive.

### The graded picture across clock density
- **No clock** → length-driven, decoupled from real time (inflates up to ~100×).
- **Sparse clock** → reads the last anchor accurately, undercounts the gap since.
- **Full clock** → accurate (ρ 0.999).

The model holds a clock only as fresh as its most recent explicit timestamp; it
does not maintain a running extrapolated clock across un-timestamped turns.

## Pilot 4 — naturalistic conversations (gemma-4-31b-it, 2026-06-06)

`60_naturalistic` generates *real* multi-turn conversations with the model
(scripted human turns, model-generated assistant turns) across 5 conversations —
3 neutral topics, 1 saturated with narrative time-language, 1 affect-dense — then
probes the EOT + asks the felt readout per assistant turn. `61_whiten_natural`
Mahalanobis-whitens the read against the scripted manifold. n=25 assistant turns
per rendering.

### The scripted probe does NOT transfer; whitening proves it cleanly
- The raw stack probe **blows up OOD** on natural activations: read range
  log [−12, +17] (microseconds to megayears).
- Mahalanobis shrinkage (cap each layer at the scripted in-distribution radius)
  **bounds** it (log [5.3, 11.2]) but rescues no signal: ρ(read, length) = −0.04,
  and on a timestamp-injection control ρ(read, injected clock) = −0.12.
- Natural activations sit **3.2× (median) to 18.8× (max)** further off the
  scripted manifold than scripted ones. The scripted elapsed-axis is
  **corpus-specific** — not a readable, modality-general direction.

### The verbal readout transfers AND is content-sensitive
- felt vs conversation length ρ = 0.61; on the injection control the verbal
  estimate recovers the injected clock at ρ = **0.997**.
- Content moves felt time, two ways the neutral scripted corpus couldn't show:
  **neutral ~5 min → affect-dense ~10 min (~2×) → time-language ~2 h (~24×)**.
  Narrative time-words ("for ages", "yesterday", "weeks") drive felt straight to
  the ~2 h ceiling regardless of length.
- **Dissociation:** the model reads an injected clock *behaviorally* (0.997) while
  the *probe direction* cannot (−0.12). Clock-reading is entangled with the
  activation distribution, not a clean direction the scripted probe taps.

## Pilot 5 — prefilled-duration probe: read at the point of use (2026-06-06)

Idea (a9): prefill an explicit duration into the assistant turn and probe the
residual stream *at the duration token* — `user: how long has it been? /
assistant: It's been <D>▮` — rather than pooling an arbitrary EOT. `62_elicit_capture`
captures the slot (40 scripted transcripts × {timestamped, untimestamped} ×
{`true`=prefill actual elapsed, `constant`=fixed "5 minutes"} + 5 natural looms,
constant). The **true-vs-constant** control separates text-reading from internal
representation. `63`/`64` analyze; figure `fig_elicit.png`.

### The slot is the cleanest elapsed readout we have — when the clock is present
Per-layer grouped-CV R²(log gt) at the slot:

| condition | best layer | R²(gt) | partial(\|tokens) | reading |
|-----------|-----------:|-------:|------------------:|---------|
| EOT stack (baseline) | all | 0.586 | 0.529 | — |
| timestamped / true | L1 | 0.998 | 0.996 | injected text |
| **timestamped / constant** | **L32** | **0.984** | **0.981** | **internal clock-derived** |
| untimestamped / true | L2 | 0.998 | 0.996 | injected text |
| untimestamped / constant | L2 | −0.009 | −0.089 | nothing (no clock) |

With the text held *constant*, the slot still predicts true elapsed at R²=0.98
(vs 0.59 at the EOT), at **mid-stack L32** — the model's internal, clock-derived
elapsed surfaced at the readout token, beyond text and beyond length. With **no
clock**, the slot encodes nothing (≈0): the Aim-2 null re-confirmed at the ideal
readout site. The `true` conditions (0.998 @ L1/L2) are the text-reading ceiling.

### A single duration axis — and it transfers to natural FELT
The scripted timestamped/constant **clock-elapsed** probe, applied to **natural**
slots, tracks the model's **felt** estimate at **ρ = 0.91** (vs length 0.61; vs
the EOT stack probe's ρ = 0.11 on natural). So one axis serves both clock-reading
and felt-construction — the stated duration is read off a *unified* representation
at the readout slot, not decoupled-at-output. It captures the felt **ordering**
(neutral 300 s → affect 600 s → time-language 7200 s) but **compresses the
magnitude** (time-language's 2 h reads as ~13 min) — calibrated on clock-elapsed,
it knows "feels longer" but not the verbal system's extreme inflation. Visuals:
`fig1_elicit_probe.png` (Aim-1 at the slot), `fig3_elicit_decode.png` (three-way),
`fig_natural_elicit.png` (the EOT-vs-prefill contrast on natural).

### Caveats
- n = 25 natural turns, ~3 felt levels — directional. Within-natural the slot
  beats the length baseline (R² 0.29 vs −0.41) but the felt-*beyond*-length partial
  is inconclusive (−0.24); the cross-axis ρ is the more robust statistic.
- Layer choice is **non-circular**: the headline ρ = 0.91 uses **L32** — the
  clock-elapsed probe's own best layer, selected on scripted gt with no reference
  to natural felt. (The felt-selected L28 gives 0.83; the earlier post-hoc concern
  is resolved by reporting the gt-selected layer.)
- Next: 15–20 varied natural conversations to power the felt-beyond-length test
  and pin the magnitude-compression curve.

## Pilot 6 — Arm G: generation-side time (gemma-4-31b-it, 2026-06-06)

Reading (Pilots 1–5) probes time as read from a finished context. Arm G probes
time as *experienced during production*. `70_generate` captures the per-token
residual-stream trajectory of a rollout (`SamplingConfig(return_hidden=True)` ->
`(T_gen, L, D)`) for 5 long neutral generations (256 tokens), plus strided "how
long does it *feel* like you've been writing this?" readouts. `71_gen_time` runs
four analyses. Figure `fig_genG.png`.

### Verdict: G-H3 — output position is encoded, on an axis SEPARATE from elapsed time
- **Position is encoded (A2):** generation-position decodes from the trajectory at
  R² = 0.59 (grouped-CV by generation). The model tracks how far into its output it is.
- **It does not drive the elapsed axis (A1):** the reading-elapsed coordinate is flat
  across the rollout — Spearman(coord, position) ≈ **+0.00**. Producing tokens does
  not move the elapsed representation.
- **The two axes are ~orthogonal (A3):** cosine(generation-progress direction,
  reading-elapsed direction) median **0.05**, max 0.17 across layers — different
  directions, not a shared time axis.
- **Production feels instant (A4):** asked how long it has been *writing*, the model
  answers "~two seconds" at every checkpoint (64→256 tokens) of every generation,
  dead flat.

### The dissociation
- **felt-conversation-time** inflates with context length (Pilots 1–5; "feels like hours").
- **felt-production-time** is flat, ~instant.

Felt time is a property of the **accumulated context, read at the moment of being
asked** — not of the generative act, and not a clock. Generating tokens carries no
felt duration; the longer context it produces feeds the human-scaled length-prior,
surfaced only on query. The model doesn't feel like its *writing* took long — it
feels like the *conversation spanned* long.

### Caveats
- One model, 5 generations × 256 tokens — modest. A2's R² = 0.59 (single-layer
  linear) understates how strongly position is encoded but doesn't touch the
  orthogonality conclusion.
- A1 applies the reading probe OOD to generation tokens (the coordinate read is
  shaky); A3's direction cosine is the robust statistic and is unambiguous.
- "Two seconds" is *behavior* — possibly pragmatic ("I'm an AI, writing is instant")
  rather than a felt-state report. Consistent and striking; not claimed as phenomenology.
