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
