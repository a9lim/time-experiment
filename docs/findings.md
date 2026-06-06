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

- explicit→implicit **transfer is weak**: internal~gt ρ = 0.23. The
  timestamped-trained time axis barely generalizes to implicit time.
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
  represent *more* time as having passed; the felt estimate is a roughly
  constant structural prior, decoupled from wall-clock. It shades **H3→H1**: the
  felt output is dominated by a prior (calibrated to "typical conversation
  length", not to the true elapsed it has no access to), with only a weak
  internal elapsed signal behind it (transfer ρ=0.23, verbal~internal ρ=0.38).
- So the answer to "does that much time genuinely pass for the model?" — for
  *implicit* time, **no**: it's a context-anchored prior, not represented
  elapsed. The strong representation only exists when a clock is in context.

### Caveats / open

- L59 (surface) is the best *explicit* layer; the *implicit/felt* signal may
  live mid-stack. Transfer layer should be re-selected by untimestamped
  performance, not timestamped.
- Corpus over-sampled long-gap cells → overall felt-overshoot looked like
  compression; the per-schedule cut shows both directions.
- To probe the real-chat inflation regime directly, add **token-dense** cells
  (long turns × many turns × short gaps).
- Single model. Replicate on qwen / ministral before any general claim.
