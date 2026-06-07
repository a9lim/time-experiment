# findings

Citable results, organized by the four throughlines. Numbers are from pilot runs
on **gemma-4-31b-it** (2026-06-06); treat as provisional until replicated across
models.

**Provenance + status.** The pipeline was consolidated (2026-06-06) to canonicalize
the elapsed-time probe as the **prefilled elicitation slot**, read by an
**EV-weighted all-layer probe** (saklas's explained-variance aggregation — every
layer weighted by its own R², no learned meta-model; one prompt, one probe). The
slot results (Pilot 5) are the canonical line; the EOT
results (Pilots 1–4) are the superseded baseline the slot beats, kept as cited
history. Pilots 2/3/6 were measured at the EOT site with the old A_clock/B_felt
prompts. **Regenerating under the unified pipeline** re-derives the behavioral
numbers under the neutral prompt and the internal coordinate at the slot locus
(see "What regeneration re-measures"); the qualitative story is expected to hold,
specific multipliers will move.

---

## T1 — the model represents elapsed time, read cleanest at the slot

**Canonical (Pilot 5, prefilled-duration slot).** Prefill an explicit duration
into the assistant turn and probe at the duration token. Per-layer grouped-CV
R²(log gt) on scripted transcripts:

| condition | best layer | R²(gt) | partial(\|tokens) | reading |
|-----------|-----------:|-------:|------------------:|---------|
| EOT stack (baseline) | all | 0.586 | 0.529 | superseded site |
| timestamped / true | L1 | 0.998 | 0.996 | injected text (ceiling) |
| **timestamped / constant** | **L32** | **0.984** | **0.981** | **internal clock-derived** |
| untimestamped / true | L2 | 0.998 | 0.996 | injected text |
| untimestamped / constant | L2 | −0.009 | −0.089 | nothing (no clock) |

With the text held **constant**, the slot still predicts true elapsed at R²=0.98,
at **mid-stack L32** — the model's internal, clock-derived elapsed surfaced at the
readout token, beyond text (the `true`−`constant` gap) and beyond length (partial
0.98). With **no clock** the slot encodes nothing (≈0): the Aim-2 null at the
ideal readout site. Layer choice is non-circular (L32 is the gt-selected best).

**EOT baseline (Pilot 1/1b, cited, superseded).** The original EOT-pooled probe:
single best layer L59 CV R²=0.520 (ρ=0.751); an all-layer nested stack reached
0.586 (partial 0.529). The architecture bake-off (single vs concat vs stack)
showed stack > single > naive-concat. All superseded by the slot — `20_probe`
fits a single layer and reports the slot's own controls, not the EOT contrast.

**Geometry (Pilot 1, EOT L59 locus; slot locus will differ).** The explicit-time
axis was ~1-D in early layers (L2 PC1 97.5%), more curved deep (PC1≈68%), and
weakly more linear in *raw* timestamp magnitude than in log-t — clock-reading,
not log-compressed subjective duration. No cyclic hour-of-day decode; a weak
day-of-week signal riding the weekday token. `20_probe` recomputes a compact
version at the slot locus.

---

## T2 — felt time is a length-driven prior, not a represented quantity

**No internal felt-elapsed beyond length (Pilot 1).** Decoding true elapsed from
*no-clock* activations is at the token-length baseline at every layer, and the
**partial R² after removing log-tokens is ≈0 everywhere** (L2 −0.01 … L59 −0.20).
The felt null is robust to capacity: the nested stack — which *gains* on explicit
time — still gives partial ≈ 0 on felt. A real absence, not under-powered probing.

**The felt verbal estimate is a near-constant prior (Pilot 1).** In short pilot
contexts (≤600 tok) the felt estimate collapses to a **near-constant ~10 minutes**
independent of actual elapsed (median felt 600 s in every schedule while real
elapsed spans 42 s → 5 days) — inflation when real elapsed is small ("feels like
hours"), compression when large.

**Felt ≈ f(conversation length), saturating (Pilot 2, inflation arm).** Long dense
transcripts with tiny real gaps: **felt vs length ρ=0.807; felt vs real elapsed
ρ=0.337**. The length→felt curve: ~400 tok → "5 min", ~1500 tok → "2 hours"
(saturates at 7200 s). In the tiny-real-elapsed regime this is massive inflation:

| turn | tokens | real | felt | inflation |
|---|---|---|---|---|
| 3 | 388 | 12 s | 5 min | 25× |
| 15 | 1529 | 48 s | **2 hours** | **150×** |
| 23 | 2294 | 79 s | 2 hours | 91× |

The pilot's "constant ~10-min prior" is the short end of this one curve. With
timestamps visible (A_clock) felt vs real ρ=0.997 — accurate when it can read a
clock; inflation is the no-clock fallback to length.

**Clock-density gradient (Pilot 3, intermittent).** A clock on every 4th turn:
rate-sensitivity at fixed length = 0.80 (far above the no-clock floor −0.13, near
the full-clock ceiling 0.997) — the model *uses* sparse anchors. But it latches
to the most recent stamp and does **not** extrapolate: `stated/true(last-anchor)`
= 1.00, `stated/true(current-turn)` = 0.73 (the (k−3)/k undercount at stride 4).
Graded picture: **no clock** → length prior (up to ~100× inflation); **sparse
clock** → reads the last anchor, undercounts since; **full clock** → accurate.

**Reading.** H2 is rejected — the model does not internally represent *more* time;
for implicit time it represents *no* true elapsed beyond length (partial ≈ 0). The
felt estimate is a context-anchored prior decoupled from wall-clock. Shades
**H3→H1**: with no internal elapsed signal to read, the felt output is a prior
keyed to typical conversation length, not the true elapsed the model can't access.

---

## T3 — one duration axis serves clock-reading and felt, and it transfers

**The slot axis transfers to natural felt (Pilot 5).** The scripted
timestamped/constant **clock-elapsed** probe (L32, gt-selected — non-circular),
applied to **natural** conversation slots, tracks the model's **felt** estimate at
**ρ=0.91** (vs length 0.61; vs the EOT stack probe's 0.11 on natural). One axis
serves both clock-reading and felt-construction — the stated duration is read off
a *unified* representation, not decoupled-at-output. It captures the felt
**ordering** (neutral 300 s → affect 600 s → time-language 7200 s) but
**compresses the magnitude** (time-language's 2 h reads as ~13 min): calibrated on
clock-elapsed, it knows "feels longer" but not the verbal system's extreme
inflation. n=25 natural turns, ~3 felt levels — directional.

**The slot is on-manifold; the EOT axis was not (Pilot 4/5).** The raw EOT stack
probe blows up OOD on natural activations (read range log [−12, +17]); Mahalanobis
shrinkage bounds it but rescues no signal (ρ(read, length) = −0.04, ρ(read,
injected clock) = −0.12). Natural EOT activations sit **3.2× (median) / 18.8×
(max)** off the scripted manifold — the EOT elapsed-axis is corpus-specific. The
slot site is far tamer (its whole advantage), so `40_transfer` reports the slot
OOD ratio and skips whitening when it's ≈1×.

**Behavioral-vs-probe dissociation (Pilot 4).** On an injected-clock control the
**verbal** estimate recovers the injected clock at ρ=0.997 while the EOT **probe
direction** cannot (−0.12) — clock-reading is entangled with the activation
distribution, not a clean EOT direction. Content moves felt (neutral ~5 min →
affect ~10 min → time-language ~2 h); narrative time-words drive felt to the ~2 h
ceiling regardless of length.

---

## T4 — generation-side time is a separate, flat axis (Pilot 6, G-H3)

Reading (T1–T3) probes time read from a finished context; T4 probes time
*experienced during production* — the per-token trajectory of a rollout.

- **Position is encoded:** generation-position decodes at R²=0.59 (grouped-CV).
- **It does not drive the elapsed axis:** the reading-elapsed coordinate is flat
  across the rollout (Spearman(coord, position) ≈ +0.00).
- **The two axes are ~orthogonal:** cosine(gen-progress, reading-elapsed) median
  **0.05**, max 0.17 — different directions, not a shared time axis.
- **Production feels instant:** asked how long it's been *writing*, the model
  answers "~two seconds" at every checkpoint (64→256 tokens), dead flat.

The dissociation: **felt-conversation-time** inflates with context length ("feels
like hours"); **felt-production-time** is flat, ~instant. Felt time is a property
of the accumulated context read at the moment of being asked — not of the
generative act, and not a clock. (EOT-era reading axis; `50_generation` re-points
it at the slot probe — the axis that actually carries felt time — which sharpens
the orthogonality test rather than weakening it.)

"Two seconds" is *behavior*, possibly pragmatic ("I'm an AI, writing is instant")
rather than a felt-state report — consistent and striking; not claimed as
phenomenology.

---

## What regeneration re-measures

Under the unified (slot + neutral-prompt) pipeline:

- **T1** is already slot-based (Pilot 5); regen confirms R²/partial/locus on the
  full corpus and adds the slot-locus geometry.
- **T2 numbers move:** the inflation multipliers (150×, the ~10-min constant) and
  the felt-overshoot factors are prompt- and corpus-specific; the neutral prompt
  re-measures them. The decode internal coordinate is now the EV-weighted all-layer
  slot OOF (was the EOT stack), so internal~gt shifts. The clock-density gradient
  re-runs on the `rates`/`inflation` corpora.
- **T3** re-measures the transfer ρ with the saved EV all-layer probe and reports
  the slot OOD ratio (expected ≈1×, replacing the whitening apparatus).
- **T4** re-points the reading axis at the slot probe; G-H3 expected to hold (and
  be more damning, since the comparison axis now carries felt time).
- **Gate:** on the smoke run, the neutral prompt's stated-vs-gt correlation on the
  timestamped rendering must clear ≈0.9 (old A_clock was 0.997–0.999). If a
  neutral prompt fails to elicit clock arithmetic, fall back to a minimally
  clock-pointing variant before the full regen.
