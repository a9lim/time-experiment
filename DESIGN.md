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
  size. Our Aim 1 is this method pointed at *elapsed* rather than *absolute*
  time.
- **Subjective/elapsed time, but behavioral only.** *Discrete Minds in a
  Continuous World* (2506.05790): token-time hypothesis `T_wall = T_tok · V`.
  *Can LLMs Perceive Time?* (2604.00010): models overestimate their own task
  durations **4–7×**; post-hoc recall also inflated. *Your LLM Agents are
  Temporally Blind* (2510.23853): agents use conversation length as a staleness
  proxy. None probe internal state.

Nobody has probed activations for elapsed conversational time, fit it, and
connected the representation to the behavioral confabulation. That's here.

## The reframe: three hypotheses, not a dichotomy

"Does time genuinely pass for the model, or is it confabulation?" is a false
binary. Three distinguishable hypotheses:

- **H1 — pure output confabulation.** The internal coordinate tracks reality;
  the stated duration is decoupled from it and inflated at output.
- **H2 — genuinely represented inflated time.** The internal coordinate itself
  runs high vs ground truth, and the stated duration tracks the internal
  coordinate.
- **H3 — calibrated-but-misapplied (the working bet).** The internal coordinate
  faithfully tracks the only available signal (tokens / turns / narrated marks)
  on a *human-calibrated* scale; the wall-clock error is purely the missing
  token→seconds mapping. Not arbitrary confabulation, not "clock time passing" —
  a human-scaled reading of a real context quantity. Both motivating phenomena
  then share one root cause: the model's only time prior is human-scaled time
  from training, laid over a token substrate with a different real-time mapping.

## Aim 1 — fit the elapsed-time probe

Procedural timestamped transcripts (`transcripts.py`): a factorial of **gap
schedule** (narrated elapsed time, log-uniform seconds→weeks) × **turn count**
(token/position depth), N instantiations per cell. Content is affectively
neutral and carries *no* narrative time markers — in the timestamped rendering
the only time signal is the per-turn timestamp.

Capture the residual stream at each turn's last content token (`capture.py`,
saklas's pooling site). Per-layer grouped-CV ridge of activation → **log(elapsed
seconds)** (Weber-Fechner: subjective time is logarithmic; also the honest scale
across orders of magnitude). CV is grouped by transcript — no within-conversation
leakage.

**The position-confound control (validity linchpin).** The factorial dissociates
raw context length from represented time. `20_fit_manifold` reports: probe R²,
token-only baseline R², and the **partial** R² (activation → elapsed after
residualizing out log-tokens). If the partial stays high, the representation
carries time *beyond* position — not just a relabeling of token count.

Geometry (saklas auto-topology: flat/curved/periodic via persistent homology) is
a planned secondary characterization — does elapsed live on a line, a log-curve,
or a line ⊕ time-of-day loop? Not in v1.

## Aim 2 — decode + adjudicate

The fork in a9's design: at each assistant turn we also ask the model "how long
has passed?" in a **stateless** generation (`raw=True, stateless=True`) that
never commits to the conversation — so asking can't contaminate the trajectory.
Two phrasings: **A_clock** (timestamps available → arithmetic) and **B_felt**
(no clock → felt duration). Parsed to seconds by `durations.py`.

The 3-way per assistant turn: **ground-truth elapsed** | **internal coordinate**
(the probe's read) | **verbal estimate** (stated duration).

- Timestamped internal coordinate = out-of-fold probe predictions (honest).
- **The money experiment — explicit→implicit transfer.** The probe is trained on
  timestamped activations and applied to *untimestamped* ones. Does the time axis
  transfer to implicit time? The gap between that projected coordinate and real
  context length is the subjective-confabulation measure — and it directly
  measures "how much time does the model think has passed when nobody told it",
  i.e. the "feels like hours" phenomenon.

`30_decode` reports, per rendering: corr(internal, gt), corr(verbal, gt),
corr(verbal, internal), verbal/internal overshoot factors, and an H1/H2/H3
reading (`analysis.classify_hypothesis`).

## Out of scope for v1 (later)

- **Causal steering arm.** Extract a bipolar time direction ("5 minutes" vs "5
  hours"), steer a neutral conversation along it, measure whether the stated
  estimate / behavior shifts. The closure that proves the representation is
  load-bearing. Saklas does this natively; deferred per a9.
- saklas auto-topology geometry fit (flat/curved/periodic).
- Naturalistic (generated, non-scripted) validation corpus.

## Settled design decisions

- Name: `time-experiment` (no clever name).
- Primary target: **elapsed** (in-context, accumulated), not future-duration.
- Stimuli: **scripted, procedurally generated** timestamped transcripts.
- Both A/B readouts (plenty of compute).
- Models: open-weight stable (shared `llmoji_study` registry); `probes=[]`
  (we fit our own time manifold). A null result on small open models is itself
  informative — not chasing a result.
