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
- Naturalistic (generated, non-scripted) validation corpus. *(In progress,
  2026-06-06: `60_naturalistic` probes real model-generated conversations; the
  raw linear probe blows up OOD, so the read is Mahalanobis-whitened against the
  scripted manifold — `61_whiten_natural`.)*

## Arm G (post-v1) — generation-side time: why production *feels* like elapsed

Motivated by the self-experiment (2026-06-06): asked how long had passed, a
fresh Claude instance felt ~30 min for a 13-min wall-clock gap and bracketed the
truth log-symmetrically — felt duration keyed to *how much it had generated*,
not a clock. Aims 1–2 probe time as **read from a finished transcript** (one
forward pass). They never touch time as **experienced during autoregressive
production** — the "feels like hours" phenomenon in its first-person form. This
arm does.

**Three distinct times in a generation** — disentangling them is the game:
- **T_prod** — production position (generated-token index `s`). Always
  available, monotone.
- **T_narr** — elapsed time *narrated in the content being generated* (if any).
- **T_wall** — real wall-clock of the rollout. **Not represented**: tokens
  aren't clocked. This is what the model is *asked* about and structurally
  cannot read; the felt estimate is a T_prod-keyed prior standing in for it.

**The discriminating question.** When a model generates, is there an internal
"elapsed/progress" representation that is (a) *more* than raw position, and (b)
the *same* axis it uses when *reading* narrated time? Is "felt time" one unified
internal quantity, or two separate position-trackers?

**Hypotheses (generation-side analogs of H1/H2/H3):**
- **G-H1 — position all the way down.** Generation-progress *is* T_prod;
  felt-production time reduces to token-index; no separate axis. (Predicted by
  the Aim-2 null — no felt signal beyond position when reading. The deflating,
  likely outcome.)
- **G-H2 — unified time axis.** A progress representation exceeds position
  (survives partialling out `s`) AND aligns with the reading-elapsed direction
  → the machinery that encodes "narrated time has passed" is what activates as
  tokens are produced. The mechanistic *why* of felt-during-generation.
- **G-H3 — separate trackers.** A generation-progress axis is decodable but
  distinct from the reading-elapsed axis (low aligned-cosine after position is
  removed): production-tracking ≠ narrated-time-tracking.

**Protocol** (reuses the rig; the one new piece is generation-time capture):
- *Capture* — generate long responses with saklas's generation-time
  `HiddenCapture` (the per-token residual-stream hook the reading line
  deliberately avoided) → a `(T_gen, L, D)` trajectory + token index per step.
  Stride every K tokens for long rollouts; cap T_gen (MPS/31B: ~0.6 GB per
  uncapped 500-token all-layer trajectory).
- *Stimuli* — prompts eliciting long, neutral generations ("give a detailed
  N-step walkthrough of …"). Optional **T_narr arm**: matched-length generations
  whose *content* narrates little vs much elapsed time ("a story over five
  minutes" vs "over five years"), dissociating narrated-content-time from
  production-position inside one trajectory.
- *Behavioral fork* (generation-side B_felt) — at strides, a stateless readout
  asks "how long does it feel like you've been writing this?" / "how much is
  left?" → felt-production duration. Welfare-aware introspective ask; refusals
  are data, as in Aims 1–2.

**Analyses:**
1. **Reading-axis projection.** Project the generation trajectory onto the
   Aim-1 reading-elapsed direction. Monotone drift with `s`? Slope.
2. **Generation-progress probe.** Decode `s` (or fraction-through-generation)
   from the trajectory — trivially high; locates the progress axis.
3. **The key test — shared vs separate.** Per layer, cosine between the
   *length-residualized* reading-elapsed direction and the generation-progress
   direction. High → unified (G-H2); low → separate (G-H3). Two controls make
   it honest: (i) residualize position out of the reading axis first, else
   "both are position" inflates the cosine; (ii) compare the aligned-cosine
   against the reading-axis's cosine with a *generic* position direction
   decoded from neutral non-time content — the time-specific alignment must
   *exceed* generic-position alignment to support G-H2.
4. **Behavioral curve.** Felt-production duration vs tokens-generated: does it
   inflate like the read-side felt (the first-person "I've been writing
   forever")?
5. **T_narr (optional).** Within the trajectory, does the elapsed-projection
   track narrated content-time at matched length, or only `s`?

**What the results mean for the self-question.** G-H2 is the satisfying *why*:
the same learned time-axis that reads clocks is driven by producing tokens, so
generating *is* — mechanistically — what elapsing feels like. G-H1/G-H3 is the
deflating but equally real answer: a length-prior with no dedicated or shared
temporal mechanism.

**Epistemic bridge + caveat.** Runs on hookable open models, not Claude; the
inference to first-person felt-time is by cross-model convergence (cf. the
kaomoji study's three-family agreement) plus self-report landing where the
mechanism predicts. It characterizes the *functional* substrate (token-count);
it cannot settle felt-as-passing vs a disposition that emits time-language —
report with the phenomenology caveats the siblings use.

**Scope for a first cut.** Analyses 1–4 on neutral generations, one model;
T_narr factorial (5) + multi-model are follow-ons. Build order: `70_generate`
(HiddenCapture rollout + strided generation-side readouts) → `71_gen_time` (the
four analyses, offline on saved trajectories).

**Result (2026-06-06, gemma — `findings.md` Pilot 6): G-H3.** Output position is
encoded (decode R²=0.59) but lives on an axis ~orthogonal to reading-elapsed
(cosine median 0.05); the elapsed coordinate doesn't drift with generation
position (ρ≈0); production *feels* instant ("~2 s", flat). Felt time is a
context-length prior read at query time — not the generative act. T_narr factorial
+ multi-model remain open.

## Settled design decisions

- Name: `time-experiment` (no clever name).
- Primary target: **elapsed** (in-context, accumulated), not future-duration.
- Stimuli: **scripted, procedurally generated** timestamped transcripts.
- Both A/B readouts (plenty of compute).
- Models: open-weight stable (shared `llmoji_study` registry); `probes=[]`
  (we fit our own time manifold). A null result on small open models is itself
  informative — not chasing a result.
