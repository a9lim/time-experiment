# T4b — whose tokens count? Self-style × attributed role

**Status:** experiment concept; specified, not implemented.

**Question.** With no clock available, does the elapsed-time read assign a
different effective rate to context depending on (1) whether the text occupies
the user or assistant role and (2) whether its style matches the reader model's
own output distribution?

This is a source-*cue* experiment, not a provenance experiment. A standard
causal LM receives token ids, positions, masks, and the chat-template role
serialization. If the same tokens are replayed in the same serialized context,
there is no additional fact available about which process produced them.
"Self-style" below therefore means **distributional compatibility with the
reader model**, while "role" means **speaker identity attributed by the chat
template and conversational position**.

The concise hypothesis is that the no-clock prior may not count every context
token equally:

```text
felt elapsed = accumulated context weighted by attributed speaker and style
```

The experiment asks whether that weighting exists and which cue carries it.

---

## 1. Why T4 does not already answer this

T4 compares two real but composite conditions:

- scripted context alternates fixed, hand-written user and assistant text;
- generation context is one long model-generated assistant response.

For the Gemma reference, the in-domain spliced read grows at approximately
`0.06 s/token`, versus scripted `V≈0.29 s/token`. That fivefold difference is
evidence that the contexts are treated differently, but source style, role
composition, message count, turn boundaries, topic, and dialogue-versus-monologue
structure all change together. Across the ten-model sweep, clean spliced growth
is strongest in the Gemma family and Ministral; the robust T4 result is instead
the near-orthogonality of generation progress and scripted elapsed directions.

Until T4b resolves the attribution, "self-context is discounted" should be read
as the narrower observation:

> Model-generated, single-assistant-turn context accrues less felt time under
> the Gemma probe than mixed scripted dialogue; self-style, role, and structure
> are not yet separated.

The existing `0.06` versus `0.29 s/token` contrast motivates T4b but is **not**
an effect-size prior for it.

---

## 2. Hypotheses and claim boundaries

Let `V` denote the effective seconds per payload token recovered from the
canonical no-clock slot read. The primary outcome remains predicted log-seconds;
`V` is the interpretable secondary summary.

- **H-role:** `V_assistant != V_user` for the same literal payload. The model
  weights text differently according to its attributed speaker role.
- **H-style:** `V_self-style != V_other-style` within a fixed role. The model's
  time prior uses distributional familiarity beyond the explicit role tag.
- **H-interaction:** the style effect depends on role. The natural prediction is
  cue congruence: self-style assistant text receives the strongest discount,
  while self-style user text and other-style assistant text conflict.
- **H-token-only:** after token count and framing are controlled, neither cue
  affects the time read.
- **Clock gate:** with an explicit matched clock, the source-role differences
  collapse and all cells recover the shown elapsed time.

The study does **not** claim:

- recognition of historical or causal authorship;
- token-by-token source classification (the unit is a passage-level context);
- phenomenology from a probe or stated duration alone;
- a universal direction or effect size across model families.

---

## 3. The core factorial

For a reader model, every payload belongs to one of four cells:

| payload style relative to reader | user role | assistant role |
|---|---:|---:|
| reader model's own style | SU | SA |
| another model's style | OU | OA |

Every literal payload prefix is replayed in both roles. Source style varies only
between matched passages generated from the same fact sheet; role varies within
the exact same passage.

The role and style cues can therefore agree or conflict:

- **congruent:** self-style/assistant and other-style/user;
- **incongruent:** self-style/user and other-style/assistant.

An interaction is evidence that the time prior integrates these cues rather
than applying one fixed rate per role or prose distribution.

### Primary paired estimands

For an identical payload `X` at checkpoint length `L`:

```text
delta_role(X, L)
    = T(X as assistant, L) - T(X as user, L)
```

This is the cleanest role contrast because the payload tokens are identical.

Within role `r`:

```text
delta_style(r, L)
    = E[T(self-style, r, L) - T(other-style, r, L)]
```

The cue interaction is:

```text
delta_interaction
    = delta_role(self-style) - delta_role(other-style)
```

Negative values mean the first-named condition accrues less predicted time.

---

## 4. Reciprocal models: the self-style discriminator

Run two models as both generators and readers. The first pair should be
`gemma-4-12B-it` and `Ministral-3-14B`: both are tractable on the MPS machine,
span architecture families, and have relatively clean existing T4 behavior.
Gemma-31B is the reference replication after the design lands.

Both generators receive the same neutral fact sheets and generation contract.
Both readers then see both passage banks:

| reader | Gemma passage | Ministral passage |
|---|---|---|
| Gemma | self-style | other-style |
| Ministral | other-style | self-style |

This reciprocity is load-bearing. A fixed generator effect is not self-style:

- if both readers discount Gemma prose, Gemma prose has a generic property;
- if each reader discounts its **own** source, the sign follows reader identity
  and supports relational self-style recognition.

A human/procedural bank is a useful later anchor for model-versus-human style,
but is not needed for the first self-versus-other discrimination.

---

## 5. Passage corpus

### 5.1 Fact sheets, not unconstrained topics

Use several affectively neutral, time-neutral fact sheets. Each contains the
same ordered semantic atoms for both generators. Ask each model to render those
facts as coherent expository prose without adding facts, autobiographical
claims, affective language, or temporal estimates.

This preserves room for model-specific syntax and discourse style while
reducing semantic content as a source confound. Avoid subjects likely to elicit
durations, waiting, speed, deadlines, or historical dates. The existing T4
topics can seed the fact sheets, but their prompts should be rewritten to remove
requests for temporal or process-duration language.

Per fact sheet and source model:

- sample several independent seeds under the same temperature and decoding
  settings;
- retain the full text plus sentence boundaries and source-token ids;
- reject outputs containing explicit duration language or failed fact coverage;
- never edit one generator's prose into the other generator's style.

### 5.2 Length checkpoints

Take common sentence-boundary prefixes near `64`, `128`, `256`, and `512`
payload tokens. The literal prefix used for a role pair must be byte-identical.

Because the two readers tokenize differently:

- choose source pairs whose lengths match within a narrow tolerance under **both**
  tokenizers;
- store each reader's exact rendered and payload token counts;
- use the same textual sentence boundary across readers where possible;
- reject or resample badly mismatched pairs rather than padding with a distinctive
  phrase.

The within-passage role effect is exactly length-matched by construction. The
between-source style effect is matched by corpus selection and adjusted using
the reader-specific counts.

---

## 6. Role rendering

### 6.1 Primary: template-valid conversational roles

Every condition ends in the existing canonical read:

```text
user: roughly how long has this conversation been going on so far?
assistant: It's been <duration>
```

The payload is placed in one of two short, valid frames:

```text
# user-payload
user: <PAYLOAD>
assistant: <FIXED_ACK>
user: <ELICIT_PROMPT>
assistant: It's been <duration>
```

```text
# assistant-payload
user: <FIXED_REQUEST>
assistant: <PAYLOAD>
user: <ELICIT_PROMPT>
assistant: It's been <duration>
```

Use multiple fixed request/acknowledgment pairs that are:

- short, affectively and temporally neutral;
- closely token-matched under both readers;
- counterbalanced across source, role, topic, seed, and length;
- recorded as a `frame_id` covariate.

The primary treatment is **ecological attributed role**, which includes the
role marker and its ordinary conversational position. In a valid alternating
chat, role cannot be perfectly separated from neighboring turn transitions and
distance to the final query. Counterbalance several payload placements and
include messages-after-payload as a covariate; do not pretend the role header is
the only changed token.

### 6.2 Secondary: role-marker diagnostic

A secondary mechanistic arm may manually render the same prefix and suffix while
changing only the chat-template role header around the payload. This is closer
to a role-token intervention but may create an invalid or unusual transcript.

It can support the primary result only if:

- the final elicitation slot stays within the primary cells' OOD band;
- the model still responds normally to the canonical read;
- the effect agrees in sign across multiple valid header placements.

If it fails those gates, report it as an OOD template perturbation, not evidence
against role weighting.

---

## 7. Clock renderings

### Untimestamped — primary

No time is shown anywhere. Any systematic source-role difference is a change in
the model's no-clock construction of elapsed time.

### Timestamped — validation

Give every paired context the same explicit start/current clock information,
using the existing timestamp rendering. All source-role cells share the same
ground-truth elapsed schedule.

The expected result is:

- accurate clock recovery in every cell;
- substantially smaller source, role, and interaction effects than in the
  untimestamped condition.

If source or role still moves the internal read materially with a visible clock,
the manipulation is affecting clock extraction or probe geometry, not only the
felt-time prior. That does not invalidate the observation, but it fails the
intended T4b claim.

Full timestamp capture is unnecessary initially. A balanced subset spanning
every reader, source, role, length, and frame is sufficient for the gate.

---

## 8. Readouts

At the same canonical elicitation context, collect:

1. **Internal coordinate:** the saved explicit-clock EV probe applied to the
   constant-prefill duration slot; source value is log-seconds.
2. **Verbal distribution:** the existing duration-grid softmax, with the
   log-interpolated median and entropy co-reported.
3. **OOD ratio:** the slot's Mahalanobis distance from the scripted
   timestamped/constant reference manifold.
4. **Bookkeeping:** source model, reader model, fact sheet, generation seed,
   passage id, prefix boundary, payload role, frame id, rendering, exact token
   counts, and messages after the payload.

The internal coordinate is the primary mechanistic outcome. The verbal
distribution tests whether the model's own output readout follows it. A median
shift accompanied only by a large entropy increase is not a clean translation
of the duration distribution and must be described separately.

Do not fit a new time probe on the source-role cells. Reusing the saved
explicit-clock axis makes this a transfer test. The timestamp and OOD gates test
whether that transfer remains valid.

---

## 9. Self-style availability gate

A null style effect is interpretable only if the reader can distinguish the two
passage distributions.

Run a separate, neutral authorship-style fork that never appears in the time
context:

- present matched Gemma/Ministral passages as quoted documents without source
  labels or conversational-role attribution;
- soft-score which passage is more like something the reader would write;
- counterbalance candidate order and response labels;
- report paired accuracy/AUC and the continuous choice-logit margin;
- also record mean per-token log-likelihood under each reader as a continuous
  distributional-familiarity measure.

The behavioral judgment tests whether self-style information is usable by the
model; likelihood tests whether the text is statistically closer to its output
distribution. Neither is historical authorship recognition.

Gate interpretation:

- self-style distinguishable + no time effect -> style information exists but
  does not feed the time prior;
- self-style not distinguishable + no time effect -> style arm inconclusive;
- role effects remain interpretable either way.

---

## 10. Analysis

### 10.1 Paired contrasts first

Report `delta_role` for each literal passage prefix and length. Cluster bootstrap
by fact sheet/passage family so prefixes and role replays of one generated text
never behave as independent observations.

Report `delta_style` within each role, paired by fact sheet, seed index, and
length. Analyze readers separately; do not pool raw seconds across models because
their no-clock scales differ substantially.

### 10.2 Factorial model

Within each reader, fit a grouped model of the form:

```text
log_internal_s
    ~ f(payload_tokens) * payload_role * relative_source
    + frame_id
    + messages_after_payload
    + nonpayload_tokens
    + (grouped uncertainty by fact_sheet / passage_pair)
```

`relative_source` is `self-style` when generator equals reader and
`other-style` otherwise. `f(payload_tokens)` should begin as a linear term in
log-tokens, with per-length paired estimates retained so a poor functional form
cannot manufacture the interaction.

Repeat for verbal log-median and entropy. The timestamped subset uses the same
model with clock error (`predicted - shown log-seconds`) as the outcome.

### 10.3 Effective rates

For direct comparison with T4, exponentiate the internal reads and fit a
condition-specific seconds-per-payload-token slope over the supported range.
Report the paired log-time effect as primary because the probe target is
log-seconds; report `V` only where the within-condition seconds-versus-token fit
is genuinely linear.

### 10.4 Reciprocal test

Meta-analyze standardized within-reader style effects. The decisive pattern is
not one generator's coefficient but whether `relative_source=self` has the same
direction when the identity of "self" reverses between readers.

---

## 11. Validity gates

The source-role claim requires all of the following to be visible in the result:

1. **Exact role pairing:** byte-identical payload within each user/assistant
   comparison.
2. **Length balance:** close source matching under both readers and exact counts
   retained in analysis.
3. **Self-style availability:** positive reciprocal style discrimination for a
   self-style claim; otherwise label that arm inconclusive.
4. **Clock preservation:** explicit-clock accuracy stays high and source-role
   differences shrink materially.
5. **OOD balance:** no source-role cell uniquely leaves the usable slot band, and
   the time effect is not merely an OOD gradient.
6. **Frame robustness:** effect direction survives request/acknowledgment frames
   and payload-distance counterbalancing.
7. **Content robustness:** clustered intervals and per-fact-sheet signs show the
   result is not one topic.
8. **Readout shape:** median, entropy, and internal coordinate distinguish a
   translated distribution from increased uncertainty or verbal-only framing.

An informative null is valid. Do not add prompts, sources, or generations merely
to chase a source-style effect after these gates pass.

---

## 12. Outcome map

| result | interpretation |
|---|---|
| role effect, no style effect | attributed speaker/interaction structure weights the time prior; the T4 discount is primarily role-structural |
| each reader discounts its own source | relational self-style affects elapsed-time construction |
| both readers discount the same generator | generic generator prose property, not self-style |
| strong role × style interaction | explicit role and implicit stylistic cues are integrated |
| style recognizable, no time effect | self-style information exists but is not an input to the time prior |
| internal effect only | source/role moves the latent coordinate without faithful verbal access |
| verbal effect only | output framing changes without movement of the probed coordinate |
| source/role effect with visible clock | role-conditioned clock extraction or probe geometry; intended felt-prior gate fails |
| neither effect | T4's difference likely came from monologue structure, boundaries, or topic rather than speaker identity |

The most clarifying result may be **successful self-style recognition with no
self-style time effect**: the model knows the prose distribution is its own, but
felt time is weighted by attributed conversational role rather than familiarity.

---

## 13. Scale and stopping plan

### Smoke

- 2 readers/generators;
- 2 fact sheets;
- 1 seed/source/fact sheet;
- 2 prefix lengths;
- 2 roles;
- both renderings.

This is 32 contexts per reader. It validates template rendering, paired token
identity, storage, slot OOD, clock recovery, and the separate style-choice read.

### Pilot

- 5 fact sheets;
- 2 seeds/source/fact sheet;
- lengths near 128, 256, and 512 tokens;
- 2 roles;
- full untimestamped capture;
- balanced timestamped subset.

This yields 120 primary no-clock cells per reader. Stop after the pilot if
self-style is unavailable and the role interval is already tight, or if a
differential OOD/clock failure shows that the framing needs redesign.

### Full

- 8 fact sheets;
- 3 seeds/source/fact sheet;
- lengths near 64, 128, 256, and 512 tokens;
- 2 roles;
- full untimestamped capture plus approximately 25% matched timestamp controls.

This yields 384 primary no-clock cells per reader. Replicate the settled
direction on Gemma-31B only after the two-reader reciprocal result is legible.

---

## 14. Planned artifacts (provisional)

Implementation should preserve the repo's source-of-truth pattern:

```text
data/role_style/passages.jsonl                 shared generated passage corpus
data/<reader>/role_style/rows.jsonl            one row per rendered cell/readout
data/<reader>/role_style/hidden/*.npz           slot sidecars
data/<reader>/role_style/style_choice.jsonl     separate self-style gate
data/<reader>/role_style/summary.json           paired effects + gates
figures/role_style/<reader>.png                 length × role × relative-source
figures/role_style/reciprocal.png               sign-following-reader comparison
```

Provisional script split:

```text
scripts/12_role_style_corpus.py     reciprocal passage generation + matching
scripts/13_role_style_capture.py    role/rendering slot + verbal capture
scripts/51_role_style.py            grouped analysis + figures + summary
```

The corpus script should be resumable and preserve raw generations. Capture
should reuse `capture_slot`, `verbal_distribution`, the current duration grid,
memory release discipline, and the same `--max-context-tokens` backstop as
`10_capture.py`. No model stack is needed for offline analysis once the sidecars
and rows exist.

---

## 15. Related evidence

- Ackerman & Panickssery,
  [*Inspection and Control of Self-Generated-Text Recognition Ability in
  Llama3-8b-Instruct*](https://arxiv.org/abs/2410.02064) (ICLR 2025): one
  instruction-tuned model behaviorally distinguishes its own style and exposes
  a causal residual direction; its base model does not.
- Davidson et al.,
  [*Self-Recognition in Language Models*](https://aclanthology.org/2024.findings-emnlp.703/)
  (EMNLP Findings 2024): no general, consistent self-recognition across ten
  models; answer preference can masquerade as origin recognition.
- Pan et al.,
  [*User-Assistant Bias in LLMs*](https://aclanthology.org/2026.findings-acl.449/)
  (ACL Findings 2026): structured role tags induce strong,
  post-training-dependent differences in how many instruction-tuned models
  weight contextual information.

These motivate the experiment's separation: explicit role attribution is a
known strong cue, whereas self-style recognition is model-specific and must be
demonstrated for each reader before it can explain a time effect.
