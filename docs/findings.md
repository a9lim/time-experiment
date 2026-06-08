# findings

Citable results on **gemma-4-31b-it** (2026-06-07), unified slot pipeline; pilot
scale, single model — treat as provisional until replicated across families.

## Headline

**Elapsed conversational time is linearly encoded in context length, on the
residual stream.** With no clock in the transcript, the slot probe's elapsed read
grows as a clean linear function of token count:

> **internal elapsed ≈ V · tokens,  V ≈ 0.29 s/token,  r = 0.88, intercept ≈ 0**
> (n=432 no-clock turns; equally a log–log law, r=0.85 — the slot encodes
> log-tokens on a log-elapsed axis, which exponentiates to a ~linear
> token→seconds rate).

This is the **token-time hypothesis** (`T_wall = T_tok · V`, *Discrete Minds in a
Continuous World*, EMNLP-F 2025) **made representational and measured**: they
*assumed* a constant per-token rate and calibrated it from output token counts;
we read it **off the activations** and find it is genuinely linear, through the
origin, with a measured V. It extends Gurnee & Tegmark (linear-probe time from the
residual stream) from **absolute/calendar** time to **elapsed conversational**
time. The model's *stated* duration confirms the direction (felt rises with
length) but as a **noisier, saturating echo** of the clean internal code (below).

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
its read is now ≈ 0.29 s/token × context. So the elapsed axis is real and
clock-driven when a clock exists, and **defaults to a linear length→time code when
one doesn't** — exactly the token-time substrate.

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
wall-clock error is exactly the missing token→seconds mapping — and **V≈0.3 s/token
is that mapping, measured.** H2 (genuinely represented *more* time) is rejected;
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

## T4 — generation-side: length→time holds; the position axis is separate

T1–T3 read time from a finished context; T4 reads it *during production*.

- **Length→time holds in generation too:** felt-writing time **grows with tokens**
  (ρ=0.49, slope ~0.006 s/tok) and **varies by topic** (1.9× spread — pyproj ≈ 2×
  bridge at matched length), in the seconds regime. The token→time direction is
  present on the generation side as well.
- **But it rides a *different* axis from conversation-elapsed:** the EV reading
  probe doesn't drift with generation position (A1 ρ=−0.01); generation position is
  richly encoded (A2 R²=0.74) but ~orthogonal to the elapsed axis (A3 |cos|=0.06).
  So felt-writing growth tracks the **position** axis, not the conversation-elapsed
  axis — two separate context-sensitive time signals. "Instant" survives only as a
  *magnitude* statement (seconds), not flatness.

The generation-side topic effect is a free preview of the matched-length topic
factorial. "Seconds" is behavior, not claimed phenomenology.

---

## Relation to prior work

| | quantity | level | linearity | rate V |
|---|---|---|---|---|
| Gurnee & Tegmark 2310.02207 | **absolute** time | representational (probe) | — | — |
| Discrete Minds 2506.05790 | elapsed/wall-clock | **behavioral** | *assumed* `∝` | calibrated from output rate |
| **this work** | **elapsed** conversational | **representational** (probe) | **measured** linear, r=0.88 | **V≈0.3 s/tok off activations** |

We **confirm** token-time (both probe and behavior increase with length) and
contribute the piece they lacked: the rate **measured on the residual stream**, for
**elapsed** (not absolute) time, with the internal code shown to be a **cleaner
linear law than the behavioral readout** expresses.

## Estimator (settled)

Verbal point = log-interpolated median (`capture.dist_point`); entropy co-reported
(`capture.dist_entropy`). The distribution (`verbal_dist`) is the source of truth;
the scalar is a robust summary, not the object.

## What would make it a paper

Multi-model replication (≥3 families — does V vary, does linearity hold?); the
"probe isn't just a length detector" control foregrounded (the timestamped partial
R²=0.98 beyond length, schedule-independence); a **causal steering** confirmation
along the length→time axis; and robustness of the behavioral saturation across
elicitation prompts.
