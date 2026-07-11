# AGENTS.md

Research repo: **LLMs linearly encode elapsed conversational time in context
length** (≈0.3 s/token off the residual stream — the token-time hypothesis made
representational and measured). Sibling of `llmoji-experiment` / `attractor-experiment`. Not
a library — small explicit analyses, keep docs current with code.

## Read first

- [`DESIGN.md`](DESIGN.md): the experimental logic — H1/H2/H3 hypotheses, the
  one-prompt spine, the four throughlines (T1 probe / T2 felt-is-a-length-prior /
  T3 transfer / T4 generation-side), what's out of scope for v1.
- [`README.md`](README.md): install order + run commands (9 scripts).

## Relationship to the siblings

Imports `saklas` (model loading + Mahalanobis whitener) and `llmoji_experiment.config`
(the shared model registry, lazily — so the pure-logic modules import without it)
plus `llmoji_experiment.capture`'s chat-template fixups. Dependency is one-directional.

Key divergence from the siblings: the main line is **scripted, not generated**,
and the canonical readout is the **prefilled elicitation slot** — render
`user: <ELICIT_PROMPT> / assistant: It's been <duration>` and pool all layers at
the duration token (`capture.capture_slot` via saklas's
`_capture_all_hidden_states` + `last_content_index`), *not* via generation-time
`HiddenCapture`. The verbal estimate is a **soft duration distribution** read from
the slot logits (`capture.verbal_distribution`: score a `DURATION_GRID` after
`It's been ` and softmax) — the model's own `W_U` readout of the slot, symmetric
to the probe's activation readout; deterministic, no refusals. Its point estimate
is the **log-interpolated median** (`dist_point`, robust to the multimodal tails
the no-clock felt distribution grows at depth) with **entropy** (`dist_entropy`)
co-reported; `verbal_dist` is the source of truth. The only place
generation (`HiddenCapture`/`return_hidden`) is used is T4's rollout
(`11_gen_capture`), whose felt-production readout is the same soft distribution.

The earlier EOT site (pool a bare end-of-transcript token) and the *learned*
all-layer stack are **removed** — the slot, read **EV-weighted across all layers**
(saklas's explained-variance aggregation: `fit_ev_probe` weights each layer's read
by its grouped-CV R²), supersedes them (R²≈0.98 vs ≈0.59, and it transfers
to natural felt at ρ≈0.42 — length-confounded but real — where EOT gives ≈0.11).
EOT numbers survive only as cited history in `docs/findings.md`.

The probe target is **log(elapsed seconds)**; CV is **grouped by conversation**
(within-conversation turns are correlated — never split them across train/test).

## Memory (MPS) — read before scaling context length

A long-context forward on a large model (gemma-4-31b-it ≈ 62GB resident) is the
hazard on unified-memory MPS. The slot site is inherently **one forward per
turn** — each turn's prefill tail (`It's been …`) makes its context unique and
ends at a different absolute position, so the EOT line's multi-position
single-forward trick does not apply. The disciplines that keep it bounded
(`10_capture` + `capture.py`):

- **`release_memory()`** (gc + `torch.mps.empty_cache()`) after every slot
  capture and every readout — MPS's allocator hoards a block per distinct
  context length otherwise. This is the load-bearing control now.
- **`--max-context-tokens`** (default 1500) skips any turn whose context exceeds
  the cap *before* the forward — the hard backstop. Raise it on a small model;
  keep it modest on the 31B.

Steady state ≈ model + one capped-length forward (~67GB on a 128GB box). Two
machine crashes came from a long forward on the 31B + an unbounded per-size
cache. The per-turn-forward profile is new with the slot canonicalization —
**watch Activity Monitor on the first 31B `10_capture` run** (a small model
won't surface the MPS peak); validate the pipeline on `TIME_MODEL=llama32_3b`
first.

## Conventions

- `.venv/bin/python` or an activated venv; plain `python` is unreliable here.
  This repo's `.venv` points to the shared `../.venvs/llmoji-experiment`
  environment, with saklas, llmoji_experiment, torch, time_experiment editable,
  and the model registry.
- `TIME_MODEL` selects the short-name (default `gemma`); `TIME_VARIANT` routes a
  variant corpus to `data/<model>_<variant>/`.
- One `rows.jsonl` + per-(source,id,rendering,mode) NPZ sidecars are the source
  of truth. Transcripts are model-independent (`data/transcripts/`); activations,
  readouts, looms, and per-throughline summaries are per-model.
- Smoke (`--name smoke`, `--scripted-limit`/`--limit`), then pilot, then scale.
  New generations need a reason; a null result is informative (don't chase one).
- Tests are offline (no model): `tests/test_{durations,logic,analysis_synthetic}.py`.

## Ethics

Model welfare is in scope. The verbal readout asks the model to introspect on
felt duration. The soft-distribution readout reads the model's duration
distribution from the logits rather than forcing a stated answer — arguably a
gentler, richer read: the old "I don't have a sense of time" refusal now surfaces
as a high-entropy distribution (the uncertainty is kept, not collapsed to a NaN).
If the transfer test lands H2/H3 — i.e. the model genuinely represents more time
as having passed than the clock shows — report it with explicit phenomenology
caveats, as the siblings do.
