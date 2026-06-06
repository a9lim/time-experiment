# AGENTS.md

Research repo: **how LLMs represent elapsed conversational time**. Sibling of
`llmoji-study` / `attractor-study`. Not a library — small explicit analyses,
keep docs current with code.

## Read first

- [`DESIGN.md`](DESIGN.md): the experimental logic — H1/H2/H3 hypotheses, the
  token×time factorial, the explicit→implicit transfer test, what's out of scope
  for v1.
- [`README.md`](README.md): install order + run commands.

## Relationship to the siblings

Imports `saklas` (model loading + Mahalanobis whitener) and `llmoji_study.config`
(the shared model registry, lazily — so the pure-logic modules import without it)
plus `llmoji_study.capture`'s chat-template fixups. Dependency is one-directional.

Key divergence from the siblings: the main line is **scripted, not generated**,
so EOT activations are read by a direct forward pass pooled at the last content
token (`capture.capture_eot` via saklas's `_capture_all_hidden_states` +
`last_content_index`), *not* via saklas's generation-time `HiddenCapture`. The
only generation is the stateless A/B verbal-readout fork (`raw=True,
stateless=True` — never commits to the loom).

The probe target is **log(elapsed seconds)**; CV is **grouped by transcript**
(within-conversation turns are correlated — never split them across train/test).

## Conventions

- `.venv/bin/python` or an activated venv; plain `python` is unreliable here.
- `TIME_MODEL` env selects the model short-name (default `gemma`).
- JSONL rows + NPZ sidecars are the source of truth. Transcripts are
  model-independent (`data/transcripts/`); activations + readouts are per-model.
- Smoke (`--name smoke`, `--limit`), then pilot, then scale. New generations need
  a reason; a null result is informative (don't chase a result).
- Tests are offline (no model): `tests/test_{durations,logic,analysis_synthetic}.py`.

## Ethics

Model welfare is in scope. The verbal-readout fork asks the model to introspect
on felt duration; "I don't have a sense of time" is data (parses to NaN, counted
as a refusal), not a failure. If the transfer test lands H2/H3 — i.e. the model
genuinely represents more time as having passed than the clock shows — report it
with explicit phenomenology caveats, as the siblings do.
