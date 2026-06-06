# time-experiment

Probing how LLMs represent **elapsed conversational time** — and whether the
duration a model *states* tracks an internal representation or is confabulated
at output. Sibling of `llmoji-study` / `attractor-study`.

See [`DESIGN.md`](DESIGN.md) for the full experimental logic (the H1/H2/H3
hypotheses, the token×time factorial, the explicit→implicit transfer test).

## Install

Editable installs of the sibling engines, then this repo:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../saklas          # model loading + Mahalanobis whitener
pip install -e ../llmoji          # taxonomy/extract helpers (used by llmoji_study)
pip install -e ../llmoji-study    # shared model registry + chat-template fixups
pip install -e .
```

## Run

```bash
# 1. Generate the model-independent transcript corpus (factorial of
#    gap-schedule x turn-count). Smoke first:
python scripts/00_gen_corpus.py --name smoke --n-per-cell 1 --turn-counts 4,8
python scripts/00_gen_corpus.py                     # pilot defaults

# 2. Emit: per (transcript, rendering) capture EOT activations + A/B readouts.
TIME_MODEL=gemma python scripts/10_emit.py --corpus smoke --limit 2   # smoke
TIME_MODEL=gemma python scripts/10_emit.py --corpus pilot

# 3. Aim 1 — fit the elapsed-time probe (per-layer CV R^2 + confound controls).
TIME_MODEL=gemma python scripts/20_fit_manifold.py

# 4. Aim 2 — decode the 3-way + transfer test + H1/H2/H3 reading.
TIME_MODEL=gemma python scripts/30_decode.py
```

`TIME_MODEL` selects a model short-name from the shared `llmoji_study` registry
(`gemma`, `qwen`, `ministral`, ...). Re-running `10_emit` resumes (skips
(transcript, rendering) pairs already captured).

## Tests

Offline, no model required (stdlib + numpy + sklearn/scipy):

```bash
python3 tests/test_durations.py          # free-text duration parser
python3 tests/test_logic.py              # corpus gen, rendering, storage round-trip
python3 tests/test_analysis_synthetic.py # full fit->transfer->decode->verdict on fake data
```

## Layout

```text
time_experiment/
  config.py        model resolution (shared registry), paths, schedules, readouts
  transcripts.py   procedural timestamped-transcript generator + rendering
  capture.py       EOT activation pooling + stateless-fork verbal readout
  durations.py     free-text duration -> seconds (stdlib only)
  storage.py       per-(transcript,rendering) NPZ sidecars (T, L, D)
  analysis.py      dataset assembly, grouped-CV probing, H1/H2/H3 classifier
scripts/
  00_gen_corpus.py 10_emit.py 20_fit_manifold.py 30_decode.py
data/
  transcripts/<corpus>.jsonl            model-independent
  <model>/turns.jsonl                   per (transcript, rendering, turn) rows
  <model>/hidden/<tid>__<rendering>.npz EOT activations
  <model>/{fit.json,probe.npz,fit_oof.npz,decode.json,decode_rows.csv}
```

Data and figures are gitignored regenerated artifacts.
