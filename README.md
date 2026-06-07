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

# 3. Aim 1 — fit the elapsed-time probe (deployable = all-layer STACK;
#    per-layer sweep + confound controls + stack power-check).
TIME_MODEL=gemma python scripts/20_fit_manifold.py
# 3b. (optional) probe-architecture bake-off: single best layer vs concat vs
#     stack, on both renderings (offline; re-fits existing sidecars).
TIME_MODEL=gemma python scripts/21_layer_probe_compare.py

# 4. Aim 2 — decode the 3-way + transfer test + H1/H2/H3 reading.
TIME_MODEL=gemma python scripts/30_decode.py

# 5. Secondary analyses (read existing captures, no re-emit).
TIME_MODEL=gemma python scripts/40_geometry.py       # log-t axis geometry
TIME_VARIANT=inflation python scripts/41_inflation.py
TIME_VARIANT=rates python scripts/42_intermittent.py

# 6. Render the headline figures to figures/<model>/ (offline; reads artifacts).
TIME_MODEL=gemma python scripts/50_figures.py

# 7. Naturalistic arm: probe real model-generated conversations. The scripted
#    EOT axis is corpus-specific and blows up OOD -> Mahalanobis-whitened read;
#    the verbal readout transfers and is content-sensitive.
TIME_MODEL=gemma python scripts/60_naturalistic.py
TIME_MODEL=gemma python scripts/61_whiten_natural.py

# 8. Prefilled-duration probe (read at the point of use, "It's been <D>"):
#    true-vs-constant control + transfer of the duration axis to natural felt.
TIME_MODEL=gemma python scripts/62_elicit_capture.py --scripted-limit 40
TIME_MODEL=gemma python scripts/63_elicit_analyze.py    # probe-site + control
TIME_MODEL=gemma python scripts/64_verbal_target.py     # felt-axis transfer
TIME_MODEL=gemma python scripts/65_elicit_figures.py    # fig_elicit.png
TIME_MODEL=gemma python scripts/66_elicit_aim_figures.py    # fig1/fig3, prefill probe
TIME_MODEL=gemma python scripts/67_natural_elicit_figure.py # EOT vs prefill on natural

# 9. Arm G — generation-side time: capture the per-token rollout trajectory +
#    felt-production readouts, then test if producing tokens drives an elapsed axis.
TIME_MODEL=gemma python scripts/70_generate.py
TIME_MODEL=gemma python scripts/71_gen_time.py          # fig_genG.png
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
  21_layer_probe_compare.py                            single vs concat vs stack (offline)
  40_geometry.py 41_inflation.py 42_intermittent.py   secondary analyses
  50_figures.py                                        headline figures (offline)
  60_naturalistic.py 61_whiten_natural.py             naturalistic arm + whitening
  62_elicit_capture.py 63_elicit_analyze.py           prefilled-duration probe
  64_verbal_target.py 65_elicit_figures.py            felt-axis transfer + figure
  66_elicit_aim_figures.py 67_natural_elicit_figure.py  fig1/fig3 + natural contrast
  70_generate.py 71_gen_time.py                       Arm G: generation-side time
data/
  transcripts/<corpus>.jsonl            model-independent
  <model>/turns.jsonl                   per (transcript, rendering, turn) rows
  <model>/hidden/<tid>__<rendering>.npz EOT activations
  <model>/{fit.json,probe.npz(stacked),fit_oof.npz,decode.json,decode_rows.csv,layer_probe_compare.json}
```

Data and figures are gitignored regenerated artifacts.
