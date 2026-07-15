# time-experiment

**LLMs linearly encode elapsed conversational time in context length.** A linear
probe on the elicitation slot reads elapsed as ≈0.3 s/token off the residual stream
(r=0.88, through the origin) — the token-time hypothesis made representational and
measured. This repo establishes that encoding and characterizes how the model's
*stated* duration tracks it (confirms the direction, saturates). It is runtime-
independent of the sibling experiments.

The elapsed-time probe is canonicalized as the **prefilled answer to a time
elicitation prompt**: ask "roughly how long has this been going on?", prefill
`It's been <D>`, and read the residual stream at the duration slot. See
[`docs/design.md`](docs/design.md) for the experimental logic (the H1/H2/H3 hypotheses, the
one-prompt spine, the four throughlines).

## Install

Install the workspace-root shared package, then this repo:

```bash
python --version  # system Python 3.12
python -m pip install -e ..
python -m pip install -e .
```

## Run

The pipeline is four throughlines (T1–T4) fed by capture + corpus, plus figures.
`TIME_MODEL` selects a short-name from the shared registry (default `gemma`);
`TIME_VARIANT` routes a variant corpus to its own `data/<model>_<variant>/`.
Capture resumes (skips (source, id, rendering, mode) already done).

```bash
# --- corpora ---
python scripts/00_corpus.py --name smoke --n-per-cell 1 --turn-counts 4,8   # smoke
python scripts/00_corpus.py                                                  # pilot
TIME_MODEL=gemma python scripts/01_natural.py                # naturalistic looms (model)

# --- capture: the elicitation slot (probe) + the verbal soft distribution ---
TIME_MODEL=gemma python scripts/10_capture.py --corpus smoke --scripted-limit 2 --peek  # smoke
TIME_MODEL=gemma python scripts/10_capture.py --corpus pilot
TIME_MODEL=gemma python scripts/10_capture.py --corpus pilot --verbal-only  # re-score verbal only
TIME_MODEL=gemma python scripts/11_gen_capture.py            # Arm G per-token trajectories

# --- the four throughlines (offline; read captures, no re-emit) ---
TIME_MODEL=gemma python scripts/20_probe.py         # T1 — the probe at the slot + controls
TIME_MODEL=gemma python scripts/30_felt.py          # T2 — felt is a length prior (decode/inflation/intermittent)
TIME_MODEL=gemma python scripts/40_transfer.py      # T3 — scripted axis -> natural felt + OOD
TIME_MODEL=gemma python scripts/50_generation.py    # T4 — generation-side: elapsed read at query time (raw flat → spliced ρ≈0.87) + its figure

# --- variant corpora for the T2 clock-density gradient ---
TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/10_capture.py --corpus inflation \
    --renderings timestamped,untimestamped --no-true --no-natural
TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/30_felt.py
TIME_MODEL=gemma TIME_VARIANT=rates python scripts/10_capture.py --corpus rates \
    --renderings timestamped,intermittent,untimestamped --no-true --no-natural
TIME_MODEL=gemma TIME_VARIANT=rates python scripts/30_felt.py

# --- headline figures (offline; reads artifacts) ---
TIME_MODEL=gemma python scripts/90_figures.py       # T1–T3 figures (T4 fig emitted by 50_generation)
```

Validate a new long-context run on a small model first
(`TIME_MODEL=llama32_3b`) — the slot is one forward per turn, so watch memory on
the 31B (see [`AGENTS.md`](AGENTS.md)). On the timestamped smoke, check that the
neutral prompt's stated-vs-gt correlation clears ≈0.9 before scaling.

## Tests

Offline, no model required (numpy + sklearn/scipy):

```bash
python3 tests/test_durations.py          # free-text duration parser
python3 tests/test_logic.py              # corpus gen, rendering, storage round-trip
python3 tests/test_analysis_synthetic.py # fit -> transfer -> decode -> verdict on fake data
```

## Layout

```text
time_experiment/
  config.py        model resolution (shared registry), paths, schedules, ELICIT_PROMPT
  transcripts.py   procedural timestamped-transcript generator + rendering
  capture.py       slot pooling + elicit-render + verbal_distribution (soft readout)
  durations.py     free-text duration -> seconds (stdlib only)
  storage.py       unified slot sidecar (source,id,rendering,mode) -> (T,L,D)
  analysis.py      assembly, grouped-CV EV-weighted all-layer probe, H1/H2/H3 classifier
scripts/
  00_corpus.py 01_natural.py                 corpora (scripted factorial + variants; looms)
  10_capture.py 11_gen_capture.py            slot+verbal capture; Arm G trajectories
  20_probe.py                                T1 — the probe
  30_felt.py                                 T2 — felt is a length prior
  40_transfer.py                             T3 — scripted axis -> natural felt
  50_generation.py                           T4 — generation-side time
  90_figures.py                              T1–T3 headline figures (offline)
data/
  transcripts/<corpus>.jsonl                 model-independent
  <model>/rows.jsonl                         per (source,id,rendering,turn,mode) rows
  <model>/hidden/<source>__<id>__<rendering>__<mode>.npz   slot activations
  <model>/natural/conversations.json         model-generated looms
  <model>/gen/                               Arm G trajectories + readouts
  <model>/{probe.npz,probe_meta.json,fit_oof.npz,felt.json,transfer.json,decode_rows.csv,natural_reads.csv}
figures/          generated figures; readout/ holds compact tracked evidence
logs/             local run logs and compact tracked status notes
tests/            offline duration, storage, and synthetic-analysis checks
docs/             design and findings
```

Data and figures are gitignored regenerated artifacts.

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
