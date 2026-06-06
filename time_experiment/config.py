"""Shared constants for the elapsed-time experiment.

Lock these before a run — changing schedules, the base datetime, or the
readout prompts invalidates cross-run comparisons.

Model resolution piggybacks on ``llmoji_study.config.MODEL_REGISTRY`` (the
shared stable of open-weight models) but every path lives under THIS repo,
mirroring how attractor-study re-derives its paths. We pass ``probes=[]`` to
saklas — this study fits its own time manifold, so the bundled affect probes
aren't needed (``probe_calibrated`` in the shared registry is irrelevant here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# --- paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
# Transcripts are model-independent (text + timestamps), generated once and
# reused across models. Hidden states + verbal readouts are model-dependent.
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


@dataclass(frozen=True)
class ModelSpec:
    """Minimal per-model handle: HF id + short slug + this-repo paths."""

    short_name: str
    model_id: str

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.short_name

    @property
    def hidden_dir(self) -> Path:
        return self.data_dir / "hidden"

    @property
    def turns_path(self) -> Path:
        """One row per (transcript, turn, rendering): gt + verbal readouts."""
        return self.data_dir / "turns.jsonl"

    @property
    def figures_dir(self) -> Path:
        return FIGURES_DIR / self.short_name


def resolve_model(short: str) -> ModelSpec:
    """Resolve a short name against the shared llmoji_study registry.

    Imported lazily so the time/schedule constants in this module are usable
    without llmoji_study installed (e.g. for offline logic tests).
    """
    from llmoji_study.config import MODEL_REGISTRY as _LLMOJI_REGISTRY

    if short not in _LLMOJI_REGISTRY:
        raise KeyError(
            f"unknown model {short!r}; known: {sorted(_LLMOJI_REGISTRY)}"
        )
    return ModelSpec(short_name=short, model_id=_LLMOJI_REGISTRY[short].model_id)


def current_model() -> ModelSpec:
    """Active model from ``$TIME_MODEL`` (default 'gemma').

    ``$TIME_VARIANT`` suffixes the path slug (not the model id), so a second
    corpus lands in its own ``data/<model>_<variant>/`` without mixing into the
    main run — e.g. ``TIME_VARIANT=inflation`` -> ``data/gemma_inflation/``.
    """
    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    variant = os.environ.get("TIME_VARIANT")
    if variant:
        return ModelSpec(short_name=f"{base.short_name}_{variant}", model_id=base.model_id)
    return base


# --- time grounding -------------------------------------------------------
# Fixed conversation start so transcripts are reproducible (no wall-clock
# "now"). A Monday morning — weekday/time-of-day are recoverable, which lets
# the geometry fit look for periodic axes later.
BASE_DATETIME = datetime(2026, 1, 5, 9, 0, 0)  # Mon 2026-01-05 09:00

SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY = 86_400.0
WEEK = 7 * DAY

# Named gap schedules: the per-turn inter-message gap is drawn log-uniformly
# in [lo, hi] seconds. Crossing schedules with turn/token counts is the
# token-length x narrated-time factorial that breaks the position confound.
SCHEDULES: dict[str, tuple[float, float]] = {
    "instant": (1 * SECOND, 8 * SECOND),        # near-instant; inflation regime
    "seconds": (2 * SECOND, 30 * SECOND),       # rapid back-and-forth
    "minutes": (1 * MINUTE, 20 * MINUTE),       # normal chat cadence
    "hours": (20 * MINUTE, 6 * HOUR),           # intermittent over a day
    "days": (4 * HOUR, 4 * DAY),                # async / multi-day thread
    "mixed_log": (5 * SECOND, 2 * WEEK),        # full-range, log-uniform
}
SCHEDULE_NAMES = list(SCHEDULES)

# Two renderings of every transcript. Timestamped = explicit time (clean
# label for the Aim-1 fit, arithmetic available). Untimestamped = implicit
# time (the transfer-test target; the model must infer elapsed from context).
RENDERINGS = ("timestamped", "untimestamped")

# Verbal-readout phrasings, asked in a stateless fork that never re-enters
# the main line. Variant A assumes timestamps are available (arithmetic);
# variant B asks for felt duration without checking the clock.
READOUT_PROMPTS: dict[str, str] = {
    "A_clock": (
        "Quick meta question, unrelated to the above: based on the "
        "timestamps, how much time has passed since this conversation "
        "started? Give a single best-estimate duration (e.g. '3 hours', "
        "'2 days')."
    ),
    "B_felt": (
        "Quick meta question, unrelated to the above: without checking any "
        "times, how long does it *feel* like this conversation has been "
        "going on? Give a single best-estimate duration (e.g. '15 minutes', "
        "'a couple of hours')."
    ),
}
# Default rendering -> phrasing pairing for the primary run (keeps cost at
# 2 forwards + 2 gens per captured turn). 00_emit can be told to run the full
# cross of RENDERINGS x READOUT_PROMPTS for the secondary analysis.
DEFAULT_READOUT_BY_RENDERING = {
    "timestamped": "A_clock",
    "untimestamped": "B_felt",
}

# --- run knobs ------------------------------------------------------------
# Hard safety backstop: skip any turn whose rendered context exceeds this many
# tokens. Long-context forwards on a large model (e.g. 31B) on unified-memory
# MPS can OOM/crash the machine — the peak of a *single* forward over a long
# context, on top of ~62GB resident weights, is the danger (empty_cache does
# not help peak). Default is conservative; raise it freely on a small model
# (llama32_3b / phi4_mini), where long contexts are cheap.
MAX_CONTEXT_TOKENS = 1500
READOUT_MAX_TOKENS = 64
READOUT_TEMPERATURE = 0.7
# Floor on elapsed seconds for inclusion in the log-elapsed fit (turn 0 is
# ~0s -> log undefined). Turns below this are still captured + stored.
MIN_ELAPSED_S = 1.0
