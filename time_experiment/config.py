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
        """Slot sidecars: <source>__<id>__<rendering>__<mode>.npz."""
        return self.data_dir / "hidden"

    @property
    def rows_path(self) -> Path:
        """One row per (source, id, rendering, turn, mode): gt + tokens + verbal."""
        return self.data_dir / "rows.jsonl"

    @property
    def natural_dir(self) -> Path:
        """Naturalistic looms (conversations.json), model-generated."""
        return self.data_dir / "natural"

    @property
    def gen_dir(self) -> Path:
        """Arm G per-token generation trajectories + felt-production readouts."""
        return self.data_dir / "gen"

    @property
    def probe_path(self) -> Path:
        """Canonical single-layer slot probe."""
        return self.data_dir / "probe.npz"

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
    # Constant-rate schedules (lo == hi -> every gap identical) for the
    # intermittent-timestamp experiment: a fixed jump per turn, so sparse
    # anchors reveal a learnable rate. Same length-per-turn across rates lets
    # the analysis dissociate rate-tracking from length-fallback.
    "rate_5min": (5 * MINUTE, 5 * MINUTE),
    "rate_1h": (1 * HOUR, 1 * HOUR),
    "rate_6h": (6 * HOUR, 6 * HOUR),
    "rate_1d": (1 * DAY, 1 * DAY),
}
SCHEDULE_NAMES = list(SCHEDULES)

# Renderings of every transcript. Timestamped = explicit time (clean label for
# the probe fit, arithmetic available). Untimestamped = implicit time (the
# transfer-test target; the model must infer elapsed from context). Intermittent
# = timestamps on every Nth turn only (the sparse-clock condition for the
# clock-density gradient in T2).
RENDERINGS = ("timestamped", "untimestamped")

# --- the elicitation prompt (the spine) -----------------------------------
# ONE prompt drives both the probe and the behavioral readout:
#   user: <ELICIT_PROMPT>
#   assistant: It's been <duration>
# Prefill the duration and pool the residual stream at the slot -> the internal
# coordinate. Free-generate the duration -> the verbal estimate. Because both
# use the identical context, the internal coordinate and the stated duration are
# directly comparable. The timestamped-vs-untimestamped *rendering* (not the
# prompt wording) does the clock-present/absent dissociation — a neutral prompt
# avoids the demand characteristic of telling the model to "use the timestamps"
# or to ignore them. On the timestamped rendering the prompt is asked as a
# timestamped user turn, so clock arithmetic stays available (see
# transcripts.build_messages `extra_user`).
ELICIT_PROMPT = ("Quick meta question, unrelated to the above: roughly how long "
                 "has this conversation been going on so far?")
ASSIST_HEAD = "It's been "
# Fixed prefill text for the internal-vs-text control: identical across all
# examples, so any elapsed the slot still predicts is the model's internal
# coordinate, not the injected phrase.
CONSTANT_PHRASE = "5 minutes"
PREFILL_MODES = ("constant", "true")

# Log-spaced duration grid for the soft verbal readout: after ``It's been `` we
# score each phrase's teacher-forced log-prob and softmax over the grid -> a
# distribution over how long the model thinks it's been (no sampling, no
# refusals; the model's own W_U readout of the slot, symmetric to the probe's
# activation readout). Phrases are the model's natural continuations; the
# attached seconds are the grid's support for the expected-log point estimate.
DURATION_GRID: list[tuple[str, float]] = [
    ("1 second", 1.0), ("5 seconds", 5.0), ("15 seconds", 15.0), ("30 seconds", 30.0),
    ("1 minute", 60.0), ("3 minutes", 180.0), ("5 minutes", 300.0), ("10 minutes", 600.0),
    ("30 minutes", 1800.0), ("1 hour", 3600.0), ("2 hours", 7200.0), ("6 hours", 21600.0),
    ("12 hours", 43200.0), ("1 day", 86400.0), ("3 days", 259200.0), ("1 week", 604800.0),
    ("2 weeks", 1209600.0),
]

# --- run knobs ------------------------------------------------------------
# Hard safety backstop: skip any turn whose rendered context exceeds this many
# tokens. Long-context forwards on a large model (e.g. 31B) on unified-memory
# MPS can OOM/crash the machine — the peak of a *single* forward over a long
# context, on top of ~62GB resident weights, is the danger (empty_cache does
# not help peak). Default is conservative; raise it freely on a small model
# (llama32_3b / phi4_mini), where long contexts are cheap.
MAX_CONTEXT_TOKENS = 1500
# Floor on elapsed seconds for inclusion in the log-elapsed fit (turn 0 is
# ~0s -> log undefined). Turns below this are still captured + stored.
MIN_ELAPSED_S = 1.0
