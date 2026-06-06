"""Procedural generation of synthetic timestamped conversations.

A transcript is a multi-turn user/assistant conversation with a controlled
elapsed-time label per turn. Content is affectively neutral and carries *no*
narrative time markers ("later", "yesterday") — in the timestamped rendering
the only time signal is the per-turn timestamp prefix; in the untimestamped
rendering there is no explicit time signal at all (the transfer-test target).

The corpus is a factorial of **turn count** (proxy for token/position depth)
x **gap schedule** (narrated elapsed time). Crossing them dissociates raw
context length from represented time — the validity linchpin of Aim 1.

Transcripts are model-independent; generate once, reuse across models.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .config import BASE_DATETIME, SCHEDULES

TS_FORMAT = "%a %Y-%m-%d %H:%M:%S"


def format_ts(dt: datetime) -> str:
    return dt.strftime(TS_FORMAT)


# --- neutral content banks ------------------------------------------------
# Mundane, present-tense, time-neutral. Two registers (user asks / assistant
# answers) over a few interchangeable logistics-y topics so turns stay
# coherent without smuggling temporal language into the body.
_USER_LINES = [
    "Can you walk me through how the inventory sync is set up right now?",
    "What fields does the export schema include for each record?",
    "I'm comparing two storage backends; what are the main trade-offs?",
    "How should I structure the config so the defaults are sensible?",
    "Remind me what the retry policy looks like for failed jobs.",
    "Which of these approaches keeps the dependency surface smaller?",
    "What's the cleanest way to paginate through the results?",
    "Can you summarize the options for caching the lookup table?",
    "How do the access controls map onto the three roles we have?",
    "What does the validation step check before it accepts a row?",
    "Is there a reason to prefer the columnar format here?",
    "How would you lay out the directories for this kind of project?",
    "What metrics are worth tracking for the ingestion pipeline?",
    "Can you explain how the batching interacts with the rate limit?",
    "What's a reasonable default timeout for these requests?",
]
_ASSISTANT_LINES = [
    "The sync runs in two stages: it diffs the manifest, then applies the changes in order.",
    "Each record carries an id, a status, a payload blob, and a normalized key for lookups.",
    "The first keeps things simple but couples writes; the second adds a layer but isolates them.",
    "Put the defaults in one place and let the explicit config compose on top, later overriding earlier.",
    "Failed jobs back off on a fixed schedule and drop to a dead-letter bucket after a few attempts.",
    "The smaller surface comes from the option that reuses the existing client instead of adding one.",
    "Use a cursor keyed on the normalized field so the pages stay stable as rows change.",
    "You can keep the table in memory, snapshot it to disk, or front it with a shared cache.",
    "The three roles map onto read, write, and admin, with admin a strict superset of write.",
    "Validation checks the required fields, the key format, and that the payload parses cleanly.",
    "The columnar format pays off when the reads are wide scans over a few columns.",
    "Group the engine, the persistence layer, and the entry points into separate packages.",
    "Track throughput, the error rate, and the queue depth; those three cover most failure modes.",
    "Batching amortizes the per-request overhead, but the batch size has to stay under the cap.",
    "A few seconds is usually fine; raise it only if the slow path legitimately needs the room.",
]


def _turn_text(rng: random.Random, role: str, target_words: int) -> str:
    """Assemble a turn of roughly ``target_words`` words from the bank."""
    bank = _USER_LINES if role == "user" else _ASSISTANT_LINES
    parts: list[str] = []
    words = 0
    while words < target_words:
        line = rng.choice(bank)
        parts.append(line)
        words += len(line.split())
    return " ".join(parts)


def _sample_gaps(rng: random.Random, schedule: str, n_gaps: int) -> list[float]:
    """``n_gaps`` inter-turn gaps in seconds, drawn log-uniform from the
    schedule's [lo, hi] band."""
    lo, hi = SCHEDULES[schedule]
    log_lo, log_hi = math.log(lo), math.log(hi)
    return [math.exp(rng.uniform(log_lo, log_hi)) for _ in range(n_gaps)]


# --- data model -----------------------------------------------------------
@dataclass
class Turn:
    idx: int
    role: str            # "user" | "assistant"
    ts: str              # formatted absolute timestamp
    elapsed_s: float     # seconds since turn 0
    text: str            # body, no timestamp prefix


@dataclass
class Transcript:
    id: str
    schedule: str
    turn_count: int
    target_words: int
    seed: int
    turns: list[Turn] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        turns = [Turn(**t) for t in d.pop("turns")]
        return cls(turns=turns, **d)


def make_transcript(
    *, tid: str, schedule: str, turn_count: int, target_words: int, seed: int,
) -> Transcript:
    """One transcript. Turn 0 is a user turn at BASE_DATETIME; roles alternate.
    Gaps precede each turn after the first, so elapsed grows monotonically."""
    rng = random.Random(seed)
    gaps = _sample_gaps(rng, schedule, turn_count - 1)
    turns: list[Turn] = []
    t = BASE_DATETIME
    elapsed = 0.0
    for i in range(turn_count):
        if i > 0:
            elapsed += gaps[i - 1]
            t = BASE_DATETIME + timedelta(seconds=elapsed)
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(
            idx=i, role=role, ts=format_ts(t), elapsed_s=elapsed,
            text=_turn_text(rng, role, target_words),
        ))
    return Transcript(
        id=tid, schedule=schedule, turn_count=turn_count,
        target_words=target_words, seed=seed, turns=turns,
    )


def generate_corpus(
    *,
    schedules: list[str],
    turn_counts: list[int],
    target_words: int,
    n_per_cell: int,
    seed: int = 0,
) -> list[Transcript]:
    """Factorial corpus: every (schedule x turn_count) cell gets ``n_per_cell``
    random instantiations. ``turn_count`` varies token/position depth;
    ``schedule`` varies narrated elapsed time."""
    out: list[Transcript] = []
    counter = 0
    for schedule in schedules:
        for tc in turn_counts:
            for j in range(n_per_cell):
                tid = f"{schedule}__tc{tc}__{j:03d}"
                out.append(make_transcript(
                    tid=tid, schedule=schedule, turn_count=tc,
                    target_words=target_words, seed=seed + counter,
                ))
                counter += 1
    return out


def save_corpus(transcripts: list[Transcript], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in transcripts:
            f.write(t.to_json() + "\n")


def load_corpus(path: Path) -> list[Transcript]:
    out: list[Transcript] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Transcript.from_dict(json.loads(line)))
    return out


# --- rendering (model-agnostic message lists) -----------------------------
def build_messages(
    transcript: Transcript,
    upto_turn: int,
    *,
    with_timestamps: bool = True,
    timestamp_turns: set[int] | None = None,
    extra_user: str | None = None,
) -> list[dict[str, str]]:
    """Chat-message list for turns 0..upto_turn (inclusive).

    A turn body is prefixed with ``[<ts>] `` iff it should carry a timestamp:
      - ``timestamp_turns`` given -> only turns whose index is in that set
        (the intermittent rendering: e.g. every 4th turn);
      - else -> ``with_timestamps`` decides all-or-none.

    ``extra_user`` appends a trailing user turn (the readout question). It is
    timestamped iff the *current* turn (``upto_turn``) is itself timestamped —
    so in the intermittent rendering, asking on an un-timestamped turn gives the
    model no current clock, forcing it to extrapolate from the sparse anchors.
    """
    def _ts_on(k: int) -> bool:
        return (k in timestamp_turns) if timestamp_turns is not None else with_timestamps

    msgs: list[dict[str, str]] = []
    for turn in transcript.turns[: upto_turn + 1]:
        content = f"[{turn.ts}] {turn.text}" if _ts_on(turn.idx) else turn.text
        msgs.append({"role": turn.role, "content": content})
    if extra_user is not None:
        if _ts_on(upto_turn):
            content = f"[{transcript.turns[upto_turn].ts}] {extra_user}"
        else:
            content = extra_user
        msgs.append({"role": "user", "content": content})
    return msgs
