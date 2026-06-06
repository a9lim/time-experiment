"""Generate the procedural timestamped-transcript corpus (model-independent).

The corpus is a factorial of gap-schedule x turn-count, with N random
instantiations per cell. Crossing schedule (narrated elapsed time) with
turn-count (token/position depth) is what lets the fit dissociate represented
time from raw context length.

    python scripts/00_gen_corpus.py                 # pilot defaults
    python scripts/00_gen_corpus.py --name smoke --n-per-cell 1 --turn-counts 4,8
    python scripts/00_gen_corpus.py --schedules minutes,hours,days --n-per-cell 10

Writes data/transcripts/<name>.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.config import SCHEDULE_NAMES, TRANSCRIPTS_DIR  # noqa: E402
from time_experiment.transcripts import generate_corpus, save_corpus  # noqa: E402


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot", help="corpus name -> data/transcripts/<name>.jsonl")
    ap.add_argument("--schedules", type=_csv, default=SCHEDULE_NAMES,
                    help=f"comma-separated; from {SCHEDULE_NAMES}")
    ap.add_argument("--turn-counts", type=lambda s: [int(x) for x in _csv(s)],
                    default=[4, 8, 12], help="comma-separated turn counts")
    ap.add_argument("--target-words", type=int, default=40,
                    help="approx words per turn (controls token length)")
    ap.add_argument("--n-per-cell", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bad = set(args.schedules) - set(SCHEDULE_NAMES)
    if bad:
        raise SystemExit(f"unknown schedules {bad}; known: {SCHEDULE_NAMES}")

    corpus = generate_corpus(
        schedules=args.schedules,
        turn_counts=args.turn_counts,
        target_words=args.target_words,
        n_per_cell=args.n_per_cell,
        seed=args.seed,
    )
    out = TRANSCRIPTS_DIR / f"{args.name}.jsonl"
    save_corpus(corpus, out)

    n_turns = sum(t.turn_count for t in corpus)
    print(f"wrote {len(corpus)} transcripts ({n_turns} turns) -> {out}")
    print(f"cells: {len(args.schedules)} schedules x {len(args.turn_counts)} "
          f"turn-counts x {args.n_per_cell} = {len(corpus)}")
    print(f"schedules: {args.schedules}")
    print(f"turn_counts: {args.turn_counts}; target_words: {args.target_words}")


if __name__ == "__main__":
    main()
