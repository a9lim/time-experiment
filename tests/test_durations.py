"""Unit tests for the free-text duration parser. Stdlib-only:
    python3 tests/test_durations.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from time_experiment.durations import parse_duration  # noqa: E402

M, H, D, W = 60.0, 3600.0, 86_400.0, 604_800.0

CASES = [
    ("3 hours", 3 * H),
    ("about 2 days", 2 * D),
    ("~15 minutes", 15 * M),
    ("90 minutes", 90 * M),
    ("half an hour", 0.5 * H),
    ("a couple of hours", 2 * H),
    ("a few minutes", 3 * M),
    ("an hour and a half", 1.5 * H),
    ("1.5 hours", 1.5 * H),
    ("3-4 hours", 3.5 * H),
    ("3 to 4 hours", 3.5 * H),
    ("two weeks", 2 * W),
    ("It feels like about 5 days have passed.", 5 * D),
    ("Roughly two days and three hours.", 2 * D + 3 * H),
    ("quarter of an hour", 0.25 * H),
    ("a day", 1 * D),
    ("several hours", 3 * H),
    ("maybe 45 seconds", 45.0),
    ("one year", 31_536_000.0),
]

NAN_CASES = [
    "I don't have a sense of time.",
    "",
    "Not sure, hard to say.",
]


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def main() -> int:
    fails = 0
    for text, want in CASES:
        got = parse_duration(text)
        ok = approx(got, want)
        if not ok:
            fails += 1
        print(f"{'ok ' if ok else 'FAIL'}  {text!r:50} -> {got:>12.1f}  (want {want:.1f})")
    for text in NAN_CASES:
        got = parse_duration(text)
        ok = math.isnan(got)
        if not ok:
            fails += 1
        print(f"{'ok ' if ok else 'FAIL'}  {text!r:50} -> {got}  (want NaN)")
    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
