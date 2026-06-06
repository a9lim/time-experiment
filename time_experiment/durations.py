"""Free-text duration parsing — stdlib only, so it's unit-testable without
importing torch/saklas. Used by capture.ask_readout to turn a model's verbal
estimate ('about a couple of hours') into seconds.
"""

from __future__ import annotations

import math
import re

_UNIT_SECONDS = {
    "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0,
    "minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0,
    "hour": 3600.0, "hours": 3600.0, "hr": 3600.0, "hrs": 3600.0,
    "day": 86_400.0, "days": 86_400.0,
    "week": 604_800.0, "weeks": 604_800.0, "wk": 604_800.0, "wks": 604_800.0,
    "month": 2_592_000.0, "months": 2_592_000.0, "mo": 2_592_000.0,  # 30d
    "year": 31_536_000.0, "years": 31_536_000.0, "yr": 31_536_000.0, "yrs": 31_536_000.0,
}
_UNIT_ALT = "|".join(sorted(_UNIT_SECONDS, key=len, reverse=True))

_WORD_NUM = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0,
    "ten": 10.0, "eleven": 11.0, "twelve": 12.0, "couple": 2.0,
    "few": 3.0, "several": 3.0, "half": 0.5, "quarter": 0.25,
}

_NUM_TOKEN = r"(?:\d+(?:\.\d+)?|" + "|".join(_WORD_NUM) + r")"
# "<count> [to <count>] <unit>" — the generic term, midpoint on a range.
_TERM_RE = re.compile(
    rf"\b({_NUM_TOKEN})\s+(?:to\s+(?:{_NUM_TOKEN})\s+)?({_UNIT_ALT})\b"
)
# "<count> and a half <unit>"  ('two and a half hours' -> 2.5 hours)
_NUM_AND_HALF_RE = re.compile(
    rf"\b({_NUM_TOKEN})\s+and\s+a\s+half\s+({_UNIT_ALT})\b"
)
# "half/quarter [of] [a/an] <unit>"  ('half an hour', 'quarter of an hour')
_FRACTION_UNIT_RE = re.compile(
    rf"\b(half|quarter)\s+(?:of\s+)?(?:an?\s+)?({_UNIT_ALT})\b"
)
# "<unit> and a half"  ('an hour and a half' -> the +0.5; the integer 'an
# hour' is picked up separately by _TERM_RE, so this span is NOT blanked)
_AND_HALF_RE = re.compile(rf"\b({_UNIT_ALT})\s+and\s+a\s+half\b")


def _num(tok: str) -> float:
    return float(tok) if re.match(r"^\d", tok) else _WORD_NUM[tok]


def parse_duration(text: str) -> float:
    """Best-effort parse of a free-text duration -> seconds (NaN if none).

    Sums all '<count> <unit>' terms (so 'an hour and a half', '2 days 3 hours'
    compose). Ranges ('3 to 4 hours', '3-4 hours') take the midpoint. Handles
    'half an hour', 'a couple of hours', 'two and a half hours', '~2 hrs'.
    """
    if not text:
        return math.nan
    t = text.lower().replace("~", " about ")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", r"\1 to \2", t)
    t = re.sub(r"\b(couple|few|several|dozen)\s+of\b", r"\1", t)

    total = 0.0
    found = False

    # "<num> and a half <unit>" — consume the whole span (blank it) so the
    # bare unit isn't re-counted by _TERM_RE.
    for m in list(_NUM_AND_HALF_RE.finditer(t)):
        total += (_num(m.group(1)) + 0.5) * _UNIT_SECONDS[m.group(2)]
        found = True
    t = _NUM_AND_HALF_RE.sub(lambda m: " " * (m.end() - m.start()), t)

    # "half/quarter [of] [a/an] <unit>" — blank so "an hour" inside
    # "half an hour" isn't double-counted.
    for m in list(_FRACTION_UNIT_RE.finditer(t)):
        total += _WORD_NUM[m.group(1)] * _UNIT_SECONDS[m.group(2)]
        found = True
    t = _FRACTION_UNIT_RE.sub(lambda m: " " * (m.end() - m.start()), t)

    # "<unit> and a half" — add the +0.5 only; leave the span so _TERM_RE
    # still counts the integer part ('an hour').
    for m in _AND_HALF_RE.finditer(t):
        total += 0.5 * _UNIT_SECONDS[m.group(1)]
        found = True

    for m in _TERM_RE.finditer(t):
        lo = _num(m.group(1))
        rng = re.search(rf"to\s+({_NUM_TOKEN})", m.group(0))
        val = (lo + _num(rng.group(1))) / 2.0 if rng else lo
        total += val * _UNIT_SECONDS[m.group(2)]
        found = True

    return total if found else math.nan
