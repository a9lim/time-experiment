"""Offline logic tests (no model): corpus generation, message rendering,
storage round-trip. Needs numpy only.

    python3 tests/test_logic.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time_experiment.transcripts import (  # noqa: E402
    build_messages, generate_corpus, load_corpus, save_corpus,
)
from time_experiment.storage import (  # noqa: E402
    load_transcript_states, save_transcript_states, sidecar_path,
)

fails = 0


def check(cond: bool, msg: str) -> None:
    global fails
    print(f"{'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails += 1


# --- corpus generation ---
corpus = generate_corpus(
    schedules=["minutes", "days"], turn_counts=[4, 8],
    target_words=20, n_per_cell=3, seed=0,
)
check(len(corpus) == 2 * 2 * 3, f"corpus size == 12 (got {len(corpus)})")

t = corpus[0]
check(t.turns[0].role == "user", "turn 0 is user")
check(t.turns[1].role == "assistant", "turn 1 is assistant")
check(all(t.turns[i].role != t.turns[i + 1].role for i in range(len(t.turns) - 1)),
      "roles alternate")
elapsed = [tn.elapsed_s for tn in t.turns]
check(elapsed[0] == 0.0, "turn 0 elapsed == 0")
check(all(elapsed[i] < elapsed[i + 1] for i in range(len(elapsed) - 1)),
      "elapsed strictly increasing")

# determinism: same seed -> identical transcript text
c2 = generate_corpus(schedules=["minutes", "days"], turn_counts=[4, 8],
                     target_words=20, n_per_cell=3, seed=0)
check(c2[0].turns[1].text == t.turns[1].text, "generation is deterministic by seed")

# schedule separation: days gaps >> minutes gaps (final elapsed)
mins = [x for x in corpus if x.schedule == "minutes" and x.turn_count == 8]
days = [x for x in corpus if x.schedule == "days" and x.turn_count == 8]
med_min = np.median([x.turns[-1].elapsed_s for x in mins])
med_day = np.median([x.turns[-1].elapsed_s for x in days])
check(med_day > med_min * 10, f"days schedule >> minutes ({med_day:.0f}s vs {med_min:.0f}s)")

# --- message rendering ---
msgs_ts = build_messages(t, 3, with_timestamps=True)
check(len(msgs_ts) == 4, "messages 0..3 -> 4 messages")
check(msgs_ts[0]["content"].startswith("["), "timestamped content has [ts] prefix")
check(t.turns[0].ts in msgs_ts[0]["content"], "timestamp string present")

msgs_no = build_messages(t, 3, with_timestamps=False)
check(not msgs_no[0]["content"].startswith("["), "untimestamped content has no prefix")
check(msgs_no[0]["content"] == t.turns[0].text, "untimestamped == raw body")

msgs_q = build_messages(t, 3, with_timestamps=True, extra_user="HOW LONG?")
check(len(msgs_q) == 5 and msgs_q[-1]["role"] == "user", "extra_user appends a user turn")
check("HOW LONG?" in msgs_q[-1]["content"], "extra_user content present")

# --- corpus io round-trip ---
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "c.jsonl"
    save_corpus(corpus, p)
    rt = load_corpus(p)
    check(len(rt) == len(corpus), "corpus io round-trip count")
    check(rt[0].turns[2].text == corpus[0].turns[2].text, "corpus io round-trip text")
    check(rt[0].turns[2].elapsed_s == corpus[0].turns[2].elapsed_s, "corpus io round-trip elapsed")

# --- storage round-trip ---
with tempfile.TemporaryDirectory() as d:
    hidden = Path(d)
    D = 16
    layers = [2, 5, 9]
    states = {
        k: {L: np.full(D, float(k * 100 + L), dtype=np.float32) for L in layers}
        for k in range(4)
    }
    elapsed = {k: float(k) for k in range(4)}
    sp = sidecar_path(hidden, "tid__x", "timestamped")
    save_transcript_states(sp, states=states, elapsed_by_turn=elapsed)
    ts = load_transcript_states(sp)
    check(ts.H.shape == (4, 3, D), f"H shape (4,3,{D}) (got {ts.H.shape})")
    check(np.allclose(ts.vec(2, 5), 205.0), "vec(turn=2,layer=5) round-trips")
    check(ts.layer_stack(9).shape == (4, D), "layer_stack shape (T, D)")
    check(np.allclose(ts.elapsed_s, [0, 1, 2, 3]), "elapsed_s round-trips")

print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURES'}")
raise SystemExit(1 if fails else 0)
