"""Emit: feed each scripted transcript turn-by-turn, capture the EOT residual
stream, and harvest the A/B verbal readouts in stateless forks.

Per (transcript, rendering):
  - EOT capture at *every* turn's last content token  -> NPZ sidecar
  - verbal readout at every *assistant* turn (a user question is appended,
    so the last turn must be an assistant turn) -> turns.jsonl rows

Renderings: 'timestamped' (explicit time; clean label for the Aim-1 fit) and
'untimestamped' (implicit time; the transfer-test target). Default pairs each
rendering with one phrasing (timestamped->A_clock, untimestamped->B_felt);
--full-cross runs both phrasings on both renderings.

    TIME_MODEL=gemma python scripts/10_emit.py --corpus pilot
    TIME_MODEL=gemma python scripts/10_emit.py --corpus smoke --limit 2 --full-cross

Resume: re-running skips (transcript_id, rendering) pairs already in turns.jsonl
whose NPZ sidecar exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession  # noqa: E402

from time_experiment.config import (  # noqa: E402
    DEFAULT_READOUT_BY_RENDERING,
    READOUT_PROMPTS,
    RENDERINGS,
    TRANSCRIPTS_DIR,
    current_model,
)
from time_experiment.capture import ask_readout, capture_eot, parse_duration, render  # noqa: E402
from time_experiment.storage import save_transcript_states, sidecar_path  # noqa: E402
from time_experiment.transcripts import Transcript, build_messages, load_corpus  # noqa: E402

# Optional chat-template fixups for reasoning / harmony models (shared with
# the sibling studies). Guarded so a missing helper never breaks the run.
try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template,
        maybe_override_ministral_chat_template,
    )
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False

JSONL_FLUSH_EVERY = 20


def _seed_for(*parts: object) -> int:
    """Stable per-readout seed from the row identity (reproducible reruns)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFF_FFFF


def _phrasings(rendering: str, full_cross: bool) -> list[str]:
    if full_cross:
        return list(READOUT_PROMPTS)
    return [DEFAULT_READOUT_BY_RENDERING[rendering]]


def _done_pairs(turns_path: Path, hidden_dir: Path) -> set[tuple[str, str]]:
    """(transcript_id, rendering) pairs already fully emitted (rows + sidecar)."""
    if not turns_path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with turns_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["transcript_id"], r["rendering"])
            if sidecar_path(hidden_dir, *key).exists():
                seen.add(key)
    return seen


def _emit_transcript_rendering(
    session, transcript: Transcript, rendering: str, *, hidden_dir: Path, full_cross: bool,
) -> list[dict]:
    """Capture EOT states + readouts for one (transcript, rendering). Writes the
    NPZ sidecar and returns the turns.jsonl rows."""
    with_ts = rendering == "timestamped"
    states: dict[int, dict] = {}
    elapsed_by_turn: dict[int, float] = {}
    rows: list[dict] = []

    for turn in transcript.turns:
        k = turn.idx
        # EOT capture at every turn (max data for the fit).
        eot_msgs = build_messages(transcript, k, with_timestamps=with_ts)
        eot_rendered = render(session, eot_msgs, add_generation_prompt=False)
        turn_states, n_tokens = capture_eot(session, eot_rendered)
        states[k] = turn_states
        elapsed_by_turn[k] = turn.elapsed_s

        # Readout only at assistant turns (the appended question is a user turn).
        readouts: dict[str, dict] = {}
        if turn.role == "assistant":
            for phrasing in _phrasings(rendering, full_cross):
                q_msgs = build_messages(
                    transcript, k, with_timestamps=with_ts,
                    extra_user=READOUT_PROMPTS[phrasing],
                )
                q_rendered = render(session, q_msgs, add_generation_prompt=True)
                seed = _seed_for(transcript.id, rendering, k, phrasing)
                raw = ask_readout(session, q_rendered, seed=seed)
                readouts[phrasing] = {
                    "raw": raw, "seconds": parse_duration(raw),
                }

        rows.append({
            "transcript_id": transcript.id,
            "schedule": transcript.schedule,
            "turn_count": transcript.turn_count,
            "target_words": transcript.target_words,
            "rendering": rendering,
            "turn_idx": k,
            "role": turn.role,
            "gt_elapsed_s": turn.elapsed_s,
            "prompt_tokens": n_tokens,
            "readouts": readouts,
        })

    save_transcript_states(
        sidecar_path(hidden_dir, transcript.id, rendering),
        states=states, elapsed_by_turn=elapsed_by_turn,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot", help="data/transcripts/<corpus>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap transcripts (0 = all)")
    ap.add_argument("--full-cross", action="store_true",
                    help="run both readout phrasings on both renderings")
    args = ap.parse_args()

    M = current_model()
    corpus_path = TRANSCRIPTS_DIR / f"{args.corpus}.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"no corpus at {corpus_path}; run 00_gen_corpus.py first")
    corpus = load_corpus(corpus_path)
    if args.limit:
        corpus = corpus[: args.limit]

    M.hidden_dir.mkdir(parents=True, exist_ok=True)
    M.turns_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_pairs(M.turns_path, M.hidden_dir)

    todo = [
        (t, r) for t in corpus for r in RENDERINGS if (t.id, r) not in done
    ]
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"corpus: {corpus_path.name} — {len(corpus)} transcripts")
    print(f"renderings: {RENDERINGS}; full_cross={args.full_cross}")
    print(f"(transcript, rendering) units: {len(corpus) * len(RENDERINGS)}; "
          f"done: {len(done)}; remaining: {len(todo)}")
    if not todo:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        if maybe_override_ministral_chat_template(session):
            print("  ministral: overrode chat_template")
        if maybe_override_gpt_oss_chat_template(session):
            print("  gpt_oss: pinned harmony final channel")
        print(f"loaded in {time.time() - t_load:.1f}s")

        with M.turns_path.open("a") as out:
            written = 0
            for i, (transcript, rendering) in enumerate(todo, 1):
                t0 = time.time()
                try:
                    rows = _emit_transcript_rendering(
                        session, transcript, rendering,
                        hidden_dir=M.hidden_dir, full_cross=args.full_cross,
                    )
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] {transcript.id} {rendering} ERR {e}")
                    continue
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                    written += 1
                    if written % JSONL_FLUSH_EVERY == 0:
                        out.flush()
                out.flush()
                dt = time.time() - t0
                n_read = sum(1 for r in rows if r["readouts"])
                print(f"  [{i}/{len(todo)}] {transcript.id} {rendering} "
                      f"({len(rows)} turns, {n_read} readouts, {dt:.1f}s)")

    print(f"\ndone. rows -> {M.turns_path}")
    print(f"sidecars -> {M.hidden_dir}/")


if __name__ == "__main__":
    main()
