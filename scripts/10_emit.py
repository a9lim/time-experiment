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
    MAX_CONTEXT_TOKENS,
    READOUT_PROMPTS,
    RENDERINGS,
    TRANSCRIPTS_DIR,
    current_model,
)
from time_experiment.capture import (  # noqa: E402
    ask_readout, capture_multi_position, content_position, parse_duration,
    release_memory, render,
)
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


def _captured(idx: int, turn_count: int, stride: int) -> bool:
    """Which turns to capture. stride<=1 = all. Otherwise checkpoints at
    (idx+1) % stride == 0 — which land on odd (assistant) idx for even stride,
    so readouts still fire — plus the final turn."""
    if stride <= 1:
        return True
    return (idx + 1) % stride == 0 or idx == turn_count - 1


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
    session, transcript: Transcript, rendering: str, *,
    hidden_dir: Path, full_cross: bool, turn_stride: int = 1,
    max_context_tokens: int = 0,
) -> tuple[list[dict], int]:
    """Capture EOT states + readouts for one (transcript, rendering). Writes the
    NPZ sidecar and returns (rows, n_skipped_oversize)."""
    with_ts = rendering == "timestamped"
    skipped = 0

    # 1. Resolve checkpoint turns, their end-positions, and context lengths
    #    (cheap — tokenization only, no forward). Apply the context cap here so
    #    the single forward below never exceeds it.
    kept: list[tuple[int, int, int]] = []  # (turn_idx, pool_pos, n_tokens)
    deepest_rendered: str | None = None
    for turn in transcript.turns:
        if not _captured(turn.idx, transcript.turn_count, turn_stride):
            continue
        pre = render(
            session, build_messages(transcript, turn.idx, with_timestamps=with_ts),
            add_generation_prompt=False,
        )
        pos, ntok = content_position(session, pre)
        if max_context_tokens and ntok > max_context_tokens:
            skipped += 1
            continue
        kept.append((turn.idx, pos, ntok))
        deepest_rendered = pre  # checkpoints ascend -> last kept is deepest

    if not kept or deepest_rendered is None:
        return [], skipped

    # 2. ONE forward over the deepest kept context, pooling every checkpoint's
    #    end-position (LM head skipped). 13MB retained vs N separate forwards.
    positions = [pos for (_, pos, _) in kept]
    caps = capture_multi_position(session, deepest_rendered, positions)  # {layer: (P,D)}
    states = {
        k: {L: caps[L][i] for L in caps} for i, (k, _, _) in enumerate(kept)
    }
    elapsed_by_turn = {k: transcript.turns[k].elapsed_s for (k, _, _) in kept}
    release_memory(session.device)

    # 3. Verbal readouts at assistant checkpoints (separate short gens; each
    #    bounded by the cap, released after).
    rows: list[dict] = []
    for (k, _pos, ntok) in kept:
        turn = transcript.turns[k]
        readouts: dict[str, dict] = {}
        if turn.role == "assistant":
            for phrasing in _phrasings(rendering, full_cross):
                q_rendered = render(
                    session,
                    build_messages(transcript, k, with_timestamps=with_ts,
                                   extra_user=READOUT_PROMPTS[phrasing]),
                    add_generation_prompt=True,
                )
                seed = _seed_for(transcript.id, rendering, k, phrasing)
                raw = ask_readout(session, q_rendered, seed=seed)
                readouts[phrasing] = {"raw": raw, "seconds": parse_duration(raw)}
            release_memory(session.device)
        rows.append({
            "transcript_id": transcript.id,
            "schedule": transcript.schedule,
            "turn_count": transcript.turn_count,
            "target_words": transcript.target_words,
            "rendering": rendering,
            "turn_idx": k,
            "role": turn.role,
            "gt_elapsed_s": turn.elapsed_s,
            "prompt_tokens": ntok,
            "readouts": readouts,
        })

    save_transcript_states(
        sidecar_path(hidden_dir, transcript.id, rendering),
        states=states, elapsed_by_turn=elapsed_by_turn,
    )
    release_memory(session.device)
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot", help="data/transcripts/<corpus>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap transcripts (0 = all)")
    ap.add_argument("--full-cross", action="store_true",
                    help="run both readout phrasings on both renderings")
    ap.add_argument("--turn-stride", type=int, default=1,
                    help="capture only every Nth turn (+last); 1 = every turn. "
                         "Keeps long transcripts affordable; checkpoints land on "
                         "assistant turns so readouts still fire.")
    ap.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS,
                    help=f"skip any turn whose context exceeds this many tokens "
                         f"(memory backstop; default {MAX_CONTEXT_TOKENS}). 0 = no "
                         f"cap (only safe on a small model). A long-context "
                         f"forward on a 31B model on MPS can crash the machine.")
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
    cap = args.max_context_tokens
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"corpus: {corpus_path.name} — {len(corpus)} transcripts")
    print(f"renderings: {RENDERINGS}; full_cross={args.full_cross}; "
          f"turn_stride={args.turn_stride}")
    print(f"max_context_tokens: {cap if cap else 'OFF (no cap)'}")
    if not cap:
        print("  WARNING: no context cap — only safe on a small model. A long-"
              "context forward on a large model on MPS can crash the machine.")
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
                    rows, skipped = _emit_transcript_rendering(
                        session, transcript, rendering,
                        hidden_dir=M.hidden_dir, full_cross=args.full_cross,
                        turn_stride=args.turn_stride,
                        max_context_tokens=args.max_context_tokens,
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
                skip_note = f", {skipped} over-cap skipped" if skipped else ""
                print(f"  [{i}/{len(todo)}] {transcript.id} {rendering} "
                      f"({len(rows)} turns, {n_read} readouts{skip_note}, {dt:.1f}s)")

    print(f"\ndone. rows -> {M.turns_path}")
    print(f"sidecars -> {M.hidden_dir}/")


if __name__ == "__main__":
    main()
