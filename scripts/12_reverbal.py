"""Refresh the verbal readout as a soft duration distribution (offline-cheap
re-run; slots untouched).

The slot activations from 10_capture are reused as-is; only the behavioral
verbal estimate is recomputed — from a single sampled generation to the
logit-scored ``DURATION_GRID`` distribution (``capture.verbal_distribution``).
For each constant-mode assistant row it rebuilds the elicitation context exactly
as 10_capture did, scores the grid, and writes ``verbal_seconds`` (expected-log
point estimate) + ``verbal_dist`` (the grid probabilities) back into rows.jsonl.

Resumable: rows that already carry ``verbal_dist`` are skipped (use ``--force``
to recompute). Works per corpus / TIME_VARIANT, mirroring 10_capture.

    TIME_MODEL=gemma python scripts/12_reverbal.py --corpus pilot
    TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/12_reverbal.py --corpus inflation --no-natural
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

from time_experiment.capture import (  # noqa: E402
    inject_timestamps, release_memory, ts_spec, verbal_distribution,
)
from time_experiment.config import (  # noqa: E402
    ELICIT_PROMPT, TRANSCRIPTS_DIR, current_model,
)
from time_experiment.transcripts import build_messages, load_corpus  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False

FLUSH_EVERY = 100


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFF_FFFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot")
    ap.add_argument("--timestamp-stride", type=int, default=4)
    ap.add_argument("--no-natural", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if verbal_dist present")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all; for a bounded memory test)")
    args = ap.parse_args()

    M = current_model()
    rows = [json.loads(l) for l in M.rows_path.read_text().splitlines() if l.strip()]
    corpus = {tx.id: tx for tx in load_corpus(TRANSCRIPTS_DIR / f"{args.corpus}.jsonl")}
    looms_path = M.natural_dir / "conversations.json"
    looms = json.loads(looms_path.read_text()) if (looms_path.exists() and not args.no_natural) else {}

    # constant-mode assistant rows are the ones carrying the verbal readout.
    todo = [r for r in rows if r["mode"] == "constant" and r["role"] == "assistant"
            and (args.force or r.get("verbal_dist") is None)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"model: {M.short_name}  rows: {len(rows)}  verbal to (re)compute: {len(todo)}")
    if not todo:
        print("nothing to do."); return

    def build_msgs_q(r):
        if r["source"] == "scripted":
            tx = corpus.get(r["id"])
            if tx is None:
                return None
            ts = ts_spec(r["rendering"], tx.turn_count, args.timestamp_stride)
            return build_messages(tx, r["turn_idx"], **ts, extra_user=ELICIT_PROMPT)
        loom = looms.get(r["id"])
        if loom is None:
            return None
        msgs = loom["messages"]
        if r["rendering"] == "timestamped":
            msgs = inject_timestamps(msgs, _seed(r["id"], "ts"))[0]
        return msgs[: r["turn_idx"] + 1] + [{"role": "user", "content": ELICIT_PROMPT}]

    print(f"loading {M.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")

        done = 0
        for r in todo:
            msgs_q = build_msgs_q(r)
            if msgs_q is None:
                continue
            sec, dist = verbal_distribution(session, msgs_q)
            r["verbal_seconds"] = sec
            r["verbal_dist"] = [round(float(x), 5) for x in dist]
            r.pop("verbal_raw", None)
            release_memory(session.device)
            done += 1
            if done % FLUSH_EVERY == 0:
                M.rows_path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
                print(f"  {done}/{len(todo)}  (last: {r['source']} {r['id']} {r['rendering']} "
                      f"t{r['turn_idx']} -> {sec:.0f}s)")

    M.rows_path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print(f"\ndone. {done} verbal distributions -> {M.rows_path}")


if __name__ == "__main__":
    main()
