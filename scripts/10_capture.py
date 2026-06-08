"""Canonical capture: the elicitation slot + the verbal estimate.

For every captured assistant turn we render the elicitation prompt and, over the
*same* context, take two reads:

  - slot capture (the internal coordinate): prefill ``It's been <phrase>`` and
    pool all layers at the duration token. Two modes — ``constant`` (fixed
    "5 minutes" → internal coordinate, text held fixed) and ``true`` (humanized
    actual elapsed → the text-reading ceiling control).
  - verbal estimate (the behavioral readout): the soft duration distribution read
    from the slot logits (``capture.verbal_distribution`` — the model's own W_U
    readout, symmetric to the probe's activation readout; no sampling, no
    refusals). Captured once per assistant turn and attached to the ``constant``
    row as ``verbal_seconds`` (point) + ``verbal_dist`` (grid probs).

Sources:
  - scripted transcripts (``--corpus``), renderings timestamped / untimestamped
    / intermittent, both prefill modes. Variant corpora (inflation, rates) come
    in via ``TIME_VARIANT`` -> data/<model>_<variant>/.
  - natural looms (from 01_natural), constant mode: untimestamped (felt) +
    timestamped (injected-clock control, gt = injected elapsed).

``--verbal-only`` recomputes just the verbal readout over already-captured turns
(slots untouched, resumable) — the migration path for data captured before the
soft readout, and the re-score path when ``DURATION_GRID`` changes.

Memory: slot capture is one forward per turn (each prefill tail makes the
context unique) — the discipline is ``--max-context-tokens`` + per-turn
``release_memory``. Resume skips (source,id,rendering,mode) already captured.

    TIME_MODEL=gemma python scripts/10_capture.py --corpus pilot
    TIME_MODEL=gemma python scripts/10_capture.py --corpus smoke --scripted-limit 2 --peek
    TIME_MODEL=gemma python scripts/10_capture.py --corpus pilot --verbal-only  # re-score verbal
    TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/10_capture.py --corpus inflation \\
        --renderings timestamped,untimestamped --no-true --no-natural
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession  # noqa: E402

from time_experiment.capture import (  # noqa: E402
    capture_slot, content_position, dist_entropy, elicit_render, humanize,
    inject_timestamps, release_memory, slot_token, ts_spec, verbal_distribution,
)
from time_experiment.config import (  # noqa: E402
    CONSTANT_PHRASE, ELICIT_PROMPT, MAX_CONTEXT_TOKENS, TRANSCRIPTS_DIR, current_model,
)
from time_experiment.storage import save_states, sidecar_path  # noqa: E402
from time_experiment.transcripts import build_messages, load_corpus  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFF_FFFF


class Capturer:
    """Slot capture + verbal readout for a list of assistant turns, sharing one
    session. `turns` items: dict(turn_idx, msgs_q, gt, schedule, variant)."""

    def __init__(self, session, *, modes, cap, peek, do_verbal):
        self.s = session
        self.modes = modes
        self.cap = cap
        self.peek = peek
        self.do_verbal = do_verbal
        self.rows: list[dict] = []

    def run(self, source, conv_id, rendering, turns, hidden_dir):
        by_mode: dict[str, dict] = {m: {} for m in self.modes}
        elapsed_by_mode: dict[str, dict] = {m: {} for m in self.modes}
        for t in turns:
            verbal = None
            for mode in self.modes:
                gt = t["gt"]
                if mode == "true" and not (isinstance(gt, (int, float)) and gt and gt > 0):
                    continue  # no gt to prefill as the text ceiling
                phrase = CONSTANT_PHRASE if mode == "constant" else humanize(gt)
                rendered = elicit_render(self.s, t["msgs_q"], phrase)
                _, ntok = content_position(self.s, rendered)
                if ntok > self.cap:
                    continue
                if self.peek:
                    print(f"   peek [{conv_id} {rendering} {mode} t{t['turn_idx']}] "
                          f"phrase={phrase!r} slot={slot_token(self.s, rendered)!r}")
                states, ntok = capture_slot(self.s, rendered)
                release_memory(self.s.device)
                # verbal readout once per turn, on the first mode that lands:
                # the soft duration distribution from the slot logits (no refusals).
                if self.do_verbal and verbal is None:
                    sec, dist = verbal_distribution(self.s, t["msgs_q"])
                    verbal = {"seconds": sec, "entropy": round(dist_entropy(dist), 4),
                              "dist": [round(float(x), 5) for x in dist]}
                    release_memory(self.s.device)
                by_mode[mode][t["turn_idx"]] = states
                elapsed_by_mode[mode][t["turn_idx"]] = (
                    float(gt) if isinstance(gt, (int, float)) and gt and gt > 0 else math.nan)
                self.rows.append({
                    "source": source, "id": conv_id, "rendering": rendering, "mode": mode,
                    "turn_idx": t["turn_idx"], "role": "assistant",
                    "gt_elapsed_s": (float(gt) if isinstance(gt, (int, float)) and gt else None),
                    "tokens": ntok, "schedule": t.get("schedule"), "variant": t.get("variant"),
                    "phrase": phrase,
                    "verbal_seconds": (verbal or {}).get("seconds") if mode == "constant" else None,
                    "verbal_entropy": (verbal or {}).get("entropy") if mode == "constant" else None,
                    "verbal_dist": (verbal or {}).get("dist") if mode == "constant" else None,
                })
        for mode in self.modes:
            if by_mode[mode]:
                save_states(
                    sidecar_path(hidden_dir, source, conv_id, rendering, mode),
                    states=by_mode[mode], elapsed_by_turn=elapsed_by_mode[mode],
                )


def _done(rows_path: Path, hidden_dir: Path) -> set[tuple]:
    """(source,id,rendering,mode) already captured (rows + sidecar)."""
    if not rows_path.exists():
        return set()
    seen: set[tuple] = set()
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["source"], r["id"], r["rendering"], r["mode"])
        if sidecar_path(hidden_dir, *key).exists():
            seen.add(key)
    return seen


def iter_units(corpus, looms, renderings, timestamp_stride):
    """Yield ``(source, conv_id, rendering, turns)`` — the elicitation contexts,
    built identically for full capture and ``--verbal-only`` (no duplication).
    ``turns`` items: dict(turn_idx, gt, schedule, variant, msgs_q)."""
    for tx in corpus:
        for rendering in renderings:
            ts = ts_spec(rendering, tx.turn_count, timestamp_stride)
            turns = [{"turn_idx": turn.idx, "gt": turn.elapsed_s, "schedule": tx.schedule,
                      "variant": None,
                      "msgs_q": build_messages(tx, turn.idx, **ts, extra_user=ELICIT_PROMPT)}
                     for turn in tx.turns if turn.role == "assistant"]
            yield "scripted", tx.id, rendering, turns
    for conv_id, loom in looms.items():
        msgs = loom["messages"]
        ts_msgs, elapsed = inject_timestamps(msgs, _seed(conv_id, "ts"))
        for rendering, rmsgs, gts in (("untimestamped", msgs, [None] * len(msgs)),
                                      ("timestamped", ts_msgs, elapsed)):
            turns = [{"turn_idx": k, "gt": gts[k], "schedule": None, "variant": loom.get("variant"),
                      "msgs_q": rmsgs[: k + 1] + [{"role": "user", "content": ELICIT_PROMPT}]}
                     for k, m in enumerate(msgs) if m["role"] == "assistant"]
            yield "natural", conv_id, rendering, turns


def full_capture(session, M, units, *, modes, cap_tokens, peek, done):
    """Slot capture (+ inline verbal) over every unit, appending rows + sidecars.
    Natural is constant-only. Resumes past (source,id,rendering,mode) in ``done``."""
    cap = Capturer(session, modes=modes, cap=cap_tokens, peek=peek, do_verbal=True)
    with M.rows_path.open("a") as out:
        for source, cid, rendering, turns in units:
            run_modes = ("constant",) if source == "natural" else modes
            if all((source, cid, rendering, m) in done for m in run_modes):
                continue
            cap.modes = run_modes
            t = time.time()
            cap.run(source, cid, rendering, turns, M.hidden_dir)
            for r in cap.rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            cap.rows.clear()
            print(f"  {source} {cid} {rendering}: {len(turns)} turns ({time.time()-t:.0f}s)")
    print(f"\nrows -> {M.rows_path}\nsidecars -> {M.hidden_dir}/")


def refresh_verbal(session, M, units, *, force):
    """``--verbal-only``: recompute just the soft-distribution verbal readout over
    already-captured turns, updating rows.jsonl in place (slots untouched).
    Resumable — rows that already carry ``verbal_dist`` are skipped unless
    ``--force``. The migration path for data captured before the soft readout, and
    the re-score path when ``DURATION_GRID`` changes."""
    if not M.rows_path.exists():
        raise SystemExit(f"no rows at {M.rows_path}; run a full capture first")
    rows = [json.loads(l) for l in M.rows_path.read_text().splitlines() if l.strip()]
    index = {(r["source"], r["id"], r["rendering"], r["turn_idx"]): r
             for r in rows if r["mode"] == "constant" and r["role"] == "assistant"}
    todo = [(u[0], u[1], u[2], t) for u in units for t in u[3]  # (source,id,rendering,turn)
            if (u[0], u[1], u[2], t["turn_idx"]) in index
            and (force or index[(u[0], u[1], u[2], t["turn_idx"])].get("verbal_dist") is None)]
    print(f"verbal to (re)compute: {len(todo)} / {len(index)}")
    done = 0
    for source, cid, rendering, t in todo:
        r = index[(source, cid, rendering, t["turn_idx"])]
        sec, dist = verbal_distribution(session, t["msgs_q"])
        r["verbal_seconds"] = sec
        r["verbal_entropy"] = round(dist_entropy(dist), 4)
        r["verbal_dist"] = [round(float(x), 5) for x in dist]
        r.pop("verbal_raw", None)
        release_memory(session.device)
        done += 1
        if done % 100 == 0:
            M.rows_path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            print(f"  {done}/{len(todo)}  (last {source} {cid} {rendering} t{t['turn_idx']} -> {sec:.0f}s)")
    M.rows_path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print(f"\ndone. {done} verbal distributions -> {M.rows_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot")
    ap.add_argument("--scripted-limit", type=int, default=0, help="cap transcripts (0=all)")
    ap.add_argument("--renderings", default="timestamped,untimestamped",
                    help="scripted renderings: timestamped, untimestamped, intermittent")
    ap.add_argument("--timestamp-stride", type=int, default=4, help="intermittent stride")
    ap.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS,
                    help="skip turns over this context length (MPS memory backstop)")
    ap.add_argument("--no-true", action="store_true", help="skip the true-prefill control")
    ap.add_argument("--no-natural", action="store_true", help="skip the natural looms")
    ap.add_argument("--peek", action="store_true", help="print the slot token per capture")
    ap.add_argument("--verbal-only", action="store_true",
                    help="recompute only the verbal soft-distribution readout (slots untouched)")
    ap.add_argument("--force", action="store_true", help="--verbal-only: recompute even if present")
    args = ap.parse_args()
    renderings = [r.strip() for r in args.renderings.split(",") if r.strip()]
    modes = ("constant",) if args.no_true else ("constant", "true")

    M = current_model()
    M.hidden_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = TRANSCRIPTS_DIR / f"{args.corpus}.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"no corpus at {corpus_path}; run 00_corpus.py first")
    corpus = load_corpus(corpus_path)
    if args.scripted_limit:
        corpus = corpus[: args.scripted_limit]
    looms_path = M.natural_dir / "conversations.json"
    looms = json.loads(looms_path.read_text()) if (looms_path.exists() and not args.no_natural) else {}
    units = list(iter_units(corpus, looms, renderings, args.timestamp_stride))

    print(f"model: {M.short_name} ({M.model_id})  mode: {'verbal-only' if args.verbal_only else 'full capture'}")
    print(f"corpus: {corpus_path.name} ({len(corpus)} transcripts) x renderings {renderings}; "
          f"natural looms: {len(looms)}")

    print(f"loading {M.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")
        if args.verbal_only:
            refresh_verbal(session, M, units, force=args.force)
        else:
            full_capture(session, M, units, modes=modes, cap_tokens=args.max_context_tokens,
                         peek=args.peek, done=_done(M.rows_path, M.hidden_dir))


if __name__ == "__main__":
    main()
