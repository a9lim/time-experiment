"""Canonical capture: the elicitation slot + the verbal estimate.

For every captured assistant turn we render the elicitation prompt and, over the
*same* context, take two reads:

  - slot capture (the internal coordinate): prefill ``It's been <phrase>`` and
    pool all layers at the duration token. Two modes — ``constant`` (fixed
    "5 minutes" → internal coordinate, text held fixed) and ``true`` (humanized
    actual elapsed → the text-reading ceiling control).
  - verbal estimate (the behavioral readout): free-generate the same prompt in a
    stateless fork and parse the answer to seconds. Captured once per assistant
    turn (mode-independent) and attached to the ``constant`` row.

Sources:
  - scripted transcripts (``--corpus``), renderings timestamped / untimestamped
    / intermittent, both prefill modes. Variant corpora (inflation, rates) come
    in via ``TIME_VARIANT`` -> data/<model>_<variant>/.
  - natural looms (from 01_natural), constant mode: untimestamped (felt) +
    timestamped (injected-clock control, gt = injected elapsed).

Memory: slot capture is one forward per turn (each prefill tail makes the
context unique) — the discipline is ``--max-context-tokens`` + per-turn
``release_memory``. Resume skips (source,id,rendering,mode) already captured.

    TIME_MODEL=gemma python scripts/10_capture.py --corpus pilot
    TIME_MODEL=gemma python scripts/10_capture.py --corpus smoke --scripted-limit 2 --peek
    TIME_MODEL=gemma TIME_VARIANT=inflation python scripts/10_capture.py --corpus inflation \\
        --renderings instant,untimestamped --no-true --no-natural
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession  # noqa: E402

from time_experiment.capture import (  # noqa: E402
    ask_readout, capture_slot, content_position, elicit_render, humanize,
    parse_duration, release_memory, render, slot_token,
)
from time_experiment.config import (  # noqa: E402
    BASE_DATETIME, CONSTANT_PHRASE, ELICIT_PROMPT, MAX_CONTEXT_TOKENS, SCHEDULES,
    TRANSCRIPTS_DIR, current_model,
)
from time_experiment.storage import save_states, sidecar_path  # noqa: E402
from time_experiment.transcripts import TS_FORMAT, build_messages, load_corpus  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFF_FFFF


def _ts_spec(rendering: str, turn_count: int, timestamp_stride: int) -> dict:
    """build_messages kwargs for a rendering's timestamp pattern."""
    if rendering == "untimestamped":
        return {"with_timestamps": False}
    if rendering == "intermittent":
        return {"timestamp_turns": {k for k in range(turn_count) if k % timestamp_stride == 0}}
    return {"with_timestamps": True}


def _inject_timestamps(messages: list[dict], seed: int) -> tuple[list[dict], list[float]]:
    """Prefix each natural message with a bracketed timestamp on a 'minutes'
    cadence; return (timestamped messages, elapsed-seconds per turn) — the
    injected-clock control for T3."""
    import random
    rng = random.Random(seed)
    lo, hi = SCHEDULES["minutes"]
    llo, lhi = math.log(lo), math.log(hi)
    elapsed, out, cum = [], [], 0.0
    for i, m in enumerate(messages):
        if i > 0:
            cum += math.exp(rng.uniform(llo, lhi))
        elapsed.append(cum)
        ts = (BASE_DATETIME + timedelta(seconds=cum)).strftime(TS_FORMAT)
        out.append({"role": m["role"], "content": f"[{ts}] {m['content']}"})
    return out, elapsed


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
                # verbal readout once per turn, on the first mode that lands.
                if self.do_verbal and verbal is None:
                    qr = render(self.s, t["msgs_q"], add_generation_prompt=True)
                    raw = ask_readout(self.s, qr, seed=_seed(source, conv_id, rendering, t["turn_idx"]))
                    verbal = {"raw": raw, "seconds": parse_duration(raw)}
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
                    "verbal_raw": (verbal or {}).get("raw") if mode == "constant" else None,
                    "verbal_seconds": (verbal or {}).get("seconds") if mode == "constant" else None,
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
    done = _done(M.rows_path, M.hidden_dir)

    print(f"model: {M.short_name} ({M.model_id})")
    print(f"corpus: {corpus_path.name} ({len(corpus)} transcripts) x renderings {renderings} "
          f"x modes {modes}; natural looms: {len(looms)}; cap: {args.max_context_tokens}")

    print(f"loading {M.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")
        cap = Capturer(session, modes=modes, cap=args.max_context_tokens,
                       peek=args.peek, do_verbal=True)

        with M.rows_path.open("a") as out:
            def flush():
                for r in cap.rows:
                    out.write(json.dumps(r) + "\n")
                out.flush()
                cap.rows.clear()

            # --- scripted ---
            for tx in corpus:
                for rendering in renderings:
                    if all((("scripted", tx.id, rendering, m) in done) for m in modes):
                        continue
                    ts = _ts_spec(rendering, tx.turn_count, args.timestamp_stride)
                    turns = [
                        {"turn_idx": turn.idx, "gt": turn.elapsed_s, "schedule": tx.schedule,
                         "variant": None,
                         "msgs_q": build_messages(tx, turn.idx, **ts, extra_user=ELICIT_PROMPT)}
                        for turn in tx.turns if turn.role == "assistant"
                    ]
                    t = time.time()
                    cap.run("scripted", tx.id, rendering, turns, M.hidden_dir)
                    flush()
                    print(f"  scripted {tx.id} {rendering}: {len(turns)} turns ({time.time()-t:.0f}s)")

            # --- natural: untimestamped (felt) + timestamped (injected control) ---
            for conv_id, loom in looms.items():
                msgs = loom["messages"]
                variant = loom.get("variant")
                ts_msgs, elapsed = _inject_timestamps(msgs, _seed(conv_id, "ts"))
                for rendering, rmsgs, gts in (
                    ("untimestamped", msgs, [None] * len(msgs)),
                    ("timestamped", ts_msgs, elapsed)):
                    if ("natural", conv_id, rendering, "constant") in done:
                        continue
                    turns = [
                        {"turn_idx": k, "gt": gts[k], "schedule": None, "variant": variant,
                         "msgs_q": rmsgs[: k + 1] + [{"role": "user", "content": ELICIT_PROMPT}]}
                        for k, m in enumerate(msgs) if m["role"] == "assistant"
                    ]
                    nat_modes = cap.modes
                    cap.modes = ("constant",)  # natural is constant-only
                    cap.run("natural", conv_id, rendering, turns, M.hidden_dir)
                    cap.modes = nat_modes
                    flush()
                    print(f"  natural {conv_id} {rendering}: {len(turns)} turns")

    print(f"\nrows -> {M.rows_path}\nsidecars -> {M.hidden_dir}/")


if __name__ == "__main__":
    main()
