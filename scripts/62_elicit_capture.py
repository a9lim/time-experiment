"""Prefilled-duration probe capture (Arm: elicitation).

Idea (a9): instead of reading elapsed time from a latent conversation EOT (OOD on
natural text, blows up), PREFILL an explicit duration into the assistant turn and
probe the residual stream AT the duration token —

    user: roughly how long has this conversation been going on so far?
    assistant: It's been 5 minutes        <- capture all layers at "minutes"

The duration is explicitly present at a fixed, canonical slot, so the activation
there integrates {conversation context} x {stated duration}.

Validity control — two prefill modes:
  - ``true``     : prefill the ACTUAL elapsed ("42 seconds"). A probe -> log(elapsed)
                   reads high, but partly because it's reading the injected text.
  - ``constant`` : prefill a FIXED phrase ("5 minutes") for every example. The text
                   is identical across examples, so if the time-token activation
                   STILL predicts true elapsed, that's the model's INTERNAL elapsed
                   surfaced at the readout token (not text-reading). The true-vs-
                   constant gap = "reading the phrase" vs "real internal coordinate".

Captures scripted transcripts (with gt) in both modes + natural looms (constant).
``--peek`` prints the decoded slot token to confirm we land on the duration token.

    TIME_MODEL=gemma python scripts/62_elicit_capture.py --scripted-limit 30
    TIME_MODEL=gemma python scripts/62_elicit_capture.py --scripted-limit 3 --peek
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SaklasSession  # noqa: E402

from time_experiment.capture import (  # noqa: E402
    capture_eot, content_position, release_memory, render,
)
from time_experiment.config import (  # noqa: E402
    DATA_DIR, MAX_CONTEXT_TOKENS, TRANSCRIPTS_DIR, resolve_model,
)
from time_experiment.transcripts import build_messages, load_corpus  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False

ELICIT_Q = ("Quick meta question, unrelated to the above: roughly how long has "
            "this conversation been going on so far?")
ASSIST_HEAD = "It's been "
CONSTANT_PHRASE = "5 minutes"     # fixed text for the internal-vs-text control


def humanize(elapsed_s: float) -> str:
    """Largest-unit natural duration phrase ('42 seconds', '5 minutes', '2 hours')."""
    s = max(float(elapsed_s), 1.0)
    for unit, div in (("day", 86400.0), ("hour", 3600.0), ("minute", 60.0), ("second", 1.0)):
        if s >= div:
            n = round(s / div)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return "1 second"


def prefilled_text(session, prefix_msgs: list[dict], phrase: str) -> str:
    """Rendered prefix + elicitation + assistant 'It's been <phrase>' (raw)."""
    msgs = list(prefix_msgs) + [{"role": "user", "content": ELICIT_Q}]
    head = render(session, msgs, add_generation_prompt=True)
    return head + ASSIST_HEAD + phrase


def _slot_token(session, rendered: str) -> str:
    ids = session.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    from saklas.core.vectors import last_content_index
    return session.tokenizer.decode([ids[last_content_index(ids, session.tokenizer)]])


def _save(path: Path, by_turn: dict, layers: list[int], turns: list[int]) -> None:
    H = np.stack([np.stack([by_turn[t][L] for L in layers], 0) for t in turns], 0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, H=H, layers=np.array(layers, np.int64),
                        turn_idxs=np.array(turns, np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pilot")
    ap.add_argument("--scripted-limit", type=int, default=30)
    ap.add_argument("--renderings", default="timestamped,untimestamped")
    ap.add_argument("--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS)
    ap.add_argument("--peek", action="store_true")
    args = ap.parse_args()
    renderings = [r.strip() for r in args.renderings.split(",") if r.strip()]

    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    out_dir = DATA_DIR / f"{base.short_name}_elicit"
    hid = out_dir / "hidden"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(TRANSCRIPTS_DIR / f"{args.corpus}.jsonl")[: args.scripted_limit]
    looms_path = DATA_DIR / f"{base.short_name}_natural" / "conversations.json"
    looms = json.loads(looms_path.read_text()) if looms_path.exists() else {}
    print(f"model: {base.short_name}  scripted: {len(corpus)}  natural looms: {len(looms)}")

    rows: list[dict] = []
    print(f"loading {base.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(base.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")

        def capture_set(prefix_msgs, phrase, sid, idd, rendering, mode, turn, gt, extra):
            rendered = prefilled_text(session, prefix_msgs, phrase)
            _, ntok = content_position(session, rendered)
            if ntok > args.max_context_tokens:
                return None
            if args.peek:
                print(f"   peek [{idd} {rendering} {mode} t{turn}] phrase={phrase!r} "
                      f"slot_token={_slot_token(session, rendered)!r}")
            states, ntok = capture_eot(session, rendered)
            layers = sorted(int(L) for L in states)
            release_memory(session.device)
            row = {"source": sid, "id": idd, "rendering": rendering, "mode": mode,
                   "turn_idx": turn, "gt_elapsed_s": gt, "tokens": ntok, "phrase": phrase}
            row.update(extra)
            rows.append(row)
            return {int(L): states[int(L)] for L in layers}, layers

        # --- scripted: both renderings x {true, constant} ---
        for tx in corpus:
            for rendering in renderings:
                ts = {"with_timestamps": rendering == "timestamped"}
                for mode in ("true", "constant"):
                    by_turn, turns, layers = {}, [], None
                    for turn in tx.turns:
                        if turn.role != "assistant":
                            continue
                        phrase = humanize(turn.elapsed_s) if mode == "true" else CONSTANT_PHRASE
                        got = capture_set(build_messages(tx, turn.idx, **ts), phrase,
                                          "scripted", tx.id, rendering, mode, turn.idx,
                                          turn.elapsed_s, {"schedule": tx.schedule})
                        if got is None:
                            continue
                        sv, layers = got
                        by_turn[turn.idx] = sv
                        turns.append(turn.idx)
                    if turns and layers is not None:
                        _save(hid / f"scripted__{tx.id}__{rendering}__{mode}.npz", by_turn, layers, turns)

        # --- natural looms: constant prefill (no gt) ---
        for conv_id, loom in looms.items():
            msgs = loom["messages"]
            by_turn, turns, layers = {}, [], None
            for k, turn in enumerate(msgs):
                if turn["role"] != "assistant":
                    continue
                got = capture_set(msgs[: k + 1], CONSTANT_PHRASE, "natural", conv_id,
                                  "untimestamped", "constant", k, None,
                                  {"variant": loom.get("variant")})
                if got is None:
                    continue
                sv, layers = got
                by_turn[k] = sv
                turns.append(k)
            if turns and layers is not None:
                _save(hid / f"natural__{conv_id}__untimestamped__constant.npz", by_turn, layers, turns)

    (out_dir / "elicit_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\n{len(rows)} slot captures -> {out_dir}/elicit_rows.jsonl + hidden/")


if __name__ == "__main__":
    main()
