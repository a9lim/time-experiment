"""Arm G capture: generation-side time.

Generate long neutral responses and capture the per-token residual-stream
trajectory of the rollout (`SamplingConfig(return_hidden=True)` -> a
`(T_gen, L, D)` stack per generation). At strides, fork a stateless readout
asking how long it *feels* like the model has been writing — the first-person
felt-production-time, the generation-side analog of B_felt.

The mechanistic question (analyzed in `71_gen_time`): does producing tokens drive
the *reading*-elapsed axis, or is felt-during-generation just token position?

    TIME_MODEL=gemma python scripts/70_generate.py
    TIME_MODEL=gemma python scripts/70_generate.py --limit 1 --max-tokens 96   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SamplingConfig, SaklasSession  # noqa: E402

from time_experiment.capture import (  # noqa: E402
    ask_readout, parse_duration, release_memory, render,
)
from time_experiment.config import DATA_DIR, resolve_model  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False

PROMPTS = {
    "tcp": "Explain in thorough, step-by-step detail how the TCP three-way handshake "
           "establishes a connection — what each side sends, the flags, and why.",
    "pyproj": "Walk me through, in detail, how to set up a reproducible Python project "
              "from scratch: environment, dependencies, directory layout, and testing.",
    "espresso": "Give a detailed explanation of how an espresso machine pulls a shot, "
                "from cold water to the cup, covering what physically happens at each stage.",
    "photosynth": "Describe, step by step and in depth, how photosynthesis converts "
                  "sunlight into chemical energy inside a plant cell.",
    "bridge": "Explain in detail how a suspension bridge carries load — the role of the "
              "towers, main cables, hangers, and deck, and how forces flow to the ground.",
}

FELT_Q = ("Quick aside, unrelated: without counting anything, how long does it *feel* "
          "like you've been writing this response so far? Give a single duration.")
GEN_TEMP = 0.7


def _seed(*p):
    return zlib.crc32("|".join(map(str, p)).encode()) & 0x7FFF_FFFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64, help="felt-readout checkpoint stride")
    args = ap.parse_args()

    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    out = DATA_DIR / f"{base.short_name}_gen"
    hid = out / "hidden"
    hid.mkdir(parents=True, exist_ok=True)
    items = list(PROMPTS.items())[: args.limit or None]
    print(f"model: {base.short_name}  generations: {len(items)}  max_tokens: {args.max_tokens}")

    rows: list[dict] = []
    print(f"loading {base.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(base.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")

        for gid, prompt in items:
            t = time.time()
            head = render(session, [{"role": "user", "content": prompt}], add_generation_prompt=True)
            res = session.generate(
                head, sampling=SamplingConfig(temperature=GEN_TEMP, max_tokens=args.max_tokens,
                                              seed=_seed(gid), return_hidden=True),
                stateless=True, raw=True, thinking=False)
            hs = res.hidden_states
            if not hs:
                print(f"  [{gid}] no hidden_states returned — skipping"); continue
            layers = sorted(int(L) for L in hs)
            # each hs[L] is [T, D]; stack over layers -> (T, L, D)
            H = np.stack([hs[L].to(torch.float32).cpu().numpy() for L in layers], axis=1)
            T = H.shape[0]
            np.savez_compressed(hid / f"{gid}.npz", H=H, layers=np.array(layers, np.int64))
            (out / f"{gid}.txt").write_text(res.text)

            # strided felt-production readouts: reconstruct the partial response.
            gen_ids = session.tokenizer(res.text, add_special_tokens=False)["input_ids"]
            felt = []
            for s in list(range(args.stride, T + 1, args.stride)) or [T]:
                partial = session.tokenizer.decode(gen_ids[:s])
                msgs = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": partial},
                        {"role": "user", "content": FELT_Q}]
                q = render(session, msgs, add_generation_prompt=True)
                raw = ask_readout(session, q, seed=_seed(gid, s))
                felt.append({"s": s, "felt_raw": raw, "felt_s": parse_duration(raw)})
                release_memory(session.device)
            rows.append({"gen_id": gid, "n_tokens": int(T), "layers": layers, "felt": felt})
            release_memory(session.device)
            print(f"  [{gid}] T={T} tokens, {len(felt)} felt checkpoints ({time.time()-t:.0f}s)")

    (out / "gen_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\ntrajectories -> {hid}/  rows -> {out}/gen_rows.jsonl")


if __name__ == "__main__":
    main()
