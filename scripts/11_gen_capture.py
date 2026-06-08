"""Arm G capture: generation-side time (the T4 corpus).

Generate long neutral responses and capture the per-token residual-stream
trajectory of the rollout (``SamplingConfig(return_hidden=True)`` -> a
``(T_gen, L, D)`` stack per generation). At strides, fork a stateless readout
asking how long it *feels* like the model has been writing — the first-person
felt-production-time, the generation-side analog of the felt readout.

``50_generation`` analyzes these offline: does producing tokens drive the
reading-elapsed axis (the slot probe's direction), or is it just position?

    TIME_MODEL=gemma python scripts/11_gen_capture.py                       # 5 prompts x 3 seeds
    TIME_MODEL=gemma python scripts/11_gen_capture.py --limit 1 --n-seeds 1 --max-tokens 96  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SamplingConfig, SaklasSession  # noqa: E402

from time_experiment.capture import (  # noqa: E402
    capture_slot, dist_entropy, elicit_render, release_memory, render, verbal_distribution,
)
from time_experiment.config import CONSTANT_PHRASE, ELICIT_PROMPT, current_model  # noqa: E402

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
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="generations per prompt (distinct seeds) — gives within-topic "
                         "variance so A1/A4 get CIs instead of n=1 anecdotes")
    args = ap.parse_args()

    M = current_model()
    hid = M.gen_dir / "hidden"
    sliced = M.gen_dir / "sliced"      # forked in-domain elicitation slots (per stride)
    hid.mkdir(parents=True, exist_ok=True)
    sliced.mkdir(parents=True, exist_ok=True)
    items = list(PROMPTS.items())[: args.limit or None]
    print(f"model: {M.short_name}  prompts: {len(items)} x {args.n_seeds} seeds = "
          f"{len(items) * args.n_seeds} generations  max_tokens: {args.max_tokens}")

    rows: list[dict] = []
    print(f"loading {M.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(M.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time()-t0:.1f}s")

        for gid, prompt in items:
            head = render(session, [{"role": "user", "content": prompt}], add_generation_prompt=True)
            for k in range(args.n_seeds):
                t = time.time()
                res = session.generate(
                    head, sampling=SamplingConfig(temperature=GEN_TEMP, max_tokens=args.max_tokens,
                                                  seed=_seed(gid, k), return_hidden=True),
                    stateless=True, raw=True, thinking=False)
                hs = res.hidden_states
                if not hs:
                    print(f"  [{gid}#{k}] no hidden_states returned — skipping"); continue
                layers = sorted(int(L) for L in hs)
                H = np.stack([hs[L].to(torch.float32).cpu().numpy() for L in layers], axis=1)  # (T,L,D)
                T = H.shape[0]
                stem = f"{gid}__s{k}"   # 50_generation parses topic = stem.split("__")[0]
                np.savez_compressed(hid / f"{stem}.npz", H=H, layers=np.array(layers, np.int64))
                (M.gen_dir / f"{stem}.txt").write_text(res.text)

                # strided felt-production readouts: reconstruct the partial response,
                # then read the felt-writing duration as the same soft grid distribution
                # used by the reading side (no sampling, no refusals).
                gen_ids = session.tokenizer(res.text, add_special_tokens=False)["input_ids"]
                felt = []
                slot_stack, slot_strides, slot_ctx, slot_layers = [], [], [], None
                for s in list(range(args.stride, T + 1, args.stride)) or [T]:
                    partial = session.tokenizer.decode(gen_ids[:s])
                    base = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": partial}]
                    # (1) writing-felt phenomenology (FELT_Q) — the distinctively
                    # generation-side question; behavioral only.
                    fs, fdist = verbal_distribution(session, base + [{"role": "user", "content": FELT_Q}])
                    # (2) spliced CANONICAL elicitation: cut here, fork, and run the
                    # exact reading-side instrument (ELICIT_PROMPT + constant prefill)
                    # so the probe reads an IN-DOMAIN slot, not the ~19x-off-manifold
                    # raw mid-stream token. Symmetric with T1/T2; 50_generation reads
                    # the EV probe off these (A1' spliced).
                    elicit = base + [{"role": "user", "content": ELICIT_PROMPT}]
                    states, ctx = capture_slot(session, elicit_render(session, elicit, CONSTANT_PHRASE))
                    es, edist = verbal_distribution(session, elicit)
                    slot_layers = sorted(states)
                    slot_stack.append(np.stack([states[L] for L in slot_layers]))   # (L,D)
                    slot_strides.append(s); slot_ctx.append(int(ctx))
                    felt.append({"s": s, "ctx_tokens": int(ctx),
                                 "felt_s": fs, "felt_entropy": round(dist_entropy(fdist), 4),
                                 "felt_dist": [round(float(x), 5) for x in fdist],
                                 "elicit_s": es, "elicit_entropy": round(dist_entropy(edist), 4)})
                    release_memory(session.device)
                if slot_stack:
                    np.savez_compressed(sliced / f"{stem}.npz",
                                        slots=np.stack(slot_stack).astype(np.float32),
                                        strides=np.array(slot_strides, np.int64),
                                        ctx_tokens=np.array(slot_ctx, np.int64),
                                        layers=np.array(slot_layers, np.int64))
                rows.append({"gen_id": gid, "seed_idx": k, "n_tokens": int(T),
                             "layers": layers, "felt": felt})
                release_memory(session.device)
                print(f"  [{gid}#{k}] T={T} tokens, {len(felt)} elicit/felt checkpoints ({time.time()-t:.0f}s)")

    (M.gen_dir / "gen_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\ntrajectories -> {hid}/  in-domain slots -> {sliced}/  rows -> {M.gen_dir}/gen_rows.jsonl")


if __name__ == "__main__":
    main()
