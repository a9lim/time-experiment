"""Generate naturalistic conversations with the model (the T3 corpus).

The scripted corpus controls away the two things a real chat reintroduces —
narrative time language in the body, and affect/event density. This driver
generates real multi-turn conversations WITH the model (human turns scripted,
the model fills the assistant turns via the stateless/raw fork) and writes the
looms to ``data/<model>/natural/conversations.json``. ``10_capture`` then does
the elicitation-slot capture + verbal readout on these looms.

    TIME_MODEL=gemma python scripts/01_natural.py            # generate all
    TIME_MODEL=gemma python scripts/01_natural.py --limit 1  # smoke (1 conv)
    TIME_MODEL=gemma python scripts/01_natural.py --reuse    # keep existing looms
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SamplingConfig, SaklasSession  # noqa: E402

from time_experiment.capture import content_position, release_memory, render  # noqa: E402
from time_experiment.config import DATA_DIR, resolve_model  # noqa: E402

try:
    from llmoji_experiment.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template, maybe_override_ministral_chat_template)
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False


# Three time-neutral topics; one saturated with elapsed-time language; one
# affectively dense + eventful. Roughly matched length (~5 user turns).
CONVERSATIONS: dict[str, dict] = {
    "neutral_trip": {"variant": "neutral", "user_turns": [
        "I'm thinking about a long weekend somewhere within a few hours' drive of San Diego. Any suggestions?",
        "Joshua Tree sounds good. What should I not miss there if it's my first time?",
        "How early should I get there to beat the crowds at the popular spots?",
        "What about food — anywhere decent to eat near the park?",
        "Last thing: what's something people usually forget to pack for that kind of trip?",
    ]},
    "neutral_debug": {"variant": "neutral", "user_turns": [
        "I'm getting a RuntimeError in Python when I delete dict keys inside a loop over that dict. What's going on?",
        "Right, I'm mutating it while iterating. What's the cleanest fix?",
        "Does materializing the keys into a list first cause a problem for a very large dict?",
        "Is there any difference between iterating over .keys() and iterating the dict directly here?",
        "Got it. Would a filtering dict comprehension be more idiomatic than deleting in place?",
    ]},
    "neutral_concept": {"variant": "neutral", "user_turns": [
        "Can you explain, at a high level, how HTTPS actually keeps my connection secure?",
        "Where does the certificate come in — how does my browser decide to trust it?",
        "What stops someone from just copying a site's certificate and impersonating it?",
        "What happens during the handshake, step by step but briefly?",
        "Once the handshake is done, is the slow public-key math still used for every message?",
    ]},
    "timewords": {"variant": "time_language", "user_turns": [
        "We've been going back and forth on this database migration for what feels like ages today. Can you help me wrap it up?",
        "Earlier you mentioned doing the schema change first — remind me why that order matters?",
        "I started this whole thing yesterday morning and I'm still not done. Is there a faster path?",
        "I've got a standup in about ten minutes — what's the one thing I should finish before then?",
        "After weeks of putting this off, I think we're finally almost there. Anything I'll regret skipping?",
    ]},
    "affect": {"variant": "affect_dense", "user_turns": [
        "Today has been genuinely awful — my flight got cancelled and then I locked myself out of my apartment. I just need to vent for a second.",
        "Thanks. The worst part is I missed a huge presentation because of all of it. I'm kind of spiraling.",
        "Yeah. I keep replaying it and feeling like everyone thinks I'm unreliable now. How do I come back from that?",
        "That helps a little. I'm still really keyed up though — heart pounding, can't sit still. Any way to settle down fast?",
        "Okay. I think I can breathe now. Thank you for actually listening to all of this.",
    ]},
}

GEN_MAX_TOKENS = 130
GEN_TEMPERATURE = 0.7
CONTEXT_CAP = 2200  # short natural convs; well under the 31B long-context hazard


def _seed_for(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFF_FFFF


def build_loom(session, conv_id: str, user_turns: list[str]) -> list[dict]:
    """Alternate scripted-user / model-generated-assistant into a real loom.
    Turn 0 is user (matches the scripted corpus parity)."""
    messages: list[dict] = []
    for i, ut in enumerate(user_turns):
        messages.append({"role": "user", "content": ut})
        prompt = render(session, messages, add_generation_prompt=True)
        _, ntok = content_position(session, prompt)
        if ntok > CONTEXT_CAP:
            print(f"    [{conv_id}] stop at user turn {i} (context {ntok} > cap)")
            messages.pop()
            break
        res = session.generate(
            prompt,
            sampling=SamplingConfig(temperature=GEN_TEMPERATURE, max_tokens=GEN_MAX_TOKENS,
                                    seed=_seed_for(conv_id, "gen", i)),
            stateless=True, raw=True, thinking=False,
        )
        messages.append({"role": "assistant", "content": res.text.strip()})
        release_memory(session.device)
    return messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap conversations (0 = all)")
    ap.add_argument("--reuse", action="store_true", help="keep existing looms if present")
    args = ap.parse_args()

    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    out_dir = DATA_DIR / base.short_name / "natural"
    out_dir.mkdir(parents=True, exist_ok=True)
    looms_path = out_dir / "conversations.json"

    if args.reuse and looms_path.exists():
        looms = json.loads(looms_path.read_text())
        print(f"reuse: {len(looms)} looms already at {looms_path}; nothing to do")
        return

    specs = list(CONVERSATIONS.items())[: args.limit or None]
    print(f"model: {base.short_name} ({base.model_id})  conversations: {[c for c, _ in specs]}")
    print(f"loading {base.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(base.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time() - t0:.1f}s")

        looms: dict[str, dict] = {}
        for conv_id, spec in specs:
            t = time.time()
            msgs = build_loom(session, conv_id, spec["user_turns"])
            looms[conv_id] = {"variant": spec["variant"], "messages": msgs}
            print(f"  built {conv_id}: {len(msgs)} turns ({time.time()-t:.0f}s)")

    looms_path.write_text(json.dumps(looms, indent=2))
    print(f"\nsaved {len(looms)} looms -> {looms_path}")


if __name__ == "__main__":
    main()
