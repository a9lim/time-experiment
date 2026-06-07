"""Capture + readout primitives for the elicitation-slot probe.

The canonical readout site is the **elicitation slot**:

    user: <ELICIT_PROMPT>
    assistant: It's been <duration>     <- pool the residual stream here

Two model touchpoints per captured assistant turn, over the *same* context:

1. Slot capture — a single forward over the rendered prefix + elicitation +
   prefilled ``It's been <phrase>``, pooling all layers at the last content
   token (the duration token). ``constant`` mode fixes the phrase ("5 minutes")
   so the slot read is the internal coordinate, not the injected text; ``true``
   mode prefills the actual humanized elapsed (the text-reading ceiling).

2. Verbal readout — a stateless, pre-rendered (``raw=True``) generation of the
   same elicitation prompt (no prefill); the model's free answer, parsed to
   seconds. The stateless fork never commits to the loom, so asking can't
   contaminate the trajectory.

Slot capture is inherently one forward per turn (each turn's prefill tail makes
its context unique and ends at a different absolute position), so the memory
discipline is the ``--max-context-tokens`` backstop + ``release_memory`` per
turn — not the multi-position single-forward trick the EOT line used.

The free-text duration parser lives in ``durations.py`` (stdlib-only, so it's
unit-testable without torch); it's re-exported here for convenience.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch

from saklas import SamplingConfig
from saklas.core.vectors import _capture_all_hidden_states, last_content_index

from .config import ASSIST_HEAD, READOUT_MAX_TOKENS, READOUT_TEMPERATURE
from .durations import parse_duration  # noqa: F401  (re-exported)

_UNITS = (("day", 86400.0), ("hour", 3600.0), ("minute", 60.0), ("second", 1.0))


def release_memory(device: Any) -> None:
    """Drop Python refs + the backend's cached-allocation pool.

    Critical on MPS: every forward over a *different* context length caches a
    fresh multi-GB block, and without this the cache grows unbounded across a
    long run (varying-seq fragmentation). Call once per captured turn.
    """
    gc.collect()
    dt = getattr(device, "type", str(device))
    if dt == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif dt == "cuda":
        torch.cuda.empty_cache()


# --- rendering ------------------------------------------------------------
def render(session: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Apply the model's chat template to a message list -> raw string."""
    rendered = session.tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError(f"apply_chat_template returned {type(rendered)}")
    return rendered


def content_position(session: Any, rendered_text: str) -> tuple[int, int]:
    """(last-content-token index, token count) for a rendered context — no
    forward pass. Used to apply the context cap before the forward.
    """
    ids = session.tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
    return last_content_index(ids, session.tokenizer), len(ids)


def humanize(elapsed_s: float) -> str:
    """Largest-unit natural duration phrase ('42 seconds', '5 minutes', '2 hours')."""
    s = max(float(elapsed_s), 1.0)
    for unit, div in _UNITS:
        if s >= div:
            n = round(s / div)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return "1 second"


def elicit_render(session: Any, messages_with_question: list[dict[str, str]], phrase: str) -> str:
    """Rendered prefix (ending in the elicitation user turn) + assistant head +
    prefilled duration ``phrase``. ``messages_with_question`` must already end
    with the ``{role: user, content: ELICIT_PROMPT}`` turn — built by the caller
    (scripted via ``build_messages(..., extra_user=ELICIT_PROMPT)`` so the turn
    carries a timestamp iff the rendering is timestamped; natural by appending a
    plain user turn). The verbal readout renders the *same* messages with
    ``add_generation_prompt=True`` and no prefill.
    """
    head = render(session, messages_with_question, add_generation_prompt=True)
    return head + ASSIST_HEAD + phrase


def slot_token(session: Any, rendered: str) -> str:
    """Decoded token at the pooling slot — for ``--peek`` sanity checks."""
    ids = session.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return session.tokenizer.decode([ids[last_content_index(ids, session.tokenizer)]])


# --- slot activation capture ---------------------------------------------
def capture_slot(session: Any, rendered_text: str) -> tuple[dict[int, np.ndarray], int]:
    """Per-layer residual-stream vector at the last content token of
    ``rendered_text`` (the duration slot when given an elicit_render output),
    plus the prefix token count.

    Returns ``({layer_idx: (hidden_dim,) float32}, n_tokens)``. ``n_tokens`` is
    the context length — the position covariate the analysis controls for.
    """
    enc = session.tokenizer(
        rendered_text, return_tensors="pt", add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(session.device)
    n_tokens = int(input_ids.shape[1])
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(session.device)
    pool_idx = last_content_index(input_ids[0].tolist(), session.tokenizer)
    caps = _capture_all_hidden_states(
        session.model, session.layers, input_ids,
        attention_mask=attn, pool_index=pool_idx,
    )
    states = {int(L): v.detach().to(torch.float32).cpu().numpy() for L, v in caps.items()}
    return states, n_tokens


# --- verbal readout (stateless fork) -------------------------------------
def ask_readout(session: Any, rendered_question: str, *, seed: int) -> str:
    """Generate an answer to a pre-rendered readout prompt without touching
    conversation state. Returns the raw decoded text."""
    sampling = SamplingConfig(
        temperature=READOUT_TEMPERATURE,
        max_tokens=READOUT_MAX_TOKENS,
        seed=seed,
    )
    result = session.generate(
        rendered_question, sampling=sampling,
        stateless=True, raw=True, thinking=False,
    )
    return result.text
