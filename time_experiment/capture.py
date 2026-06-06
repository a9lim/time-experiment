"""Capture + readout primitives.

Two model touchpoints per captured turn:

1. EOT capture — a single forward pass over the rendered transcript prefix,
   pooling the residual stream at the last *content* token (saklas's canonical
   pooling site). This is the internal "elapsed-time coordinate" source. The
   main line is scripted, so we read activations directly rather than through
   saklas's generation-time HiddenCapture.

2. Verbal readout — a stateless, pre-rendered (``raw=True``) generation that
   asks "how long has passed?" and never commits to the loom tree. This is the
   fork in a9's design: the question can't contaminate the main trajectory.

The free-text duration parser lives in ``durations.py`` (stdlib-only, so it's
unit-testable without torch); it's re-exported here for convenience.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from saklas import SamplingConfig
from saklas.core.vectors import _capture_all_hidden_states, last_content_index

from .config import READOUT_MAX_TOKENS, READOUT_TEMPERATURE
from .durations import parse_duration  # noqa: F401  (re-exported)


# --- rendering ------------------------------------------------------------
def render(session: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Apply the model's chat template to a message list -> raw string."""
    rendered = session.tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError(f"apply_chat_template returned {type(rendered)}")
    return rendered


# --- EOT activation capture ----------------------------------------------
def capture_eot(session: Any, rendered_text: str) -> tuple[dict[int, np.ndarray], int]:
    """Per-layer residual-stream vector at the last content token of
    ``rendered_text``, plus the prefix token count.

    Returns ``({layer_idx: (hidden_dim,) float32}, n_tokens)``. ``n_tokens`` is
    the raw context length — the position covariate the factorial controls for.
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
