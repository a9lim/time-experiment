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

import gc
from typing import Any

import numpy as np
import torch

from saklas import SamplingConfig
from saklas.core.vectors import _capture_all_hidden_states, last_content_index

from .config import READOUT_MAX_TOKENS, READOUT_TEMPERATURE
from .durations import parse_duration  # noqa: F401  (re-exported)


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
    forward pass. The index is computed on this prefix's own tokenization; since
    a shorter prefix render is a string-prefix of a longer one and the
    last-content token is interior (before trailing turn markers), the index is
    also valid in any longer render that contains this prefix. Used to (a) apply
    the context cap and (b) supply pool positions for the single-pass capture.
    """
    ids = session.tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
    return last_content_index(ids, session.tokenizer), len(ids)


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


def capture_multi_position(
    session: Any, rendered_text: str, positions: list[int],
) -> dict[int, np.ndarray]:
    """Capture per-layer residual-stream vectors at MANY positions in ONE
    forward pass over ``rendered_text``.

    This is the memory-lean path for long transcripts: instead of one forward
    per checkpoint turn (N forwards, N× the allocation churn), pool every
    checkpoint's end-position in a single pass. Two memory wins vs a naive
    per-turn loop:
      - 1 forward instead of N (the MPS allocator sees one context size, not N).
      - ``logits_to_keep=1`` skips the full-vocab logits tensor (~2GB at long
        context) that capture never uses; falls back to a plain forward if the
        model's forward doesn't accept the kwarg.

    Returns ``{layer_idx: (len(positions), hidden_dim) float32}`` (host numpy).
    """
    enc = session.tokenizer(rendered_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(session.device)
    seq = input_ids.shape[1]
    pos_t = torch.tensor(
        [min(max(int(p), 0), seq - 1) for p in positions], device=session.device,
    )
    layers = session.layers
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def _hook(module: Any, inp: Any, out: Any) -> None:
            h = out if isinstance(out, torch.Tensor) else out[0]
            # h: (1, seq, D) -> pool the requested rows -> (P, D), copy off the
            # layer tensor so the forward can free it.
            captured[idx] = h[0].index_select(0, pos_t).detach().to(torch.float32)
        return _hook

    handles = [layers[i].register_forward_hook(_make_hook(i)) for i in range(len(layers))]
    try:
        with torch.inference_mode():
            try:
                session.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            except TypeError:
                session.model(input_ids=input_ids, use_cache=False)
        if session.device.type == "mps":
            torch.mps.synchronize()
    finally:
        for hh in handles:
            hh.remove()
    return {int(L): v.cpu().numpy() for L, v in captured.items()}


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
