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

import copy
import gc
import math
import random
from datetime import timedelta
from typing import Any

import numpy as np
import torch

from saklas.core.vectors import _capture_all_hidden_states, last_content_index

from .config import ASSIST_HEAD, BASE_DATETIME, DURATION_GRID, SCHEDULES
from .durations import parse_duration  # noqa: F401  (re-exported)
from .transcripts import TS_FORMAT

_GRID_SECONDS = np.array([s for _, s in DURATION_GRID], dtype=np.float64)
_GRID_LOG = np.log(_GRID_SECONDS)

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


# --- soft-distribution summaries (shared by capture + migration + analyses) ---
def dist_point(p: np.ndarray) -> float:
    """Robust point estimate of a grid distribution: the **log-interpolated
    median** (the 0.5 crossing of the CDF interpolated in log-seconds). Unlike the
    geometric mean ``exp(Σ p·log s)``, it is not dragged by the spurious multimodal
    tails the no-clock felt distribution grows at depth — it summarizes a split
    vote by its central mass, not by a fictitious midpoint between the modes."""
    p = np.asarray(p, dtype=np.float64)
    c = np.cumsum(p)
    j = int(np.searchsorted(c, 0.5))
    if j <= 0:
        return float(_GRID_SECONDS[0])
    if j >= len(_GRID_SECONDS):
        return float(_GRID_SECONDS[-1])
    frac = (0.5 - c[j - 1]) / max(c[j] - c[j - 1], 1e-12)
    return float(math.exp(_GRID_LOG[j - 1] + frac * (_GRID_LOG[j] - _GRID_LOG[j - 1])))


def dist_entropy(p: np.ndarray) -> float:
    """Shannon entropy (bits) of a grid distribution — the spread/uncertainty of
    the verbal estimate, co-reported so multimodality stays visible in a scalar
    (a high-entropy felt read is the gentle surfacing of 'I don't have a sense of
    time' that the soft readout was built to keep, rather than a refusal/NaN)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


# --- verbal readout (soft duration distribution) -------------------------
def verbal_distribution(session: Any, messages_with_question: list[dict[str, str]],
                        ) -> tuple[float, np.ndarray]:
    """The verbal estimate as a SOFT DISTRIBUTION over durations, read from the
    logits at the slot (the model's own ``W_U`` readout — symmetric to the
    probe's activation readout of the same slot).

    After ``It's been ``, score each ``DURATION_GRID`` phrase by its
    teacher-forced log-prob and softmax over the grid -> a distribution over how
    long the model thinks it's been. No sampling (deterministic, denoised) and no
    refusals (every turn yields a distribution); the spread is the model's
    uncertainty. Efficient: one forward over the shared prefix, then each
    multi-token candidate scored against a copy of the prefix KV cache (cheap
    continuation forwards, validated identical to brute per-candidate forwards).

    Returns ``(point_seconds, probs)`` — ``point_seconds`` is the robust
    log-interpolated median (``dist_point``); ``probs`` is the grid distribution
    (len == ``DURATION_GRID``) for offline soft analyses (entropy/mode/spread)."""
    tok, model, dev = session.tokenizer, session.model, session.device
    prefix = render(session, messages_with_question, add_generation_prompt=True) + ASSIST_HEAD
    pids = tok(prefix, add_special_tokens=False)["input_ids"]
    # candidate continuation tokens = suffix after the prefix (robust to the
    # tokenizer merging the leading space into the first duration token).
    cand_ids = []
    for phrase, _ in DURATION_GRID:
        full = tok(prefix + phrase, add_special_tokens=False)["input_ids"]
        cand_ids.append(full[len(pids):])

    with torch.inference_mode():
        po = model(torch.tensor([pids], device=dev), use_cache=True)
        past = po.past_key_values
        last_lp = torch.log_softmax(po.logits[0, -1].float(), dim=-1).cpu()
        logps = np.full(len(cand_ids), -1e30, dtype=np.float64)
        for i, cids in enumerate(cand_ids):
            if not cids:
                continue
            s = float(last_lp[cids[0]])
            if len(cids) > 1:
                past_c = copy.deepcopy(past)
                co = model(torch.tensor([cids], device=dev),
                           past_key_values=past_c, use_cache=True)
                lp = torch.log_softmax(co.logits[0].float(), dim=-1).cpu()
                del co, past_c   # free this candidate's cache copy before the next (MPS)
                for j in range(len(cids) - 1):
                    s += float(lp[j, cids[j + 1]])
            logps[i] = s
        del past
    p = np.exp(logps - logps.max())
    p /= p.sum()
    return dist_point(p), p


# --- elicitation rendering helpers (shared by capture + reverbal) ---------
def ts_spec(rendering: str, turn_count: int, timestamp_stride: int) -> dict:
    """``build_messages`` kwargs for a rendering's timestamp pattern."""
    if rendering == "untimestamped":
        return {"with_timestamps": False}
    if rendering == "intermittent":
        return {"timestamp_turns": {k for k in range(turn_count) if k % timestamp_stride == 0}}
    return {"with_timestamps": True}


def inject_timestamps(messages: list[dict], seed: int) -> tuple[list[dict], list[float]]:
    """Prefix each natural message with a bracketed timestamp on a 'minutes'
    cadence; return (timestamped messages, elapsed-seconds per turn) — the
    injected-clock control for T3."""
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
