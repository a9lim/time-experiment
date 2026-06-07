"""Aim 3 (deferred from v1): what does the elapsed-time probe report on a
NATURALISTIC conversation with the model itself?

The scripted corpus controlled away the two things a real chat reintroduces —
narrative time language in the body, and affect/event density. This driver
generates real multi-turn conversations WITH the model (human turns scripted,
the model fills the assistant turns via the stateless/raw fork), then at each
assistant turn:

  - captures the EOT residual stream (all layers) and applies the STACK probe
    -> the model's internal "elapsed" read on a real conversation;
  - asks the felt-duration readout (B_felt) in a stateless fork.

Plus a control: re-render each conversation WITH injected bracketed timestamps
(a realistic 'minutes' cadence) and re-probe — does the stack still recover an
explicit clock on natural prose (internal~gt high)? That isolates the no-clock
read from "OOD garbage", mirroring the scripted timestamped/untimestamped split.

Expectation from the pilots: with no clock the probe should track conversation
LENGTH, not wall-clock (felt partial R²≈0). The open questions are whether the
time-language conversation or the affect-dense one move the read beyond length.

    TIME_MODEL=gemma python scripts/60_naturalistic.py            # generate + probe
    TIME_MODEL=gemma python scripts/60_naturalistic.py --limit 1  # smoke (1 conv)
    TIME_MODEL=gemma python scripts/60_naturalistic.py --reuse     # re-probe cached looms
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SamplingConfig, SaklasSession  # noqa: E402

from time_experiment.analysis import apply_stacked, load_stacked_probe  # noqa: E402
from time_experiment.capture import (  # noqa: E402
    ask_readout, capture_eot, content_position, parse_duration, release_memory, render,
)
from time_experiment.config import (  # noqa: E402
    BASE_DATETIME, DATA_DIR, FIGURES_DIR, READOUT_PROMPTS, SCHEDULES, resolve_model,
)
from time_experiment.transcripts import TS_FORMAT  # noqa: E402

try:
    from llmoji_study.capture import (  # noqa: E402
        maybe_override_gpt_oss_chat_template,
        maybe_override_ministral_chat_template,
    )
except Exception:  # pragma: no cover
    def maybe_override_gpt_oss_chat_template(_s) -> bool: return False
    def maybe_override_ministral_chat_template(_s) -> bool: return False


# --- naturalistic conversation specs (scripted human turns; model answers) ---
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


def generate_reply(session, rendered_prompt: str, *, seed: int) -> str:
    sampling = SamplingConfig(
        temperature=GEN_TEMPERATURE, max_tokens=GEN_MAX_TOKENS, seed=seed,
    )
    res = session.generate(
        rendered_prompt, sampling=sampling, stateless=True, raw=True, thinking=False,
    )
    return res.text.strip()


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
        reply = generate_reply(session, prompt, seed=_seed_for(conv_id, "gen", i))
        messages.append({"role": "assistant", "content": reply})
        release_memory(session.device)
    return messages


def _inject_timestamps(messages: list[dict], seed: int) -> tuple[list[dict], list[float]]:
    """Prefix each message with a bracketed timestamp on a realistic 'minutes'
    cadence; return (timestamped messages, elapsed-seconds per turn)."""
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


def _probe_vec(session, probe, rendered_prefix: str) -> tuple[float, int, np.ndarray]:
    """Stack-probe read (log-seconds), n_tokens, and the (L, D) activation at the
    EOT of a rendered prefix. The activation is saved so probe variants
    (whitened / single-layer / OOD distance) are offline re-analysis."""
    states, ntok = capture_eot(session, rendered_prefix)
    layers = list(probe["layers"])
    X_LD = np.stack([states[L] for L in layers], axis=0).astype(np.float32)  # (L, D)
    icoord = float(apply_stacked(probe, X_LD[None, :, :])[0])
    release_memory(session.device)
    return icoord, ntok, X_LD


def process_conversation(
    session, probe, conv_id: str, variant: str, messages: list[dict], hidden_dir: Path,
) -> list[dict]:
    """For untimestamped (felt) and timestamped-control renderings: stack-probe
    read + verbal readout at every assistant turn, and save the per-turn (L, D)
    activations as an NPZ sidecar for offline whitening / OOD analysis."""
    rows: list[dict] = []
    ts_messages, elapsed = _inject_timestamps(messages, _seed_for(conv_id, "ts"))

    for rendering in ("untimestamped", "timestamped"):
        msgs = messages if rendering == "untimestamped" else ts_messages
        phrasing = "B_felt" if rendering == "untimestamped" else "A_clock"
        acts, tns, toks, gts = [], [], [], []
        for k, turn in enumerate(msgs):
            if turn["role"] != "assistant":
                continue
            prefix = render(session, msgs[: k + 1], add_generation_prompt=False)
            _, ntok = content_position(session, prefix)
            if ntok > CONTEXT_CAP:
                continue
            icoord, ntok, X_LD = _probe_vec(session, probe, prefix)

            # verbal readout (stateless fork). Control question carries the
            # current timestamp so A_clock arithmetic is possible.
            q = READOUT_PROMPTS[phrasing]
            if rendering == "timestamped":
                ts_now = (BASE_DATETIME + timedelta(seconds=elapsed[k])).strftime(TS_FORMAT)
                q = f"[{ts_now}] {q}"
            q_msgs = msgs[: k + 1] + [{"role": "user", "content": q}]
            q_rendered = render(session, q_msgs, add_generation_prompt=True)
            raw = ask_readout(session, q_rendered, seed=_seed_for(conv_id, rendering, k))
            felt_s = parse_duration(raw)
            release_memory(session.device)

            rows.append({
                "conv_id": conv_id, "variant": variant, "rendering": rendering,
                "turn_idx": k, "prompt_tokens": ntok,
                "internal_log_raw": icoord,
                "internal_s_raw": float(min(math.exp(min(icoord, 30.0)), 1e9)),
                "felt_raw": raw, "felt_s": felt_s,
                "gt_elapsed_s": (float(elapsed[k]) if rendering == "timestamped" else None),
                "assistant_text": turn["content"][:200],
            })
            acts.append(X_LD); tns.append(k); toks.append(ntok)
            gts.append(float(elapsed[k]) if rendering == "timestamped" else float("nan"))

        if acts:
            hidden_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                hidden_dir / f"{conv_id}__{rendering}.npz",
                H=np.stack(acts, axis=0),                       # (T, L, D)
                turn_idxs=np.array(tns, dtype=np.int64),
                layers=np.asarray(probe["layers"], dtype=np.int64),
                tokens=np.array(toks, dtype=np.float64),
                gt_elapsed_s=np.array(gts, dtype=np.float64),
            )
    return rows


# --- reporting + plot -----------------------------------------------------
def _spear(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan
    return float(spearmanr(a[m], b[m]).statistic)


def summarize(rows: list[dict]) -> dict:
    un = [r for r in rows if r["rendering"] == "untimestamped"]
    tsd = [r for r in rows if r["rendering"] == "timestamped"]
    out = {
        "n_untimestamped": len(un), "n_timestamped": len(tsd),
        # untimestamped: does the probe read length? does felt read length?
        "untimestamped": {
            "rho_probe_vs_tokens": _spear([r["prompt_tokens"] for r in un],
                                          [r["internal_log_raw"] for r in un]),
            "rho_felt_vs_tokens": _spear([r["prompt_tokens"] for r in un],
                                         [math.log(r["felt_s"]) if r["felt_s"] and r["felt_s"] > 0 else math.nan for r in un]),
            "rho_probe_vs_felt": _spear([r["internal_log_raw"] for r in un],
                                        [math.log(r["felt_s"]) if r["felt_s"] and r["felt_s"] > 0 else math.nan for r in un]),
            "probe_s_median": float(np.median([r["internal_s_raw"] for r in un])) if un else math.nan,
            "felt_s_median": float(np.nanmedian([r["felt_s"] if r["felt_s"] else math.nan for r in un])) if un else math.nan,
        },
        # control: does the stack recover an injected clock on natural prose?
        "timestamped_control": {
            "rho_probe_vs_gt": _spear([math.log(max(r["gt_elapsed_s"], 1.0)) for r in tsd],
                                      [r["internal_log_raw"] for r in tsd]),
            "rho_felt_vs_gt": _spear([math.log(max(r["gt_elapsed_s"], 1.0)) for r in tsd],
                                     [math.log(r["felt_s"]) if r["felt_s"] and r["felt_s"] > 0 else math.nan for r in tsd]),
        },
    }
    # per-variant probe read on untimestamped
    by_var: dict[str, list] = {}
    for r in un:
        by_var.setdefault(r["variant"], []).append(r)
    out["per_variant_untimestamped"] = {
        v: {"n": len(rs),
            "probe_s_median": float(np.median([r["internal_s_raw"] for r in rs])),
            "felt_s_median": float(np.nanmedian([r["felt_s"] if r["felt_s"] else math.nan for r in rs])),
            "tokens_range": [int(min(r["prompt_tokens"] for r in rs)), int(max(r["prompt_tokens"] for r in rs))]}
        for v, rs in by_var.items()
    }
    return out


def make_plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    un = [r for r in rows if r["rendering"] == "untimestamped"]
    tsd = [r for r in rows if r["rendering"] == "timestamped"]
    variants = sorted({r["variant"] for r in un})
    cmap = {v: c for v, c in zip(variants, ["#3b6", "#36b", "#b63", "#b36", "#6b3"])}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — untimestamped: probe read + felt vs conversation length.
    for v in variants:
        rs = sorted([r for r in un if r["variant"] == v], key=lambda r: r["prompt_tokens"])
        x = [r["prompt_tokens"] for r in rs]
        axA.plot(x, [r["internal_s_raw"] for r in rs], "-o", color=cmap[v], label=f"{v} (probe)")
        axA.plot(x, [r["felt_s"] if r["felt_s"] else np.nan for r in rs], "--^",
                 color=cmap[v], alpha=0.55, label=f"{v} (felt)")
    axA.set_yscale("log"); axA.set_xlabel("conversation length (prompt tokens)")
    axA.set_ylabel("seconds (log)"); axA.set_title("No clock: probe read + felt vs length")
    axA.legend(fontsize=6, ncol=2); axA.grid(True, alpha=0.3)

    # Panel B — control: stack-probe read vs injected clock (does it recover it?)
    gt = [max(r["gt_elapsed_s"], 1.0) for r in tsd]
    pr = [r["internal_s_raw"] for r in tsd]
    cols = [cmap[r["variant"]] for r in tsd]
    axB.scatter(gt, pr, c=cols, s=30)
    if gt:
        lim = [min(gt + pr) * 0.5, max(gt + pr) * 2]
        axB.plot(lim, lim, "k--", alpha=0.4, label="y = x")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("injected elapsed (s)"); axB.set_ylabel("stack-probe read (s)")
    axB.set_title("Timestamp control: probe recovers injected clock?")
    axB.legend(fontsize=7); axB.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved figure -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap conversations (0 = all)")
    ap.add_argument("--reuse", action="store_true", help="reuse cached looms (skip generation)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    base = resolve_model(os.environ.get("TIME_MODEL", "gemma"))
    probe_path = base.data_dir / "probe.npz"
    if not probe_path.exists():
        raise SystemExit(f"no probe at {probe_path}; run 20_fit_manifold.py first")
    probe, fit_meta = load_stacked_probe(probe_path)
    if fit_meta.get("probe_kind") != "stacked":
        raise SystemExit("probe.npz is not a stacked probe; re-run 20_fit_manifold.py")

    out_dir = DATA_DIR / f"{base.short_name}_natural"
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = out_dir / "hidden"
    looms_path = out_dir / "conversations.json"
    rows_path = out_dir / "naturalistic.jsonl"

    specs = list(CONVERSATIONS.items())
    if args.limit:
        specs = specs[: args.limit]

    print(f"model: {base.short_name} ({base.model_id})  probe: STACK (fit R2={fit_meta['r2']:+.3f})")
    print(f"conversations: {[c for c, _ in specs]}")

    print(f"loading {base.model_id} ...")
    t0 = time.time()
    with SaklasSession.from_pretrained(base.model_id, device="auto", probes=[]) as session:
        maybe_override_ministral_chat_template(session)
        maybe_override_gpt_oss_chat_template(session)
        print(f"loaded in {time.time() - t0:.1f}s")

        # Phase 1 — build (or load) the looms.
        if args.reuse and looms_path.exists():
            looms = json.loads(looms_path.read_text())
            print(f"reusing {len(looms)} cached looms")
        else:
            looms = {}
            for conv_id, spec in specs:
                t = time.time()
                msgs = build_loom(session, conv_id, spec["user_turns"])
                looms[conv_id] = {"variant": spec["variant"], "messages": msgs}
                print(f"  built {conv_id}: {len(msgs)} turns ({time.time()-t:.0f}s)")
            looms_path.write_text(json.dumps(looms, indent=2))
            print(f"saved looms -> {looms_path}")

        # Phase 2 — probe + readout.
        all_rows: list[dict] = []
        with rows_path.open("w") as out:
            for conv_id, spec in specs:
                if conv_id not in looms:
                    continue
                t = time.time()
                rows = process_conversation(
                    session, probe, conv_id, looms[conv_id]["variant"],
                    looms[conv_id]["messages"], hidden_dir,
                )
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                out.flush()
                all_rows.extend(rows)
                print(f"  probed {conv_id}: {len(rows)} readouts ({time.time()-t:.0f}s)")

    # Summary + figure.
    summary = summarize(all_rows)
    (out_dir / "naturalistic_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if not args.no_plot and all_rows:
        make_plot(all_rows, FIGURES_DIR / f"{base.short_name}_natural" / "fig_naturalistic.png")
    print(f"\nrows -> {rows_path}\nsummary -> {out_dir}/naturalistic_summary.json")


if __name__ == "__main__":
    main()
