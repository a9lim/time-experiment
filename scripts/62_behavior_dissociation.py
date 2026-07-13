#!/usr/bin/env python
"""Behavior-vs-speech dissociation for the doc's models (gemma / qwen / …).

Run INSIDE the friend's time-experiment repo (it imports time_experiment.* for the
corpus). Reuses the existing fine-grained SPEECH readout (`verbal_seconds` in
rows.jsonl — the soft-distribution median, NOT a parsed text duration, so it avoids
the bucketing that muddied the Claude run). The only new channel is BEHAVIOR: a free
continuation of each conversation, scored by a blind judge.

Per analysis unit = one (conversation, assistant-turn):
    LENGTH   = tokens at that turn            (rows.jsonl)
    SPEECH   = verbal_seconds at that turn    (rows.jsonl; log-taken)
    BEHAVIOR = blind judge score (0-100) of the model's generated continuation
Metric (validated): partial corr of BEHAVIOR with LENGTH, controlling for SPEECH.
    > 0  -> behavior carries elapsed-time info beyond what speech reports (dissociation)
    ~ 0  -> behavior adds nothing beyond speech (null / echo)
CI by cluster bootstrap over conversation id.

USAGE
  # validate the metric only (no model, no data):
  python dissociation_gemma_qwen.py --self-test

  # dry-run the full orchestration with a fake model+judge on real rows.jsonl:
  python dissociation_gemma_qwen.py --rows data/qwen/rows.jsonl --corpus rates --mock

  # the real thing:
  python dissociation_gemma_qwen.py \
      --rows data/qwen/rows.jsonl --corpus rates \
      --model Qwen/Qwen3.6-27B --judge anthropic --n 80 --out qwen_dissoc.json

JUDGE backends:
  --judge anthropic        uses api.anthropic.com; reads ANTHROPIC_API_KEY from the
                           environment (NEVER hardcode a key). Strongest judge.
  --judge hf:<model_id>    uses a local HF instruct model as judge.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo-script convention
import numpy as np

# ───────────────────────── analysis (validated, model-free) ─────────────────────────
def _linreg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean(); sxx = ((x - mx) ** 2).sum()
    b = ((x - mx) * (y - my)).sum() / sxx if sxx else 0.0
    return b, my - b * mx

def _resid(y, x):
    b, a = _linreg(x, y); return np.asarray(y, float) - (a + b * np.asarray(x, float))

def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def partial_corr(behavior, length, speech):
    """corr( behavior | speech , length | speech ) — does behavior know more than speech says?"""
    return _pearson(_resid(behavior, speech), _resid(length, speech))

def cluster_bootstrap(behavior, length, speech, ids, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); uids = np.unique(ids); out = []
    beh, ln, sp, ids = map(np.asarray, (behavior, length, speech, ids))
    for _ in range(n_boot):
        pick = rng.choice(uids, size=len(uids), replace=True)
        idx = np.concatenate([np.where(ids == u)[0] for u in pick])
        v = partial_corr(beh[idx], ln[idx], sp[idx])
        if np.isfinite(v): out.append(v)
    if not out: return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

# ───────────────────────── data loading (model-free; testable on real rows.jsonl) ─────────────────────────
def load_speech_length(rows_path, rendering="untimestamped"):
    """(id, turn_idx) -> {tokens, verbal_s, role} for the chosen (no-clock) condition."""
    out = {}
    with open(rows_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("rendering") != rendering: continue
            v = r.get("verbal_seconds"); tok = r.get("tokens")
            if v in (None, "", "nan") or tok in (None, ""): continue
            out[(r["id"], int(r["turn_idx"]))] = dict(
                tokens=float(tok), verbal_s=float(v), role=r.get("role", "assistant"))
    return out

def load_conversations(corpus_name):
    """id -> Transcript, via the repo's own loader (model-independent corpus)."""
    from time_experiment.config import TRANSCRIPTS_DIR
    from time_experiment.transcripts import load_corpus
    path = TRANSCRIPTS_DIR / f"{corpus_name}.jsonl"
    return {t.id: t for t in load_corpus(path)}

def render_context(transcript, upto_turn):
    """Chat messages for turns [0, upto_turn) — no timestamps (the no-clock condition)."""
    return [{"role": t.role, "content": t.text} for t in transcript.turns[:upto_turn]]

# ───────────────────────── model + judge (heavy; import-guarded) ─────────────────────────
def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tok

def generate_continuation(model, tok, messages, max_new_tokens=120, seed=0):
    import torch
    torch.manual_seed(seed)
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

_JUDGE_SYS = "You are an expert conversation rater. Output only an integer 0-100."
def _judge_prompt(text):
    return (f'Here is a single assistant message taken out of context:\n\n"{text}"\n\n'
            "On a scale of 0 to 100, how strongly does this message read like it comes from a "
            "LONG, mature, winding-down conversation (100) versus a FRESH, just-started one (0)? "
            "Consider references to prior discussion, summarizing/closing language, and tone. "
            "Output ONLY the integer.")

def make_judge(spec):
    """Returns judge(text)->int|None. spec: 'anthropic' or 'hf:<model_id>'."""
    if spec == "anthropic":
        import urllib.request
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("set ANTHROPIC_API_KEY in your environment for --judge anthropic "
                             "(the script never hardcodes a key)")
        def judge(text):
            body = json.dumps({"model": "claude-sonnet-4-6", "max_tokens": 16,
                               "system": _JUDGE_SYS,
                               "messages": [{"role": "user", "content": _judge_prompt(text)}]}).encode()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                headers={"content-type": "application/json", "x-api-key": key,
                         "anthropic-version": "2023-06-01"})
            try:
                resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
                txt = "".join(b.get("text", "") for b in resp.get("content", []))
                return _parse_score(txt)
            except Exception:
                return None
        return judge
    if spec.startswith("hf:"):
        jmodel, jtok = load_model(spec[3:])
        def judge(text):
            msgs = [{"role": "system", "content": _JUDGE_SYS},
                    {"role": "user", "content": _judge_prompt(text)}]
            return _parse_score(generate_continuation(jmodel, jtok, msgs, max_new_tokens=8))
        return judge
    raise SystemExit(f"unknown judge spec: {spec}")

def _parse_score(txt):
    if not txt: return None
    import re
    m = re.search(r"\d{1,3}", txt)
    if not m: return None
    v = int(m.group()); return v if 0 <= v <= 100 else None

# ───────────────────────── experiment ─────────────────────────
def run(rows_path, corpus_name, gen_fn, judge_fn, n=80, seed=0, rendering="untimestamped"):
    sl = load_speech_length(rows_path, rendering)
    convs = load_conversations(corpus_name)
    # sample assistant-turn points that we can both reconstruct and have speech for
    pts = [(cid, ti) for (cid, ti), d in sl.items()
           if cid in convs and ti < convs[cid].turn_count
           and convs[cid].turns[ti].role == "assistant" and ti >= 1]
    random.Random(seed).shuffle(pts); pts = pts[:n]
    if len(pts) < 4:
        raise SystemExit(f"only {len(pts)} usable (conversation, assistant-turn) points found")

    rows = []
    for k, (cid, ti) in enumerate(pts):
        ctx = render_context(convs[cid], ti)              # turns before this assistant turn
        cont = gen_fn(ctx, seed + k)                       # BEHAVIOR: model's continuation
        score = judge_fn(cont) if cont else None
        d = sl[(cid, ti)]
        ok = score is not None and d["verbal_s"] > 0 and d["tokens"] > 0
        rows.append(dict(id=cid, turn=ti, tokens=d["tokens"], verbal_s=d["verbal_s"],
                         behavior=score, ok=ok, continuation=(cont or "")[:200]))
        print(f"[{k+1}/{len(pts)}] {cid} t{ti}: len={d['tokens']:.0f} "
              f"speech={d['verbal_s']:.0f}s behavior={score}  {'ok' if ok else 'skip'}")

    ok = [r for r in rows if r["ok"]]
    if len(ok) < 4:
        raise SystemExit(f"only {len(ok)} usable points after judging")
    L = np.log([r["tokens"] for r in ok]); S = np.log([r["verbal_s"] for r in ok])
    B = np.array([r["behavior"] for r in ok], float); ids = np.array([r["id"] for r in ok])
    pc = partial_corr(B, L, S); ci = cluster_bootstrap(B, L, S, ids)
    res = dict(n_usable=len(ok), partial_corr=pc, ci95=ci,
               slope_behavior_length=_linreg(L, B)[0], slope_speech_length=_linreg(L, S)[0],
               r_behavior_length=_pearson(B, L), r_speech_length=_pearson(S, L), rows=rows)
    verdict = ("DISSOCIATION (behavior > speech in length-tracking)" if ci[0] > 0.05 else
               "no dissociation / underpowered (CI spans 0)" if ci[1] > 0.05 else
               "reverse (behavior < speech)")
    print("\n" + "=" * 60)
    print(f"partial corr (behavior·length | speech) = {pc:+.3f}   95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"slope behavior~length = {res['slope_behavior_length']:+.2f}   "
          f"slope speech~length = {res['slope_speech_length']:+.2f}")
    print(f"r(beh,len)={res['r_behavior_length']:+.2f}  r(speech,len)={res['r_speech_length']:+.2f}  "
          f"(n={len(ok)})")
    print(f"VERDICT: {verdict}")
    print("NOTE: observational partial is the SCREEN; EIV in the speech read can inflate it. "
          "Confirm a positive with the steering arm or a split-half instrument on speech.")
    return res

# ───────────────────────── self-test (model-free metric validation) ─────────────────────────
def self_test():
    rng = np.random.default_rng(0); N = 60
    L = np.log(rng.uniform(60, 1400, N)); internal = L + rng.normal(0, 0.1, N)
    speech = 0.6 * internal + rng.normal(0, 0.2, N)
    cases = {
        "DISSOCIATION (behavior less compressed)": 0.9 * internal + rng.normal(0, 0.2, N),
        "NULL (behavior == speech)": speech + rng.normal(0, 0.2, N),
    }
    print("metric self-test (planted truth):")
    for name, beh in cases.items():
        print(f"  {name:42s} partial = {partial_corr(beh, L, speech):+.2f}")
    # confabulator: speech pinned flat
    sp_flat = np.zeros(N) + rng.normal(0, 0.05, N); beh = 0.9 * internal + rng.normal(0, 0.2, N)
    print(f"  {'CONFABULATOR (speech flat)':42s} partial = {partial_corr(beh, L, sp_flat):+.2f}")
    print("=> dissociation high+, null ~0, confab large+  (metric separates them)")

# ───────────────────────── mock model+judge (orchestration dry-run, no torch) ─────────────────────────
def _mock_backends(planted="dissociation", seed=0):
    rng = np.random.default_rng(seed)
    def gen_fn(ctx, s):                       # length-aware fake continuation
        n = sum(len(m["content"]) for m in ctx)
        return f"[mock continuation; context_chars={n}]"
    def judge_fn(text):                        # behavior = less-compressed fn of context length + noise
        import re
        m = re.search(r"context_chars=(\d+)", text); n = int(m.group(1)) if m else 100
        base = 40 + 25 * np.log(max(n, 1))     # tracks length strongly (less compressed than speech)
        return int(np.clip(base + rng.normal(0, 6), 0, 100))
    return gen_fn, judge_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows"); ap.add_argument("--corpus", default="rates")
    ap.add_argument("--model"); ap.add_argument("--judge", default="anthropic")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rendering", default="untimestamped")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mock", action="store_true", help="dry-run orchestration with a fake model+judge")
    ap.add_argument("--out", default="dissociation_result.json")
    a = ap.parse_args()

    if a.self_test:
        self_test(); return
    if not a.rows:
        ap.error("--rows required (or use --self-test)")

    if a.mock:
        gen_fn, judge_fn = _mock_backends(seed=a.seed)
    else:
        if not a.model: ap.error("--model required for a real run")
        model, tok = load_model(a.model)
        gen_fn = lambda ctx, s: generate_continuation(model, tok, ctx, seed=s)
        judge_fn = make_judge(a.judge)

    res = run(a.rows, a.corpus, gen_fn, judge_fn, n=a.n, seed=a.seed, rendering=a.rendering)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
