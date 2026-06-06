"""Shared analysis utilities: dataset assembly from turns.jsonl + sidecars,
grouped-CV linear probing, and probe save/apply.

The probe target is log(elapsed seconds) — Weber-Fechner says subjective time is
logarithmic, and it's the geometrically honest scale across the seconds->weeks
span. CV is grouped by transcript so correlated within-conversation turns never
straddle the train/test split (the leakage that would inflate R²).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import MIN_ELAPSED_S
from .storage import TranscriptStates, load_transcript_states, sidecar_path

RIDGE_ALPHAS = np.logspace(-1, 5, 13)


# --- loading --------------------------------------------------------------
def load_rows(model: Any) -> list[dict]:
    rows: list[dict] = []
    with model.turns_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class StatesCache:
    """Lazy (transcript_id, rendering) -> TranscriptStates loader."""

    def __init__(self, hidden_dir: Path) -> None:
        self.hidden_dir = hidden_dir
        self._cache: dict[tuple[str, str], TranscriptStates] = {}

    def get(self, tid: str, rendering: str) -> TranscriptStates:
        key = (tid, rendering)
        if key not in self._cache:
            self._cache[key] = load_transcript_states(
                sidecar_path(self.hidden_dir, tid, rendering)
            )
        return self._cache[key]


def available_layers(model: Any, rows: list[dict]) -> list[int]:
    cache = StatesCache(model.hidden_dir)
    r = rows[0]
    return [int(L) for L in cache.get(r["transcript_id"], r["rendering"]).layer_idxs]


# --- dataset assembly -----------------------------------------------------
def assemble_layer(
    model: Any,
    rows: list[dict],
    layer: int,
    *,
    rendering: str,
    min_elapsed: float = MIN_ELAPSED_S,
    roles: tuple[str, ...] | None = None,
    cache: StatesCache | None = None,
) -> dict[str, Any]:
    """Build (X, y_log, groups, meta) for one layer + rendering.

    ``roles`` filters captured turns (e.g. ('assistant',)); None keeps all.
    Only turns with elapsed >= ``min_elapsed`` are included (log domain).
    """
    cache = cache or StatesCache(model.hidden_dir)
    X, y_log, groups, tokens, turn_idx, schedule, role = [], [], [], [], [], [], []
    for r in rows:
        if r["rendering"] != rendering:
            continue
        if roles is not None and r["role"] not in roles:
            continue
        if r["gt_elapsed_s"] < min_elapsed:
            continue
        ts = cache.get(r["transcript_id"], r["rendering"])
        X.append(ts.vec(r["turn_idx"], layer))
        y_log.append(math.log(r["gt_elapsed_s"]))
        groups.append(r["transcript_id"])
        tokens.append(r["prompt_tokens"])
        turn_idx.append(r["turn_idx"])
        schedule.append(r["schedule"])
        role.append(r["role"])
    return {
        "X": np.asarray(X, dtype=np.float64),
        "y_log": np.asarray(y_log, dtype=np.float64),
        "groups": np.asarray(groups),
        "tokens": np.asarray(tokens, dtype=np.float64),
        "turn_idx": np.asarray(turn_idx, dtype=np.int64),
        "schedule": np.asarray(schedule),
        "role": np.asarray(role),
    }


# --- probing --------------------------------------------------------------
def _n_splits(groups: np.ndarray, requested: int = 5) -> int:
    return max(2, min(requested, len(np.unique(groups))))


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               *, n_splits: int = 5) -> tuple[np.ndarray, float, float]:
    """Grouped out-of-fold predictions + (R^2, Spearman) over them."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full_like(y, np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=_n_splits(groups, n_splits))
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
    r2 = float(r2_score(y, oof))
    rho = float(spearmanr(y, oof).statistic)
    return oof, r2, rho


def fit_full(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit a probe on all data; return params for save/apply (no pickle)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs, y)
    return {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "coef": ridge.coef_.astype(np.float64),
        "intercept": float(ridge.intercept_),
        "alpha": float(ridge.alpha_),
    }


def apply_probe(probe: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predicted log-elapsed for activations X under a saved probe."""
    Xs = (X - probe["mean"]) / probe["scale"]
    return Xs @ probe["coef"] + probe["intercept"]


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual of y after a least-squares linear fit on z (z is 1-D or 2-D)."""
    Z = z.reshape(len(y), -1)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta


def save_probe(path: Path, probe: dict[str, Any], *, layer: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, mean=probe["mean"], scale=probe["scale"], coef=probe["coef"],
        intercept=np.float64(probe["intercept"]), alpha=np.float64(probe["alpha"]),
        layer=np.int64(layer), meta=json.dumps(meta),
    )


def classify_hypothesis(
    *, corr_verbal_internal: float, corr_internal_gt: float,
    overshoot_internal: float, overshoot_verbal: float,
) -> str:
    """Heuristic H1/H2/H3 reading from the untimestamped (implicit-time) decode.

    H1 output confabulation : internal tracks gt, verbal decoupled from internal
    H2 represented inflated  : verbal tracks internal AND internal runs high vs gt
    H3 calibrated-misapplied : verbal tracks internal, internal ~ gt, verbal high
    """
    vi, ig = corr_verbal_internal, corr_internal_gt
    if not (math.isfinite(vi) and math.isfinite(ig)):
        return "insufficient parsed data for a call"
    if vi < 0.3 and ig > 0.4:
        return ("H1 — internal coordinate tracks reality but the verbal estimate "
                "is decoupled from it: confabulated at output")
    if vi > 0.5 and overshoot_internal > 2.0:
        return ("H2 — verbal tracks the internal coordinate AND the internal "
                "coordinate itself runs high vs ground truth: represented elapsed "
                "time is genuinely inflated")
    if vi > 0.5 and overshoot_internal < 2.0 and overshoot_verbal > 2.0:
        return ("H3 — internal coordinate is well-calibrated to the available "
                "signal; the verbal overshoot is the token->seconds gap, not "
                "representational (calibrated-but-misapplied)")
    return ("mixed / between hypotheses — see corr_verbal_internal, "
            "overshoot_internal, and the transfer correlation")


def load_probe(path: Path) -> tuple[dict[str, Any], int, dict]:
    d = np.load(path, allow_pickle=False)
    probe = {
        "mean": d["mean"], "scale": d["scale"], "coef": d["coef"],
        "intercept": float(d["intercept"]), "alpha": float(d["alpha"]),
    }
    return probe, int(d["layer"]), json.loads(str(d["meta"]))
