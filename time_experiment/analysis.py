"""Shared analysis: dataset assembly from rows.jsonl + slot sidecars, grouped-CV
linear probing, and the EV-weighted all-layer probe save/apply.

The canonical probe reads the elicitation slot across **all layers**, combined by
the saklas idiom: fit a single-layer ridge at every layer and take an
**explained-variance-weighted mean** of the per-layer log-elapsed predictions
(each layer's weight is its own grouped-CV R²; ``fit_ev_probe`` / ``apply_ev_probe``).
No learned meta-model — the weights ARE the fit qualities. The target is
log(elapsed seconds) (Weber-Fechner; the geometrically honest scale across
seconds->weeks). CV is grouped by conversation so correlated within-conversation
turns never straddle the train/test split (the leakage that would inflate R²).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import MIN_ELAPSED_S
from .storage import ConvStates, load_states, sidecar_path

RIDGE_ALPHAS = np.logspace(-1, 5, 13)
# Floor on total EV before the cross-layer weighting falls back to uniform —
# mirrors saklas's ``_MIN_EV_WEIGHT`` (keeps the EV-weighted mean from collapsing
# when every layer's fit is degenerate).
_MIN_EV_WEIGHT = 1e-6


# --- loading --------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    """Read a rows.jsonl (one row per captured (source,id,rendering,turn,mode))."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class StatesCache:
    """Lazy (source, id, rendering, mode) -> ConvStates loader."""

    def __init__(self, hidden_dir: Path) -> None:
        self.hidden_dir = hidden_dir
        self._cache: dict[tuple[str, str, str, str], ConvStates] = {}

    def get(self, source: str, conv_id: str, rendering: str, mode: str) -> ConvStates:
        key = (source, conv_id, rendering, mode)
        if key not in self._cache:
            self._cache[key] = load_states(
                sidecar_path(self.hidden_dir, *key)
            )
        return self._cache[key]


# --- dataset assembly -----------------------------------------------------
def assemble(
    rows: list[dict],
    cache: StatesCache,
    *,
    source: str,
    rendering: str,
    mode: str,
    roles: tuple[str, ...] | None = ("assistant",),
    need_gt: bool = True,
    min_elapsed: float = MIN_ELAPSED_S,
) -> dict[str, Any]:
    """Build (X3d, gt_log, groups, covariates) for one (source, rendering, mode).

    X3d is ``(N, L, D)`` — every layer kept so the per-layer sweep can pick the
    best. The caller slices ``X3d[:, li, :]`` for a single-layer fit/apply.

    ``roles`` filters captured turns (default assistant-only). With
    ``need_gt=True`` only rows with elapsed >= ``min_elapsed`` are kept (the log
    domain); with ``need_gt=False`` (natural transfer) rows with no gt are kept
    and ``gt_log`` is NaN for them. A row is included only if its activation
    sidecar actually has the turn (capture may have skipped over-cap turns).
    """
    X, gt_log, groups, tokens, turn_idx = [], [], [], [], []
    schedule, role, variant, verbal_s = [], [], [], []
    layers: list[int] | None = None
    for r in rows:
        if r["source"] != source or r["rendering"] != rendering or r["mode"] != mode:
            continue
        if roles is not None and r["role"] not in roles:
            continue
        gt = r.get("gt_elapsed_s")
        has_gt = isinstance(gt, (int, float)) and math.isfinite(gt) and gt >= min_elapsed
        if need_gt and not has_gt:
            continue
        st = cache.get(source, r["id"], rendering, mode)
        if not st.has_turn(r["turn_idx"]):
            continue
        if layers is None:
            layers = [int(L) for L in st.layers]
        X.append(st.turn_all_layers(r["turn_idx"]))           # (L, D)
        gt_log.append(math.log(gt) if has_gt else math.nan)
        groups.append(r["id"])
        tokens.append(r.get("tokens", math.nan))
        turn_idx.append(r["turn_idx"])
        schedule.append(r.get("schedule"))
        role.append(r["role"])
        variant.append(r.get("variant"))
        verbal_s.append(r.get("verbal_seconds"))
    return {
        "X3d": np.asarray(X, dtype=np.float32),               # (N, L, D)
        "gt_log": np.asarray(gt_log, dtype=np.float64),
        "groups": np.asarray(groups),
        "tokens": np.asarray(tokens, dtype=np.float64),
        "turn_idx": np.asarray(turn_idx, dtype=np.int64),
        "schedule": np.asarray(schedule),
        "role": np.asarray(role),
        "variant": np.asarray(variant),
        "verbal_s": np.asarray(verbal_s, dtype=np.float64),
        "layers": layers or [],
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


def best_layer_sweep(X3d: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     *, n_splits: int = 5) -> tuple[int, float, list[float]]:
    """Per-layer grouped-CV R² sweep -> (best_layer_index, best_r2, all_r2).

    The best layer is the representational locus of the elapsed coordinate; it's
    selected on gt R² (so layer choice is non-circular w.r.t. any downstream
    felt / natural read)."""
    L = X3d.shape[1]
    r2s = [cv_predict(X3d[:, li, :], y, groups, n_splits=n_splits)[1] for li in range(L)]
    bi = int(np.argmax(r2s))
    return bi, r2s[bi], r2s


def residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual of y after a least-squares linear fit on z (z is 1-D or 2-D)."""
    Z = z.reshape(len(y), -1)
    Z1 = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    return y - Z1 @ beta


# --- single-layer probe components ----------------------------------------
def fit_full(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit a single-layer ridge on all data; return params for save/apply
    (plain arrays — no pickle). The per-layer base learner of the EV probe."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scaler.transform(X), y)
    return {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "coef": ridge.coef_.astype(np.float64),
        "intercept": float(ridge.intercept_),
        "alpha": float(ridge.alpha_),
    }


def apply_probe(probe: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Predicted log-elapsed for activations X (N, D) under a single-layer probe."""
    Xs = (X - probe["mean"]) / probe["scale"]
    return Xs @ probe["coef"] + probe["intercept"]


def probe_direction(probe: dict[str, Any]) -> np.ndarray:
    """Unit reading-elapsed direction in raw activation space (coef / scale,
    normalized) — the axis Arm G projects a generation trajectory onto."""
    w = np.asarray(probe["coef"], np.float64) / np.asarray(probe["scale"], np.float64)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


# --- EV-weighted all-layer probe (saklas idiom) ---------------------------
# saklas reads a trait across layers by an *explained-variance-weighted* mean of
# per-layer readings (`Monitor._layer_geometry` + `ev_weights`,
# `manifold.explained_variance / Σ`, floored, uniform fallback). The analog here:
# fit a single-layer ridge at every layer, weight each layer's log-elapsed
# prediction by its own grouped-CV R² (= the variance it explains), and sum. No
# learned meta-model to overfit — the weights ARE the fit qualities.
def perlayer_oof(X3d: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 *, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer grouped-CV out-of-fold predictions (N, L) + per-layer R² (L,)."""
    from sklearn.metrics import r2_score
    L = X3d.shape[1]
    Z = np.column_stack([cv_predict(X3d[:, li, :], y, groups, n_splits=n_splits)[0]
                         for li in range(L)])
    r2 = np.array([r2_score(y, Z[:, li]) for li in range(L)])
    return Z, r2


def ev_weights(r2_per_layer: np.ndarray) -> np.ndarray:
    """Normalized EV weights from per-layer R²: w_L = relu(R²_L) / Σ relu(R²).

    A layer that predicts worse than the mean (R²<0) explains no variance and
    gets weight 0; if every layer is degenerate the weights fall back to uniform
    (saklas's ``_MIN_EV_WEIGHT`` floor behavior)."""
    ev = np.clip(np.asarray(r2_per_layer, np.float64), 0.0, None)
    total = ev.sum()
    if total <= _MIN_EV_WEIGHT:
        return np.full(len(ev), 1.0 / len(ev))
    return ev / total


def ev_combined_oof(X3d: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    *, n_splits: int = 5) -> dict[str, Any]:
    """Honest combined read: per-layer OOF preds EV-weighted into one OOF series.

    The per-layer predictions are out-of-fold; the EV weights are derived from
    those same OOF R²s (L smooth scalars, saklas computes EV once at fit time —
    not nested), so the combined-OOF optimism is second-order. Returns the
    combined OOF, its R²/Spearman, the weights, and the per-layer R² profile."""
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    Z, r2_per_layer = perlayer_oof(X3d, y, groups, n_splits=n_splits)
    w = ev_weights(r2_per_layer)
    oof = Z @ w
    return {"oof": oof, "r2": float(r2_score(y, oof)),
            "spearman": float(spearmanr(y, oof).statistic),
            "weights": w, "r2_per_layer": r2_per_layer}


def fit_ev_probe(X3d: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 layers: list[int], *, n_splits: int = 5) -> dict[str, Any]:
    """Deployable EV-weighted all-layer probe: per-layer base ridges fit on ALL
    data (rectangular arrays — every layer shares D, no pickle) + the EV weights
    from per-layer grouped-CV R². ``apply_ev_probe`` reads it as Σ_L w_L·read_L."""
    n, L, D = X3d.shape
    base_mean = np.empty((L, D))
    base_scale = np.empty((L, D))
    base_coef = np.empty((L, D))
    base_intercept = np.empty(L)
    for li in range(L):
        p = fit_full(X3d[:, li, :], y)
        base_mean[li] = p["mean"]
        base_scale[li] = p["scale"]
        base_coef[li] = p["coef"]
        base_intercept[li] = p["intercept"]
    _, r2_per_layer = perlayer_oof(X3d, y, groups, n_splits=n_splits)
    return {
        "layers": np.asarray(layers, dtype=np.int64),
        "weights": ev_weights(r2_per_layer),
        "r2_per_layer": r2_per_layer,
        "base_mean": base_mean, "base_scale": base_scale,
        "base_coef": base_coef, "base_intercept": base_intercept,
    }


def apply_ev_probe(probe: dict[str, Any], X3d: np.ndarray) -> np.ndarray:
    """EV-weighted log-elapsed read for all-layer activations X3d (N, L, D). The
    layer axis must match ``probe['layers']`` (sorted sidecar layer order)."""
    w = probe["weights"]
    n, L, D = X3d.shape
    out = np.zeros(n, dtype=np.float64)
    for li in range(L):
        Xs = (X3d[:, li, :] - probe["base_mean"][li]) / probe["base_scale"][li]
        out += w[li] * (Xs @ probe["base_coef"][li] + probe["base_intercept"][li])
    return out


def ev_layer_direction(probe: dict[str, Any], li: int) -> np.ndarray:
    """Unit reading-elapsed direction at layer index ``li`` of the EV probe —
    for Arm G's per-layer cosine (EV-weighted across layers by the caller)."""
    return probe_direction({"coef": probe["base_coef"][li], "scale": probe["base_scale"][li]})


# --- off-manifold scoring (saklas Mahalanobis idiom) ----------------------
def maha_scorer(reference_X3d: np.ndarray, layers):
    """Mahalanobis scorer whitened on a reference manifold (saklas LayerWhitener).

    ``reference_X3d`` is ``(N_ref, L, D)`` — typically the scripted slot manifold.
    Returns a callable ``score(query_X3d) -> ratios`` giving, per query row, the
    median-over-layers Mahalanobis distance divided by the reference's own median
    distance (≈1 on-manifold, >1 off it). The whitener (small N×N Woodbury kernel)
    is built **once**; call the scorer on as many query sets as needed. Returns
    ``None`` if saklas's Mahalanobis is unavailable (keeps this module importable
    without torch/saklas). Mirrors T3's ``maha_ratio`` so the natural-slot and
    generation-token off-manifold numbers are computed the same way.
    """
    try:
        import torch
        from saklas.core.mahalanobis import LayerWhitener
    except Exception:  # pragma: no cover - exercised only with saklas installed
        return None
    ref = {int(L): torch.from_numpy(np.ascontiguousarray(reference_X3d[:, i, :])).float()
           for i, L in enumerate(layers)}
    means = {L: ref[L].mean(0) for L in ref}
    w = LayerWhitener.from_neutral_activations(ref, means, ridge_scale=1.0)

    def _norms(L: int, V: torch.Tensor) -> np.ndarray:
        """Batched Mahalanobis norm ``sqrt(vᵀΣ⁻¹v)`` over every row of ``V``
        ``(N, D)``. ``apply_inv`` batches its leading dim through one Woodbury
        pass, so this is a single set of BLAS calls per layer — bit-identical to
        looping ``mahalanobis_norm`` per row, but ~N× fewer Python/linalg calls.
        (The per-(token,layer) scalar loop made the gen-token OOD — ~700k calls —
        take ~80 min single-threaded; this collapses it to one call per layer.)"""
        si = w.apply_inv(L, V).float()
        return torch.sqrt((V * si).sum(dim=1).clamp_min(0.0)).numpy()

    med_ref: dict[int, float] = {}
    for L in ref:
        if not w.covers(L):
            continue
        med_ref[L] = float(np.median(_norms(L, ref[L] - means[L])))

    def score(query_X3d: np.ndarray) -> np.ndarray:
        cols = []  # one (N_query,) ratio column per covered layer
        for i, L in enumerate(layers):
            L = int(L)
            if L not in med_ref or med_ref[L] <= 0:
                continue
            V = torch.from_numpy(np.ascontiguousarray(query_X3d[:, i, :])).float() - means[L]
            cols.append(_norms(L, V) / med_ref[L])
        if not cols:
            return np.zeros(0, dtype=np.float64)
        return np.median(np.stack(cols, axis=1), axis=1)  # median over layers, per token

    return score


def save_ev_probe(path: Path, probe: dict[str, Any], *, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, kind=np.str_("ev"), layers=probe["layers"], weights=probe["weights"],
        r2_per_layer=probe["r2_per_layer"],
        base_mean=probe["base_mean"], base_scale=probe["base_scale"],
        base_coef=probe["base_coef"], base_intercept=probe["base_intercept"],
        meta=json.dumps(meta),
    )


def load_ev_probe(path: Path) -> tuple[dict[str, Any], dict]:
    d = np.load(path, allow_pickle=False)
    if str(d["kind"]) != "ev":
        raise ValueError(f"{path} is not an EV probe (kind={d['kind']!r})")
    probe = {
        "layers": d["layers"], "weights": d["weights"], "r2_per_layer": d["r2_per_layer"],
        "base_mean": d["base_mean"], "base_scale": d["base_scale"],
        "base_coef": d["base_coef"], "base_intercept": d["base_intercept"],
    }
    return probe, json.loads(str(d["meta"]))


# --- H1/H2/H3 reading -----------------------------------------------------
def classify_hypothesis(
    *, corr_verbal_internal: float, corr_internal_gt: float,
    overshoot_internal: float, overshoot_verbal: float,
) -> str:
    """Heuristic H1/H2/H3 reading from the untimestamped (implicit-time) decode.

    H1 output confabulation : internal tracks gt, verbal decoupled from internal
    H2 represented inflated  : verbal tracks internal AND internal runs high vs gt
    H3 calibrated-misapplied : verbal tracks internal, internal ~ gt, verbal high

    NB: the overshoot cutoffs were tuned to the EOT era. The slot internal
    coordinate sits on a different scale (Pilot 5: it *undershoots* the verbal),
    so re-validate these thresholds against the regenerated decode before
    trusting the verdict string.
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
