"""NPZ storage for per-conversation slot activations.

One sidecar per ``(source, id, rendering, mode)``: a stacked
``(n_turns, n_layers, hidden_dim)`` array of the residual-stream vectors pooled
at the elicitation slot, plus the turn and layer indices and (when known) the
ground-truth elapsed seconds per turn. This is the trajectory unit — the
captured turns of one conversation in order — which the probe + decode consume.

All other covariates (tokens, schedule, role, prefill mode, verbal readout,
natural variant) live in the sibling ``rows.jsonl``; the sidecar is purely the
activation tensor + its indices, so it stays model-agnostic about metadata.
``elapsed_s`` is NaN-filled for natural conversations (no clock / no gt).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sidecar_path(
    hidden_dir: Path, source: str, conv_id: str, rendering: str, mode: str,
) -> Path:
    """``{source}__{id}__{rendering}__{mode}.npz`` under ``hidden_dir``."""
    return hidden_dir / f"{source}__{conv_id}__{rendering}__{mode}.npz"


def save_states(
    path: Path,
    *,
    states: dict[int, dict[int, np.ndarray]],   # turn_idx -> {layer_idx: (D,)}
    elapsed_by_turn: dict[int, float] | None = None,
) -> None:
    """Stack per-turn per-layer vectors into (T, L, D) and write the sidecar.

    ``elapsed_by_turn`` may be ``None`` (natural, no gt) -> elapsed stored as
    NaN, so a single loader handles scripted-with-gt and natural-without-gt.
    """
    turn_idxs = sorted(states)
    if not turn_idxs:
        raise ValueError("no turn states to save")
    layer_idxs = sorted(states[turn_idxs[0]])
    H = np.stack([
        np.stack([states[t][L] for L in layer_idxs], axis=0)
        for t in turn_idxs
    ], axis=0).astype(np.float32)  # (T, L, D)
    if elapsed_by_turn is None:
        elapsed = np.full(len(turn_idxs), np.nan, dtype=np.float64)
    else:
        elapsed = np.array([elapsed_by_turn[t] for t in turn_idxs], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H=H,
        layers=np.array(layer_idxs, dtype=np.int64),
        turn_idxs=np.array(turn_idxs, dtype=np.int64),
        elapsed_s=elapsed,
    )


class ConvStates:
    """Loaded sidecar with convenient lookups."""

    def __init__(self, H: np.ndarray, layers: np.ndarray,
                 turn_idxs: np.ndarray, elapsed_s: np.ndarray) -> None:
        self.H = H                       # (T, L, D)
        self.layers = layers             # (L,)
        self.turn_idxs = turn_idxs       # (T,)
        self.elapsed_s = elapsed_s       # (T,)
        self._layer_pos = {int(L): i for i, L in enumerate(layers)}
        self._turn_pos = {int(t): i for i, t in enumerate(turn_idxs)}

    def has_turn(self, turn: int) -> bool:
        return int(turn) in self._turn_pos

    def layer_stack(self, layer: int) -> np.ndarray:
        """All turns' vectors at one layer: (T, D)."""
        return self.H[:, self._layer_pos[layer], :]

    def vec(self, turn: int, layer: int) -> np.ndarray:
        return self.H[self._turn_pos[turn], self._layer_pos[layer], :]

    def turn_all_layers(self, turn: int) -> np.ndarray:
        """All layers' vectors at one turn: (L, D) — for the per-layer sweep."""
        return self.H[self._turn_pos[turn]]


def load_states(path: Path) -> ConvStates:
    d = np.load(path)
    return ConvStates(
        H=d["H"], layers=d["layers"],
        turn_idxs=d["turn_idxs"], elapsed_s=d["elapsed_s"],
    )
