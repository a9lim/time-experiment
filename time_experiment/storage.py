"""NPZ storage for per-transcript EOT activations.

One sidecar per (transcript, rendering): a stacked ``(n_turns, n_layers,
hidden_dim)`` array plus the turn indices, layer indices, and ground-truth
elapsed seconds. This is the trajectory unit — turns of one conversation in
order — which is what the longitudinal decode + trajectory analysis consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sidecar_path(hidden_dir: Path, transcript_id: str, rendering: str) -> Path:
    return hidden_dir / f"{transcript_id}__{rendering}.npz"


def save_transcript_states(
    path: Path,
    *,
    states: dict[int, dict[int, np.ndarray]],  # turn_idx -> {layer_idx: (D,)}
    elapsed_by_turn: dict[int, float],
) -> None:
    """Stack per-turn per-layer vectors into (T, L, D) and write the sidecar."""
    turn_idxs = sorted(states)
    if not turn_idxs:
        raise ValueError("no turn states to save")
    layer_idxs = sorted(states[turn_idxs[0]])
    H = np.stack([
        np.stack([states[t][L] for L in layer_idxs], axis=0)
        for t in turn_idxs
    ], axis=0).astype(np.float32)  # (T, L, D)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H=H,
        layer_idxs=np.array(layer_idxs, dtype=np.int64),
        turn_idxs=np.array(turn_idxs, dtype=np.int64),
        elapsed_s=np.array([elapsed_by_turn[t] for t in turn_idxs], dtype=np.float64),
    )


class TranscriptStates:
    """Loaded sidecar with convenient lookups."""

    def __init__(self, H: np.ndarray, layer_idxs: np.ndarray,
                 turn_idxs: np.ndarray, elapsed_s: np.ndarray) -> None:
        self.H = H                       # (T, L, D)
        self.layer_idxs = layer_idxs     # (L,)
        self.turn_idxs = turn_idxs       # (T,)
        self.elapsed_s = elapsed_s       # (T,)
        self._layer_pos = {int(L): i for i, L in enumerate(layer_idxs)}
        self._turn_pos = {int(t): i for i, t in enumerate(turn_idxs)}

    def layer_stack(self, layer: int) -> np.ndarray:
        """All turns' vectors at one layer: (T, D)."""
        return self.H[:, self._layer_pos[layer], :]

    def vec(self, turn: int, layer: int) -> np.ndarray:
        return self.H[self._turn_pos[turn], self._layer_pos[layer], :]

    def turn_all_layers(self, turn: int) -> np.ndarray:
        """All layers' vectors at one turn: (L, D) — for the all-layer probes."""
        return self.H[self._turn_pos[turn]]


def load_transcript_states(path: Path) -> TranscriptStates:
    d = np.load(path)
    return TranscriptStates(
        H=d["H"], layer_idxs=d["layer_idxs"],
        turn_idxs=d["turn_idxs"], elapsed_s=d["elapsed_s"],
    )
