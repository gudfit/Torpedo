"""Synthetic data generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(slots=True)
class CTMCSimulator:
    """Generate synthetic marked CTMC sequences for pre-training."""

    transition_matrix: np.ndarray
    intensity_matrix: np.ndarray
    mark_params: Dict[str, Tuple[float, float]]

    def simulate(self, num_paths: int, horizon: float) -> Dict[str, np.ndarray]:
        """Return simulated event times, marks, and sizes."""

        rng = np.random.default_rng()
        states = np.zeros((num_paths,), dtype=np.int64)
        times = [[] for _ in range(num_paths)]
        marks = [[] for _ in range(num_paths)]
        sizes = [[] for _ in range(num_paths)]

        for path in range(num_paths):
            t = 0.0
            while t < horizon:
                state = states[path]
                lambda_state = self.intensity_matrix[state, state]
                if lambda_state <= 0:
                    break
                dt = rng.exponential(1.0 / lambda_state)
                t += dt
                if t >= horizon:
                    break
                probs = self.transition_matrix[state]
                next_state = rng.choice(len(probs), p=probs)
                states[path] = next_state
                mark_key = str(next_state)
                mu, sigma = self.mark_params.get(mark_key, (0.0, 0.0))
                marks[path].append(next_state)
                sizes[path].append(float(np.exp(rng.normal(mu, sigma))))
                times[path].append(t)

        def _pad(records: Iterable[float]) -> np.ndarray:
            lengths = [len(seq) for seq in records]
            max_len = max(lengths, default=0)
            padded = np.zeros((len(records), max_len), dtype=float)
            for idx, seq in enumerate(records):
                padded[idx, : len(seq)] = seq
            return padded

        return {
            "event_times": _pad(times),
            "marks": _pad(marks),
            "sizes": _pad(sizes),
        }


__all__ = ["CTMCSimulator"]
