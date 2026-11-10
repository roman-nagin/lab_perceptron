"""Synthetic dataset generation."""

from __future__ import annotations

import random


def make_blobs(
    n_samples: int = 200,
    seed: int = 42,
    center_pos: tuple[float, float] = (2.0, 2.0),
    center_neg: tuple[float, float] = (-2.0, -2.0),
    spread: float = 0.8,
) -> tuple[list[list[float]], list[int]]:
    """Generate two linearly separable Gaussian blobs in 2D.

    Args:
        n_samples: Total number of samples (half per class).
        seed: Random seed for reproducibility.
        center_pos: Center of the positive class.
        center_neg: Center of the negative class.
        spread: Standard deviation of each coordinate.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a list of ``[x1, x2]`` and
        ``y`` is a list of labels in ``{0, 1}``.
    """
    if n_samples <= 0:
        msg = "n_samples must be positive"
        raise ValueError(msg)

    rng = random.Random(seed)
    half = n_samples // 2
    X: list[list[float]] = []
    y: list[int] = []

    for _ in range(half):
        X.append(
            [
                rng.gauss(center_pos[0], spread),
                rng.gauss(center_pos[1], spread),
            ]
        )
        y.append(1)
        X.append(
            [
                rng.gauss(center_neg[0], spread),
                rng.gauss(center_neg[1], spread),
            ]
        )
        y.append(0)

    if n_samples % 2 == 1:
        X.append([rng.gauss(center_pos[0], spread), rng.gauss(center_pos[1], spread)])
        y.append(1)

    return X, y
