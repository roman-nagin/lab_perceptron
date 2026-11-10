"""Tests for the Perceptron implementation."""

from __future__ import annotations

import pytest

from perceptron.data import make_blobs
from perceptron.model import Perceptron


def test_activate_threshold() -> None:
    """activate() must return 1 for z >= 0 and 0 otherwise."""
    p = Perceptron(n_features=2)
    assert p.activate(0) == 1
    assert p.activate(-1) == 0
    assert p.activate(0.0001) == 1
    assert p.activate(-0.0001) == 0


def test_weights_length_after_init() -> None:
    """Weights list must have length == n_features and start at zero."""
    p = Perceptron(n_features=3)
    assert len(p.weights) == 3
    assert all(w == 0.0 for w in p.weights)
    assert p.bias == 0.0


def test_predict_clear_separable_data() -> None:
    """predict() returns the correct class on obvious examples."""
    p = Perceptron(n_features=2)
    p.weights = [1.0, 1.0]
    p.bias = 0.0
    assert p.predict([2.0, 2.0]) == 1
    assert p.predict([-2.0, -2.0]) == 0


def test_train_separates_blobs() -> None:
    """After training on separable blobs, accuracy should be high."""
    X, y = make_blobs(n_samples=200, seed=42)
    p = Perceptron(n_features=2, learning_rate=0.1, epochs=50)
    p.train(X, y)
    assert p.score(X, y) >= 0.95


def test_score_in_range() -> None:
    """score() must return a value in [0, 1]."""
    X, y = make_blobs(n_samples=50, seed=7)
    p = Perceptron(n_features=2, learning_rate=0.1, epochs=10)
    p.train(X, y)
    s = p.score(X, y)
    assert 0.0 <= s <= 1.0


def test_invalid_init_raises() -> None:
    """Invalid constructor arguments must raise ValueError."""
    with pytest.raises(ValueError):
        Perceptron(n_features=0)
    with pytest.raises(ValueError):
        Perceptron(n_features=2, learning_rate=0)
    with pytest.raises(ValueError):
        Perceptron(n_features=2, epochs=-1)
