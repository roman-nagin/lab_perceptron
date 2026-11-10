"""Single-layer perceptron implementation."""

from __future__ import annotations


class Perceptron:
    """A simple single-layer perceptron for binary classification.

    The model computes ``z = w . x + b`` and applies a step function:
    returns ``1`` if ``z >= 0`` and ``0`` otherwise. Training uses the
    classic perceptron update rule.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Initialize the perceptron.

        Args:
            n_features: Number of input features.
            learning_rate: Step size for weight updates.
            epochs: Number of passes over the training data.
        """
        if n_features <= 0:
            msg = "n_features must be positive"
            raise ValueError(msg)
        if learning_rate <= 0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if epochs < 0:
            msg = "epochs must be non-negative"
            raise ValueError(msg)

        self.n_features: int = n_features
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.weights: list[float] = [0.0] * n_features
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []

    def activate(self, z: float) -> int:
        """Return 1 if ``z >= 0``, otherwise 0."""
        return 1 if z >= 0 else 0

    def _raw(self, x: list[float]) -> float:
        """Compute the weighted sum plus bias."""
        total = self.bias
        for wi, xi in zip(self.weights, x, strict=True):
            total += wi * xi
        return total

    def predict(self, x: list[float]) -> int:
        """Predict the class for a single sample."""
        if len(x) != self.n_features:
            msg = f"expected {self.n_features} features, got {len(x)}"
            raise ValueError(msg)
        return self.activate(self._raw(x))

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Train the perceptron on the given data.

        Uses the perceptron update rule:
        ``w_i += lr * (target - pred) * x_i`` and ``b += lr * (target - pred)``.

        Args:
            X: Feature vectors.
            y: Target labels (0 or 1).
        """
        if len(X) != len(y):
            msg = "X and y must have the same length"
            raise ValueError(msg)
        if not X:
            return

        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    for i in range(self.n_features):
                        self.weights[i] += self.learning_rate * error * xi[i]
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break

    def score(self, X: list[list[float]], y: list[int]) -> float:
        """Return the fraction of correctly predicted labels."""
        if not X:
            return 0.0
        correct = 0
        for xi, yi in zip(X, y, strict=True):
            if self.predict(xi) == yi:
                correct += 1
        return correct / len(X)
