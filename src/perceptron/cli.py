"""Command-line interface for the perceptron lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from perceptron.data import make_blobs
from perceptron.model import Perceptron


def positive_float(value: str) -> float:
    """Argparse type: a strictly positive float."""
    result = float(value)
    if result <= 0:
        msg = f"must be positive, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return result


def positive_int(value: str) -> int:
    """Argparse type: a strictly positive int."""
    result = int(value)
    if result <= 0:
        msg = f"must be positive, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return result


def non_negative_int(value: str) -> int:
    """Argparse type: a non-negative int."""
    result = int(value)
    if result < 0:
        msg = f"must be non-negative, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="perceptron",
        description="Train a single-layer perceptron on synthetic 2D data.",
    )
    parser.add_argument("--samples", type=positive_int, default=200)
    parser.add_argument("--lr", type=positive_float, default=0.1)
    parser.add_argument("--epochs", type=non_negative_int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show matplotlib plots",
    )
    parser.add_argument(
        "--iris",
        action="store_true",
        help="Also evaluate on the iris dataset",
    )
    return parser.parse_args(argv)


def plot_results(
    model: Perceptron,
    X: list[list[float]],
    y: list[int],
    out_dir: Path,
) -> None:
    """Save the decision boundary and the error curve as PNG files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    xs0 = [x[0] for x, label in zip(X, y, strict=True) if label == 0]
    ys0 = [x[1] for x, label in zip(X, y, strict=True) if label == 0]
    xs1 = [x[0] for x, label in zip(X, y, strict=True) if label == 1]
    ys1 = [x[1] for x, label in zip(X, y, strict=True) if label == 1]

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(xs0, ys0, c="tab:red", label="class 0", alpha=0.6)
    ax1.scatter(xs1, ys1, c="tab:blue", label="class 1", alpha=0.6)

    w0, w1 = model.weights
    if abs(w1) > 1e-12:
        x_min = min(x[0] for x in X) - 1
        x_max = max(x[0] for x in X) + 1
        x_vals = [x_min, x_max]
        y_vals = [-(w0 * xv + model.bias) / w1 for xv in x_vals]
        ax1.plot(x_vals, y_vals, "k--", label="decision boundary")

    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title(f"Decision boundary (accuracy={model.score(X, y):.3f})")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "decision_boundary.png", dpi=120)
    plt.close(fig1)

    if model.errors_per_epoch:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(
            range(1, len(model.errors_per_epoch) + 1),
            model.errors_per_epoch,
        )
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("misclassified samples")
        ax2.set_title("Training errors per epoch")
        fig2.tight_layout()
        fig2.savefig(out_dir / "errors_per_epoch.png", dpi=120)
        plt.close(fig2)


def run_iris() -> None:
    """Train and evaluate the perceptron on the iris dataset."""
    from sklearn.datasets import load_iris  # type: ignore[import-untyped]

    data = load_iris()
    X_full = data.data
    y_full = data.target

    X: list[list[float]] = []
    y: list[int] = []
    for row, label in zip(X_full, y_full, strict=True):
        if label in (0, 1):
            X.append([float(row[0]), float(row[1])])
            y.append(int(label))

    model = Perceptron(n_features=2, learning_rate=0.1, epochs=100)
    model.train(X, y)
    print(f"[iris] accuracy: {model.score(X, y):.3f}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the perceptron CLI."""
    args = parse_args(argv)

    X, y = make_blobs(n_samples=args.samples, seed=args.seed)

    model = Perceptron(
        n_features=2,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    model.train(X, y)
    accuracy = model.score(X, y)

    print(f"Perceptron trained on {len(X)} samples")
    print(f"Final accuracy: {accuracy:.3f}")
    print(f"Weights: {model.weights}, bias: {model.bias:.4f}")

    if not args.no_plot:
        out_dir = Path("plots")
        plot_results(model, X, y, out_dir)
        print(f"Plots saved to: {out_dir.resolve()}")

    if args.iris:
        run_iris()
