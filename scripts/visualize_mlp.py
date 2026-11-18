"""Plot measured MLP learning curves and compare with the two baseline runs."""
import argparse
import os
from pathlib import Path

import pandas as pd

from visualize_baselines import COLORS, LABELS, ROOT, comparison_figures, evaluation_map, plt


def plot_learning(history, output):
    folds = history.loc[history["fold"].notna()]
    summary = folds.groupby("epoch")[["train_objective", "validation_log_loss"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    lines = [
        ("train_objective", "Training objective", "#0f766e"),
        ("validation_log_loss", "Held-out candidate log loss", "#b45309"),
    ]
    for metric, name, color in lines:
        mean, std = summary[metric]["mean"], summary[metric]["std"]
        ax.plot(summary.index, mean, marker="o", markersize=4, label=name, color=color)
        ax.fill_between(summary.index, mean - std, mean + std, color=color, alpha=.14)
    ax.set(xlabel="Epoch", ylabel="Loss", xticks=list(summary.index),
           title="MLP learning curves across five trip-grouped folds")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/mlp")
    parser.add_argument("--baselines", type=Path, default=ROOT / "experiments/baselines/metrics.csv")
    parser.add_argument("--trip-id", default="00000007")
    args = parser.parse_args()

    metrics = pd.concat([pd.read_csv(args.baselines), pd.read_csv(args.output / "metrics.csv")])
    labels = {**LABELS, "mlp": "MLP"}
    palette = {**COLORS, "mlp": "#be185d"}
    comparison_figures(metrics, args.output, labels, palette)
    plot_learning(pd.read_csv(args.output / "epochs.csv"), args.output)
    evaluation_map(args.processed_dir, args.archive, args.output, args.trip_id,
                   {"mlp": labels["mlp"]}, palette)


if __name__ == "__main__":
    main()
