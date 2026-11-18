"""Plot matched student-test classifier and decoder results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_candidates import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/motion_features")
    args = parser.parse_args()
    names = ["logistic_regression", "knn", "mlp"]
    labels = ["Logistic Regression", "KNN", "MLP"]
    prior, enhanced = (
        pd.read_csv(args.results / f"viterbi_{variant}_summary.csv").set_index("model").loc[names]
        for variant in ("prior", "enhanced")
    )
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained", sharey=True)
    for ax, (metric, title) in zip(axes, [
        ("top1_accuracy", "Classifier only"), ("viterbi_accuracy", "Classifier + Viterbi"),
    ]):
        for i, (frame, label, color) in enumerate([
            (prior, "Prior tuned features", "#64748b"),
            (enhanced, "Selected direction/motion features", "#087986"),
        ]):
            scores = 100 * frame[metric]
            bars = ax.bar(x + (i - .5)*.35, scores, width=.35, label=label, color=color)
            ax.bar_label(bars, labels=[f"{value:.1f}%" for value in scores], padding=5)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 108), title=title)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Student-test aligned observation accuracy (%)")
    axes[0].legend(loc="lower right")
    fig.savefig(args.results / "accuracy_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
