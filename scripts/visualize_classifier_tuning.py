"""Plot recorded training-search trials and held-out classifier accuracy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_candidates import ROOT


LABELS = {"logistic_regression": "Logistic Regression", "knn": "KNN", "mlp": "MLP"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/classifier_tuning")
    args = parser.parse_args()
    results = args.results
    comparison = pd.read_csv(results / "student_test_comparison.csv")
    accuracy = 100 * comparison.pivot(index="model", columns="variant", values="top1_accuracy")
    names = list(LABELS)
    labels = list(LABELS.values())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for i, (variant, color) in enumerate([("untuned", "#708198"), ("tuned", "#087986")]):
        values = accuracy.loc[names, variant]
        bars = ax.bar(x + (i - .5) * .35, values, width=.35, color=color, label=variant.title())
        ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 100), ylabel="Top-1 accuracy (%)",
           title="Classifier-only accuracy on the ten student-test trips")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(results / "student_test_top1.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(12, 4), layout="constrained", sharey=True)
    for ax, (name, label) in zip(axes, LABELS.items()):
        trials = pd.read_csv(results / name / "search_trials.csv")
        chosen = json.loads((results / name / "selection.json").read_text(encoding="utf-8"))
        ax.plot(trials["number"], 100 * trials["value"], "o-", color="#52677f", linewidth=1)
        ax.scatter(chosen["selected_trial"], 100 * chosen["selected_trial_cv_top1"],
                   color="#be185d", s=75, zorder=3, label="Selected trial")
        ax.set(title=label, xlabel="Trial number", xticks=trials["number"])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean training-fold top-1 (%)")
    axes[-1].legend()
    fig.savefig(results / "search_objective.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
