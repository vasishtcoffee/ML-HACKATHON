"""Build baseline comparison figures and an offline Folium evaluation map."""
import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from score_baseline import KEY
from train_baselines import MODELS, OBS_KEY, top_candidates
from view_trip import ROOT, load_trip


LABELS = {"logistic_regression": "Logistic Regression", "knn": "KNN"}
COLORS = {"logistic_regression": "#b45309", "knn": "#7e22ce"}


def comparison_figures(metrics, output):
    cv = metrics.loc[metrics["phase"].eq("cv")]
    test = metrics.loc[metrics["phase"].eq("student_test")].set_index("model").loc[list(MODELS)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    titles = [
        ("Grouped CV macro F1", "macro_f1", "cv"),
        ("Student-test macro F1", "macro_f1", "test"),
        ("Student-test top-1 accuracy", "top1_accuracy", "test"),
        ("Student-test candidate log loss", "candidate_log_loss", "test"),
    ]
    names = [LABELS[name] for name in MODELS]
    colors = [COLORS[name] for name in MODELS]
    for ax, (title, metric, phase) in zip(axes.ravel(), titles):
        if phase == "cv":
            grouped = cv.groupby("model")[metric]
            values = grouped.mean().loc[list(MODELS)]
            error = grouped.std().loc[list(MODELS)]
        else:
            values = test[metric]
            error = None
        bars = ax.bar(names, values, color=colors, yerr=error, capsize=4)
        ax.bar_label(bars, labels=[f"{x:.3f}" for x in values], padding=6)
        ax.set_title(title)
        ax.set_ylim(0, 1 if metric != "candidate_log_loss" else max(values) * 1.30)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(10, 4.4), layout="constrained")
    for ax, (name, row) in zip(axes, test.iterrows()):
        counts = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, vmin=0, vmax=1, cmap="Blues")
        for i, j in np.ndindex(2, 2):
            ax.text(j, i, f"{counts[i,j]:,}\n{shares[i,j]:.1%}", ha="center", va="center",
                    color="white" if shares[i, j] > 0.55 else "#10243c")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Other", "Reference"],
               yticklabels=["Other", "Reference"], xlabel="Predicted candidate class",
               ylabel="Actual candidate class", title=LABELS[name])
    fig.savefig(output / "student_test_confusion.png", dpi=160)
    plt.close(fig)


def evaluation_map(processed, archive_path, output, trip_id):
    split = "student_test"
    track, nodes, arcs = load_trip(processed / "observations.csv.gz", archive_path, split, trip_id)
    gps = track[["latitude", "longitude"]].to_numpy()
    latlon = nodes[:, ::-1]
    segments = latlon[arcs]
    lo, hi = gps.min(axis=0), gps.max(axis=0)
    margin = np.maximum((hi - lo) * .05, .001)
    near = ((segments.max(axis=1) >= lo - margin) & (segments.min(axis=1) <= hi + margin)).all(axis=1)

    inventory = pd.read_csv(processed / "archive_inventory.csv", dtype={"trip_id": str})
    member = inventory.loc[
        inventory["split"].eq(split) & inventory["trip_id"].eq(trip_id)
        & inventory["kind"].eq("route"), "member"
    ].item()
    with ZipFile(archive_path) as archive, archive.open(member) as source:
        route = np.loadtxt(source, dtype=np.int64, ndmin=1)

    report_map = folium.Map(location=gps.mean(axis=0).tolist(), prefer_canvas=True)
    roads = folium.FeatureGroup(name="Nearby road network")
    if near.any():
        folium.PolyLine(segments[near].tolist(), color="#64748b", weight=1, opacity=.4).add_to(roads)
    roads.add_to(report_map)
    measured = folium.FeatureGroup(name="GPS observations")
    for point in gps:
        folium.CircleMarker(point.tolist(), radius=1.7, stroke=False, fill=True,
                            fill_color="#1d4ed8", fill_opacity=.55).add_to(measured)
    measured.add_to(report_map)
    reference = folium.FeatureGroup(name="Supplied reference route")
    folium.PolyLine(segments[route].tolist(), color="#15803d", weight=6, opacity=.75).add_to(reference)
    reference.add_to(report_map)

    candidates = pd.read_parquet(
        processed / f"candidates_{split}.parquet", columns=KEY + ["candidate_rank"],
        filters=[("trip_id", "=", trip_id)],
    )
    for name in MODELS:
        scores = pd.read_parquet(output / f"{name}_student_scores.parquet",
                                 filters=[("trip_id", "=", trip_id)])
        scored = candidates.merge(scores, on=KEY, validate="one_to_one")
        winners = top_candidates(scored, scored["score"].to_numpy())
        ordered = track[OBS_KEY].merge(winners, on=OBS_KEY, how="left", validate="one_to_one")
        arc_ids = ordered["arc_id"].dropna().astype(int).unique()
        layer = folium.FeatureGroup(name=LABELS[name] + " top-1 arcs", show=name == "logistic_regression")
        if len(arc_ids):
            folium.PolyLine(segments[arc_ids].tolist(), color=COLORS[name], weight=3,
                            opacity=.9).add_to(layer)
        layer.add_to(report_map)

    report_map.fit_bounds([lo.tolist(), hi.tolist()])
    folium.LayerControl(collapsed=False).add_to(report_map)
    panel = f"""
    <aside style="position:fixed;z-index:1000;bottom:24px;left:16px;background:white;
                  padding:12px 15px;border:1px solid #cbd5e1;border-radius:6px;font:14px system-ui">
      <strong>Student-test trip {trip_id}</strong><br>
      <span style="color:#64748b">━</span> Road network<br>
      <span style="color:#1d4ed8">●</span> Measured GPS<br>
      <span style="color:#15803d">━</span> Reference route<br>
      <span style="color:#b45309">━</span> Logistic Regression<br>
      <span style="color:#7e22ce">━</span> KNN
    </aside>"""
    if not trip_id.isascii() or not trip_id.isdecimal():
        raise ValueError("Expected a numeric trip ID.")
    report_map.get_root().html.add_child(folium.Element(panel))
    html = report_map.get_root().render()
    clean_html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (output / "student_test_map.html").write_text(clean_html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get(
        "PATHSNAP_DATA_ZIP", ROOT / "data" / "raw" / "Student folder.zip",
    )))
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/baselines")
    parser.add_argument("--trip-id", default="00000007")
    args = parser.parse_args()
    comparison_figures(pd.read_csv(args.output / "metrics.csv"), args.output)
    evaluation_map(args.processed_dir, args.archive, args.output, args.trip_id)


if __name__ == "__main__":
    main()
