"""Profile slowest student-test trip and verify cached graph routes exactly.

No student-test labels enter inference; profiling does not select tuning
parameters. Uses the KNN scored-export, not a newly trained model.
"""
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.decoder import decode
from pathsnap.routing import RoadGraph
from pathsnap.scoring import match_exported_scores, SCORE_FILES
from scripts.view_trip import ROOT, load_trip, render_trip
from scripts.benchmark_trained_decoding import assemble

TRIP = "00000009"
DATA = ROOT / "data/processed"
OUT = ROOT / "experiments/viterbi_tuning"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "data/raw/Student folder.zip"
    track, nodes, arcs = load_trip(DATA / "observations.csv.gz", zip_path, "student_test", TRIP)
    rows = pq.read_table(DATA / "candidates_student_test.parquet",
                         filters=[("split", "=", "student_test"), ("trip_id", "=", TRIP)]).to_pandas()
    file = SCORE_FILES["knn"]
    scores = match_exported_scores(
        rows, pq.read_table(file, filters=[("trip_id", "=", TRIP)],
                            columns=["split","trip_id","obs_id","arc_id","score"]).to_pandas())
    timeline, _ = assemble(track, rows, scores)

    measurements = []
    results = []
    for cache_size in (0, 20000):
        graph = RoadGraph(nodes, arcs, shortest_cache_size=cache_size)
        start = perf_counter()
        result = decode(graph, timeline)
        elapsed = perf_counter() - start
        map_start = perf_counter()
        folium_map, _ = render_trip(track, nodes, arcs, "student_test", TRIP,
                                    route_segments=result.segments)
        # Render HTML in memory to measure actual Folium generation.
        html = folium_map.get_root().render()
        map_time = perf_counter() - map_start
        results.append(result)
        measurements.append({
            "trip_id": TRIP, "model": "knn", "cache_size": cache_size,
            "gps_observations": len(track), "candidate_rows": len(rows),
            "route_segments": len(result.segments),
            "route_breaks": max(0, len(result.segments)-1),
            "decode_seconds": elapsed,
            "road_routing_seconds": graph.route_seconds,
            "dijkstra_seconds": graph.shortest_search_seconds,
            "dijkstra_searches": graph.shortest_searches,
            "shortest_cache_hits": graph.shortest_cache_hits,
            "map_render_seconds": map_time,
            "map_html_bytes": len(html.encode("utf-8")),
        })
        print(json.dumps(measurements[-1]), flush=True)
    same_arcs = results[0].arcs_by_observation == results[1].arcs_by_observation
    same_route = results[0].segments == results[1].segments
    if not same_arcs or not same_route:
        raise AssertionError("Shortest-path cache changed decoded route or GPS matching!")
    saved = measurements[0]["decode_seconds"] - measurements[1]["decode_seconds"]
    profile = {
        "status": "exact route and GPS arc predictions verified identical",
        "same_arcs_by_observation": same_arcs,
        "same_segment_routes": same_route,
        "decode_seconds_saved": saved,
        "decode_percent_saved": 100 * saved / measurements[0]["decode_seconds"],
        "runs": measurements,
        "note": "One pair of cold runs on a GitHub Actions runner; timings can vary, do not infer cross-machine speedups. Folium HTML rendering is measured separately.",
    }
    (OUT / "profile_trip_00000009.json").write_text(json.dumps(profile, indent=2)+"\n")
    print("COMPARISON", json.dumps(profile), flush=True)


if __name__ == "__main__":
    main()
