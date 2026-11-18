"""Render one development trip's measured GPS and road network with Folium.

Uses the preprocessing notebook's coordinate/ID conventions and bounding-box
road selection. Does not read reference routes or require trained models.
"""
import argparse
import html
import json
import os
from pathlib import Path
from zipfile import ZipFile

import folium
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_trip(observations_path, archive_path, split, trip_id):
    observations = pd.read_csv(observations_path, dtype={"trip_id": str})
    track = observations.loc[
        observations["split"].eq(split) & observations["trip_id"].eq(trip_id)
    ].sort_values(["timestamp_s", "obs_id"], kind="stable").copy()
    if track.empty:
        raise ValueError(f"No observations for {split}/{trip_id}. Use --list-trips.")
    if track["obs_id"].duplicated().any():
        raise ValueError("Duplicate observation IDs in selected trip.")
    values = track[["longitude", "latitude", "timestamp_s"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite coordinates or timestamps in processed observations.")
    if not (track.longitude.between(-180, 180) & track.latitude.between(-90, 90)).all():
        raise ValueError("GPS coordinates out of bounds.")

    with ZipFile(archive_path) as archive:
        def member(kind):
            suffix = f"{split}/{trip_id}/{trip_id}.{kind}"
            matches = [n for n in archive.namelist()
                       if n == suffix or n.endswith('/' + suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {suffix}; found {len(matches)}.")
            return matches[0]

        with archive.open(member("nodes")) as stream:
            nodes = np.loadtxt(stream, ndmin=2)
        with archive.open(member("arcs")) as stream:
            arcs = np.loadtxt(stream, dtype=np.int64, ndmin=2)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Nodes and arcs must each have two columns.")
    if (not np.isfinite(nodes).all() or np.any(np.abs(nodes[:, 0]) > 180)
            or np.any(np.abs(nodes[:, 1]) > 90)):
        raise ValueError("Invalid road-node coordinates.")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Arc endpoint is outside the node table.")
    return track, nodes, arcs


def render_trip(track, nodes, arcs, split, trip_id, *, route_segments=None):
    # File order is longitude/latitude. Folium locations use latitude/longitude.
    gps = track[["latitude", "longitude"]].to_numpy()
    latlon = nodes[:, ::-1]
    starts, ends = latlon[arcs[:, 0]], latlon[arcs[:, 1]]
    lo, hi = gps.min(axis=0), gps.max(axis=0)
    margin = np.maximum((hi - lo) * 0.05, 0.001)
    near = ((np.maximum(starts, ends) >= lo - margin)
            & (np.minimum(starts, ends) <= hi + margin)).all(axis=1)

    trip_map = folium.Map(location=gps.mean(axis=0).tolist(),
                          tiles="OpenStreetMap", prefer_canvas=True,
                          control_scale=True, zoom_control=False)
    road_layer = folium.FeatureGroup(name="Road network (near GPS extent)")
    lines = np.stack([starts[near], ends[near]], axis=1).tolist()
    if lines:
        # Separate segments: never join unrelated arcs into a single line.
        folium.PolyLine(lines, color="#64748b", weight=2, opacity=0.55).add_to(road_layer)
    road_layer.add_to(trip_map)

    gps_layer = folium.FeatureGroup(name="Measured GPS observations")
    gap_layer = folium.FeatureGroup(name="GPS gaps over 5 seconds")
    dt = track.timestamp_s.diff()
    for i, row in enumerate(track.itertuples(index=False)):
        popup = (f"Observation {int(row.obs_id)}<br>Time: {row.timestamp_s:g} s"
                 f"<br>Longitude: {row.longitude:.6f}<br>Latitude: {row.latitude:.6f}")
        folium.CircleMarker(gps[i].tolist(), radius=2.5, color="#2563eb",
                            weight=1, fill=True, fill_opacity=0.85,
                            popup=popup).add_to(gps_layer)
        if dt.iloc[i] > 5:
            folium.CircleMarker(gps[i].tolist(), radius=7, color="#d97706",
                                weight=2, fill=False,
                                tooltip=f"After a {dt.iloc[i]:g}-second gap").add_to(gap_layer)
    gps_layer.add_to(trip_map)
    gap_layer.add_to(trip_map)
    endpoints = folium.FeatureGroup(name="First / last observation")
    for point, label, color in [(gps[0], "First observation", "green"),
                                (gps[-1], "Last observation", "red")]:
        folium.Marker(point.tolist(), tooltip=label,
                      icon=folium.Icon(color=color, icon="info-sign")).add_to(endpoints)
    endpoints.add_to(trip_map)
    if route_segments is not None:
        decoded_layer = folium.FeatureGroup(name="Decoded route (distance baseline)")
        decoded_lines = [
            [starts[arc_id].tolist(), ends[arc_id].tolist()]
            for segment in route_segments for arc_id in segment
        ]
        if decoded_lines:
            folium.PolyLine(decoded_lines, color="#be185d", weight=4).add_to(decoded_layer)
        decoded_layer.add_to(trip_map)
    folium.LayerControl(collapsed=False, position="topright").add_to(trip_map)
    trip_map.fit_bounds([lo.tolist(), hi.tolist()], padding=(35, 35))

    summary = {"split": split, "trip_id": trip_id, "gps_observations": len(track),
               "road_nodes": len(nodes), "road_arcs": len(arcs),
               "displayed_road_arcs": int(near.sum()),
               "gaps_over_5s": int(dt.gt(5).sum()),
               "duration_seconds": float(track.timestamp_s.iloc[-1] - track.timestamp_s.iloc[0])}
    title = html.escape(f"{split} / {trip_id}")
    mode = ("Development prediction · Distance baseline"
            if route_segments is not None else "Exploration only · No model predictions.")
    panel = f"""
    <style>
      .leaflet-container {{ background: #e9eef4; }}
      .trip-panel {{ position:fixed; left:18px; bottom:28px; z-index:1000;
        background:white; padding:18px 22px; border-radius:12px;
        box-shadow:0 4px 24px #0f172a26; font:14px/1.5 system-ui,sans-serif;
        color:#0f172a; max-width:330px; }}
      .trip-panel h1 {{ font-size:22px; margin:0 0 3px; }}
      .trip-panel .muted {{ color:#64748b; font-size:12px; }}
      .trip-panel .counts {{ margin:10px 0; font-weight:600; }}
      @media(max-width:600px) {{ .trip-panel {{ left:8px; bottom:20px;
        padding:10px; max-width:235px; font-size:12px; }} }}
    </style>
    <aside class="trip-panel">
      <div class="muted">PATHSNAP · DATA EXPLORER</div>
      <h1>Trip {html.escape(trip_id)}</h1><div class="muted">{title}</div>
      <div class="counts">{len(track):,} GPS points · {int(near.sum()):,} visible arcs</div>
      <div><span style="color:#2563eb">●</span> Recorded GPS observations</div>
      <div><span style="color:#64748b">━</span> Road segments</div>
      <div><span style="color:#d97706">○</span> After a gap &gt; 5 s ({summary['gaps_over_5s']})</div>
      <div class="muted" style="margin-top:10px">{mode}<br>
      GPS points are not joined across gaps.<br>Map tiles and libraries need internet access.</div>
    </aside>"""
    trip_map.get_root().html.add_child(folium.Element(panel))
    trip_map.get_root().header.add_child(folium.Element("<title>PathSnap · Trip " + html.escape(trip_id) + "</title>"))
    return trip_map, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "student_test"], default="train")
    parser.add_argument("--trip-id", help="Original string ID, including leading zeros")
    parser.add_argument("--list-trips", action="store_true")
    parser.add_argument("--observations", type=Path,
                        default=ROOT / "data/processed/observations.csv.gz")
    parser.add_argument("--data-zip", type=Path,
                        default=Path(os.environ.get("PATHSNAP_DATA_ZIP",
                                                   ROOT / "data/raw/Student folder.zip")))
    parser.add_argument("--output", type=Path, help="Map output path ending in .html")
    args = parser.parse_args()
    if args.output is not None and args.output.suffix != ".html":
        parser.error("--output must end in .html")
    observations = pd.read_csv(args.observations, dtype={"trip_id": str})
    counts = observations.loc[observations.split.eq(args.split)].groupby("trip_id").size()
    if counts.empty:
        parser.error(f"No trips for {args.split}")
    if args.list_trips:
        print(counts.rename("gps_observations").to_string())
        return
    trip_id = args.trip_id or (counts - counts.median()).abs().idxmin()
    try:
        track, nodes, arcs = load_trip(args.observations, args.data_zip.expanduser(), args.split, trip_id)
        trip_map, summary = render_trip(track, nodes, arcs, args.split, trip_id)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = args.output or ROOT / "data/viewer" / f"{args.split}_{trip_id}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    trip_map.save(str(output))
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Map: {output.resolve()}")


if __name__ == "__main__":
    main()
