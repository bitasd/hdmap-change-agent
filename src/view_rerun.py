"""Open one TbV log in the Rerun viewer: LiDAR sweeps in the city frame over time, plus HD map and ego path.

Usage: .venv/bin/python src/view_rerun.py data/<LOG_ID> [--every 1] [--max-range 60]
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import rerun as rr

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_feather

# AV2 ego origin is the rear axle center on the ground; the car extends ~3.9 m forward, ~1 m back.
EGO_CENTER = [1.45, 0.0, 0.8]
EGO_HALF_SIZE = [2.45, 1.0, 0.8]


def intensity_to_rgb(intensity: np.ndarray, vmax: float = 60.0) -> np.ndarray:
    """Map LiDAR intensity (0-255) to RGB; clipped low so retroreflective paint stands out."""
    return (matplotlib.colormaps["inferno"](np.clip(intensity / vmax, 0, 1))[:, :3] * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--every", type=int, default=1, help="use every Nth sweep")
    parser.add_argument("--max-range", type=float, default=60.0, help="drop points farther than this from the car (m)")
    args = parser.parse_args()

    rr.init("tbv_log_viewer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Static layers: HD map (already in the city frame) and the full ego path.
    avm = ArgoverseStaticMap.from_json(next((args.log_dir / "map").glob("log_map_archive_*.json")))
    lane_bounds = [b.xyz for ls in avm.get_scenario_lane_segments()
                   for b in (ls.left_lane_boundary, ls.right_lane_boundary)]
    rr.log("world/map/lane_boundaries", rr.LineStrips3D(lane_bounds, colors=[255, 160, 0], radii=0.05), static=True)
    crosswalks = [pc.polygon for pc in avm.get_scenario_ped_crossings()]  # closed 5-point loops
    rr.log("world/map/crosswalks", rr.LineStrips3D(crosswalks, colors=[0, 200, 255], radii=0.08), static=True)

    city_SE3_ego = read_city_SE3_ego(args.log_dir)
    pose_ts = np.array(sorted(city_SE3_ego))
    path = np.array([city_SE3_ego[t].translation for t in pose_ts])
    rr.log("world/ego_path", rr.LineStrips3D([path], colors=[255, 50, 50], radii=0.1), static=True)
    rr.log("world/ego/box", rr.Boxes3D(centers=[EGO_CENTER], half_sizes=[EGO_HALF_SIZE], colors=[255, 50, 50]),
           static=True)

    # Per sweep: set the timeline, move the car, and log the points transformed ego -> city.
    sweep_paths = sorted((args.log_dir / "sensors" / "lidar").glob("*.feather"))[:: args.every]
    t0 = int(sweep_paths[0].stem)
    for sweep_path in sweep_paths:
        ts = int(sweep_path.stem)
        rr.set_time("log_time", duration=(ts - t0) * 1e-9)
        pose = city_SE3_ego[pose_ts[np.abs(pose_ts - ts).argmin()]]

        rr.log("world/ego", rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation))

        pts = read_feather(sweep_path)[["x", "y", "z", "intensity"]].to_numpy(dtype=np.float64)
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < args.max_range]
        rr.log("world/lidar", rr.Points3D(pose.transform_point_cloud(pts[:, :3]),
                                          colors=intensity_to_rgb(pts[:, 3]), radii=0.03))

    print(f"logged {len(sweep_paths)} sweeps over {(int(sweep_paths[-1].stem) - t0) * 1e-9:.1f} s")


if __name__ == "__main__":
    main()
