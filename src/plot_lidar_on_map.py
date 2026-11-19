"""Plot one TbV log's LiDAR points in the city (map) frame over the HD map lane boundaries.

Usage: .venv/bin/python src/plot_lidar_on_map.py data/<LOG_ID> [--every 5] [--out docs/lidar_on_map.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_feather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--every", type=int, default=5, help="use every Nth sweep (10 Hz -> 2 Hz by default)")
    parser.add_argument("--max-range", type=float, default=40.0, help="drop points farther than this from the car (m)")
    parser.add_argument("--out", type=Path, default=Path("lidar_on_map.png"))
    args = parser.parse_args()

    # 1. Ego poses: timestamp -> SE(3) transform from ego-vehicle frame to city frame.
    city_SE3_ego = read_city_SE3_ego(args.log_dir)
    pose_ts = np.array(sorted(city_SE3_ego))

    # 2. LiDAR sweeps are stored in the ego-vehicle frame. Move each into the city frame
    #    with the pose closest in time (poses ~50 Hz, sweeps 10 Hz).
    sweep_paths = sorted((args.log_dir / "sensors" / "lidar").glob("*.feather"))[:: args.every]
    xyz_city, intensity = [], []
    for path in sweep_paths:
        ts = int(path.stem)
        nearest_ts = pose_ts[np.abs(pose_ts - ts).argmin()]
        pts = read_feather(path)[["x", "y", "z", "intensity"]].to_numpy(dtype=np.float64)
        keep = np.linalg.norm(pts[:, :2], axis=1) < args.max_range
        xyz_city.append(city_SE3_ego[nearest_ts].transform_point_cloud(pts[keep, :3]))
        intensity.append(pts[keep, 3])
    xyz_city = np.concatenate(xyz_city)
    intensity = np.concatenate(intensity)

    # 3. HD map: vector map JSON lives in <log_dir>/map/log_map_archive_*.json, also in city frame.
    map_json = next((args.log_dir / "map").glob("log_map_archive_*.json"))
    avm = ArgoverseStaticMap.from_json(map_json)

    # 4. Plot: LiDAR points colored by intensity (bright = retroreflective paint), map on top.
    fig, ax = plt.subplots(figsize=(12, 12))
    order = np.argsort(intensity)  # draw bright points last so paint stays visible
    ax.scatter(xyz_city[order, 0], xyz_city[order, 1], c=intensity[order], s=0.05,
               cmap="gray", vmin=0, vmax=60, rasterized=True)
    for ls in avm.get_scenario_lane_segments():
        for bnd in (ls.left_lane_boundary, ls.right_lane_boundary):
            ax.plot(bnd.xyz[:, 0], bnd.xyz[:, 1], color="tab:orange", lw=0.8)
    for pc in avm.get_scenario_ped_crossings():
        poly = pc.polygon
        ax.fill(poly[:, 0], poly[:, 1], color="tab:cyan", alpha=0.3)
    traj = np.array([city_SE3_ego[t].translation for t in pose_ts])
    ax.plot(traj[:, 0], traj[:, 1], color="tab:red", lw=1.5, label="ego trajectory")

    ax.set_facecolor("black")
    ax.set_aspect("equal")
    pad = args.max_range
    ax.set_xlim(traj[:, 0].min() - pad, traj[:, 0].max() + pad)
    ax.set_ylim(traj[:, 1].min() - pad, traj[:, 1].max() + pad)
    ax.set_xlabel("city x (m)")
    ax.set_ylabel("city y (m)")
    ax.set_title(f"{args.log_dir.name}: {len(sweep_paths)} sweeps, {len(xyz_city):,} pts\n"
                 "orange = map lane boundaries, cyan = map crosswalks")
    ax.legend(loc="upper right")
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}  ({len(xyz_city):,} points from {len(sweep_paths)} sweeps)")


if __name__ == "__main__":
    main()
