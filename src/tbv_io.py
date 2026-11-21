"""Shared loaders for one TbV log."""

from pathlib import Path

import numpy as np

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_feather


def load_city_points(log_dir: Path, every: int = 1, max_range: float = 40.0):
    """Load LiDAR sweeps and move them from the ego-vehicle frame into the city frame.

    Each sweep uses the ego pose closest in time (poses ~170 Hz, sweeps 10 Hz).

    Returns:
        xyz_city: (N, 3) points in the city frame.
        intensity: (N,) LiDAR return intensity, 0-255.
        rng: (N,) horizontal distance from the car when the point was measured (m).
        city_SE3_ego: timestamp -> SE(3) pose mapping, for plotting the trajectory.
    """
    city_SE3_ego = read_city_SE3_ego(log_dir)
    pose_ts = np.array(sorted(city_SE3_ego))

    sweep_paths = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))[::every]
    xyz_city, intensity, rng = [], [], []
    for path in sweep_paths:
        ts = int(path.stem)
        pose = city_SE3_ego[pose_ts[np.abs(pose_ts - ts).argmin()]]
        pts = read_feather(path)[["x", "y", "z", "intensity"]].to_numpy(dtype=np.float64)
        r = np.linalg.norm(pts[:, :2], axis=1)
        keep = r < max_range
        xyz_city.append(pose.transform_point_cloud(pts[keep, :3]))
        intensity.append(pts[keep, 3])
        rng.append(r[keep])
    return np.concatenate(xyz_city), np.concatenate(intensity), np.concatenate(rng), city_SE3_ego


def load_map(log_dir: Path) -> ArgoverseStaticMap:
    """Load the log's HD vector map (lane segments, crosswalks), already in the city frame."""
    return ArgoverseStaticMap.from_json(next((log_dir / "map").glob("log_map_archive_*.json")))
