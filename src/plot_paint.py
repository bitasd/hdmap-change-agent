"""Phase 2: plot only the lane-paint points of one TbV log, over the HD map.

Usage: .venv/bin/python src/plot_paint.py data/<LOG_ID> [--min-intensity 80] [--max-intensity 120] [--out docs/paint_on_map.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paint import ground_mask, paint_mask
from tbv_io import load_city_points, load_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--every", type=int, default=2, help="use every Nth sweep")
    parser.add_argument("--min-intensity", type=float, default=80.0)
    parser.add_argument("--max-intensity", type=float, default=120.0)
    parser.add_argument("--paint-range", type=float, default=25.0, help="only trust paint seen within this range (m)")
    parser.add_argument("--out", type=Path, default=Path("docs/paint_on_map.png"))
    args = parser.parse_args()

    xyz, intensity, rng, city_SE3_ego = load_city_points(args.log_dir, every=args.every)
    is_ground = ground_mask(xyz)
    is_paint = paint_mask(intensity, rng, is_ground, args.min_intensity, args.max_intensity, args.paint_range)
    print(f"{len(xyz):,} pts -> ground {is_ground.sum():,} ({is_ground.mean():.0%}) "
          f"-> paint {is_paint.sum():,} ({is_paint.mean():.2%})")

    avm = load_map(args.log_dir)
    traj = np.array([city_SE3_ego[t].translation for t in sorted(city_SE3_ego)])

    # Left: ground vs non-ground (sanity check). Right: paint points only, with the map.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    sub = slice(None, None, 10)
    axes[0].scatter(*xyz[sub][~is_ground[sub], :2].T, s=0.05, c="tab:green", rasterized=True, label="non-ground")
    axes[0].scatter(*xyz[sub][is_ground[sub], :2].T, s=0.05, c="0.6", rasterized=True, label="ground")
    axes[0].set_title("ground (gray) vs non-ground (green), 10% of points")

    axes[1].scatter(*xyz[is_paint, :2].T, s=0.1, c="white", rasterized=True)
    for ls in avm.get_scenario_lane_segments():
        for bnd in (ls.left_lane_boundary, ls.right_lane_boundary):
            axes[1].plot(*bnd.xyz[:, :2].T, color="tab:orange", lw=0.6)
    for pc in avm.get_scenario_ped_crossings():
        axes[1].plot(*pc.polygon[:, :2].T, color="tab:cyan", lw=1.0)
    axes[1].set_title(f"paint points (ground, {args.min_intensity:.0f} <= intensity < {args.max_intensity:.0f}, range < {args.paint_range:.0f} m)\n"
                      "orange = map lane boundaries, cyan = map crosswalks")

    for ax in axes:
        ax.plot(*traj[:, :2].T, color="tab:red", lw=1.0)
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_xlabel("city x (m)")
    axes[0].set_ylabel("city y (m)")
    axes[0].set_xlim(traj[:, 0].min() - 30, traj[:, 0].max() + 30)
    axes[0].set_ylim(traj[:, 1].min() - 30, traj[:, 1].max() + 30)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
