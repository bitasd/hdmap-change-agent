"""Regenerate every figure in docs/walkthrough.md from one TbV log.

Usage: .venv/bin/python src/make_figures.py data/<LOG_ID> [--out docs/figures]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow

from av2.utils.io import read_city_SE3_ego, read_feather
from map_geom import lane_boundaries, on_map_paint
from paint import ground_mask, paint_mask
from tbv_io import load_city_points, load_map

# Palette: light surface, recessive ink, categorical slots in fixed order (blue, orange, aqua).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INTENSITY_CMAP = LinearSegmentedColormap.from_list("intensity", ["#e4e3df", "#8a8984", INK])  # neutral, so blue = paint

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def draw_map(ax, avm, unpainted=False):
    """Lane boundaries (orange) and crosswalks (aqua); optionally unpainted boundaries dotted."""
    for xyz, mark in lane_boundaries(avm):
        if mark != "NONE":
            ax.plot(*xyz[:, :2].T, color=ORANGE, lw=1.0)
        elif unpainted:
            ax.plot(*xyz[:, :2].T, color=INK_2, lw=0.6, ls=(0, (1, 2)))
    for pc in avm.get_scenario_ped_crossings():
        ax.plot(*pc.polygon[:, :2].T, color=AQUA, lw=1.6)


def draw_car(ax, x, y, heading, size=4.0, color=INK):
    ax.add_patch(FancyArrow(x, y, size * np.cos(heading), size * np.sin(heading),
                            width=size * 0.35, head_width=size * 0.7, head_length=size * 0.45,
                            length_includes_head=True, color=color, zorder=5))


def bev(ax, xlabel="city x (m)", ylabel="city y (m)"):
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save(fig, out_dir, name):
    fig.savefig(out_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}

    avm = load_map(args.log_dir)
    city_SE3_ego = read_city_SE3_ego(args.log_dir)
    pose_ts = np.array(sorted(city_SE3_ego))
    traj = np.array([city_SE3_ego[t].translation for t in pose_ts])
    sweep_paths = sorted((args.log_dir / "sensors" / "lidar").glob("*.feather"))
    t0 = int(sweep_paths[0].stem)

    def sweep(i):
        ts = int(sweep_paths[i].stem)
        pts = read_feather(sweep_paths[i])[["x", "y", "z", "intensity"]].to_numpy(dtype=np.float64)
        return ts, pts, city_SE3_ego[pose_ts[np.abs(pose_ts - ts).argmin()]]

    city_xlim = (traj[:, 0].min() - 30, traj[:, 0].max() + 30)
    city_ylim = (traj[:, 1].min() - 30, traj[:, 1].max() + 30)
    stats["n_sweeps"] = len(sweep_paths)
    stats["duration_s"] = round((int(sweep_paths[-1].stem) - t0) * 1e-9, 1)
    stats["n_poses"] = len(pose_ts)

    # 1. One sweep, as the sensor sees it: ego frame, car at the origin.
    _, pts, _ = sweep(50)
    near = np.linalg.norm(pts[:, :2], axis=1) < 40
    fig, ax = plt.subplots(figsize=(8, 7))
    order = np.argsort(pts[near, 3])
    sc = ax.scatter(*pts[near][order, :2].T, c=pts[near][order, 3], cmap=INTENSITY_CMAP, vmin=0, vmax=60,
                    s=0.3, rasterized=True)
    draw_car(ax, 0, 0, 0, size=4.5)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="LiDAR intensity (clipped at 60)")
    ax.set_title("1. One LiDAR sweep, in the car's own frame")
    bev(ax, "ego x: forward (m)", "ego y: left (m)")
    stats["points_per_sweep"] = len(pts)
    save(fig, args.out, "01_single_sweep_ego.png")

    # 2. Where the car was: poses in the city frame.
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_map(ax, avm)
    ax.plot(*traj[:, :2].T, color=INK, lw=1.5)
    for i in range(0, len(sweep_paths), 30):
        ts, _, pose = sweep(i)
        heading = np.arctan2(pose.rotation[1, 0], pose.rotation[0, 0])
        draw_car(ax, *pose.translation[:2], heading)
        ax.annotate(f"t = {(ts - t0) * 1e-9:.0f} s", pose.translation[:2], xytext=(10, 8),
                    textcoords="offset points", color=INK_2)
    ax.set_xlim(city_xlim)
    ax.set_ylim(city_ylim)
    ax.set_title("2. Ego poses: the car's position + heading in the city frame")
    bev(ax)
    save(fig, args.out, "02_poses_city_frame.png")

    # 3. Why poses matter: three sweeps before and after the ego -> city transform.
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    rng = np.random.default_rng(0)
    for i, color in zip((0, 60, 120), (BLUE, ORANGE, AQUA)):
        ts, pts, pose = sweep(i)
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 30]
        pts = pts[rng.choice(len(pts), 15000, replace=False)]
        label = f"sweep at t = {(ts - t0) * 1e-9:.0f} s"
        axes[0].scatter(*pts[:, :2].T, s=0.4, color=color, rasterized=True, label=label)
        axes[1].scatter(*pose.transform_point_cloud(pts[:, :3])[:, :2].T, s=0.4, color=color, rasterized=True,
                        label=label)
    axes[0].set_title("3a. Car frame: sweeps don't line up")
    bev(axes[0], "ego x (m)", "ego y (m)")
    axes[1].set_title("3b. City frame: the same walls overlap")
    bev(axes[1])
    for ax in axes:
        ax.set_anchor("N")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3, markerscale=12)
    save(fig, args.out, "03_sweeps_before_after_transform.png")

    # Accumulated cloud for steps 4-8.
    xyz, intensity, rng_m, _ = load_city_points(args.log_dir, every=2)
    is_ground = ground_mask(xyz)
    is_paint = paint_mask(intensity, rng_m, is_ground)
    stats["accumulated_points"] = len(xyz)
    stats["ground_frac"] = round(float(is_ground.mean()), 3)
    stats["paint_points"] = int(is_paint.sum())

    # 4. All sweeps in the city frame, over the HD map.
    fig, ax = plt.subplots(figsize=(10, 9))
    sub = np.arange(0, len(xyz), 3)
    sub = sub[np.argsort(intensity[sub])]
    ax.scatter(*xyz[sub, :2].T, c=intensity[sub], cmap=INTENSITY_CMAP, vmin=0, vmax=60, s=0.05, rasterized=True)
    draw_map(ax, avm)
    ax.plot(*traj[:, :2].T, color=INK, lw=1.5)
    ax.plot([], [], color=ORANGE, label="map: painted lane boundary")
    ax.plot([], [], color=AQUA, label="map: crosswalk")
    ax.plot([], [], color=INK, label="ego path")
    ax.legend(loc="upper right")
    ax.set_xlim(city_xlim)
    ax.set_ylim(city_ylim)
    ax.set_title(f"4. {len(sweep_paths) // 2} sweeps in the city frame, over the HD map")
    bev(ax)
    save(fig, args.out, "04_lidar_on_map.png")

    # 5. Ground segmentation: a 1 m-thick slice across the road, seen from the side.
    _, _, pose = sweep(40)
    local = pose.inverse().transform_point_cloud(xyz)
    in_slice = (np.abs(local[:, 0]) < 0.5) & (np.abs(local[:, 1]) < 25) & (local[:, 2] < 6)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for mask, color, label in ((~is_ground, ORANGE, "non-ground"), (is_ground, BLUE, "ground")):
        m = in_slice & mask
        ax.scatter(local[m, 1], local[m, 2], s=1.5, color=color, label=label, rasterized=True)
    ax.invert_xaxis()  # left of the car on the left of the plot, as if looking forward
    ax.set_xlabel("lateral offset from the car (m, + = left)")
    ax.set_ylabel("height above car's ground (m)")
    ax.legend(markerscale=6, loc="upper right")
    ax.set_title("5. Ground vs everything else: a 1 m slice across the road, seen from behind the car")
    save(fig, args.out, "05_ground_cross_section.png")

    # 6. Intensity: paint vs not-paint (using the map as a rough answer key).
    close_ground = is_ground & (rng_m < 25)
    on_map = on_map_paint(xyz[close_ground, :2], avm)
    I = intensity[close_ground]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(0, 256, 4)
    ax.axvspan(80, 120, color=MUTED, alpha=0.35, lw=0)
    ax.text(100, 0.97, "paint band\n80 - 120", ha="center", va="top", color=INK_2, transform=ax.get_xaxis_transform())
    ax.hist(I[~on_map], bins=bins, histtype="step", lw=2, color=INK_2, label="ground elsewhere (asphalt, curb, sidewalk)")
    ax.hist(I[on_map], bins=bins, histtype="step", lw=2, color=BLUE, label="ground on map paint (lane line / crosswalk)")
    ax.set_yscale("log")
    ax.set_xlabel("LiDAR intensity")
    ax.set_ylabel("points (log scale)")
    ax.legend(loc="upper right")
    ax.set_title("6. Paint returns brighter than road surface")
    band = (I >= 80) & (I < 120)
    stats["band_on_map_frac"] = round(float(on_map[band].mean()), 3)
    stats["ge50_on_map_frac"] = round(float(on_map[I >= 50].mean()), 3)
    save(fig, args.out, "06_intensity_histogram.png")

    # 7. Paint points over the map.
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(*xyz[is_paint, :2].T, s=0.2, color=BLUE, rasterized=True)
    draw_map(ax, avm, unpainted=True)
    ax.scatter([], [], s=12, color=BLUE, label="detected paint (LiDAR)")
    ax.plot([], [], color=ORANGE, label="map: painted lane boundary")
    ax.plot([], [], color=INK_2, lw=0.6, ls=(0, (1, 2)), label="map: unpainted (virtual) boundary")
    ax.plot([], [], color=AQUA, label="map: crosswalk")
    ax.legend(loc="upper right")
    ax.set_xlim(city_xlim)
    ax.set_ylim(city_ylim)
    ax.set_title("7. Detected paint vs the HD map")
    bev(ax)
    marks = [m for _, m in lane_boundaries(avm)]
    stats["boundaries_total"] = len(marks)
    stats["boundaries_unpainted"] = marks.count("NONE")
    save(fig, args.out, "07_paint_on_map.png")

    # 8. Zoom on the crosswalk with the most paint.
    crossings = avm.get_scenario_ped_crossings()
    centers = np.array([pc.polygon[:, :2].mean(axis=0) for pc in crossings])
    paint_xy = xyz[is_paint, :2]
    counts = [(np.linalg.norm(paint_xy - c, axis=1) < 4).sum() for c in centers]
    cx, cy = centers[int(np.argmax(counts))]
    win = 9
    box = (np.abs(xyz[:, 0] - cx) < win) & (np.abs(xyz[:, 1] - cy) < win)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    g = box & is_ground
    order = np.argsort(intensity[g])
    axes[0].scatter(*xyz[g][order, :2].T, c=intensity[g][order], cmap=INTENSITY_CMAP, vmin=0, vmax=120, s=0.6,
                    rasterized=True)
    axes[0].set_title("8a. Ground points, colored by intensity")
    axes[1].scatter(*xyz[box & is_paint, :2].T, s=0.8, color=BLUE, rasterized=True)
    axes[1].set_title("8b. Paint points vs map crosswalk (aqua)")
    for ax in axes:
        draw_map(ax, avm)
        ax.set_xlim(cx - win, cx + win)
        ax.set_ylim(cy - win, cy + win)
        bev(ax)
    save(fig, args.out, "08_crosswalk_zoom.png")

    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
