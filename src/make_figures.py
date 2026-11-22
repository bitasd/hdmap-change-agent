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

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.io import read_city_SE3_ego, read_feather
from deviation import inside_crosswalks, painted_boundaries, signed_offsets
from map_geom import lane_boundaries, on_map_paint
from measure_deviation import labeled_change_points, measure_log
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
    parser.add_argument("--labels", type=Path, default=Path("data/tbv_labels/tbv_val_split_annotations.json"))
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

    # Phase 3: measure every painted element.
    label_pts = labeled_change_points(args.labels, args.log_dir.name)
    table, shifts = measure_log(xyz, intensity, rng_m, avm, label_pts)
    boundaries = {eid: xy for eid, _, xy in painted_boundaries(avm)}

    # 9. How one lane line is measured: straighten it, look at the sideways offsets.
    eid = "lane_33047006_L"
    row = table.set_index("element").loc[eid]
    lane_paint = paint_xy[~inside_crosswalks(paint_xy, avm)]
    offsets, arc, _ = signed_offsets(lane_paint, boundaries[eid])
    inl = np.abs(offsets - row.offset_cm / 100) < 0.15
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    axes[0].axhline(0, color=ORANGE, lw=2, label="map line")
    axes[0].scatter(arc[~inl], 100 * offsets[~inl], s=4, color=MUTED, label="other paint in the 1 m corridor")
    axes[0].scatter(arc[inl], 100 * offsets[inl], s=4, color=BLUE, label="paint on the line (inliers)")
    axes[0].axhline(row.offset_cm, color=INK, lw=1, ls="--")
    axes[0].set_xlabel("distance along the map line (m)")
    axes[0].set_ylabel("sideways offset (cm, + = left)")
    axes[0].legend(loc="upper right", markerscale=3)
    axes[0].set_title(f"9a. One dashed lane line, straightened ({eid})")
    axes[1].hist(100 * offsets, bins=np.arange(-100, 101, 5), orientation="horizontal", color=MUTED)
    axes[1].hist(100 * offsets[inl], bins=np.arange(-100, 101, 5), orientation="horizontal", color=BLUE)
    axes[1].axhline(row.offset_cm, color=INK, lw=1, ls="--")
    axes[1].annotate(f"median of inliers\n= {row.offset_cm:+.0f} cm", (axes[1].get_xlim()[1] * 0.35, row.offset_cm),
                     xytext=(0, 14), textcoords="offset points", color=INK)
    axes[1].set_xlabel("paint points")
    axes[1].set_title("9b. Offset histogram")
    save(fig, args.out, "09_measure_one_line.png")

    # 10. Every element's status, with the TbV change label.
    status_style = {"measured": (BLUE, 1.8, "-"), "faint_paint": (AQUA, 1.8, "-"),
                    "paint_missing": (ORANGE, 3.0, "-"), "not_observed": (MUTED, 0.8, (0, (1, 2)))}
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(*paint_xy.T, s=0.1, color="#c9d6e8", rasterized=True)
    polys = boundaries | {f"xwalk_{pc.id}": pc.polygon[:, :2] for pc in crossings}
    for eid, st in table[["element", "status"]].values:
        color, lw, ls = status_style[st]
        ax.plot(*polys[eid].T, color=color, lw=lw, ls=ls)
    if label_pts is not None and len(label_pts):
        ax.plot(*label_pts.T, color=INK, lw=1.5, ls="--")
        ax.plot([], [], color=INK, lw=1.5, ls="--", label="TbV label: line removed (bike lane)")
    counts = table.status.value_counts()
    for st, (color, lw, ls) in status_style.items():
        ax.plot([], [], color=color, lw=lw, ls=ls, label=f"{st.replace('_', ' ')} ({counts.get(st, 0)})")
    ax.plot(*traj[:, :2].T, color=INK, lw=0.8)
    ax.legend(loc="upper right")
    ax.set_xlim(city_xlim)
    ax.set_ylim(city_ylim)
    ax.set_title("10. Each map element: paint found, faint, missing, or not seen")
    bev(ax)
    save(fig, args.out, "10_element_status.png")

    # 11. Synthetic test: shift map elements by a known amount, check the measured change.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    jitter = np.random.default_rng(1)
    for kind, color, label in ((False, ORANGE, "lane lines"), (True, AQUA, "crosswalks")):
        sub_s = shifts[(shifts.type == "CROSSWALK") == kind]
        x = sub_s.injected_cm + jitter.uniform(-1.5, 1.5, len(sub_s)) + (2 if kind else -2)
        ax.scatter(x, sub_s.error_cm.clip(-60, 60), s=40, color=color, label=label, zorder=3,
                   edgecolor=SURFACE, linewidth=1)
    ax.axhspan(-5, 5, color=MUTED, alpha=0.35, lw=0)
    ax.annotate("±5 cm", (1.005, 0), xycoords=("axes fraction", "data"), va="center", color=INK_2)
    lost = shifts[shifts.error_cm.isna()].groupby("injected_cm").size()
    for shift, n in lost.items():
        ax.annotate(f"{n} lane line{'s' if n > 1 else ''}\nnot found", (shift, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="top", color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(sorted(shifts.injected_cm.unique()))
    ax.set_xlabel("injected map shift (cm)")
    ax.set_ylabel("recovery error (cm, clipped at ±60)")
    ax.legend(loc="lower left")
    ax.set_title("11. Synthetic test: move the map on purpose, measure how much it moved")
    save(fig, args.out, "11_synthetic_shift.png")

    # 12. Check with the cameras: HD map elements projected into camera images, colored by status.
    cam_dir = args.log_dir / "sensors" / "cameras"
    views = [(cam, sorted((cam_dir / cam).glob("*.jpg"))[0]) for cam in ("ring_front_center", "ring_rear_left")
             if (cam_dir / cam).is_dir() and any((cam_dir / cam).glob("*.jpg"))]
    if views:
        polys3d = {f"xwalk_{pc.id}": pc.polygon for pc in crossings}
        for ls in avm.get_scenario_lane_segments():
            polys3d[f"lane_{ls.id}_L"] = ls.left_lane_boundary.xyz
            polys3d[f"lane_{ls.id}_R"] = ls.right_lane_boundary.xyz
        widths = [plt.imread(img).shape[1] / plt.imread(img).shape[0] for _, img in views]
        fig, axes = plt.subplots(1, len(views), figsize=(8 * sum(widths), 8), gridspec_kw={"width_ratios": widths})
        for ax, (cam_name, img_path) in zip(np.atleast_1d(axes), views):
            cam = PinholeCamera.from_feather(args.log_dir, cam_name)
            ts = int(img_path.stem)
            ego_SE3_city = city_SE3_ego[pose_ts[np.abs(pose_ts - ts).argmin()]].inverse()
            ax.imshow(plt.imread(img_path))
            for eid, st in table[["element", "status"]].values:
                color, lw, ls_ = status_style[st]
                dense = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.25)))
                                        for a, b in zip(polys3d[eid][:-1], polys3d[eid][1:])])
                uv, pts_cam, valid = cam.project_ego_to_img(ego_SE3_city.transform_point_cloud(dense))
                valid &= pts_cam[:, 2] < 40  # only the next 40 m, farther lines get too thin to read
                uv[~valid] = np.nan  # breaks the line where it leaves the image
                ax.plot(*uv.T, color=color, lw=2 * lw, ls=ls_)
            ax.set_xlim(0, cam.width_px)
            ax.set_ylim(cam.height_px, 0)
            ax.axis("off")
            ax.set_title(f"{cam_name.replace('_', ' ')}, t = {max(0.0, (ts - t0) * 1e-9):.1f} s", fontsize=11)
        for st, (color, lw, ls_) in status_style.items():
            fig.axes[0].plot([], [], color=color, lw=2 * lw, ls=ls_, label=st.replace("_", " "))
        fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
        fig.suptitle("12. HD map elements projected into the cameras, colored by LiDAR status",
                     x=0.01, ha="left", fontweight="bold", fontsize=13)
        save(fig, args.out, "12_map_in_camera.png")

    measured = table[table.status == "measured"]
    good = shifts.error_cm.abs() <= 5
    stats |= {"elements_total": len(table), "status_counts": counts.to_dict(),
              "labeled_elements": int(table.labeled_change.sum()) if label_pts is not None else None,
              "labeled_paint_missing": int((table.labeled_change & (table.status == "paint_missing")).sum())
              if label_pts is not None else None,
              "measured_abs_offset_median_cm": round(float(measured.offset_cm.abs().median()), 1),
              "synthetic_within_5cm_lanes": round(float(good[shifts.type != "CROSSWALK"].mean()), 2),
              "synthetic_within_5cm_crosswalks": round(float(good[shifts.type == "CROSSWALK"].mean()), 2)}

    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
