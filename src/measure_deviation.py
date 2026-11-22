"""Phase 3: per-element deviation table (cm) for one TbV log, plus a synthetic shift test.

Usage: .venv/bin/python src/measure_deviation.py data/<LOG_ID> [--labels data/tbv_labels/tbv_val_split_annotations.json]

Writes docs/results/deviation_<LOG_ID>.csv and docs/results/synthetic_shift_<LOG_ID>.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deviation import inside_crosswalks, measure_boundary, measure_crosswalk, painted_boundaries
from paint import ground_mask, paint_mask
from tbv_io import load_city_points, load_map

SHIFTS_M = (0.10, 0.25, 0.50, -0.25)  # synthetic map errors to inject (+ = left / along road)
LABEL_RADIUS_M = 3.0  # element counts as "in a labeled change" if this close to the label polyline


def labeled_change_points(labels_path: Path | None, log_id: str) -> np.ndarray | None:
    """City-frame points describing this log's labeled map changes (None if no labels file)."""
    if labels_path is None or not labels_path.exists():
        return None
    pts = []
    for entry in json.loads(labels_path.read_text()):
        if entry["log_id"] != log_id:
            continue
        for ev in entry["events"]:
            pts += [p["City Coords"] for p in ev.get("change_endpoints", [])]
            pts += ev.get("polygon", [])
            if "City Coords" in ev:
                pts.append(ev["City Coords"])
    return np.array(pts, dtype=float).reshape(-1, 2)


def near_label(xy: np.ndarray, label_pts: np.ndarray | None) -> bool | None:
    """True if a polyline comes within LABEL_RADIUS_M of the labeled change (densified label line)."""
    if label_pts is None:
        return None
    if len(label_pts) == 0:
        return False
    dense = np.concatenate([np.linspace(a, b, 50) for a, b in zip(label_pts[:-1], label_pts[1:])] or [label_pts])
    return bool((np.linalg.norm(xy[:, None] - dense[None], axis=-1) < LABEL_RADIUS_M).any())


def measure_log(xyz, intensity, rng, avm, label_pts=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure every painted map element, then re-measure with known shifts injected.

    Returns:
        table: one row per element (status, offset, coverage, ...), all statuses included.
        shifts: one row per (measured element, injected shift) with the recovery error.
    """
    is_ground = ground_mask(xyz)
    is_paint = paint_mask(intensity, rng, is_ground)
    paint_xy = xyz[is_paint, :2]
    near_ground = is_ground & (rng < 25)  # same range limit as paint
    ground_xy, ground_i = xyz[near_ground, :2], intensity[near_ground]
    lane_paint = paint_xy[~inside_crosswalks(paint_xy, avm)]  # crosswalk stripes would pollute lane lines

    rows, shift_rows = [], []
    elements = [(eid, mark, xy, lambda s, xy=xy: measure_boundary(lane_paint, ground_xy, ground_i, xy, s))
                for eid, mark, xy in painted_boundaries(avm)]
    elements += [(f"xwalk_{pc.id}", "CROSSWALK", pc.polygon[:, :2],
                  lambda s, pc=pc: measure_crosswalk(paint_xy, ground_xy, ground_i, pc, s))
                 for pc in avm.get_scenario_ped_crossings()]

    for eid, kind, xy, measure in elements:
        base = measure(0.0)
        rows.append({"element": eid, "type": kind} | base | {"labeled_change": near_label(xy, label_pts)})
        if np.isnan(base["offset_cm"]):
            continue
        for shift in SHIFTS_M:
            shifted = measure(shift)["offset_cm"]
            recovered = base["offset_cm"] - shifted  # map moved +d -> paint looks d further the other way
            shift_rows.append({"element": eid, "type": kind, "base_offset_cm": base["offset_cm"],
                               "injected_cm": round(100 * shift), "recovered_cm": round(recovered, 1),
                               "error_cm": round(recovered - 100 * shift, 1)})
    return pd.DataFrame(rows), pd.DataFrame(shift_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("data/tbv_labels/tbv_val_split_annotations.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_id = args.log_dir.name

    xyz, intensity, rng, _ = load_city_points(args.log_dir, every=2)
    avm = load_map(args.log_dir)
    table, shifts = measure_log(xyz, intensity, rng, avm, labeled_change_points(args.labels, log_id))

    order = {"paint_missing": 0, "measured": 1, "faint_paint": 2}
    observed = (table[table.status != "not_observed"]
                .assign(_o=lambda t: t.status.map(order), _a=lambda t: -t.offset_cm.abs())
                .sort_values(["_o", "_a"]).drop(columns=["_o", "_a"]))
    observed.to_csv(args.out / f"deviation_{log_id}.csv", index=False)
    shifts.to_csv(args.out / f"synthetic_shift_{log_id}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{len(table)} painted map elements:", table.status.value_counts().to_dict(), "\n")
    print(observed.drop(columns=["n_points", "length_m"]).to_string(index=False))
    print("\nSynthetic shift test: |error| <= 5 cm, share of measured elements")
    ok = shifts.assign(ok=shifts.error_cm.abs() <= 5)
    print(ok.pivot_table(index="type", columns="injected_cm", values="ok", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()
