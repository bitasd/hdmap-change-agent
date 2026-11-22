"""Phase 3: how far is the observed paint from each HD map element?"""

import numpy as np
from matplotlib.path import Path as MplPath

from av2.map.map_api import ArgoverseStaticMap

MIN_POINTS = 30  # fewer line points than this -> element counts as not observed
MIN_OBSERVED_M = 2.0  # line points must cover at least this much of the element (m)
MIN_GROUND_COVERAGE = 0.5  # no paint but ground seen along >= this share of the element -> paint missing
FAINT_INTENSITY = 25.0  # on-line 90th-pct intensity above bare asphalt (~6): dim paint (yellow/worn), not missing
ROBUST_SIGMA = 1.4826  # MAD * this = standard deviation, for normally distributed noise


def signed_offsets(xy: np.ndarray, polyline: np.ndarray, corridor: float = 1.0):
    """Signed perpendicular distance of each point to a 2D polyline (+ = left of its direction).

    Only points whose closest spot on the line is strictly inside it (not past either end)
    and within `corridor` m are kept, so paint beyond the element's ends doesn't count.

    Returns:
        offsets: (M,) signed distances (m) of the kept points.
        arc: (M,) distance along the polyline (m) of each kept point's foot, for coverage.
        keep: (N,) boolean mask into `xy`.
    """
    # Only points near the polyline's bounding box can be within the corridor.
    lo, hi = polyline.min(0) - corridor, polyline.max(0) + corridor
    cand = np.flatnonzero(np.all((xy >= lo) & (xy <= hi), axis=1))
    keep = np.zeros(len(xy), dtype=bool)
    xy = xy[cand]

    a, b = polyline[:-1], polyline[1:]
    seg = b - a
    seg_len = np.linalg.norm(seg, axis=1)
    # Projection of every point on every segment: t in [0, 1] along the segment.
    rel = xy[:, None, :] - a[None]
    t = np.clip((rel * seg[None]).sum(-1) / seg_len[None] ** 2, 0, 1)
    foot = a[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(xy[:, None, :] - foot, axis=-1)
    nearest = dist.argmin(axis=1)
    rows = np.arange(len(xy))
    t_near = t[rows, nearest]

    past_start = (nearest == 0) & (t_near <= 0)
    past_end = (nearest == len(seg) - 1) & (t_near >= 1)
    kept = (dist[rows, nearest] < corridor) & ~past_start & ~past_end
    keep[cand[kept]] = True

    # Sign from the 2D cross product: positive if the point is to the left of the segment.
    s, r = seg[nearest[kept]], rel[rows[kept], nearest[kept]]
    sign = np.sign(s[:, 0] * r[:, 1] - s[:, 1] * r[:, 0])
    offsets = sign * dist[rows[kept], nearest[kept]]
    arc = np.concatenate([[0], np.cumsum(seg_len)])[nearest[kept]] + t_near[kept] * seg_len[nearest[kept]]
    return offsets, arc, keep


def painted_boundaries(avm: ArgoverseStaticMap):
    """Unique painted lane boundaries as (element_id, mark_type, xy polyline).

    Neighboring lanes share a boundary (one lane's left = the next lane's right), so each
    physical line is listed once.
    """
    seen, out = set(), []
    for ls in avm.get_scenario_lane_segments():
        for side, bnd, mark in (("L", ls.left_lane_boundary, ls.left_mark_type),
                                ("R", ls.right_lane_boundary, ls.right_mark_type)):
            if mark.value == "NONE":
                continue
            xy = bnd.xyz[:, :2]
            key = frozenset((tuple(np.round(xy[0], 1)), tuple(np.round(xy[-1], 1))))
            if key in seen:
                continue
            seen.add(key)
            out.append((f"lane_{ls.id}_{side}", mark.value, xy))
    return out


def line_fit(offsets: np.ndarray, along: np.ndarray, bin_m: float = 0.5, inlier_m: float = 0.15) -> dict:
    """Find the painted line inside a corridor of signed offsets.

    A painted line shows up as a dense cluster of offsets; blobs (arrows, symbols) and other
    parallel lines add points elsewhere. So: find the offset whose +-`inlier_m` window holds the
    most points, keep those points, and report their median offset. `observed_m` is how much of the element's
    length those points cover; blobs are short, real lines are long.
    """
    if len(offsets) == 0:
        return {"n_points": 0, "observed_m": 0.0, "inlier_frac": np.nan, "offset_cm": np.nan, "spread_cm": np.nan}
    srt = np.sort(offsets)
    in_window = np.searchsorted(srt, srt + inlier_m) - np.searchsorted(srt, srt - inlier_m)
    peak = srt[in_window.argmax()]
    inl = np.abs(offsets - peak) < inlier_m
    observed_m = np.unique((along[inl] / bin_m).astype(int)).size * bin_m
    row = {"n_points": int(inl.sum()), "observed_m": observed_m, "inlier_frac": round(float(inl.mean()), 2)}
    if inl.sum() < MIN_POINTS or observed_m < MIN_OBSERVED_M:
        return row | {"offset_cm": np.nan, "spread_cm": np.nan}
    med = np.median(offsets[inl])
    return row | {"offset_cm": round(100 * med, 1),
                  "spread_cm": round(100 * ROBUST_SIGMA * np.median(np.abs(offsets[inl] - med)), 1)}


def status(offset_cm: float, ground_coverage: float, line_intensity_p90: float) -> str:
    """measured: paint found in the paint band.
    faint_paint: road seen, nothing in the band, but brighter than asphalt (yellow or worn paint).
    paint_missing: road clearly seen and it looks like bare asphalt -> the paint is gone.
    not_observed: the road along the element was not seen well enough to say.
    """
    if not np.isnan(offset_cm):
        return "measured"
    if ground_coverage < MIN_GROUND_COVERAGE:
        return "not_observed"
    return "faint_paint" if line_intensity_p90 >= FAINT_INTENSITY else "paint_missing"


def measure_boundary(paint_xy: np.ndarray, ground_xy: np.ndarray, ground_i: np.ndarray, xy: np.ndarray,
                     shift_m: float = 0.0, bin_m: float = 0.5) -> dict:
    """Deviation of one lane boundary (cm, + = paint is left of the map line).

    `ground_xy` (+ intensity `ground_i`) tells "no paint here" apart from "never looked here".
    `shift_m` moves the map line sideways first (+ = left), to test that a known error is recovered.
    """
    if shift_m:
        xy = shift_polyline(xy, shift_m)
    offsets, arc, _ = signed_offsets(paint_xy, xy)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    _, ground_arc, on_line = signed_offsets(ground_xy, xy, corridor=0.2)
    n_bins = max(1, int(np.ceil(length / bin_m)))
    coverage = np.unique(np.minimum((ground_arc / bin_m).astype(int), n_bins - 1)).size / n_bins
    p90 = float(np.percentile(ground_i[on_line], 90)) if on_line.any() else np.nan
    row = {"length_m": round(length, 1), "ground_coverage": round(coverage, 2),
           "line_intensity_p90": round(p90, 1)} | line_fit(offsets, arc)
    return row | {"status": status(row["offset_cm"], coverage, p90)}


def shift_polyline(xy: np.ndarray, d: float) -> np.ndarray:
    """Move a 2D polyline sideways by d m (+ = left), along each vertex's averaged normal."""
    seg = np.diff(xy, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    tangent = np.vstack([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return xy + d * np.c_[-tangent[:, 1], tangent[:, 0]]


def crosswalk_axes(pc):
    """Crosswalk frame: origin at the center, v along the walking direction, u along the road.

    The map's two edges sit at u = -width/2 and u = +width/2.
    """
    e1, e2 = pc.get_edges_2d()
    v = (e1[1] - e1[0]) + (e2[1] - e2[0])
    v /= np.linalg.norm(v)
    u = np.array([-v[1], v[0]])
    center = (e1.mean(0) + e2.mean(0)) / 2
    width = abs((e2.mean(0) - e1.mean(0)) @ u)
    half_len = max(abs((p - center) @ v) for p in (*e1, *e2))
    return center, u, v, width, half_len


def measure_crosswalk(paint_xy: np.ndarray, ground_xy: np.ndarray, ground_i: np.ndarray, pc, shift_m: float = 0.0,
                      margin: float = 1.0, bin_m: float = 0.1) -> dict:
    """Deviation of one crosswalk along the road (cm): where the painted band is vs the map's edges.

    Paint is projected on the road direction u. Zebra stripes make one solid band there that
    also spreads across the crossing (along v). Lane lines run along u through the crosswalk but
    sit at one v, and a stop line forms a separate band, so: a 10 cm u-bin is "filled" only if
    its paint spreads over many 0.5 m v-bins, and the biggest contiguous run of filled bins wins. `shift_m` moves the map crosswalk along u first (for the synthetic test).
    """
    center, u, v, width, half_len = crosswalk_axes(pc)
    center = center + shift_m * u
    result = _crosswalk_band(paint_xy, center, u, v, width, half_len, margin, bin_m)

    # Ground coverage: share of 0.5 m cells inside the (shifted) crosswalk that have ground points.
    g = ground_xy - center
    gu, gv = g @ u, g @ v
    inside = (np.abs(gu) < width / 2) & (np.abs(gv) < half_len)
    cells = np.unique(np.floor(np.c_[gu[inside], gv[inside]] / 0.5), axis=0).shape[0]
    coverage = min(1.0, cells / max(1, int(np.ceil(width / 0.5)) * int(np.ceil(2 * half_len / 0.5))))
    p90 = float(np.percentile(ground_i[inside], 90)) if inside.any() else np.nan
    return result | {"ground_coverage": round(coverage, 2), "line_intensity_p90": round(p90, 1),
                     "status": status(result["offset_cm"], coverage, p90)}


def _crosswalk_band(paint_xy, center, u, v, width, half_len, margin, bin_m) -> dict:
    rel = paint_xy - center
    pu, pv = rel @ u, rel @ v
    near = (np.abs(pv) < half_len) & (np.abs(pu) < width / 2 + margin)
    pu, pv = pu[near], pv[near]
    row = {"length_m": round(2 * half_len, 1), "width_m": round(width, 2)}
    empty = {"n_points": int(near.sum()), "observed_m": 0.0, "offset_cm": np.nan, "width_diff_cm": np.nan}
    if near.sum() < MIN_POINTS:
        return row | empty

    edges = np.arange(-width / 2 - margin, width / 2 + margin + bin_m, bin_m)
    u_bin = np.digitize(pu, edges) - 1
    v_bin = np.floor(pv / 0.5).astype(int)
    spread = np.array([np.unique(v_bin[u_bin == i]).size for i in range(len(edges) - 1)])
    filled = spread >= max(3, 0.3 * spread.max())
    # Contiguous run of filled bins containing the most points.
    runs, start = [], None
    for i, f in enumerate(np.append(filled, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return row | empty
    lo, hi = max(runs, key=lambda r: ((pu >= edges[r[0]]) & (pu < edges[r[1]])).sum())
    inl = (pu >= edges[lo]) & (pu < edges[hi])
    obs_lo, obs_hi = np.percentile(pu[inl], [1, 99])
    observed_m = np.unique((pv[inl] / 0.5).astype(int)).size * 0.5
    # Under half the map width painted = we saw a sliver, not the whole crosswalk.
    if inl.sum() < MIN_POINTS or observed_m < MIN_OBSERVED_M or obs_hi - obs_lo < 0.5 * width:
        return row | empty | {"n_points": int(inl.sum()), "observed_m": observed_m}
    return row | {"n_points": int(inl.sum()), "observed_m": observed_m,
                  "offset_cm": round(100 * (obs_lo + obs_hi) / 2, 1),
                  "width_diff_cm": round(100 * ((obs_hi - obs_lo) - width), 1)}


def inside_crosswalks(xy: np.ndarray, avm: ArgoverseStaticMap) -> np.ndarray:
    inside = np.zeros(len(xy), dtype=bool)
    for pc in avm.get_scenario_ped_crossings():
        inside |= MplPath(pc.polygon[:, :2]).contains_points(xy)
    return inside
