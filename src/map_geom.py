"""HD map geometry helpers: painted vs unpainted boundaries, and which points lie on map paint."""

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from av2.map.map_api import ArgoverseStaticMap


def lane_boundaries(avm: ArgoverseStaticMap) -> list[tuple[np.ndarray, str]]:
    """All lane boundaries as (xyz polyline, mark type), e.g. "SOLID_WHITE" or "NONE".

    The mark type is stored on the lane segment, not on the boundary polyline itself.
    """
    return [(bnd.xyz, mark.value)
            for ls in avm.get_scenario_lane_segments()
            for bnd, mark in ((ls.left_lane_boundary, ls.left_mark_type),
                              (ls.right_lane_boundary, ls.right_mark_type))]


def densify(polyline: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Resample a 2D polyline so consecutive points are at most `step` m apart."""
    pieces = [np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / step))))
              for a, b in zip(polyline[:-1, :2], polyline[1:, :2])]
    return np.concatenate(pieces)


def on_map_paint(xy: np.ndarray, avm: ArgoverseStaticMap, tol: float = 0.3) -> np.ndarray:
    """True for points within `tol` m of a painted lane boundary or inside a crosswalk."""
    painted = [densify(xyz) for xyz, mark in lane_boundaries(avm) if mark != "NONE"]
    dist, _ = cKDTree(np.concatenate(painted)).query(xy, distance_upper_bound=tol)
    on = np.isfinite(dist)
    for pc in avm.get_scenario_ped_crossings():
        on |= MplPath(pc.polygon[:, :2]).contains_points(xy)
    return on
