"""Find lane paint in a LiDAR point cloud: ground points first, then bright ones among them."""

import numpy as np


def ground_mask(xyz: np.ndarray, cell: float = 1.0, height_tol: float = 0.2, pct: float = 10.0) -> np.ndarray:
    """Mark points lying on the ground, using the LiDAR itself (not the map's ground surface).

    Split the x-y plane into `cell`-sized squares. In each square, the ground height is a low
    percentile of z (robust to a few noisy below-ground returns). A point is ground if it is
    within `height_tol` of that height. Cars, poles and trees sit higher and drop out.
    """
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, cell_id = np.unique(ij, axis=0, return_inverse=True)

    # Sort by cell, then z within cell, so each cell's points form a sorted block.
    order = np.lexsort((xyz[:, 2], cell_id))
    counts = np.bincount(cell_id)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ground_z = xyz[order[starts + ((counts - 1) * pct / 100).astype(np.int64)], 2]

    return xyz[:, 2] < ground_z[cell_id] + height_tol


def paint_mask(intensity: np.ndarray, rng: np.ndarray, is_ground: np.ndarray,
               min_intensity: float = 80.0, max_intensity: float = 120.0, max_range: float = 25.0) -> np.ndarray:
    """Lane paint = ground points in a high-intensity band, seen close to the car.

    Paint has glass beads that reflect the laser back, so it returns much brighter than asphalt
    (median ~10-17). Concrete and curbs reach ~40-80, so the lower bar sits above them. Returns
    >= 120 are almost never paint (signs/reflectors that slipped into ground). Band picked on log
    01bb304d: 67% of these points fall on painted map elements vs 36% for intensity >= 50.
    Intensity fades with distance, so far points are dropped rather than given a lower bar.
    """
    return is_ground & (intensity >= min_intensity) & (intensity < max_intensity) & (rng < max_range)
