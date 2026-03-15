"""Pure processing utilities for height and slope based costmap generation.

This module is ROS-message agnostic except for expecting PointCloud2-like
attributes when parsing cloud bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# sensor_msgs/msg/PointField datatype ids.
POINT_FIELD_INT8 = 1
POINT_FIELD_UINT8 = 2
POINT_FIELD_INT16 = 3
POINT_FIELD_UINT16 = 4
POINT_FIELD_INT32 = 5
POINT_FIELD_UINT32 = 6
POINT_FIELD_FLOAT32 = 7
POINT_FIELD_FLOAT64 = 8


@dataclass(frozen=True)
class GridSpec:
    """Grid geometry for binning point cloud data."""

    resolution_m: float
    width: int
    height: int
    origin_x_m: float
    origin_y_m: float

    def __post_init__(self) -> None:
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be > 0")


@dataclass(frozen=True)
class GridBinningResult:
    """Per-cell statistics computed from 3D points."""

    counts: np.ndarray
    min_height_m: np.ndarray
    max_height_m: np.ndarray


@dataclass(frozen=True)
class HeightCostmapResult:
    """Outputs from the deterministic pure costmap pipeline."""

    points_used: int
    counts: np.ndarray
    known_mask: np.ndarray
    min_height_m: np.ndarray
    max_height_m: np.ndarray
    floor_height_m: Optional[float]
    relative_height_m: np.ndarray
    slope_deg: np.ndarray
    binary_costmap: np.ndarray
    debug_height_costmap: np.ndarray


def requires_transform(source_frame: str, target_frame: str) -> bool:
    """Return True when source and target frames differ."""
    if not source_frame:
        raise ValueError("source_frame must be non-empty")
    if not target_frame:
        raise ValueError("target_frame must be non-empty")
    return source_frame != target_frame


def _pointfield_to_numpy_dtype(datatype: int, is_bigendian: bool) -> np.dtype:
    """Map PointField datatype id to numpy dtype with endian handling."""
    endian = ">" if is_bigendian else "<"
    if datatype == POINT_FIELD_INT8:
        return np.dtype("i1")
    if datatype == POINT_FIELD_UINT8:
        return np.dtype("u1")
    if datatype == POINT_FIELD_INT16:
        return np.dtype(f"{endian}i2")
    if datatype == POINT_FIELD_UINT16:
        return np.dtype(f"{endian}u2")
    if datatype == POINT_FIELD_INT32:
        return np.dtype(f"{endian}i4")
    if datatype == POINT_FIELD_UINT32:
        return np.dtype(f"{endian}u4")
    if datatype == POINT_FIELD_FLOAT32:
        return np.dtype(f"{endian}f4")
    if datatype == POINT_FIELD_FLOAT64:
        return np.dtype(f"{endian}f8")
    raise ValueError(f"Unsupported PointField datatype: {datatype}")


def pointcloud2_to_xyz_array(cloud: Any, skip_nans: bool = True) -> np.ndarray:
    """Extract xyz points from a PointCloud2-like object as (N, 3) float64.

    Expected attributes on ``cloud``:
    - ``width``, ``height``, ``point_step``, ``row_step``, ``is_bigendian``
    - ``data`` bytes-like
    - ``fields`` iterable with field attributes ``name``, ``offset``,
      ``datatype``, ``count``
    """
    width = int(cloud.width)
    height = int(cloud.height)
    point_step = int(cloud.point_step)
    row_step = int(cloud.row_step)
    if width < 0 or height < 0:
        raise ValueError("width and height must be >= 0")
    if point_step <= 0:
        raise ValueError("point_step must be > 0")
    if row_step < width * point_step:
        raise ValueError("row_step is smaller than width * point_step")

    field_map = {}
    for field in cloud.fields:
        field_map[str(field.name)] = field
    for axis in ("x", "y", "z"):
        if axis not in field_map:
            raise ValueError(f"PointCloud2 field '{axis}' is required")

    names = ["x", "y", "z"]
    offsets = []
    formats = []
    for name in names:
        field = field_map[name]
        count = int(getattr(field, "count", 1))
        if count != 1:
            raise ValueError(f"PointCloud2 field '{name}' must have count == 1")
        offset = int(field.offset)
        if offset < 0:
            raise ValueError(f"PointCloud2 field '{name}' has negative offset")
        dtype = _pointfield_to_numpy_dtype(
            int(field.datatype),
            bool(cloud.is_bigendian),
        )
        if offset + dtype.itemsize > point_step:
            raise ValueError(f"PointCloud2 field '{name}' exceeds point_step")
        offsets.append(offset)
        formats.append(dtype)

    expected_data_size = row_step * height
    if len(cloud.data) < expected_data_size:
        raise ValueError(
            "PointCloud2 data buffer is smaller than row_step * height: "
            f"{len(cloud.data)} < {expected_data_size}"
        )

    xyz_dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        }
    )

    total_points = width * height
    if total_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    packed_row_size = width * point_step
    data_view = memoryview(cloud.data)
    structured = np.empty(total_points, dtype=xyz_dtype)

    for row_idx in range(height):
        row_start = row_idx * row_step
        row_end = row_start + packed_row_size
        row_array = np.frombuffer(
            data_view[row_start:row_end],
            dtype=xyz_dtype,
            count=width,
        )
        start = row_idx * width
        structured[start : start + width] = row_array

    points = np.empty((total_points, 3), dtype=np.float64)
    points[:, 0] = structured["x"].astype(np.float64, copy=False)
    points[:, 1] = structured["y"].astype(np.float64, copy=False)
    points[:, 2] = structured["z"].astype(np.float64, copy=False)

    if skip_nans:
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]

    return points


def preprocess_points(
    points_xyz: np.ndarray,
    range_max_m: Optional[float] = None,
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
) -> np.ndarray:
    """Drop invalid points and apply deterministic range/z cropping."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    if range_max_m is not None and range_max_m <= 0.0:
        raise ValueError("range_max_m must be > 0 when provided")
    if z_min_m is not None and z_max_m is not None and z_min_m > z_max_m:
        raise ValueError("z_min_m must be <= z_max_m")

    valid = np.isfinite(points).all(axis=1)
    if range_max_m is not None:
        r2 = points[:, 0] ** 2 + points[:, 1] ** 2
        valid &= r2 <= (range_max_m * range_max_m)
    if z_min_m is not None:
        valid &= points[:, 2] >= z_min_m
    if z_max_m is not None:
        valid &= points[:, 2] <= z_max_m
    return points[valid]


def bin_points_to_grid(points_xyz: np.ndarray, grid: GridSpec) -> GridBinningResult:
    """Bin points into a fixed XY grid and compute count/min/max height per cell."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    total_cells = grid.width * grid.height
    counts_flat = np.zeros(total_cells, dtype=np.int32)
    min_flat = np.full(total_cells, np.inf, dtype=np.float64)
    max_flat = np.full(total_cells, -np.inf, dtype=np.float64)

    if points.shape[0] > 0:
        gx = np.floor((points[:, 0] - grid.origin_x_m) / grid.resolution_m).astype(np.int64)
        gy = np.floor((points[:, 1] - grid.origin_y_m) / grid.resolution_m).astype(np.int64)
        in_bounds = (
            (gx >= 0)
            & (gx < grid.width)
            & (gy >= 0)
            & (gy < grid.height)
        )
        if np.any(in_bounds):
            gx_in = gx[in_bounds]
            gy_in = gy[in_bounds]
            z_in = points[in_bounds, 2]
            flat_indices = gy_in * grid.width + gx_in
            np.add.at(counts_flat, flat_indices, 1)
            np.minimum.at(min_flat, flat_indices, z_in)
            np.maximum.at(max_flat, flat_indices, z_in)

    counts = counts_flat.reshape((grid.height, grid.width))
    min_height = min_flat.reshape((grid.height, grid.width))
    max_height = max_flat.reshape((grid.height, grid.width))

    unknown_mask = counts == 0
    min_height[unknown_mask] = np.nan
    max_height[unknown_mask] = np.nan

    return GridBinningResult(
        counts=counts,
        min_height_m=min_height,
        max_height_m=max_height,
    )


def estimate_floor_height(
    min_height_m: np.ndarray,
    counts: np.ndarray,
    min_points_per_cell: int,
    floor_percentile: float,
) -> Optional[float]:
    """Estimate floor height from low-percentile of known-cell minimum heights.

    Assumption: a substantial fraction of known cells represent terrain/floor.
    """
    if min_points_per_cell <= 0:
        raise ValueError("min_points_per_cell must be > 0")
    if floor_percentile < 0.0 or floor_percentile > 100.0:
        raise ValueError("floor_percentile must be in [0, 100]")
    if min_height_m.shape != counts.shape:
        raise ValueError("min_height_m and counts must have the same shape")

    known_mask = counts >= min_points_per_cell
    samples = min_height_m[known_mask]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None
    return float(np.percentile(samples, floor_percentile))


def compute_height_grid(
    max_height_m: np.ndarray,
    floor_height_m: Optional[float],
) -> np.ndarray:
    """Compute per-cell relative height (cell max height - floor reference)."""
    relative = np.asarray(max_height_m, dtype=np.float64).copy()
    if floor_height_m is None or not math.isfinite(floor_height_m):
        relative[:] = np.nan
        return relative

    finite_mask = np.isfinite(relative)
    relative[finite_mask] = relative[finite_mask] - floor_height_m
    relative[~finite_mask] = np.nan
    return relative


def compute_slope_grid(
    surface_height_m: np.ndarray,
    resolution_m: float,
    known_mask: np.ndarray,
    min_valid_neighbors: int = 2,
) -> np.ndarray:
    """Compute per-cell slope angle in degrees from finite differences.

    Formula:
      gradient = sqrt((dz/dx)^2 + (dz/dy)^2)
      slope_deg = atan(gradient) * 180 / pi

    Derivatives use central differences when both neighbors exist, otherwise
    one-sided differences. Cells with insufficient neighbor support are NaN.
    """
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be > 0")
    if min_valid_neighbors < 1 or min_valid_neighbors > 2:
        raise ValueError("min_valid_neighbors must be 1 or 2")

    heights = np.asarray(surface_height_m, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    if heights.shape != known.shape:
        raise ValueError("surface_height_m and known_mask must have the same shape")

    grid_h, grid_w = heights.shape
    slope_deg = np.full((grid_h, grid_w), np.nan, dtype=np.float64)

    for gy in range(grid_h):
        for gx in range(grid_w):
            if not known[gy, gx] or not np.isfinite(heights[gy, gx]):
                continue

            center = heights[gy, gx]
            axes_available = 0
            dzdx = 0.0
            dzdy = 0.0

            has_left = gx > 0 and known[gy, gx - 1] and np.isfinite(heights[gy, gx - 1])
            has_right = (
                gx < (grid_w - 1)
                and known[gy, gx + 1]
                and np.isfinite(heights[gy, gx + 1])
            )
            if has_left and has_right:
                dzdx = (heights[gy, gx + 1] - heights[gy, gx - 1]) / (2.0 * resolution_m)
                axes_available += 1
            elif has_right:
                dzdx = (heights[gy, gx + 1] - center) / resolution_m
                axes_available += 1
            elif has_left:
                dzdx = (center - heights[gy, gx - 1]) / resolution_m
                axes_available += 1

            has_down = gy > 0 and known[gy - 1, gx] and np.isfinite(heights[gy - 1, gx])
            has_up = (
                gy < (grid_h - 1)
                and known[gy + 1, gx]
                and np.isfinite(heights[gy + 1, gx])
            )
            if has_down and has_up:
                dzdy = (heights[gy + 1, gx] - heights[gy - 1, gx]) / (2.0 * resolution_m)
                axes_available += 1
            elif has_up:
                dzdy = (heights[gy + 1, gx] - center) / resolution_m
                axes_available += 1
            elif has_down:
                dzdy = (center - heights[gy - 1, gx]) / resolution_m
                axes_available += 1

            if axes_available < min_valid_neighbors:
                continue

            gradient = math.sqrt((dzdx * dzdx) + (dzdy * dzdy))
            slope_deg[gy, gx] = math.degrees(math.atan(gradient))

    return slope_deg


def build_binary_costmap(
    relative_height_m: np.ndarray,
    slope_deg: np.ndarray,
    known_mask: np.ndarray,
    obstacle_height_threshold_m: float,
    slope_threshold_deg: float,
    unknown_is_free: bool = False,
) -> np.ndarray:
    """Build binary OccupancyGrid values: -1/0/100 for unknown/free/occupied."""
    if relative_height_m.shape != slope_deg.shape or relative_height_m.shape != known_mask.shape:
        raise ValueError("relative_height_m, slope_deg, and known_mask shape mismatch")

    if unknown_is_free:
        costmap = np.zeros(relative_height_m.shape, dtype=np.int8)
    else:
        costmap = np.full(relative_height_m.shape, -1, dtype=np.int8)

    known = np.asarray(known_mask, dtype=bool)
    criteria_available = np.isfinite(relative_height_m) | np.isfinite(slope_deg)
    decision_mask = known & criteria_available
    costmap[decision_mask] = 0

    height_unsafe = np.isfinite(relative_height_m) & (relative_height_m > obstacle_height_threshold_m)
    slope_unsafe = np.isfinite(slope_deg) & (slope_deg > slope_threshold_deg)
    unsafe = decision_mask & (height_unsafe | slope_unsafe)
    costmap[unsafe] = 100

    return costmap


def build_debug_height_costmap(
    relative_height_m: np.ndarray,
    known_mask: np.ndarray,
    max_display_height_m: float,
    unknown_is_free: bool = False,
) -> np.ndarray:
    """Build OccupancyGrid debug layer by scaling relative height to [0, 100]."""
    if max_display_height_m <= 0.0:
        raise ValueError("max_display_height_m must be > 0")
    if relative_height_m.shape != known_mask.shape:
        raise ValueError("relative_height_m and known_mask shape mismatch")

    if unknown_is_free:
        debug = np.zeros(relative_height_m.shape, dtype=np.int8)
    else:
        debug = np.full(relative_height_m.shape, -1, dtype=np.int8)

    valid = np.asarray(known_mask, dtype=bool) & np.isfinite(relative_height_m)
    if np.any(valid):
        clamped = np.clip(relative_height_m[valid], 0.0, max_display_height_m)
        scaled = np.rint((clamped / max_display_height_m) * 100.0).astype(np.int8)
        debug[valid] = scaled
    return debug


def run_height_costmap_pipeline(
    points_xyz: np.ndarray,
    grid: GridSpec,
    min_points_per_cell: int,
    floor_percentile: float,
    obstacle_height_threshold_m: float,
    slope_threshold_deg: float,
    slope_min_valid_neighbors: int = 2,
    range_max_m: Optional[float] = None,
    z_min_m: Optional[float] = None,
    z_max_m: Optional[float] = None,
    unknown_is_free: bool = False,
    debug_height_max_m: float = 0.5,
) -> HeightCostmapResult:
    """Execute full deterministic height+slope costmap processing."""
    filtered_points = preprocess_points(
        points_xyz=points_xyz,
        range_max_m=range_max_m,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
    )
    binning = bin_points_to_grid(filtered_points, grid)
    known_mask = binning.counts >= min_points_per_cell

    floor_height_m = estimate_floor_height(
        min_height_m=binning.min_height_m,
        counts=binning.counts,
        min_points_per_cell=min_points_per_cell,
        floor_percentile=floor_percentile,
    )
    relative_height_m = compute_height_grid(
        max_height_m=binning.max_height_m,
        floor_height_m=floor_height_m,
    )
    slope_deg = compute_slope_grid(
        surface_height_m=binning.min_height_m,
        resolution_m=grid.resolution_m,
        known_mask=known_mask,
        min_valid_neighbors=slope_min_valid_neighbors,
    )
    binary_costmap = build_binary_costmap(
        relative_height_m=relative_height_m,
        slope_deg=slope_deg,
        known_mask=known_mask,
        obstacle_height_threshold_m=obstacle_height_threshold_m,
        slope_threshold_deg=slope_threshold_deg,
        unknown_is_free=unknown_is_free,
    )
    debug_height_costmap = build_debug_height_costmap(
        relative_height_m=relative_height_m,
        known_mask=known_mask,
        max_display_height_m=debug_height_max_m,
        unknown_is_free=unknown_is_free,
    )

    return HeightCostmapResult(
        points_used=int(filtered_points.shape[0]),
        counts=binning.counts,
        known_mask=known_mask,
        min_height_m=binning.min_height_m,
        max_height_m=binning.max_height_m,
        floor_height_m=floor_height_m,
        relative_height_m=relative_height_m,
        slope_deg=slope_deg,
        binary_costmap=binary_costmap,
        debug_height_costmap=debug_height_costmap,
    )
