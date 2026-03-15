"""Unit tests for pure height/slope costmap processing logic."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest

from rover_perception.costmap_processing import (
    GridSpec,
    POINT_FIELD_FLOAT32,
    bin_points_to_grid,
    build_binary_costmap,
    compute_height_grid,
    compute_slope_grid,
    estimate_floor_height,
    pointcloud2_to_xyz_array,
    preprocess_points,
    requires_transform,
    run_height_costmap_pipeline,
)


@dataclass(frozen=True)
class FakeField:
    """PointField-like object for deterministic parser tests."""

    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class FakeCloud:
    """PointCloud2-like object with only required parser fields."""

    width: int
    height: int
    point_step: int
    row_step: int
    is_bigendian: bool
    fields: Sequence[FakeField]
    data: bytes


def _build_cloud(
    points: Sequence[tuple[float, float, float]],
    *,
    width: int,
    height: int,
    row_padding_bytes: int = 0,
    is_bigendian: bool = False,
) -> FakeCloud:
    """Build a simple xyz float32 cloud with optional row padding."""
    assert len(points) == width * height
    endian = ">" if is_bigendian else "<"
    point_step = 12
    row_step = width * point_step + row_padding_bytes

    buf = bytearray()
    for row in range(height):
        row_start = row * width
        row_points = points[row_start : row_start + width]
        for point in row_points:
            buf.extend(struct.pack(f"{endian}fff", point[0], point[1], point[2]))
        buf.extend(b"\x00" * row_padding_bytes)

    return FakeCloud(
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=is_bigendian,
        fields=[
            FakeField(name="x", offset=0, datatype=POINT_FIELD_FLOAT32),
            FakeField(name="y", offset=4, datatype=POINT_FIELD_FLOAT32),
            FakeField(name="z", offset=8, datatype=POINT_FIELD_FLOAT32),
        ],
        data=bytes(buf),
    )


def test_pointcloud2_parsing_little_endian_skip_nans() -> None:
    cloud = _build_cloud(
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (float("nan"), 1.0, 2.0),
        ],
        width=3,
        height=1,
    )

    xyz = pointcloud2_to_xyz_array(cloud, skip_nans=True)
    assert xyz.shape == (2, 3)
    np.testing.assert_allclose(
        xyz,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ]
        ),
    )


def test_pointcloud2_parsing_with_row_padding() -> None:
    cloud = _build_cloud(
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
        width=2,
        height=2,
        row_padding_bytes=8,
    )
    xyz = pointcloud2_to_xyz_array(cloud, skip_nans=False)
    np.testing.assert_allclose(
        xyz,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_pointcloud2_parsing_big_endian() -> None:
    cloud = _build_cloud(
        points=[(1.5, -2.0, 3.25)],
        width=1,
        height=1,
        is_bigendian=True,
    )
    xyz = pointcloud2_to_xyz_array(cloud, skip_nans=True)
    np.testing.assert_allclose(xyz, np.array([[1.5, -2.0, 3.25]]))


def test_requires_transform_behavior() -> None:
    assert requires_transform("odom", "base_link")
    assert not requires_transform("odom", "odom")
    with pytest.raises(ValueError):
        requires_transform("", "odom")
    with pytest.raises(ValueError):
        requires_transform("odom", "")


def test_preprocess_points_filters_nans_and_ranges() -> None:
    points = np.array(
        [
            [np.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    filtered = preprocess_points(points, range_max_m=5.0, z_min_m=-1.0, z_max_m=1.0)
    np.testing.assert_allclose(filtered, np.array([[0.0, 0.0, 0.0]]))


def test_grid_binning_uses_floor_not_truncation_for_negative_coords() -> None:
    grid = GridSpec(
        resolution_m=1.0,
        width=3,
        height=3,
        origin_x_m=0.0,
        origin_y_m=0.0,
    )
    points = np.array(
        [
            [-0.1, 0.5, 3.0],  # must be out-of-grid
            [0.1, 0.5, 2.0],   # in (0,0)
        ]
    )
    result = bin_points_to_grid(points, grid)
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[0, 0] = 1
    np.testing.assert_array_equal(result.counts, expected)
    assert result.max_height_m[0, 0] == pytest.approx(2.0)


def test_height_accumulation_per_cell() -> None:
    grid = GridSpec(
        resolution_m=1.0,
        width=2,
        height=2,
        origin_x_m=0.0,
        origin_y_m=0.0,
    )
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.3, 0.3],
            [1.1, 0.1, -0.2],
        ]
    )
    result = bin_points_to_grid(points, grid)
    np.testing.assert_array_equal(result.counts, np.array([[2, 1], [0, 0]], dtype=np.int32))
    np.testing.assert_allclose(result.min_height_m[0, 0], 0.1)
    np.testing.assert_allclose(result.max_height_m[0, 0], 0.3)
    np.testing.assert_allclose(result.min_height_m[0, 1], -0.2)
    np.testing.assert_allclose(result.max_height_m[0, 1], -0.2)
    assert np.isnan(result.min_height_m[1, 0])
    assert np.isnan(result.max_height_m[1, 0])


def test_estimate_floor_and_relative_height() -> None:
    counts = np.full((2, 2), 5, dtype=np.int32)
    min_height = np.array([[1.0, 1.0], [1.0, 1.2]])
    max_height = np.array([[1.0, 1.1], [1.3, 1.25]])
    floor = estimate_floor_height(
        min_height_m=min_height,
        counts=counts,
        min_points_per_cell=1,
        floor_percentile=25.0,
    )
    assert floor == pytest.approx(1.0)
    rel = compute_height_grid(max_height, floor)
    np.testing.assert_allclose(rel, np.array([[0.0, 0.1], [0.3, 0.25]]))


def test_compute_slope_grid_flat_surface() -> None:
    heights = np.zeros((3, 3), dtype=np.float64)
    known = np.ones((3, 3), dtype=bool)
    slope = compute_slope_grid(
        surface_height_m=heights,
        resolution_m=1.0,
        known_mask=known,
        min_valid_neighbors=2,
    )
    np.testing.assert_allclose(slope, np.zeros((3, 3)))


def test_compute_slope_grid_linear_x_slope() -> None:
    heights = np.tile(np.array([0.0, 0.1, 0.2, 0.3]), (3, 1))
    known = np.ones_like(heights, dtype=bool)
    slope = compute_slope_grid(
        surface_height_m=heights,
        resolution_m=1.0,
        known_mask=known,
        min_valid_neighbors=2,
    )
    expected = np.degrees(np.arctan(0.1))
    np.testing.assert_allclose(slope, np.full_like(heights, expected), atol=1e-10)


def test_binary_costmap_logic_height_and_slope() -> None:
    relative = np.array(
        [
            [0.0, 0.25, np.nan],
            [0.05, 0.05, 0.05],
        ]
    )
    slope = np.array(
        [
            [1.0, 5.0, 30.0],
            [20.0, np.nan, 3.0],
        ]
    )
    known = np.array(
        [
            [True, True, False],
            [True, True, False],
        ]
    )
    costmap = build_binary_costmap(
        relative_height_m=relative,
        slope_deg=slope,
        known_mask=known,
        obstacle_height_threshold_m=0.2,
        slope_threshold_deg=10.0,
        unknown_is_free=False,
    )
    np.testing.assert_array_equal(
        costmap,
        np.array(
            [
                [0, 100, -1],
                [100, 0, -1],
            ],
            dtype=np.int8,
        ),
    )


def test_pipeline_empty_cloud() -> None:
    result = run_height_costmap_pipeline(
        points_xyz=np.empty((0, 3)),
        grid=GridSpec(0.5, 4, 4, 0.0, 0.0),
        min_points_per_cell=2,
        floor_percentile=10.0,
        obstacle_height_threshold_m=0.1,
        slope_threshold_deg=10.0,
        unknown_is_free=False,
    )
    assert result.points_used == 0
    assert result.floor_height_m is None
    np.testing.assert_array_equal(result.counts, np.zeros((4, 4), dtype=np.int32))
    np.testing.assert_array_equal(result.binary_costmap, np.full((4, 4), -1, dtype=np.int8))
    np.testing.assert_array_equal(
        result.debug_height_costmap,
        np.full((4, 4), -1, dtype=np.int8),
    )


def test_pipeline_single_obstacle_on_floor() -> None:
    points = []
    for gy in range(3):
        for gx in range(3):
            cx = gx + 0.25
            cy = gy + 0.25
            points.append((cx, cy, 0.0))
            points.append((cx + 0.05, cy + 0.05, 0.0))
    points.extend(
        [
            (1.2, 1.2, 0.3),
            (1.3, 1.3, 0.3),
        ]
    )
    result = run_height_costmap_pipeline(
        points_xyz=np.asarray(points),
        grid=GridSpec(1.0, 3, 3, 0.0, 0.0),
        min_points_per_cell=2,
        floor_percentile=10.0,
        obstacle_height_threshold_m=0.1,
        slope_threshold_deg=30.0,
        unknown_is_free=False,
    )
    assert result.floor_height_m == pytest.approx(0.0)
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[1, 1] = 100
    np.testing.assert_array_equal(result.binary_costmap, expected)


def test_pipeline_sloped_surface_flags_by_slope() -> None:
    points = []
    for gy in range(3):
        for gx in range(4):
            x = gx + 0.1
            y = gy + 0.1
            z = 0.2 * gx
            points.append((x, y, z))
            points.append((x + 0.2, y + 0.1, z))
            points.append((x + 0.3, y + 0.2, z))

    result = run_height_costmap_pipeline(
        points_xyz=np.asarray(points),
        grid=GridSpec(1.0, 4, 3, 0.0, 0.0),
        min_points_per_cell=3,
        floor_percentile=0.0,
        obstacle_height_threshold_m=10.0,
        slope_threshold_deg=10.0,
        unknown_is_free=False,
    )
    assert int(np.count_nonzero(result.binary_costmap == 100)) == 12


def test_pipeline_mixed_known_unknown_and_sparse_cells() -> None:
    points = np.array(
        [
            [0.2, 0.2, 0.0],  # sparse -> unknown (min_points=2)
            [1.2, 0.2, 0.0],
            [1.3, 0.3, 0.0],  # known safe
            [2.2, 0.2, 0.3],
            [2.3, 0.3, 0.3],  # known obstacle by height
        ]
    )
    result = run_height_costmap_pipeline(
        points_xyz=points,
        grid=GridSpec(1.0, 3, 2, 0.0, 0.0),
        min_points_per_cell=2,
        floor_percentile=0.0,
        obstacle_height_threshold_m=0.1,
        slope_threshold_deg=60.0,
        unknown_is_free=False,
    )
    np.testing.assert_array_equal(
        result.binary_costmap,
        np.array(
            [
                [-1, 0, 100],
                [-1, -1, -1],
            ],
            dtype=np.int8,
        ),
    )
