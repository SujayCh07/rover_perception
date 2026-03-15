"""ROS node that builds a height+slope binary costmap from PointCloud2."""

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from rover_perception.costmap_processing import (
    GridSpec,
    pointcloud2_to_xyz_array,
    requires_transform,
    run_height_costmap_pipeline,
)


class HeightCostmap(Node):
    """Build and publish slope-aware binary and debug-height OccupancyGrids."""

    def __init__(self) -> None:
        super().__init__("height_costmap")

        self.declare_parameter("input_topic", "/cloud_in_target_frame")
        self.declare_parameter("output_topic", "/height_costmap")
        self.declare_parameter("debug_height_topic", "/height_costmap_debug_height")
        self.declare_parameter("publish_debug_height", True)
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("enforce_target_frame", True)
        self.declare_parameter("tf_timeout_s", 0.20)
        self.declare_parameter("use_latest_tf_when_stamp_zero", True)

        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width", 100)
        self.declare_parameter("height", 100)
        self.declare_parameter("origin_x", -5.0)
        self.declare_parameter("origin_y", -5.0)

        self.declare_parameter("range_max_m", 6.0)
        self.declare_parameter("z_min_m", -0.30)
        self.declare_parameter("z_max_m", 0.50)
        self.declare_parameter("min_points_per_cell", 5)
        self.declare_parameter("floor_percentile", 15.0)
        self.declare_parameter("obstacle_height_threshold_m", 0.10)
        self.declare_parameter("slope_threshold_deg", 15.0)
        self.declare_parameter("slope_min_valid_neighbors", 2)
        self.declare_parameter("unknown_is_free", False)
        self.declare_parameter("debug_height_max_m", 0.40)
        self.declare_parameter("log_period_s", 2.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.debug_height_topic = str(self.get_parameter("debug_height_topic").value)
        self.publish_debug_height = bool(self.get_parameter("publish_debug_height").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.enforce_target_frame = bool(self.get_parameter("enforce_target_frame").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        self.use_latest_tf_when_stamp_zero = bool(
            self.get_parameter("use_latest_tf_when_stamp_zero").value
        )

        self.grid = GridSpec(
            resolution_m=float(self.get_parameter("resolution").value),
            width=int(self.get_parameter("width").value),
            height=int(self.get_parameter("height").value),
            origin_x_m=float(self.get_parameter("origin_x").value),
            origin_y_m=float(self.get_parameter("origin_y").value),
        )

        self.range_max_m = float(self.get_parameter("range_max_m").value)
        self.z_min_m = float(self.get_parameter("z_min_m").value)
        self.z_max_m = float(self.get_parameter("z_max_m").value)
        self.min_points_per_cell = int(self.get_parameter("min_points_per_cell").value)
        self.floor_percentile = float(self.get_parameter("floor_percentile").value)
        self.obstacle_height_threshold_m = float(
            self.get_parameter("obstacle_height_threshold_m").value
        )
        self.slope_threshold_deg = float(self.get_parameter("slope_threshold_deg").value)
        self.slope_min_valid_neighbors = int(
            self.get_parameter("slope_min_valid_neighbors").value
        )
        self.unknown_is_free = bool(self.get_parameter("unknown_is_free").value)
        self.debug_height_max_m = float(self.get_parameter("debug_height_max_m").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)

        if not self.target_frame:
            raise ValueError("Parameter 'target_frame' must be non-empty")
        if self.tf_timeout_s <= 0.0:
            raise ValueError("Parameter 'tf_timeout_s' must be > 0")
        if self.min_points_per_cell <= 0:
            raise ValueError("Parameter 'min_points_per_cell' must be > 0")
        if self.log_period_s <= 0.0:
            raise ValueError("Parameter 'log_period_s' must be > 0")
        if self.debug_height_max_m <= 0.0:
            raise ValueError("Parameter 'debug_height_max_m' must be > 0")

        self.range_max_m = self.range_max_m if self.range_max_m > 0.0 else None
        self._last_log_time = self.get_clock().now()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        self._binary_pub = self.create_publisher(OccupancyGrid, self.output_topic, 10)
        self._debug_pub = None
        if self.publish_debug_height:
            self._debug_pub = self.create_publisher(
                OccupancyGrid,
                self.debug_height_topic,
                10,
            )

        self.get_logger().info(
            "HeightCostmap configured: "
            f"input={self.input_topic} output={self.output_topic} "
            f"debug_topic={self.debug_height_topic} publish_debug={self.publish_debug_height} "
            f"frame={self.target_frame} grid={self.grid.width}x{self.grid.height}@{self.grid.resolution_m:.3f}m "
            f"floor_percentile={self.floor_percentile:.1f} "
            f"height_thr={self.obstacle_height_threshold_m:.3f}m "
            f"slope_thr={self.slope_threshold_deg:.2f}deg"
        )

    def _throttled_log(self, message: str, warn: bool = False) -> None:
        now = self.get_clock().now()
        elapsed = (now - self._last_log_time).nanoseconds * 1e-9
        if elapsed >= self.log_period_s:
            if warn:
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)
            self._last_log_time = now

    def _lookup_time(self, cloud_msg: PointCloud2) -> rclpy.time.Time:
        stamp = cloud_msg.header.stamp
        if (
            self.use_latest_tf_when_stamp_zero
            and stamp.sec == 0
            and stamp.nanosec == 0
        ):
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(stamp)

    def _ensure_target_frame(self, cloud_msg: PointCloud2) -> PointCloud2 | None:
        source_frame = cloud_msg.header.frame_id.strip()
        if not source_frame:
            self._throttled_log("Dropping cloud with empty frame_id", warn=True)
            return None

        if not requires_transform(source_frame, self.target_frame):
            return cloud_msg

        if not self.enforce_target_frame:
            self._throttled_log(
                f"Input cloud frame {source_frame} differs from target frame "
                f"{self.target_frame}, but enforce_target_frame is disabled.",
                warn=True,
            )
            return cloud_msg

        try:
            transform = self._tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                self._lookup_time(cloud_msg),
                timeout=Duration(seconds=self.tf_timeout_s),
            )
            transformed = do_transform_cloud(cloud_msg, transform)
            transformed.header.frame_id = self.target_frame
            return transformed
        except TransformException as exc:
            self._throttled_log(
                f"TF lookup failed for {source_frame} -> {self.target_frame}: {exc}",
                warn=True,
            )
            return None

    def _build_grid_msg(self, frame_id: str, stamp, data: np.ndarray) -> OccupancyGrid:
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = frame_id
        grid_msg.header.stamp = stamp

        info = MapMetaData()
        info.resolution = float(self.grid.resolution_m)
        info.width = int(self.grid.width)
        info.height = int(self.grid.height)
        origin = Pose()
        origin.position.x = float(self.grid.origin_x_m)
        origin.position.y = float(self.grid.origin_y_m)
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        info.origin = origin
        grid_msg.info = info
        grid_msg.data = data.flatten(order="C").astype(np.int8).tolist()
        return grid_msg

    def _cloud_callback(self, cloud_msg: PointCloud2) -> None:
        target_cloud = self._ensure_target_frame(cloud_msg)
        if target_cloud is None:
            return

        try:
            points_xyz = pointcloud2_to_xyz_array(target_cloud, skip_nans=True)
        except ValueError as exc:
            self._throttled_log(f"PointCloud2 parsing failed: {exc}", warn=True)
            return

        result = run_height_costmap_pipeline(
            points_xyz=points_xyz,
            grid=self.grid,
            min_points_per_cell=self.min_points_per_cell,
            floor_percentile=self.floor_percentile,
            obstacle_height_threshold_m=self.obstacle_height_threshold_m,
            slope_threshold_deg=self.slope_threshold_deg,
            slope_min_valid_neighbors=self.slope_min_valid_neighbors,
            range_max_m=self.range_max_m,
            z_min_m=self.z_min_m,
            z_max_m=self.z_max_m,
            unknown_is_free=self.unknown_is_free,
            debug_height_max_m=self.debug_height_max_m,
        )

        stamp = target_cloud.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        frame_id = self.target_frame if self.enforce_target_frame else target_cloud.header.frame_id
        binary_msg = self._build_grid_msg(
            frame_id=frame_id,
            stamp=stamp,
            data=result.binary_costmap,
        )
        self._binary_pub.publish(binary_msg)

        if self.publish_debug_height and self._debug_pub is not None:
            debug_msg = self._build_grid_msg(
                frame_id=frame_id,
                stamp=stamp,
                data=result.debug_height_costmap,
            )
            self._debug_pub.publish(debug_msg)

        known_cells = int(np.count_nonzero(result.known_mask))
        unsafe_cells = int(np.count_nonzero(result.binary_costmap == 100))
        floor_str = (
            "None"
            if result.floor_height_m is None
            else f"{result.floor_height_m:.3f}m"
        )
        self._throttled_log(
            "Published costmaps: "
            f"points_used={result.points_used} known_cells={known_cells} "
            f"unsafe_cells={unsafe_cells} floor_height={floor_str}"
        )


def main(args=None) -> None:
    """Run the height costmap node."""
    rclpy.init(args=args)
    node = HeightCostmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
