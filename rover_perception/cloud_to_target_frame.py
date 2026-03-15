"""ROS node for explicit PointCloud2 frame transformation."""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from rover_perception.costmap_processing import requires_transform


class CloudToTargetFrame(Node):
    """Transform incoming clouds into a configured target frame."""

    def __init__(self) -> None:
        super().__init__("cloud_to_target_frame")

        self.declare_parameter("input_topic", "/point_cloud/cloud_registered")
        self.declare_parameter("output_topic", "/cloud_in_target_frame")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("tf_timeout_s", 0.20)
        self.declare_parameter("use_latest_tf_when_stamp_zero", True)
        self.declare_parameter("log_period_s", 2.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        self.use_latest_tf_when_stamp_zero = bool(
            self.get_parameter("use_latest_tf_when_stamp_zero").value
        )
        self.log_period_s = float(self.get_parameter("log_period_s").value)

        if not self.target_frame:
            raise ValueError("Parameter 'target_frame' must be non-empty")
        if self.tf_timeout_s <= 0.0:
            raise ValueError("Parameter 'tf_timeout_s' must be > 0")
        if self.log_period_s <= 0.0:
            raise ValueError("Parameter 'log_period_s' must be > 0")

        self._last_log_time = self.get_clock().now()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        self._pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "CloudToTargetFrame configured: "
            f"input={self.input_topic} output={self.output_topic} "
            f"target_frame={self.target_frame} tf_timeout={self.tf_timeout_s:.2f}s"
        )

    def _throttled_warn(self, message: str) -> None:
        now = self.get_clock().now()
        elapsed = (now - self._last_log_time).nanoseconds * 1e-9
        if elapsed >= self.log_period_s:
            self.get_logger().warn(message)
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

    def _cloud_callback(self, cloud_msg: PointCloud2) -> None:
        source_frame = cloud_msg.header.frame_id.strip()
        if not source_frame:
            self._throttled_warn("Dropping cloud with empty frame_id")
            return

        if not requires_transform(source_frame, self.target_frame):
            self._pub.publish(cloud_msg)
            return

        try:
            transform = self._tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                self._lookup_time(cloud_msg),
                timeout=Duration(seconds=self.tf_timeout_s),
            )
            transformed = do_transform_cloud(cloud_msg, transform)
            transformed.header.frame_id = self.target_frame
            self._pub.publish(transformed)
        except TransformException as exc:
            self._throttled_warn(
                f"TF lookup failed for {source_frame} -> {self.target_frame}: {exc}"
            )


def main(args=None) -> None:
    """Run the cloud transform node."""
    rclpy.init(args=args)
    node = CloudToTargetFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
