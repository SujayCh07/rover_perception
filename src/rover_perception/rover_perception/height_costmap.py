import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs_py import point_cloud2


class HeightCostmap(Node):
    def __init__(self):
        super().__init__('height_costmap')

        # Parameters (tweak later)
        self.declare_parameter('input_topic', 'cloud_in_target_frame')
        self.declare_parameter('output_topic', 'costmap_height')
        self.declare_parameter('target_frame', 'zed_cam_xright_yfwd_zup')

        self.declare_parameter('resolution', 0.10)          # m/cell
        self.declare_parameter('width', 100)                # cells
        self.declare_parameter('height', 100)               # cells
        self.declare_parameter('origin_x', -5.0)            # meters (map lower-left)
        self.declare_parameter('origin_y', -5.0)            # meters
        self.declare_parameter('obstacle_height', 0.15)     # meters

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.resolution = float(self.get_parameter('resolution').value)
        self.w = int(self.get_parameter('width').value)
        self.h = int(self.get_parameter('height').value)
        self.origin_x = float(self.get_parameter('origin_x').value)
        self.origin_y = float(self.get_parameter('origin_y').value)
        self.obstacle_h = float(self.get_parameter('obstacle_height').value)

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, self.output_topic, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'HeightCostmap listening on {self.input_topic} -> publishing {self.output_topic} '
            f'({self.w}x{self.h} @ {self.resolution}m, origin=({self.origin_x},{self.origin_y}), '
            f'obstacle_h={self.obstacle_h}, target_frame={self.target_frame})'
        )

    def cb(self, msg: PointCloud2):
        cloud = msg

        # If incoming frame isn't target_frame, transform it
        if msg.header.frame_id != self.target_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id, rclpy.time.Time())
                cloud = do_transform_cloud(msg, tf)
            except Exception as e:
                self.get_logger().warn(f'TF to target frame failed: {e}')
                return

        # Grid stores max Z per cell; start with -inf (no data)
        max_z = np.full((self.h, self.w), -np.inf, dtype=np.float32)

        # Read points (skip NaNs)
        for (x, y, z) in point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True):
            # Convert metric x,y to grid indices
            gx = int((x - self.origin_x) / self.resolution)
            gy = int((y - self.origin_y) / self.resolution)

            if 0 <= gx < self.w and 0 <= gy < self.h:
                if z > max_z[gy, gx]:
                    max_z[gy, gx] = z

        # Build OccupancyGrid: -1 unknown, 0 free, 100 occupied
        data = np.full((self.h, self.w), -1, dtype=np.int8)

        has_data = max_z > -np.inf
        data[has_data] = 0
        data[(has_data) & (max_z > self.obstacle_h)] = 100

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.target_frame  # MVP frame

        info = MapMetaData()
        info.resolution = self.resolution
        info.width = self.w
        info.height = self.h

        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        info.origin = origin

        grid.info = info
        grid.data = data.flatten(order='C').tolist()

        self.pub.publish(grid)
        self.get_logger().info('Published height costmap')


def main():
    rclpy.init()
    node = HeightCostmap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()