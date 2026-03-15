#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class SlopeCostmapNode(Node):
    def __init__(self):
        super().__init__("slope_costmap")

        # ---------------- Parameters ----------------
        self.declare_parameter("cloud_topic", "/cloud_with_normals")
        self.declare_parameter("grid_topic", "/slope_costmap")

        # Match these to your height costmap
        self.declare_parameter("resolution", 0.05)      # m/cell
        self.declare_parameter("width", 120)            # cells
        self.declare_parameter("height", 120)           # cells
        self.declare_parameter("origin_x", -3.0)        # meters in map frame
        self.declare_parameter("origin_y", -3.0)        # meters in map frame

        # Slope thresholds
        self.declare_parameter("slope_safe_deg", 8.0)
        self.declare_parameter("slope_max_deg", 25.0)

        # Minimum number of points per cell
        self.declare_parameter("min_points_per_cell", 5)

        # Optional ROI filters
        self.declare_parameter("z_min", -10.0)
        self.declare_parameter("z_max", 10.0)
        self.declare_parameter("range_max", 20.0)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.grid_topic = self.get_parameter("grid_topic").value

        self.resolution = float(self.get_parameter("resolution").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)

        self.slope_safe_deg = float(self.get_parameter("slope_safe_deg").value)
        self.slope_max_deg = float(self.get_parameter("slope_max_deg").value)
        self.min_points_per_cell = int(self.get_parameter("min_points_per_cell").value)

        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.range_max = float(self.get_parameter("range_max").value)

        # ---------------- ROS interfaces ----------------
        self.sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            10
        )

        self.pub = self.create_publisher(
            OccupancyGrid,
            self.grid_topic,
            10
        )

        self.get_logger().info(
            f"Slope costmap node started\n"
            f"  cloud_topic: {self.cloud_topic}\n"
            f"  grid_topic: {self.grid_topic}\n"
            f"  resolution: {self.resolution}\n"
            f"  width x height: {self.width} x {self.height}\n"
            f"  origin: ({self.origin_x}, {self.origin_y})\n"
            f"  slope_safe_deg: {self.slope_safe_deg}\n"
            f"  slope_max_deg: {self.slope_max_deg}\n"
            f"  min_points_per_cell: {self.min_points_per_cell}"
        )

    def slope_to_cost(self, slope_deg: float) -> int:
        if slope_deg <= self.slope_safe_deg:
            return 0
        if slope_deg >= self.slope_max_deg:
            return 100

        c = 100.0 * (slope_deg - self.slope_safe_deg) / (
            self.slope_max_deg - self.slope_safe_deg
        )
        return int(round(c))

    def cloud_callback(self, msg: PointCloud2):
        # Accumulators
        normal_sum = np.zeros((self.height, self.width, 3), dtype=np.float32)
        count = np.zeros((self.height, self.width), dtype=np.int32)

        # Read x,y,z,nx,ny,nz
        try:
            points_iter = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "nx", "ny", "nz"),
                skip_nans=True
            )
        except Exception as e:
            self.get_logger().error(f"Failed to read cloud fields x,y,z,nx,ny,nz: {e}")
            return

        total_points = 0
        kept_points = 0

        for p in points_iter:
            total_points += 1
            x, y, z, nx, ny, nz = p

            # Basic validity checks
            if not np.isfinite([x, y, z, nx, ny, nz]).all():
                continue

            # ROI filters
            if z < self.z_min or z > self.z_max:
                continue

            r = math.sqrt(x * x + y * y + z * z)
            if r > self.range_max:
                continue

            gx = int((x - self.origin_x) / self.resolution)
            gy = int((y - self.origin_y) / self.resolution)

            if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
                continue

            normal_sum[gy, gx, 0] += nx
            normal_sum[gy, gx, 1] += ny
            normal_sum[gy, gx, 2] += nz
            count[gy, gx] += 1
            kept_points += 1

        # Build OccupancyGrid
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = msg.header.stamp
        grid.header.frame_id = msg.header.frame_id

        grid.info = MapMetaData()
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin = Pose()
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        # Default unknown
        data = np.full((self.height, self.width), -1, dtype=np.int8)

        valid_cells = count >= self.min_points_per_cell

        for gy in range(self.height):
            for gx in range(self.width):
                if not valid_cells[gy, gx]:
                    continue

                n = normal_sum[gy, gx]
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-6:
                    continue

                n = n / n_norm

                # Assuming frame is gravity-aligned and +Z is up
                nz = float(np.clip(n[2], -1.0, 1.0))
                slope_deg = math.degrees(math.acos(nz))

                data[gy, gx] = self.slope_to_cost(slope_deg)

        # Flatten row-major
        grid.data = data.flatten().tolist()
        self.pub.publish(grid)

        self.get_logger().debug(
            f"Processed cloud: total={total_points}, kept={kept_points}, "
            f"valid_cells={int(np.count_nonzero(valid_cells))}"
        )


def main():
    rclpy.init()
    node = SlopeCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()