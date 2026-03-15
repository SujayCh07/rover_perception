import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class CloudToTarget(Node):
    def __init__(self):
        super().__init__('cloud_to_target_frame')
        self.sub = self.create_subscription(
            PointCloud2,
            'point_cloud/cloud_registered',
            self.cb,
            10
        )
        self.pub = self.create_publisher(PointCloud2, 'cloud_in_target_frame', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_frame = 'zed_cam_xright_yfwd_zup'

    def cb(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
            cloud_out = do_transform_cloud(msg, tf)
            cloud_out.header.frame_id = self.target_frame
            self.pub.publish(cloud_out)
            self.get_logger().info(f'Transformed {msg.header.frame_id} -> {self.target_frame}')
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}')

def main():
    rclpy.init()
    node = CloudToTarget()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
