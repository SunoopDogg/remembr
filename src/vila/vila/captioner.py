import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage


class Captioner(Node):

    def __init__(self):
        super().__init__('captioner')

        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        self.image_subscriber  # prevent unused variable warning

    def image_callback(self, msg):
        # Process the incoming image message
        pass


def main(args=None):
    rclpy.init(args=args)

    captioner = Captioner()

    rclpy.spin(captioner)

    captioner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
