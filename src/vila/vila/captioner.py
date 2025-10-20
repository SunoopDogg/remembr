import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage

import time
import concurrent.futures
from PIL import Image as im

import llava


class Captioner(Node):

    def __init__(self):
        super().__init__('captioner')

        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        self.image_subscriber  # prevent unused variable warning

        self.start_time = 0
        self.segment_time = 3  # seconds

        self.vila_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.image_buffer = []

        self.model = llava.load("NVILA-Lite-8B")

    def image_callback(self, msg):
        # Process the incoming image message
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
        pil_image = im.fromarray(cv_image)
        self.image_buffer.append(pil_image)

        current_time = msg.header.stamp
        converted_time = float(str(current_time.sec) + '.' + str(current_time.nanosec))

        if converted_time - self.start_time >= self.segment_time:
            if self.image_buffer:
                self.vila_executor.submit(self.process_images, self.image_buffer)

            self.start_time = converted_time
            self.image_buffer = []

    def process_images(self, images):
        # Process the list of images using the Vila model
        print(f"Processing {len(images)} images with Vila model...")
        # captions = []
        # for image in images:
        #     caption = self.model.generate_caption(image)
        #     captions.append(caption)
        # for i, caption in enumerate(captions):
        #     self.get_logger().info(f'Image {i}: {caption}')


def main(args=None):
    rclpy.init(args=args)

    captioner = Captioner()

    rclpy.spin(captioner)

    captioner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
