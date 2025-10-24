import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage

import os
import time
import concurrent.futures
from PIL import Image as im
from transformers import GenerationConfig

import llava
from llava import conversation as clib
from llava.media import Image
from llava.utils.logging import logger


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on environment variables."""
    # Get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        logger.info(f"Num look close: {num_look_close}")
        model.num_look_close = int(num_look_close)
    if num_token_look_close is not None:
        logger.info(f"Num token look close: {num_token_look_close}")
        model.num_token_look_close = int(num_token_look_close)
    if select_num_each_scale is not None:
        logger.info(f"Select num each scale: {select_num_each_scale}")
        select_num_each_scale = [int(x)
                                 for x in select_num_each_scale.split("+")]
        model.get_vision_tower(
        ).vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        logger.info(f"Look close mode: {look_close_mode}")
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        logger.info(f"Smooth selection prob: {smooth_selection_prob}")
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(
                f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, int(
            num_look_close) * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, int(
            num_token_look_close) // 4 + 1024)
    context_length = max(
        getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


class Captioner(Node):

    def __init__(self,
                 model_path="NVILA-Lite-8B",
                 prompt_text="Describe what you see in these images.",
                 conv_mode="vicuna_v1"):
        super().__init__('captioner')

        # ROS2 subscriber setup
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        self.image_subscriber  # prevent unused variable warning

        self.image_buffer = []

        self.start_time = 0
        self.segment_time = 3  # seconds

        self.vila_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Store prompt text for VILA queries
        self.prompt_text = prompt_text

        # Load VILA model
        logger.info(f"Loading VILA model from: {model_path}")
        self.model = llava.load(model_path, model_base=None)
        logger.info("VILA model loaded successfully")

        # Configure PS3 and context length
        configure_ps3_and_context_length(self.model)

        # Set conversation mode
        clib.default_conversation = clib.conv_templates[conv_mode].copy()
        logger.info(f"Using conversation mode: {conv_mode}")

    def image_callback(self, msg):
        # Process the incoming image message
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
        pil_image = im.fromarray(cv_image)
        self.image_buffer.append(pil_image)

        current_time = msg.header.stamp
        converted_time = float(str(current_time.sec) + '.' + str(current_time.nanosec))

        if converted_time - self.start_time >= self.segment_time:
            before_time = time.time()

            if self.image_buffer:
                self.vila_executor.submit(self.process_image_segment, self.image_buffer)

            after_time = time.time()
            logger.info(f"Processing time for segment: {after_time - before_time} seconds")

            self.start_time = converted_time
            self.image_buffer = []

    def process_image_segment(self, images):
        try:
            logger.info(f"Processing segment with {len(images)} images")

            # Build the prompt: [Image1, Image2, ..., ImageN, prompt_text]
            prompt = images.copy()

            # Add the text query
            prompt.append(self.prompt_text)

            # Generate response using VILA
            logger.info("Generating caption with VILA...")
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)

            # Log the response
            logger.info("=" * 60)
            logger.info("VILA Response:")
            logger.info(response)
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during VILA inference: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)

    captioner = Captioner()

    rclpy.spin(captioner)

    captioner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
