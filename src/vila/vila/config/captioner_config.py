from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


@dataclass(frozen=True, slots=True)
class CaptionerConfig:
    """Captioner node configuration with ROS2 parameter defaults."""

    model_path: str = 'NVILA-Lite-8B'
    prompt_text: str = 'Describe what you see in these images.'
    conv_mode: str = 'vicuna_v1'
    segment_time: float = 3.0
    compressed_image_topic: str = '/camera/camera/color/image_raw/compressed'
    raw_image_topic: str = '/camera/image_raw'
    odom_topic: str = '/odom'
    output_topic: str = '/caption_with_pose'
    max_buffer_size: int = 90
    temperature: float = 0.2
    max_new_tokens: int = 512

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'CaptionerConfig':
        """Create config from ROS2 node by declaring and retrieving parameters."""
        # Get default values from field definitions
        # (slots=True makes cls.attr return a descriptor, not the default value)
        defaults = {
            f.name: f.default if f.default is not MISSING else f.default_factory()
            for f in fields(cls)
        }

        # Declare parameters with defaults
        node.declare_parameter('model_path', defaults['model_path'])
        node.declare_parameter('prompt_text', defaults['prompt_text'])
        node.declare_parameter('conv_mode', defaults['conv_mode'])
        node.declare_parameter('segment_time', defaults['segment_time'])
        node.declare_parameter('compressed_image_topic', defaults['compressed_image_topic'])
        node.declare_parameter('raw_image_topic', defaults['raw_image_topic'])
        node.declare_parameter('odom_topic', defaults['odom_topic'])
        node.declare_parameter('output_topic', defaults['output_topic'])
        node.declare_parameter('max_buffer_size', defaults['max_buffer_size'])
        node.declare_parameter('temperature', defaults['temperature'])
        node.declare_parameter('max_new_tokens', defaults['max_new_tokens'])

        # Get parameter values and construct config
        return cls(
            model_path=node.get_parameter('model_path').value,
            prompt_text=node.get_parameter('prompt_text').value,
            conv_mode=node.get_parameter('conv_mode').value,
            segment_time=node.get_parameter('segment_time').value,
            compressed_image_topic=node.get_parameter('compressed_image_topic').value,
            raw_image_topic=node.get_parameter('raw_image_topic').value,
            odom_topic=node.get_parameter('odom_topic').value,
            output_topic=node.get_parameter('output_topic').value,
            max_buffer_size=node.get_parameter('max_buffer_size').value,
            temperature=node.get_parameter('temperature').value,
            max_new_tokens=node.get_parameter('max_new_tokens').value,
        )
