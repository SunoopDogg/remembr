from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


@dataclass(frozen=True, slots=True)
class CaptionerConfig:

    model_name: str = 'gemma-4'
    vlm_base_url: str = 'http://192.168.0.151:8000'
    prompt_text: str = 'Describe what you see in these images.'
    segment_time: float = 3.0
    compressed_image_topic: str = '/camera/camera/color/image_raw/compressed'
    raw_image_topic: str = '/camera/image_raw'
    odom_topic: str = '/odom'
    output_topic: str = '/caption_with_pose'
    max_buffer_size: int = 90
    max_caption_frames: int = 10
    temperature: float = 0.2
    max_tokens: int = 512

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'CaptionerConfig':
        defaults = {
            f.name: f.default if f.default is not MISSING else f.default_factory()
            for f in fields(cls)
        }

        node.declare_parameter('model_name', defaults['model_name'])
        node.declare_parameter('vlm_base_url', defaults['vlm_base_url'])
        node.declare_parameter('prompt_text', defaults['prompt_text'])
        node.declare_parameter('segment_time', defaults['segment_time'])
        node.declare_parameter('compressed_image_topic', defaults['compressed_image_topic'])
        node.declare_parameter('raw_image_topic', defaults['raw_image_topic'])
        node.declare_parameter('odom_topic', defaults['odom_topic'])
        node.declare_parameter('output_topic', defaults['output_topic'])
        node.declare_parameter('max_buffer_size', defaults['max_buffer_size'])
        node.declare_parameter('max_caption_frames', defaults['max_caption_frames'])
        node.declare_parameter('temperature', defaults['temperature'])
        node.declare_parameter('max_tokens', defaults['max_tokens'])

        return cls(
            model_name=node.get_parameter('model_name').value,
            vlm_base_url=node.get_parameter('vlm_base_url').value,
            prompt_text=node.get_parameter('prompt_text').value,
            segment_time=node.get_parameter('segment_time').value,
            compressed_image_topic=node.get_parameter('compressed_image_topic').value,
            raw_image_topic=node.get_parameter('raw_image_topic').value,
            odom_topic=node.get_parameter('odom_topic').value,
            output_topic=node.get_parameter('output_topic').value,
            max_buffer_size=node.get_parameter('max_buffer_size').value,
            temperature=node.get_parameter('temperature').value,
            max_tokens=node.get_parameter('max_tokens').value,
        )
