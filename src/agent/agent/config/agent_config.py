from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """ReMEmbR Agent configuration with ROS2 parameter defaults."""

    model: str = 'gpt-oss:20b'
    num_ctx: int = 8192
    temperature: float = 0.0
    max_tool_calls: int = 3
    use_json_format: bool = True
    use_functions_wrapper: bool = False
    novelty_weight: float = 0.0

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'AgentConfig':
        """Create config from ROS2 node by declaring and retrieving parameters."""
        defaults = {
            'model': 'gpt-oss:20b',
            'num_ctx': 8192,
            'temperature': 0.0,
            'max_tool_calls': 3,
            'use_json_format': True,
            'use_functions_wrapper': False,
            'novelty_weight': 0.0,
        }
        node.declare_parameter('model', defaults['model'])
        node.declare_parameter('num_ctx', defaults['num_ctx'])
        node.declare_parameter('temperature', defaults['temperature'])
        node.declare_parameter('max_tool_calls', defaults['max_tool_calls'])
        node.declare_parameter('use_json_format', defaults['use_json_format'])
        node.declare_parameter('use_functions_wrapper', defaults['use_functions_wrapper'])
        node.declare_parameter('novelty_weight', defaults['novelty_weight'])

        return cls(
            model=node.get_parameter('model').value,
            num_ctx=node.get_parameter('num_ctx').value,
            temperature=node.get_parameter('temperature').value,
            max_tool_calls=node.get_parameter('max_tool_calls').value,
            use_json_format=node.get_parameter('use_json_format').value,
            use_functions_wrapper=node.get_parameter('use_functions_wrapper').value,
            novelty_weight=node.get_parameter('novelty_weight').value,
        )
