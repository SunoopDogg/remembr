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

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'AgentConfig':
        """Create config from ROS2 node by declaring and retrieving parameters."""
        node.declare_parameter('model', 'gpt-oss:20b')
        node.declare_parameter('num_ctx', 8192)
        node.declare_parameter('temperature', 0.0)
        node.declare_parameter('max_tool_calls', 3)

        return cls(
            model=node.get_parameter('model').value,
            num_ctx=node.get_parameter('num_ctx').value,
            temperature=node.get_parameter('temperature').value,
            max_tool_calls=node.get_parameter('max_tool_calls').value,
        )
