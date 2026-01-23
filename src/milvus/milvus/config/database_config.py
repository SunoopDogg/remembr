import os
from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


# Time normalization constant
FIXED_SUBTRACT = 1721761000


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Milvus database configuration with ROS2 parameter defaults."""

    db_dir: str = '/root/remembr/src/milvus/db'
    db_filename: str = 'milvus_demo.db'
    collection_name: str = 'robot_memories'
    embedding_model: str = 'mixedbread-ai/mxbai-embed-large-v1'
    input_topic: str = '/caption_with_pose'

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'DatabaseConfig':
        """Create config from ROS2 node by declaring and retrieving parameters."""
        defaults = {
            f.name: f.default if f.default is not MISSING else f.default_factory()
            for f in fields(cls)
        }

        node.declare_parameter('db_dir', defaults['db_dir'])
        node.declare_parameter('db_filename', defaults['db_filename'])
        node.declare_parameter('collection_name', defaults['collection_name'])
        node.declare_parameter('embedding_model', defaults['embedding_model'])
        node.declare_parameter('input_topic', defaults['input_topic'])

        return cls(
            db_dir=node.get_parameter('db_dir').value,
            db_filename=node.get_parameter('db_filename').value,
            collection_name=node.get_parameter('collection_name').value,
            embedding_model=node.get_parameter('embedding_model').value,
            input_topic=node.get_parameter('input_topic').value,
        )

    @property
    def db_path(self) -> str:
        """Compute database path from directory and filename."""
        return os.path.join(self.db_dir, self.db_filename)

    def ensure_db_dir(self) -> None:
        """Ensure database directory exists."""
        os.makedirs(self.db_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.collection_name:
            raise ValueError("Collection name cannot be empty")
        if not self.collection_name.replace('_', '').isalnum():
            raise ValueError("Collection name must be alphanumeric with underscores")
