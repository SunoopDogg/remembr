from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node

TIMESTAMP_NORMALIZATION_EPOCH = 1721761000

EMBEDDING_MODEL_DIMS = {
    'mixedbread-ai/mxbai-embed-large-v1': 1024,
    'jinaai/jina-embeddings-v3': 1024,
}


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Qdrant database configuration with ROS2 parameter defaults."""

    qdrant_url: str = 'http://localhost:6333'
    collection_name: str = 'robot_memories'
    embedding_model: str = 'mixedbread-ai/mxbai-embed-large-v1'
    embedding_dim: int = 0  # 0 means auto-detect
    input_topic: str = '/caption_with_pose'
    max_id_length: int = 1000
    max_caption_length: int = 3000

    @classmethod
    def from_ros_node(cls, node: 'Node') -> 'DatabaseConfig':
        """Create config from ROS2 node by declaring and retrieving parameters."""
        defaults = {
            f.name: f.default if f.default is not MISSING else f.default_factory()
            for f in fields(cls)
        }

        node.declare_parameter('qdrant_url', defaults['qdrant_url'])
        node.declare_parameter('collection_name', defaults['collection_name'])
        node.declare_parameter('embedding_model', defaults['embedding_model'])
        node.declare_parameter('embedding_dim', defaults['embedding_dim'])
        node.declare_parameter('input_topic', defaults['input_topic'])

        return cls(
            qdrant_url=node.get_parameter('qdrant_url').value,
            collection_name=node.get_parameter('collection_name').value,
            embedding_model=node.get_parameter('embedding_model').value,
            embedding_dim=node.get_parameter('embedding_dim').value,
            input_topic=node.get_parameter('input_topic').value,
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.collection_name:
            raise ValueError('Collection name cannot be empty')
        if not self.collection_name.replace('_', '').isalnum():
            raise ValueError('Collection name must be alphanumeric with underscores')

    def get_known_embedding_dim(self) -> int | None:
        """Get known embedding dimension for the configured model."""
        return EMBEDDING_MODEL_DIMS.get(self.embedding_model)
