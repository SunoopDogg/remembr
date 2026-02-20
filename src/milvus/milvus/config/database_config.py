import os
from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


# Time normalization constant (Unix timestamp for 2024-07-24 00:00:00 UTC)
# Used to normalize timestamps closer to 0 for better float precision
# Note: This offset should be updated for deployments significantly after 2024
TIMESTAMP_NORMALIZATION_EPOCH = 1721761000

EMBEDDING_MODEL_DIMS = {
    'mixedbread-ai/mxbai-embed-large-v1': 1024,
    'jinaai/jina-embeddings-v3': 1024,
}


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Milvus database configuration with ROS2 parameter defaults."""

    db_dir: str = '/root/remembr/src/milvus/db'
    db_filename: str = ''  # Empty means auto-generate from model
    collection_name: str = 'robot_memories'
    embedding_model: str = 'mixedbread-ai/mxbai-embed-large-v1'
    embedding_dim: int = 0  # 0 means auto-detect
    input_topic: str = '/caption_with_pose'

    # Index parameters
    text_index_nlist: int = 1024
    spatial_index_nlist: int = 2

    # Field length limits
    max_id_length: int = 1000
    max_caption_length: int = 3000

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
        node.declare_parameter('embedding_dim', defaults['embedding_dim'])
        node.declare_parameter('input_topic', defaults['input_topic'])

        return cls(
            db_dir=node.get_parameter('db_dir').value,
            db_filename=node.get_parameter('db_filename').value,
            collection_name=node.get_parameter('collection_name').value,
            embedding_model=node.get_parameter('embedding_model').value,
            embedding_dim=node.get_parameter('embedding_dim').value,
            input_topic=node.get_parameter('input_topic').value,
        )

    @property
    def resolved_db_filename(self) -> str:
        """Get DB filename, auto-generating from model if not specified."""
        if self.db_filename:
            return self.db_filename
        # Auto-generate from model name and dimension
        model_short = self.embedding_model.split('/')[-1]
        dim = self.get_known_embedding_dim() or 'auto'
        return f"milvus_{model_short}_{dim}.db"

    @property
    def db_path(self) -> str:
        """Compute database path from directory and filename."""
        return os.path.join(self.db_dir, self.resolved_db_filename)

    def ensure_db_dir(self) -> None:
        """Ensure database directory exists."""
        os.makedirs(self.db_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.collection_name:
            raise ValueError("Collection name cannot be empty")
        if not self.collection_name.replace('_', '').isalnum():
            raise ValueError("Collection name must be alphanumeric with underscores")

    def get_known_embedding_dim(self) -> int | None:
        """Get known embedding dimension for the configured model."""
        return EMBEDDING_MODEL_DIMS.get(self.embedding_model)
