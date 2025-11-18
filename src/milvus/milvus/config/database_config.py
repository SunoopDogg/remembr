"""Database configuration for Milvus memory builder."""

import os
from dataclasses import dataclass, field


# Configuration Constants
DB_DIR = "/root/remembr/src/milvus/db"
DB_FILENAME = "milvus_demo.db"
COLLECTION_NAME = "robot_memories"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


@dataclass(slots=True)
class DatabaseConfig:
    """Configuration for Milvus database connection."""
    db_dir: str = DB_DIR
    db_filename: str = DB_FILENAME
    collection_name: str = COLLECTION_NAME
    db_path: str = field(init=False)

    def __post_init__(self):
        """Compute db_path and ensure directory exists."""
        self.db_path = os.path.join(self.db_dir, self.db_filename)
        os.makedirs(self.db_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.collection_name:
            raise ValueError("Collection name cannot be empty")
        if not self.collection_name.replace('_', '').isalnum():
            raise ValueError("Collection name must be alphanumeric with underscores")
