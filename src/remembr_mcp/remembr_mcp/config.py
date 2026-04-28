import logging
import os

from qdrant.config.database_config import DatabaseConfig


class SimpleLogger:
    """Logger adapter implementing qdrant.utils.protocols.Logger."""

    def __init__(self) -> None:
        self._log = logging.getLogger('remembr_mcp')

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def debug(self, msg: str) -> None:
        self._log.debug(msg)

    def fatal(self, msg: str) -> None:
        self._log.critical(msg)


def load_config() -> DatabaseConfig:
    """Load DatabaseConfig from environment variables."""
    return DatabaseConfig(
        qdrant_url=os.environ.get('QDRANT_URL', 'http://localhost:6333'),
        collection_name=os.environ.get('QDRANT_COLLECTION', 'robot_memories'),
        embedding_model=os.environ.get(
            'EMBEDDING_MODEL', 'mixedbread-ai/mxbai-embed-large-v1'
        ),
    )
