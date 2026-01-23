import traceback
from typing import Protocol

from langchain_huggingface import HuggingFaceEmbeddings


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class EmbeddingService:
    """Embedding model management and encoding service."""

    DEFAULT_EMBEDDING_DIM = 1024

    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name
        self.logger = logger
        self.model = None

    def load_model(self) -> None:
        """Load HuggingFaceEmbeddings model."""
        try:
            self.logger.info(f'Loading {self.model_name} model (HuggingFaceEmbeddings)...')
            self.model = HuggingFaceEmbeddings(model_name=self.model_name)
            self.logger.info(
                f'Embedding model loaded (output dim: {self.embedding_dimension})')
        except Exception as e:
            self.logger.error(f'Failed to load embedding model: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def encode(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")
        return self.model.embed_query(text)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        return self.DEFAULT_EMBEDDING_DIM

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model:
            del self.model
            self.model = None
            self.logger.info('Embedding model resources released')
