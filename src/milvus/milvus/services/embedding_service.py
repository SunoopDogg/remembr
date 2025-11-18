"""EmbeddingService - manages embedding model loading and encoding."""

import traceback
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Manages embedding model loading and encoding."""

    def __init__(self, model_name: str, logger):
        """
        Initialize embedding service.

        Args:
            model_name: Name of the SentenceTransformer model to load
            logger: ROS2 logger instance
        """
        self.model_name = model_name
        self.logger = logger
        self.model = None

    def load_model(self) -> None:
        """Load SentenceTransformer model."""
        try:
            self.logger.info(f'Loading {self.model_name} model...')
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(
                f'Embedding model loaded (output dim: {self.embedding_dimension})')
        except Exception as e:
            self.logger.error(f'Failed to load embedding model: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def encode(self, text: str) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to encode

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_tensor=False
        ).tolist()

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()

    def cleanup(self) -> None:
        """Cleanup resources and free memory."""
        if self.model:
            del self.model
            self.model = None
            self.logger.info('Embedding model resources released')
