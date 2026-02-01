import traceback

from langchain_huggingface import HuggingFaceEmbeddings

from ..config.database_config import EMBEDDING_MODEL_DIMS
from ..utils.protocols import Logger


class EmbeddingService:
    """Embedding model management and encoding service."""

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        expected_dim: int = 0,
    ) -> None:
        self.model_name = model_name
        self.logger = logger
        self.expected_dim = expected_dim
        self.model = None
        self._detected_dim: int | None = None

    def load_model(self) -> None:
        """Load HuggingFaceEmbeddings model and detect embedding dimension."""
        try:
            self.logger.info(f'Loading {self.model_name} model (HuggingFaceEmbeddings)...')
            self.model = HuggingFaceEmbeddings(model_name=self.model_name)

            self._detect_dimension()

            self.logger.info(
                f'Embedding model loaded (output dim: {self.embedding_dimension})')
        except Exception as e:
            self.logger.error(f'Failed to load embedding model: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def _detect_dimension(self) -> None:
        """Detect actual embedding dimension from the model."""
        if self.expected_dim > 0:
            self._detected_dim = self.expected_dim
            self.logger.info(f'Using configured embedding dimension: {self.expected_dim}')
            return

        known_dim = EMBEDDING_MODEL_DIMS.get(self.model_name)
        if known_dim:
            self._detected_dim = known_dim
            self.logger.info(f'Using known embedding dimension for {self.model_name}: {known_dim}')
            return

        self.logger.info('Detecting embedding dimension from model...')
        test_embedding = self.model.embed_query("dimension test")
        self._detected_dim = len(test_embedding)
        self.logger.info(f'Detected embedding dimension: {self._detected_dim}')

    def encode(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")
        return self.model.embed_query(text)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._detected_dim is None:
            raise RuntimeError("Embedding dimension not detected. Call load_model() first.")
        return self._detected_dim

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model:
            del self.model
            self.model = None
            self._detected_dim = None
            self.logger.info('Embedding model resources released')
