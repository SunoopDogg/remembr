import time
import traceback

import httpx

from ..utils.protocols import Logger

RETRIEVAL_INSTRUCTION = 'Given a query, retrieve relevant passages that answer the query'


class EmbeddingService:
    """HTTP client for vLLM embedding server."""

    def __init__(self, model_name: str, base_url: str, logger: Logger) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._logger = logger
        self._client: httpx.Client | None = None
        self._detected_dim: int | None = None

    def load_model(self, max_retries: int = 30, retry_interval: float = 5.0) -> None:
        """Connect to vLLM server and detect embedding dimension with retry."""
        self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        for attempt in range(1, max_retries + 1):
            try:
                self._logger.info(
                    f'Connecting to embedding server at {self._base_url} '
                    f'(attempt {attempt}/{max_retries})...'
                )
                test_embedding = self.encode_document('dimension test')
                self._detected_dim = len(test_embedding)
                self._logger.info(
                    f'Embedding server ready (model={self._model_name}, '
                    f'dim={self._detected_dim})'
                )
                return
            except Exception as e:
                if attempt == max_retries:
                    self._logger.error(
                        f'Failed to connect after {max_retries} attempts: {e}'
                    )
                    self._logger.error(traceback.format_exc())
                    raise
                self._logger.warning(
                    f'Server not ready ({e}), retrying in {retry_interval}s...'
                )
                time.sleep(retry_interval)

    def _request(self, text: str) -> list[float]:
        if self._client is None:
            raise RuntimeError(
                'Embedding service not connected. Call load_model() first.'
            )
        response = self._client.post('/v1/embeddings', json={
            'model': self._model_name,
            'input': text,
        })
        response.raise_for_status()
        return response.json()['data'][0]['embedding']

    def encode_document(self, text: str) -> list[float]:
        """Encode a document (caption) without instruction prefix."""
        return self._request(text)

    def encode_query(self, text: str) -> list[float]:
        """Encode a search query with retrieval instruction prefix."""
        return self._request(
            f'Instruct: {RETRIEVAL_INSTRUCTION}\nQuery:{text}'
        )

    @property
    def embedding_dimension(self) -> int:
        """Get detected embedding dimension."""
        if self._detected_dim is None:
            raise RuntimeError(
                'Embedding dimension not available. Call load_model() first.'
            )
        return self._detected_dim

    def cleanup(self) -> None:
        """Close HTTP client connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._detected_dim = None
            self._logger.info('Embedding service connection closed')
