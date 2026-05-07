import traceback

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..config.database_config import DatabaseConfig
from ..models.qdrant_record import QdrantMemoryRecord
from ..utils.protocols import Logger

TEXT_VECTOR = 'text_embedding'
POSITION_VECTOR = 'position'
TIME_VECTOR = 'time'


class QdrantService:
    """Qdrant database operations service."""

    def __init__(
        self,
        config: DatabaseConfig,
        logger: Logger,
        embedding_dim: int = 0,
    ) -> None:
        self._config = config
        self._logger = logger
        self._embedding_dim = embedding_dim
        self._client = None

    def connect(self) -> None:
        try:
            self._logger.info('Connecting to Qdrant...')
            self._client = QdrantClient(url=self._config.qdrant_url)
            self._logger.info(f'Connected to Qdrant at {self._config.qdrant_url}')
        except Exception as e:
            self._logger.error(f'Failed to connect to Qdrant: {e}')
            self._logger.error(traceback.format_exc())
            raise

    def setup_collection(self) -> None:
        """Set up Named Vectors schema; no-op if collection already exists."""
        name = self._config.collection_name
        if self._client.collection_exists(name):
            self._logger.info(f'Collection {name!r} already exists')
            return

        if not self._embedding_dim:
            raise RuntimeError(
                'embedding_dim must be set before creating collection'
            )

        self._client.create_collection(
            collection_name=name,
            vectors_config={
                TEXT_VECTOR:     VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
                POSITION_VECTOR: VectorParams(size=3,                  distance=Distance.EUCLID),
                TIME_VECTOR:     VectorParams(size=2,                  distance=Distance.EUCLID),
            },
        )
        self._logger.info(f'Created collection {name!r} (text_embedding dim={self._embedding_dim})')

    def reset_database(self) -> None:
        name = self._config.collection_name
        self._logger.info('Starting database reset...')
        try:
            if self._client.collection_exists(name):
                self._logger.info(f'Dropping collection {name!r}...')
                self._client.delete_collection(name)
                self._logger.info(f'Collection {name!r} dropped')
            self.setup_collection()
            self._logger.info('Database reset complete')
        except Exception as e:
            self._logger.error(f'Error during database reset: {e}')
            self._logger.error(traceback.format_exc())
            raise

    def upsert_record(self, record: QdrantMemoryRecord) -> None:
        try:
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=[record.to_point()],
                wait=True,
            )
        except Exception as e:
            self._logger.error(f'Failed to upsert record: {e}')
            self._logger.error(traceback.format_exc())
            raise

    def search(self, vector: list, using: str, limit: int) -> list:
        """Query named vector index; using must be one of the named vector constants."""
        return self._client.query_points(
            collection_name=self._config.collection_name,
            query=vector,
            using=using,
            limit=limit,
            with_payload=True,
        ).points

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._config.collection_name)

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
                self._logger.info('Qdrant connection closed')
            except Exception as e:
                self._logger.warning(f'Error closing Qdrant connection: {e}')
