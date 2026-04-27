import traceback

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..config.database_config import DatabaseConfig
from ..models.qdrant_record import QdrantMemoryRecord
from ..utils.protocols import Logger


class QdrantService:
    """Qdrant database operations service."""

    def __init__(
        self,
        config: DatabaseConfig,
        logger: Logger,
        embedding_dim: int | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.embedding_dim = (
            embedding_dim
            or config.get_known_embedding_dim()
            or 1024
        )
        self.client = None

    def connect(self) -> None:
        """Connect to Qdrant server."""
        try:
            self.logger.info('Connecting to Qdrant...')
            self.client = QdrantClient(url=self.config.qdrant_url)
            self.logger.info(f'Connected to Qdrant at {self.config.qdrant_url}')
        except Exception as e:
            self.logger.error(f'Failed to connect to Qdrant: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def setup_collection(self) -> None:
        """Set up Qdrant collection with Named Vectors schema."""
        if self.client.collection_exists(self.config.collection_name):
            self.logger.info(f'Collection "{self.config.collection_name}" already exists')
            return

        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                'text_embedding': VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                'position':       VectorParams(size=3,                  distance=Distance.EUCLID),
                'time':           VectorParams(size=2,                  distance=Distance.EUCLID),
            },
        )
        self.logger.info(f'Created collection "{self.config.collection_name}"')

    def reset_database(self) -> None:
        """Reset database by dropping and recreating collection."""
        self.logger.info('Starting database reset...')
        try:
            if self.client.collection_exists(self.config.collection_name):
                self.logger.info(f'Dropping collection "{self.config.collection_name}"...')
                self.client.delete_collection(self.config.collection_name)
                self.logger.info(f'Collection "{self.config.collection_name}" dropped')
            self.setup_collection()
            self.logger.info('Database reset complete')
        except Exception as e:
            self.logger.error(f'Error during database reset: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def upsert_record(self, record: QdrantMemoryRecord) -> None:
        """Upsert record into Qdrant collection."""
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[record.to_point()],
                wait=True,
            )
        except Exception as e:
            self.logger.error(f'Failed to upsert record: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def search(self, vector: list, using: str, limit: int) -> list:
        """Execute Named Vector search on the configured collection."""
        return self.client.query_points(
            collection_name=self.config.collection_name,
            query=vector,
            using=using,
            limit=limit,
            with_payload=True,
        ).points

    def collection_exists(self) -> bool:
        """Check if collection exists."""
        return self.client.collection_exists(self.config.collection_name)

    def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            try:
                self.client.close()
                self.logger.info('Qdrant connection closed')
            except Exception as e:
                self.logger.warning(f'Error closing Qdrant connection: {e}')
