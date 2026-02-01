import traceback

from pymilvus import MilvusClient, DataType

from ..config.database_config import DatabaseConfig
from ..models.milvus_record import MilvusMemoryRecord
from ..utils.protocols import Logger


class MilvusService:
    """Milvus database operations service."""

    def __init__(
        self,
        config: DatabaseConfig,
        logger: Logger,
        embedding_dim: int = 1024,
    ) -> None:
        self.config = config
        self.logger = logger
        self.embedding_dim = embedding_dim
        self.client = None

    def connect(self) -> None:
        """Connect to Milvus database."""
        try:
            self.logger.info('Connecting to Milvus...')
            self.client = MilvusClient(self.config.db_path)
            self.logger.info(f'Connected to Milvus at {self.config.db_path}')
        except Exception as e:
            self.logger.error(f'Failed to connect to Milvus: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def setup_collection(self) -> None:
        """Set up Milvus collection with schema and indexes."""
        if self.client.has_collection(self.config.collection_name):
            self.logger.info(f'Collection "{self.config.collection_name}" already exists')
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=False,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=self.config.max_id_length,
        )
        schema.add_field(
            field_name="text_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim,
        )
        schema.add_field(
            field_name="position",
            datatype=DataType.FLOAT_VECTOR,
            dim=3,
        )
        schema.add_field(field_name="theta", datatype=DataType.FLOAT)
        schema.add_field(
            field_name="time",
            datatype=DataType.FLOAT_VECTOR,
            dim=2,
        )
        schema.add_field(
            field_name="caption",
            datatype=DataType.VARCHAR,
            max_length=self.config.max_caption_length,
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="text_embedding",
            metric_type="L2",
            index_type="IVF_FLAT",
            params={"nlist": self.config.text_index_nlist},
        )

        index_params.add_index(
            field_name="position",
            metric_type="L2",
            index_type="IVF_FLAT",
            params={"nlist": self.config.spatial_index_nlist},
        )

        index_params.add_index(
            field_name="time",
            metric_type="L2",
            index_type="IVF_FLAT",
            params={"nlist": self.config.spatial_index_nlist},
        )

        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )

        self.logger.info(f'Created collection "{self.config.collection_name}"')

    def reset_database(self) -> None:
        """Reset database by dropping and recreating collection."""
        self.logger.info('Starting database reset...')

        try:
            # Drop collection if it exists
            if self.client.has_collection(self.config.collection_name):
                self.logger.info(f'Dropping collection "{self.config.collection_name}"...')
                self.client.drop_collection(self.config.collection_name)
                self.logger.info(f'Collection "{self.config.collection_name}" dropped')

            self.setup_collection()
            self.logger.info('Database reset complete')

        except Exception as e:
            self.logger.error(f'Error during database reset: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def insert_record(self, record: MilvusMemoryRecord) -> dict:
        """Insert record into Milvus collection."""
        try:
            result = self.client.insert(
                collection_name=self.config.collection_name,
                data=[record.to_dict()]
            )
            return result
        except Exception as e:
            self.logger.error(f'Failed to insert record: {e}')
            self.logger.error(traceback.format_exc())
            raise

    def has_collection(self) -> bool:
        """Check if collection exists."""
        return self.client.has_collection(self.config.collection_name)

    def close(self) -> None:
        """Close Milvus connection."""
        if self.client and hasattr(self.client, 'close'):
            try:
                self.client.close()
                self.logger.info('Milvus connection closed')
            except Exception as e:
                self.logger.warning(f'Error closing Milvus connection: {e}')
