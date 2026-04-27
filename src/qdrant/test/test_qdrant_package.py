import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDatabaseConfig:
    def test_default_url(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig()
        assert config.qdrant_url == 'http://localhost:6333'

    def test_default_collection(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig()
        assert config.collection_name == 'robot_memories'

    def test_validate_empty_collection_raises(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig(collection_name='')
        with pytest.raises(ValueError, match='Collection name cannot be empty'):
            config.validate()

    def test_validate_invalid_collection_raises(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig(collection_name='invalid-name!')
        with pytest.raises(ValueError, match='alphanumeric'):
            config.validate()

    def test_get_known_embedding_dim(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig(embedding_model='mixedbread-ai/mxbai-embed-large-v1')
        assert config.get_known_embedding_dim() == 1024

    def test_get_known_embedding_dim_unknown(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig(embedding_model='unknown/model')
        assert config.get_known_embedding_dim() is None

    def test_timestamp_normalization_epoch(self):
        from qdrant.config.database_config import TIMESTAMP_NORMALIZATION_EPOCH
        assert TIMESTAMP_NORMALIZATION_EPOCH == 1721761000


class TestQdrantMemoryRecord:
    def _make_record(self, **kwargs):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        defaults = dict(
            id='test-uuid',
            text_embedding=[0.1, 0.2, 0.3],
            position=[1.0, 2.0, 3.0],
            theta=1.57,
            time=[100.0, 0.0],
            caption='robot sees the kitchen',
        )
        defaults.update(kwargs)
        return QdrantMemoryRecord(**defaults)

    def test_to_point_id(self):
        record = self._make_record()
        point = record.to_point()
        assert point.id == 'test-uuid'

    def test_to_point_named_vectors(self):
        record = self._make_record()
        point = record.to_point()
        assert point.vector['text_embedding'] == [0.1, 0.2, 0.3]
        assert point.vector['position'] == [1.0, 2.0, 3.0]
        assert point.vector['time'] == [100.0, 0.0]

    def test_to_point_payload_scalars(self):
        record = self._make_record()
        point = record.to_point()
        assert point.payload['caption'] == 'robot sees the kitchen'
        assert point.payload['theta'] == 1.57

    def test_to_point_payload_includes_position_and_time(self):
        record = self._make_record()
        point = record.to_point()
        assert point.payload['position'] == [1.0, 2.0, 3.0]
        assert point.payload['time'] == [100.0, 0.0]

    def test_validation_empty_id(self):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        with pytest.raises(ValueError, match='ID cannot be empty'):
            QdrantMemoryRecord(
                id='',
                text_embedding=[0.1],
                position=[1.0, 2.0, 3.0],
                theta=0.0,
                time=[100.0, 0.0],
                caption='test',
            )

    def test_validation_id_too_long(self):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        with pytest.raises(ValueError, match='ID exceeds max length'):
            QdrantMemoryRecord(
                id='x' * 1001,
                text_embedding=[0.1],
                position=[1.0, 2.0, 3.0],
                theta=0.0,
                time=[100.0, 0.0],
                caption='test',
            )

    def test_validation_caption_too_long(self):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        with pytest.raises(ValueError, match='Caption exceeds max length'):
            QdrantMemoryRecord(
                id='test-id',
                text_embedding=[0.1],
                position=[1.0, 2.0, 3.0],
                theta=0.0,
                time=[100.0, 0.0],
                caption='x' * 3001,
            )

    def test_validation_time_wrong_dimension(self):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        with pytest.raises(ValueError, match='Time must be 2D'):
            QdrantMemoryRecord(
                id='test-id',
                text_embedding=[0.1],
                position=[1.0, 2.0, 3.0],
                theta=0.0,
                time=[100.0],
                caption='test',
            )

    def test_from_caption_and_vectors_generates_uuid(self):
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        from unittest.mock import MagicMock
        caption_data = MagicMock()
        caption_data.caption = 'test caption'
        vector_data = MagicMock()
        vector_data.text_embedding = [0.1, 0.2]
        vector_data.position_vector = [1.0, 2.0, 3.0]
        vector_data.theta = 0.5
        vector_data.time = [50.0, 0.0]

        record = QdrantMemoryRecord.from_caption_and_vectors(caption_data, vector_data)
        assert len(record.id) == 36
        assert record.caption == 'test caption'
