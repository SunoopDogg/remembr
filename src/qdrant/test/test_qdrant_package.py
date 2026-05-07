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

    def test_default_embedding_url(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig()
        assert config.embedding_url == 'http://localhost:8080'

    def test_default_embedding_model(self):
        from qdrant.config.database_config import DatabaseConfig
        config = DatabaseConfig()
        assert config.embedding_model == 'Qwen/Qwen3-Embedding-4B'

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


class TestQdrantService:
    def _make_service(self, mock_client, embedding_dim=4096):
        from qdrant.services.qdrant_service import QdrantService
        from qdrant.config.database_config import DatabaseConfig
        from unittest.mock import MagicMock
        config = DatabaseConfig()
        logger = MagicMock()
        service = QdrantService(config, logger, embedding_dim=embedding_dim)
        service.client = mock_client
        return service

    def test_embedding_dim_explicit(self):
        from qdrant.services.qdrant_service import QdrantService
        from qdrant.config.database_config import DatabaseConfig
        from unittest.mock import MagicMock
        service = QdrantService(DatabaseConfig(), MagicMock(), embedding_dim=4096)
        assert service.embedding_dim == 4096

    def test_embedding_dim_default_zero(self):
        from qdrant.services.qdrant_service import QdrantService
        from qdrant.config.database_config import DatabaseConfig
        from unittest.mock import MagicMock
        service = QdrantService(DatabaseConfig(), MagicMock())
        assert service.embedding_dim == 0

    def test_setup_collection_skips_if_exists(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        service = self._make_service(mock_client)
        service.setup_collection()
        mock_client.create_collection.assert_not_called()

    def test_setup_collection_raises_without_dim(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        service = self._make_service(mock_client, embedding_dim=0)
        with pytest.raises(RuntimeError, match='embedding_dim must be set'):
            service.setup_collection()

    def test_setup_collection_creates_with_named_vectors(self):
        from unittest.mock import MagicMock
        from qdrant_client.models import Distance
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        service = self._make_service(mock_client)
        service.setup_collection()
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        vectors_config = call_kwargs['vectors_config']
        assert 'text_embedding' in vectors_config
        assert 'position' in vectors_config
        assert 'time' in vectors_config
        assert vectors_config['text_embedding'].distance == Distance.COSINE
        assert vectors_config['position'].distance == Distance.EUCLID
        assert vectors_config['time'].distance == Distance.EUCLID

    def test_upsert_record_calls_client(self):
        from unittest.mock import MagicMock
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        mock_client = MagicMock()
        service = self._make_service(mock_client)
        record = QdrantMemoryRecord(
            id='test-id',
            text_embedding=[0.1] * 4096,
            position=[1.0, 2.0, 3.0],
            theta=0.0,
            time=[100.0, 0.0],
            caption='test',
        )
        service.upsert_record(record)
        mock_client.upsert.assert_called_once()

    def test_search_calls_query_points(self):
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.query_points.return_value = MagicMock(points=[])
        service = self._make_service(mock_client)
        service.search(vector=[0.1, 0.2], using='text_embedding', limit=5)
        mock_client.query_points.assert_called_once_with(
            collection_name='robot_memories',
            query=[0.1, 0.2],
            using='text_embedding',
            limit=5,
            with_payload=True,
        )


class TestSearchService:
    def _make_service(self):
        from qdrant.services.search_service import SearchService
        from unittest.mock import MagicMock
        qdrant_service = MagicMock()
        embedding_service = MagicMock()
        logger = MagicMock()
        return SearchService(qdrant_service, embedding_service, logger)

    def test_parse_time_string_hms_returns_float(self):
        service = self._make_service()
        result = service._parse_time_string('00:00:00')
        assert isinstance(result, float)

    def test_parse_time_string_invalid_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match='HH:MM:SS'):
            service._parse_time_string('invalid')

    def test_parse_time_string_full_datetime(self):
        service = self._make_service()
        result = service._parse_time_string('07/24/2024 00:00:00')
        assert isinstance(result, float)

    def test_process_search_results_empty(self):
        service = self._make_service()
        assert service._process_search_results([]) == []

    def test_process_search_results_extracts_payload(self):
        from unittest.mock import MagicMock
        service = self._make_service()
        hit = MagicMock()
        hit.payload = {
            'caption': 'robot sees door',
            'position': [1.0, 2.0, 3.0],
            'theta': 1.5,
            'time': [200.0, 0.0],
        }
        hit.score = 0.85
        results = service._process_search_results([hit])
        assert len(results) == 1
        assert results[0]['text'] == 'robot sees door'
        assert results[0]['position'] == [1.0, 2.0, 3.0]
        assert results[0]['orientation'] == 1.5
        assert results[0]['time'] == 200.0
        assert results[0]['distance'] == 0.85

    def test_search_by_text_calls_encode_query(self):
        from unittest.mock import MagicMock
        from qdrant.services.search_service import SearchService
        qdrant_service = MagicMock()
        qdrant_service.search.return_value = []
        embedding_service = MagicMock()
        embedding_service.encode_query.return_value = [0.1] * 4096
        service = SearchService(qdrant_service, embedding_service, MagicMock())
        service.search_by_text('where is the kitchen?')
        embedding_service.encode_query.assert_called_once_with('where is the kitchen?')

    def test_search_by_position_wrong_dim_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match='3D'):
            service.search_by_position((1.0, 2.0))

    def test_format_results_empty(self):
        service = self._make_service()
        result = service.format_results([], 'test query')
        assert 'No memories found' in result

    def test_format_results_no_score(self):
        service = self._make_service()
        doc = {
            'text': 'robot sees the kitchen',
            'position': [1.0, 2.0, 0.0],
            'orientation': 0.5,
            'time': 100.0,
            'distance': 0.9876,
        }
        result = service.format_results([doc], 'test')
        assert 'relevance_score' not in result
        assert '0.9876' not in result
        assert 'POSITION' in result
        assert 'DESCRIPTION' in result
        assert 'robot sees the kitchen' in result

    def test_format_results_includes_required_fields(self):
        service = self._make_service()
        doc = {
            'text': 'a red door',
            'position': [3.14, 2.71, 0.0],
            'orientation': 1.57,
            'time': 0.0,
            'distance': 0.5,
        }
        result = service.format_results([doc])
        assert '[Result 1]' in result
        assert 'POSITION' in result
        assert 'ORIENTATION' in result
        assert 'TIME' in result
        assert 'DESCRIPTION' in result


class TestDataPipeline:
    def test_process_ros_message_returns_record(self):
        from qdrant.services.data_pipeline import DataPipeline
        from qdrant.models.qdrant_record import QdrantMemoryRecord
        from unittest.mock import MagicMock

        embedding_service = MagicMock()
        embedding_service.encode_document.return_value = [0.1] * 4096

        pipeline = DataPipeline(embedding_service)

        msg = MagicMock()
        msg.caption = 'robot in kitchen'
        msg.position_x = 1.0
        msg.position_y = 2.0
        msg.position_z = 0.0
        msg.theta = 0.5
        msg.timestamp.sec = 1721761100
        msg.timestamp.nanosec = 0

        record = pipeline.process_ros_message(msg)

        assert isinstance(record, QdrantMemoryRecord)
        assert record.caption == 'robot in kitchen'
        assert len(record.id) == 36
        assert len(record.text_embedding) == 4096
        assert record.position == [1.0, 2.0, 0.0]
        assert record.time[1] == 0.0
        embedding_service.encode_document.assert_called_once_with('robot in kitchen')
