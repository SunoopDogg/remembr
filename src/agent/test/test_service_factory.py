import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCleanupServices:
    def test_cleanup_calls_close_and_cleanup(self):
        from unittest.mock import MagicMock
        from agent.services.service_factory import cleanup_services

        qdrant_service = MagicMock()
        embedding_service = MagicMock()
        logger = MagicMock()

        cleanup_services(qdrant_service, embedding_service, logger)

        qdrant_service.close.assert_called_once()
        embedding_service.cleanup.assert_called_once()

    def test_cleanup_handles_none_services(self):
        from agent.services.service_factory import cleanup_services
        from unittest.mock import MagicMock

        cleanup_services(None, None, MagicMock())

    def test_cleanup_handles_close_exception(self):
        from unittest.mock import MagicMock
        from agent.services.service_factory import cleanup_services

        qdrant_service = MagicMock()
        qdrant_service.close.side_effect = RuntimeError('connection lost')
        logger = MagicMock()

        cleanup_services(qdrant_service, None, logger)
        logger.warning.assert_called_once()


class TestCreateServicesUsesQdrant:
    def test_imports_qdrant_not_milvus(self):
        import inspect
        from agent.services import service_factory
        source = inspect.getsource(service_factory)
        assert 'from qdrant.services import' in source
        assert 'from milvus.services import' not in source

    def test_create_services_calls_embedding_with_url(self):
        from unittest.mock import MagicMock, patch
        from agent.services.service_factory import create_services

        db_config = MagicMock()
        db_config.embedding_model = 'Qwen/Qwen3-Embedding-4B'
        db_config.embedding_url = 'http://localhost:8080'
        db_config.collection_name = 'robot_memories'
        logger = MagicMock()

        mock_embed_instance = MagicMock()
        mock_embed_instance.embedding_dimension = 4096
        mock_qdrant_instance = MagicMock()
        mock_search_instance = MagicMock()

        MockQdrantService = MagicMock(return_value=mock_qdrant_instance)
        MockEmbeddingService = MagicMock(return_value=mock_embed_instance)
        MockSearchService = MagicMock(return_value=mock_search_instance)

        mock_qdrant_module = MagicMock()
        mock_qdrant_module.QdrantService = MockQdrantService
        mock_qdrant_module.EmbeddingService = MockEmbeddingService
        mock_qdrant_module.SearchService = MockSearchService

        with patch.dict(sys.modules, {'qdrant.services': mock_qdrant_module}):
            qdrant_svc, embed_svc, search_svc = create_services(db_config, logger)

        MockEmbeddingService.assert_called_once_with(
            'Qwen/Qwen3-Embedding-4B', 'http://localhost:8080', logger
        )
        assert qdrant_svc is mock_qdrant_instance
        assert embed_svc is mock_embed_instance
        assert search_svc is mock_search_instance
