import pytest
import sys
import os
import importlib.util


def _load_service_factory():
    """Load service_factory.py directly, bypassing package __init__.py."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'agent', 'services', 'service_factory.py'
    )
    # models.protocols is a dependency - load it too
    proto_path = os.path.join(
        os.path.dirname(__file__), '..', 'agent', 'models', 'protocols.py'
    )
    proto_spec = importlib.util.spec_from_file_location('agent.models.protocols', proto_path)
    proto_mod = importlib.util.module_from_spec(proto_spec)
    sys.modules['agent.models.protocols'] = proto_mod
    proto_spec.loader.exec_module(proto_mod)

    spec = importlib.util.spec_from_file_location('agent.services.service_factory', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['agent.services.service_factory'] = mod
    spec.loader.exec_module(mod)
    return mod


_sf = _load_service_factory()
create_services = _sf.create_services
cleanup_services = _sf.cleanup_services


class TestCleanupServices:
    def test_cleanup_calls_close_and_cleanup(self):
        from unittest.mock import MagicMock

        qdrant_service = MagicMock()
        embedding_service = MagicMock()
        logger = MagicMock()

        cleanup_services(qdrant_service, embedding_service, logger)

        qdrant_service.close.assert_called_once()
        embedding_service.cleanup.assert_called_once()

    def test_cleanup_handles_none_services(self):
        from unittest.mock import MagicMock
        cleanup_services(None, None, MagicMock())

    def test_cleanup_handles_close_exception(self):
        from unittest.mock import MagicMock

        qdrant_service = MagicMock()
        qdrant_service.close.side_effect = RuntimeError("connection lost")
        logger = MagicMock()

        cleanup_services(qdrant_service, None, logger)
        logger.warning.assert_called_once()


class TestCreateServicesUsesQdrant:
    def test_imports_qdrant_not_milvus(self):
        import inspect
        source = inspect.getsource(_sf)
        assert "from qdrant.services import" in source
        assert "from milvus.services import" not in source

    def test_create_services_calls_embedding_with_url(self):
        from unittest.mock import MagicMock, patch

        db_config = MagicMock()
        db_config.embedding_model = "Qwen/Qwen3-Embedding-4B"
        db_config.embedding_url = "http://localhost:8080"
        db_config.collection_name = "robot_memories"
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

        with patch.dict(sys.modules, {"qdrant.services": mock_qdrant_module}):
            qdrant_svc, embed_svc, search_svc = create_services(db_config, logger)

        MockEmbeddingService.assert_called_once_with(
            "Qwen/Qwen3-Embedding-4B", "http://localhost:8080", logger
        )
        assert qdrant_svc is mock_qdrant_instance
        assert embed_svc is mock_embed_instance
        assert search_svc is mock_search_instance
