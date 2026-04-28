import os
import pytest
from remembr_mcp.config import load_config, SimpleLogger


def test_load_config_defaults():
    config = load_config()
    assert config.qdrant_url == 'http://localhost:6333'
    assert config.collection_name == 'robot_memories'
    assert config.embedding_model == 'mixedbread-ai/mxbai-embed-large-v1'


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv('QDRANT_URL', 'http://myhost:9999')
    monkeypatch.setenv('QDRANT_COLLECTION', 'test_col')
    monkeypatch.setenv('EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3')

    config = load_config()

    assert config.qdrant_url == 'http://myhost:9999'
    assert config.collection_name == 'test_col'
    assert config.embedding_model == 'jinaai/jina-embeddings-v3'


def test_simple_logger_has_required_methods():
    logger = SimpleLogger()
    assert hasattr(logger, 'info')
    assert hasattr(logger, 'error')
    assert hasattr(logger, 'warning')
    logger.info('test info')
    logger.error('test error')
    logger.warning('test warning')
