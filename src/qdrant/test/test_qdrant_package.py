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
