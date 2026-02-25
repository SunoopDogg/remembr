from ..models.protocols import Logger


def create_services(db_config, logger: Logger) -> tuple:
    """Create and initialize Milvus, Embedding, and Search services.

    Returns (MilvusService, EmbeddingService, SearchService).
    """
    from milvus.services import MilvusService, EmbeddingService, SearchService

    # Load embedding model first to detect dimension
    embedding = EmbeddingService(
        db_config.embedding_model,
        logger,
        expected_dim=db_config.embedding_dim,
    )
    embedding.load_model()
    embedding_dim = embedding.embedding_dimension
    logger.info(f'Embedding service ready (dim={embedding_dim})')

    # Initialize Milvus with detected dimension
    milvus = MilvusService(db_config, logger, embedding_dim=embedding_dim)
    milvus.connect()
    milvus.setup_collection()
    logger.info(f'Milvus collection "{db_config.collection_name}" ready')

    search = SearchService(milvus, embedding, logger)
    logger.info('Search service ready')

    return milvus, embedding, search


def cleanup_services(milvus_service, embedding_service, logger: Logger) -> None:
    """Clean up services gracefully."""
    if milvus_service:
        try:
            milvus_service.close()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to close Milvus service: {e}")
    if embedding_service:
        try:
            embedding_service.cleanup()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup embedding service: {e}")
