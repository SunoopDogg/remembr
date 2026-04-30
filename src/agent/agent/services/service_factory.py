from ..models.protocols import Logger


def create_services(db_config, logger: Logger) -> tuple:
    from qdrant.services import QdrantService, EmbeddingService, SearchService

    embedding = EmbeddingService(
        db_config.embedding_model,
        db_config.embedding_url,
        logger,
    )
    embedding.load_model()
    embedding_dim = embedding.embedding_dimension
    logger.info(f"Embedding service ready (dim={embedding_dim})")

    qdrant = QdrantService(db_config, logger, embedding_dim=embedding_dim)
    qdrant.connect()
    qdrant.setup_collection()
    logger.info(f'Qdrant collection "{db_config.collection_name}" ready')

    search = SearchService(qdrant, embedding, logger)
    logger.info("Search service ready")

    return qdrant, embedding, search


def cleanup_services(qdrant_service, embedding_service, logger: Logger) -> None:
    if qdrant_service:
        try:
            qdrant_service.close()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to close Qdrant service: {e}")
    if embedding_service:
        try:
            embedding_service.cleanup()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup embedding service: {e}")
