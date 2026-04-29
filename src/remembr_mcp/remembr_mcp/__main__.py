import logging
import rclpy

from qdrant.services import EmbeddingService, QdrantService, SearchService

from .config import SimpleLogger, load_config
from .ros_publisher import RosPublisher
from .server import init_services, mcp

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = load_config()
    logger = SimpleLogger()

    rclpy.init()
    ros_publisher = RosPublisher()

    qdrant_svc = QdrantService(config, logger)
    qdrant_svc.connect()

    embed_svc = EmbeddingService(config.embedding_model, config.embedding_url, logger)
    embed_svc.load_model()

    search_svc = SearchService(qdrant_svc, embed_svc, logger)
    logger.info('All services initialized')

    init_services(search_svc, ros_publisher)

    try:
        mcp.run()
    finally:
        embed_svc.cleanup()
        ros_publisher.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
