import argparse
import sys
import traceback
from typing import Optional

import rclpy
from rclpy.node import Node

from vila_msgs.msg import CaptionWithPose

from .config import DatabaseConfig
from .services import QdrantService, EmbeddingService, DataPipeline, SearchService


class MemoryBuilder(Node):
    """ROS2 node for storing VILA captions in Qdrant vector database."""

    def __init__(self, reset_db: bool = False) -> None:
        super().__init__('memory_builder')

        self.get_logger().info('Initializing Memory Builder node...')

        self.caption_count = 0

        self._config = DatabaseConfig.from_ros_node(self)
        self._config.validate()

        self.get_logger().info(f'Qdrant URL: {self._config.qdrant_url}')
        self.get_logger().info(f'Embedding server: {self._config.embedding_url}')
        self.get_logger().info(f'Embedding model: {self._config.embedding_model}')
        self.get_logger().info(f'Collection: {self._config.collection_name}')

        self._embedding_service = EmbeddingService(
            self._config.embedding_model,
            self._config.embedding_url,
            self.get_logger(),
        )
        self._qdrant_service = None
        self._data_pipeline = None
        self._search_service = None

        self._initialize_services(reset_db)

        self._caption_subscription = self.create_subscription(
            CaptionWithPose,
            self._config.input_topic,
            self.caption_callback,
            10,
        )

        self.get_logger().info('Memory Builder node initialized successfully')
        self.get_logger().info(f'Waiting for caption data on {self._config.input_topic}...')

    def _log_error(self, context: str, error: Exception, reraise: bool = True) -> None:
        self.get_logger().error(f'{context}: {error}')
        self.get_logger().error(traceback.format_exc())
        if reraise:
            raise

    def _initialize_services(self, reset_db: bool) -> None:
        try:
            self._embedding_service.load_model()
            embedding_dim = self._embedding_service.embedding_dimension

            self._qdrant_service = QdrantService(
                self._config,
                self.get_logger(),
                embedding_dim=embedding_dim,
            )
            self._qdrant_service.connect()

            if reset_db:
                self._qdrant_service.reset_database()
            else:
                self._qdrant_service.setup_collection()

            self.get_logger().info(
                f'Qdrant collection "{self._config.collection_name}" ready '
                f'(embedding_dim={embedding_dim})')

            self._data_pipeline = DataPipeline(self._embedding_service)

            self._search_service = SearchService(
                self._qdrant_service,
                self._embedding_service,
                self.get_logger(),
            )
            self.get_logger().info('Search service initialized')

        except Exception as e:
            self._log_error('Service initialization failed', e)

    def caption_callback(self, msg: CaptionWithPose) -> None:
        try:
            self.caption_count += 1

            self.get_logger().info(
                f'Caption #{self.caption_count}: "{msg.caption[:30]}..." '
                f'at ({msg.position_x:.1f}, {msg.position_y:.1f})'
            )

            self.get_logger().debug(f'  Full caption: "{msg.caption}"')
            self.get_logger().debug(
                f'  Position: ({msg.position_x:.3f}, {msg.position_y:.3f}, {msg.position_z:.3f})')
            self.get_logger().debug(f'  Orientation (theta): {msg.theta:.3f} rad')
            self.get_logger().debug(f'  Timestamp: {msg.timestamp.sec}.{msg.timestamp.nanosec}')

            self._store_to_qdrant(msg)

        except Exception as e:
            self._log_error('Error in caption callback', e, reraise=False)

    def _store_to_qdrant(self, msg: CaptionWithPose) -> None:
        try:
            record = self._data_pipeline.process_ros_message(msg)
            self._qdrant_service.upsert_record(record)

            self.get_logger().info(
                f'Stored to Qdrant: "{record.caption[:50]}..." '
                f'(ID: {record.id})'
            )

        except Exception as e:
            self._log_error('Failed to store to Qdrant', e, reraise=False)

    @property
    def search_service(self) -> SearchService:
        return self._search_service

    def destroy_node(self) -> None:
        self.get_logger().info('Cleaning up resources...')

        if self._qdrant_service is not None:
            try:
                self._qdrant_service.close()
                self.get_logger().info('Qdrant service closed')
            except Exception as e:
                self.get_logger().warning(f'Error closing Qdrant service: {e}')

        if self._embedding_service is not None:
            try:
                self._embedding_service.cleanup()
                self.get_logger().info('Embedding service cleaned up')
            except Exception as e:
                self.get_logger().warning(f'Error cleaning up embedding service: {e}')

        if self._data_pipeline is not None:
            del self._data_pipeline

        super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Memory Builder Node - Stores robot captions in Qdrant')
    parser.add_argument('--reset', action='store_true',
                        help='Reset collection for current embedding model')

    if args is None:
        args = sys.argv[1:]

    custom_args = [arg for arg in args if not arg.startswith('__')]
    parsed_args = parser.parse_args(custom_args)

    rclpy.init(args=args if args != sys.argv[1:] else None)

    memory_builder = None
    try:
        memory_builder = MemoryBuilder(reset_db=parsed_args.reset)
        rclpy.spin(memory_builder)
    except KeyboardInterrupt:
        if memory_builder:
            memory_builder.get_logger().info('Keyboard interrupt, shutting down...')
    except Exception as e:
        if memory_builder:
            memory_builder.get_logger().error(f'Unexpected error: {e}')
            memory_builder.get_logger().error(traceback.format_exc())
        raise
    finally:
        if memory_builder:
            memory_builder.get_logger().info(
                f'Total captions received: {memory_builder.caption_count}')
            memory_builder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
