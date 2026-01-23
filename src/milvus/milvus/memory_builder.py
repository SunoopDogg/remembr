import argparse
import sys
import traceback
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from vila_msgs.msg import CaptionWithPose

from .config import DatabaseConfig
from .services import MilvusService, EmbeddingService, DataPipeline, SearchService


class MemoryBuilder(Node):
    """ROS2 node for storing VILA captions in Milvus vector database."""

    def __init__(self, reset_db: bool = False) -> None:
        super().__init__('memory_builder')

        self.get_logger().info('Initializing Memory Builder node...')

        self.caption_count = 0

        # Load configuration from ROS2 parameters
        self._config = DatabaseConfig.from_ros_node(self)
        self._config.validate()
        self._config.ensure_db_dir()
        self.get_logger().info(f'Database path: {self._config.db_path}')

        # Initialize services
        self._milvus_service = MilvusService(self._config, self.get_logger())
        self._embedding_service = EmbeddingService(
            self._config.embedding_model,
            self.get_logger(),
        )
        self._data_pipeline = DataPipeline(self._embedding_service)
        self._search_service = None

        self._initialize_services(reset_db)

        # Subscribe to caption with pose topic
        self._caption_subscription = self.create_subscription(
            CaptionWithPose,
            self._config.input_topic,
            self.caption_callback,
            10,
        )

        self.get_logger().info('Memory Builder node initialized successfully')
        self.get_logger().info(f'Waiting for caption data on {self._config.input_topic}...')

    def _log_error(self, context: str, error: Exception, reraise: bool = True) -> None:
        """Log error with traceback."""
        self.get_logger().error(f'{context}: {error}')
        self.get_logger().error(traceback.format_exc())
        if reraise:
            raise

    def _initialize_services(self, reset_db: bool) -> None:
        """Initialize all services."""
        try:
            self._milvus_service.connect()

            if reset_db:
                self._milvus_service.reset_database()
            else:
                self._milvus_service.setup_collection()

            self.get_logger().info(
                f'Milvus collection "{self._config.collection_name}" ready')

            self._embedding_service.load_model()

            self._search_service = SearchService(
                self._milvus_service,
                self._embedding_service,
                self.get_logger(),
            )
            self.get_logger().info('Search service initialized')

        except Exception as e:
            self._log_error('Service initialization failed', e)

    def caption_callback(self, msg: CaptionWithPose) -> None:
        """Process incoming caption with pose message."""
        try:
            self.caption_count += 1

            # Concise info logging
            self.get_logger().info(
                f'Caption #{self.caption_count}: "{msg.caption[:30]}..." '
                f'at ({msg.position_x:.1f}, {msg.position_y:.1f})'
            )

            # Detailed debug logging (only if debug level enabled)
            self.get_logger().debug('Detailed caption data:')
            self.get_logger().debug(f'  Full caption: "{msg.caption}"')
            self.get_logger().debug(
                f'  Position: ({msg.position_x:.3f}, {msg.position_y:.3f}, {msg.position_z:.3f})')
            self.get_logger().debug(f'  Orientation (theta): {msg.theta:.3f} rad')
            self.get_logger().debug(f'  Timestamp: {msg.timestamp.sec}.{msg.timestamp.nanosec}')
            self.get_logger().debug(f'  Image count: {msg.image_count}')

            # Store to Milvus
            self._store_to_milvus(msg)

        except Exception as e:
            self._log_error('Error in caption callback', e, reraise=False)

    def _store_to_milvus(self, msg: CaptionWithPose) -> None:
        """Store caption data to Milvus vector database."""
        try:
            record = self._data_pipeline.process_ros_message(msg)
            result = self._milvus_service.insert_record(record)

            self.get_logger().info(
                f'Stored to Milvus: "{record.caption_text[:50]}..." '
                f'(ID: {result["ids"][0] if result.get("ids") else "N/A"})'
            )

        except Exception as e:
            self._log_error('Failed to store to Milvus', e, reraise=False)

    def search_by_text(self, query: str) -> str:
        """Search memories by text using vector similarity."""
        try:
            results = self._search_service.search_by_text(query, limit=5)
            return self._search_service.format_results(results, f"text: '{query}'")

        except Exception as e:
            self.get_logger().error(f'Error in search_by_text: {e}')
            return f"Error searching by text: {str(e)}"

    def search_by_position(self, position: Tuple) -> str:
        """Search memories by spatial position."""
        try:
            results = self._search_service.search_by_position(position, limit=5)
            return self._search_service.format_results(results, f"position: {position}")

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            self.get_logger().error(f'Error in search_by_position: {e}')
            return f"Error searching by position: {str(e)}"

    def search_by_time(self, time_str: str) -> str:
        """Search memories by time in HH:MM:SS format."""
        try:
            results = self._search_service.search_by_time(time_str, limit=5)
            return self._search_service.format_results(results, f"time: {time_str}")

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            self.get_logger().error(f'Error in search_by_time: {e}')
            return f"Error searching by time: {str(e)}"

    def destroy_node(self) -> None:
        """Clean up resources before node shutdown."""
        self.get_logger().info('Cleaning up resources...')

        if hasattr(self, '_milvus_service'):
            try:
                self._milvus_service.close()
                self.get_logger().info('Milvus service closed')
            except Exception as e:
                self.get_logger().warning(f'Error closing Milvus service: {e}')

        if hasattr(self, '_embedding_service'):
            try:
                self._embedding_service.cleanup()
                self.get_logger().info('Embedding service cleaned up')
            except Exception as e:
                self.get_logger().warning(f'Error cleaning up embedding service: {e}')

        if hasattr(self, '_data_pipeline'):
            del self._data_pipeline

        super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    """Main entry point for the memory_builder node."""
    parser = argparse.ArgumentParser(
        description='Memory Builder Node - Stores robot captions in Milvus')
    parser.add_argument('--reset', action='store_true',
                        help='Reset database on startup')

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
