import argparse
import os
import shutil
import sys
import traceback
from typing import Optional

import rclpy
from rclpy.node import Node

from memory_msgs.msg import CaptionWithPose

from .config import DatabaseConfig
from .models.caption_data import CaptionData
from .models.qdrant_record import QdrantMemoryRecord
from .services import QdrantService, EmbeddingService


class MemoryBuilder(Node):

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

        self._initialize_services(reset_db)

        self._caption_subscription = self.create_subscription(
            CaptionWithPose,
            self._config.input_topic,
            self.caption_callback,
            10,
        )

        self.get_logger().info('Memory Builder node initialized successfully')
        self.get_logger().info(f'Waiting for caption data on {self._config.input_topic}...')

    def _log_exception(self, context: str, exc: Exception) -> None:
        self.get_logger().error(f'{context}: {exc}')
        self.get_logger().error(traceback.format_exc())

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
                if os.path.exists(self._config.images_dir):
                    shutil.rmtree(self._config.images_dir)
                    self.get_logger().info(f'Removed images: {self._config.images_dir}')
            else:
                self._qdrant_service.setup_collection()

            self.get_logger().info(
                f'Qdrant collection "{self._config.collection_name}" ready '
                f'(embedding_dim={embedding_dim})')

        except Exception as e:
            self._log_exception('Service initialization failed', e)
            raise

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
            self._log_exception('Error in caption callback', e)

    def _store_to_qdrant(self, msg: CaptionWithPose) -> None:
        try:
            caption_data = CaptionData.from_ros_msg(msg)
            text_embedding = self._embedding_service.encode_document(caption_data.caption)
            record = QdrantMemoryRecord.from_caption_data(caption_data, text_embedding)
            self._qdrant_service.upsert_record(record)
            self._save_images(record.id, [img.data for img in msg.images])

            self.get_logger().info(
                f'Stored to Qdrant: "{record.caption[:50]}..." '
                f'(ID: {record.id})'
            )

        except Exception as e:
            self._log_exception('Failed to store to Qdrant', e)

    def _save_images(self, record_id: str, images: list) -> None:
        out_dir = os.path.join(self._config.images_dir, record_id)
        os.makedirs(out_dir, exist_ok=True)
        for i, data in enumerate(images):
            with open(os.path.join(out_dir, f'image_{i}.jpg'), 'wb') as f:
                f.write(data)
        self.get_logger().info(f'Saved {len(images)} images to {out_dir}')

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
