import json
import traceback
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from milvus.config import DatabaseConfig
from milvus.services import MilvusService, EmbeddingService, SearchService
from agent_msgs.srv import Query

from .config import AgentConfig
from .services import ReMEmbRAgent


class ReMEmbRAgentNode(Node):
    """ROS2 node for ReMEmbR Agent query interface."""

    def __init__(self) -> None:
        super().__init__('remembr_agent')

        self.get_logger().info('Initializing ReMEmbR Agent node...')

        self._query_count = 0

        # Load configuration from ROS2 parameters
        self._config = AgentConfig.from_ros_node(self)
        self._db_config = DatabaseConfig.from_ros_node(self)
        self._db_config.validate()

        self.get_logger().info(f'Agent model: {self._config.model}')
        self.get_logger().info(f'Database path: {self._db_config.db_path}')

        # Initialize services
        self._milvus_service = None
        self._embedding_service = None
        self._search_service = None
        self._agent = None

        self._initialize_services()
        self._setup_ros_interfaces()

        self.get_logger().info('ReMEmbR Agent node initialized successfully')

    def _log_error(self, context: str, error: Exception, reraise: bool = True) -> None:
        """Log error with traceback."""
        self.get_logger().error(f'{context}: {error}')
        self.get_logger().error(traceback.format_exc())
        if reraise:
            raise

    def _initialize_services(self) -> None:
        """Initialize all services."""
        try:
            # Initialize Milvus
            self._milvus_service = MilvusService(self._db_config, self.get_logger())
            self._milvus_service.connect()
            self._milvus_service.setup_collection()
            self.get_logger().info(
                f'Milvus collection "{self._db_config.collection_name}" ready')

            # Initialize embedding service
            self._embedding_service = EmbeddingService(
                self._db_config.embedding_model,
                self.get_logger(),
            )
            self._embedding_service.load_model()
            self.get_logger().info('Embedding service ready')

            # Initialize search service
            self._search_service = SearchService(
                self._milvus_service,
                self._embedding_service,
                self.get_logger(),
            )
            self.get_logger().info('Search service ready')

            # Initialize agent
            self._agent = ReMEmbRAgent(self._config, self.get_logger())
            self._agent.set_search_service(self._search_service)
            self.get_logger().info('Agent ready')

        except Exception as e:
            self._log_error('Service initialization failed', e)

    def _setup_ros_interfaces(self) -> None:
        """Setup ROS2 service and topic interfaces."""
        # Query service
        self._query_service = self.create_service(
            Query,
            '~/query',
            self._query_service_callback,
        )
        self.get_logger().info(f'Query service: {self.get_name()}/query')

        # Status service
        self._status_service = self.create_service(
            Trigger,
            '~/status',
            self._status_callback,
        )
        self.get_logger().info(f'Status service: {self.get_name()}/status')

        # Topic-based query interface
        self._query_subscriber = self.create_subscription(
            String,
            '~/query_topic',
            self._query_topic_callback,
            10,
        )
        self._response_publisher = self.create_publisher(
            String,
            '~/response_topic',
            10,
        )
        self.get_logger().info(f'Query topic: {self.get_name()}/query_topic')
        self.get_logger().info(f'Response topic: {self.get_name()}/response_topic')

    def _status_callback(self, request, response):
        """Handle status service request."""
        response.success = self._agent is not None
        response.message = (
            f'Agent ready. Queries processed: {self._query_count}. '
            f'Model: {self._config.model}'
        )
        return response

    def _query_service_callback(self, request, response):
        """Handle query service request."""
        result = self._execute_query(request.question)

        response.success = 'error' not in result
        response.type = result.get('type') or ''
        response.text = result.get('text') or ''
        response.binary = result.get('binary') or ''
        response.position = list(result.get('position') or [])
        response.orientation = float(result.get('orientation') or 0.0)
        response.time = float(result.get('time') or 0.0)
        response.duration = float(result.get('duration') or 0.0)
        response.error = result.get('error', '')

        return response

    def _query_topic_callback(self, msg: String) -> None:
        """Handle query from topic."""
        result = self._execute_query(msg.data)

        # Publish response as JSON
        response_msg = String()
        response_msg.data = json.dumps(result, default=str)
        self._response_publisher.publish(response_msg)

    def _execute_query(self, question: str) -> dict:
        """Execute a query against the agent."""
        if self._agent is None:
            self.get_logger().error('Agent not initialized')
            return {'error': 'Agent not initialized'}

        try:
            self._query_count += 1
            self.get_logger().info(f'Query #{self._query_count}: "{question[:50]}..."')

            result = self._agent.query(question)

            self.get_logger().info(
                f'Query #{self._query_count} completed: '
                f'"{result.text[:30] if result.text else "N/A"}..."'
            )

            return {
                'type': result.type,
                'text': result.text,
                'binary': result.binary,
                'position': result.position,
                'orientation': result.orientation,
                'time': result.time,
                'duration': result.duration,
            }

        except Exception as e:
            self._log_error(f'Query #{self._query_count} failed', e, reraise=False)
            return {'error': str(e)}

    def destroy_node(self) -> None:
        """Clean up resources before node shutdown."""
        self.get_logger().info('Cleaning up resources...')

        cleanup_tasks = [
            (self._milvus_service, 'close', 'Milvus service'),
            (self._embedding_service, 'cleanup', 'Embedding service'),
        ]

        for service, method, name in cleanup_tasks:
            if service:
                try:
                    getattr(service, method)()
                    self.get_logger().info(f'{name} closed')
                except Exception as e:
                    self.get_logger().warning(f'Error closing {name}: {e}')

        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for the remembr_agent node."""
    rclpy.init(args=args)

    agent_node = None
    try:
        agent_node = ReMEmbRAgentNode()
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        if agent_node:
            agent_node.get_logger().info('Keyboard interrupt, shutting down...')
    except Exception as e:
        if agent_node:
            agent_node.get_logger().error(f'Unexpected error: {e}')
            agent_node.get_logger().error(traceback.format_exc())
        raise
    finally:
        if agent_node:
            agent_node.get_logger().info(
                f'Total queries processed: {agent_node._query_count}')
            agent_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
