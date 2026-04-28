import json
from unittest.mock import MagicMock, patch

# conftest.py가 rclpy mock을 먼저 등록하므로 여기서는 바로 import 가능
from remembr_mcp.ros_publisher import RosPublisher


def _make_publisher():
    """rclpy.create_node와 ROS2 메시지를 모킹하여 RosPublisher를 생성한다."""
    mock_node = MagicMock()
    mock_goal_pub = MagicMock()
    mock_response_pub = MagicMock()
    mock_node.create_publisher.side_effect = [mock_goal_pub, mock_response_pub]

    with patch('rclpy.create_node', return_value=mock_node):
        pub = RosPublisher()

    pub._goal_pub = mock_goal_pub
    pub._response_pub = mock_response_pub
    return pub


def test_publish_response_always_called():
    pub = _make_publisher()
    result = {'type': 'text', 'text': 'hello', 'position': None,
              'orientation': None, 'time': None, 'duration': None, 'binary': None}
    pub.publish(result)
    pub._response_pub.publish.assert_called_once()


def test_goal_pose_not_published_without_position():
    pub = _make_publisher()
    result = {'type': 'text', 'text': 'hello', 'position': None,
              'orientation': None, 'time': None, 'duration': None, 'binary': None}
    pub.publish(result)
    pub._goal_pub.publish.assert_not_called()


def test_goal_pose_published_with_position():
    pub = _make_publisher()
    result = {'type': 'position', 'text': 'sofa here',
              'position': [1.0, 2.0, 0.0], 'orientation': 0.5,
              'time': None, 'duration': None, 'binary': None}
    pub.publish(result)
    pub._goal_pub.publish.assert_called_once()


def test_response_topic_json_contains_result_fields():
    pub = _make_publisher()
    result = {'type': 'text', 'text': 'answer', 'position': None,
              'orientation': None, 'time': 3.5, 'duration': None, 'binary': None}

    with patch('remembr_mcp.ros_publisher.String') as MockString:
        instance = MagicMock()
        MockString.return_value = instance
        pub.publish(result)

    assert pub._response_pub.publish.called
    pub._response_pub.publish.assert_called_once_with(instance)
