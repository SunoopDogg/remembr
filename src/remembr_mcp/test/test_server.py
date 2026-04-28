import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def reset_server_globals():
    """각 테스트 전후로 server 전역 상태를 초기화한다."""
    import remembr_mcp.server as srv
    srv._search_service = None
    srv._ros_publisher = None
    yield
    srv._search_service = None
    srv._ros_publisher = None


def _inject(search_svc=None, ros_pub=None):
    import remembr_mcp.server as srv
    srv._search_service = search_svc or MagicMock()
    srv._ros_publisher = ros_pub or MagicMock()


# --- retrieve_from_text ---

def test_retrieve_from_text_calls_search_by_text():
    mock_search = MagicMock()
    mock_search.search_by_text.return_value = []
    mock_search.format_results.return_value = 'No memories found for query: text: sofa'
    _inject(search_svc=mock_search)

    from remembr_mcp.server import retrieve_from_text
    result = retrieve_from_text('sofa')

    mock_search.search_by_text.assert_called_once_with('sofa')
    mock_search.format_results.assert_called_once_with([], 'text: sofa')
    assert 'No memories found' in result


def test_retrieve_from_text_returns_formatted_results():
    mock_search = MagicMock()
    mock_search.search_by_text.return_value = [{'text': 'saw a sofa'}]
    mock_search.format_results.return_value = '[Result 1] DESCRIPTION: saw a sofa'
    _inject(search_svc=mock_search)

    from remembr_mcp.server import retrieve_from_text
    result = retrieve_from_text('sofa')

    assert '[Result 1]' in result


# --- retrieve_from_position ---

def test_retrieve_from_position_passes_tuple():
    mock_search = MagicMock()
    mock_search.search_by_position.return_value = []
    mock_search.format_results.return_value = 'No memories found for query: position: (1.0,2.0,0.0)'
    _inject(search_svc=mock_search)

    from remembr_mcp.server import retrieve_from_position
    retrieve_from_position(1.0, 2.0, 0.0)

    mock_search.search_by_position.assert_called_once_with((1.0, 2.0, 0.0))


# --- retrieve_from_time ---

def test_retrieve_from_time_passes_time_string():
    mock_search = MagicMock()
    mock_search.search_by_time.return_value = []
    mock_search.format_results.return_value = 'No memories found for query: time: 08:02:03'
    _inject(search_svc=mock_search)

    from remembr_mcp.server import retrieve_from_time
    retrieve_from_time('08:02:03')

    mock_search.search_by_time.assert_called_once_with('08:02:03')


# --- submit_result ---

def test_submit_result_returns_done():
    mock_pub = MagicMock()
    _inject(ros_pub=mock_pub)

    from remembr_mcp.server import submit_result
    result = submit_result(
        type='text',
        type_reasoning='descriptive question',
        answer_reasoning='found relevant info',
        text='The sofa is in the living room',
    )

    assert result == '완료'


def test_submit_result_calls_publisher_with_correct_fields():
    mock_pub = MagicMock()
    _inject(ros_pub=mock_pub)

    from remembr_mcp.server import submit_result
    submit_result(
        type='position',
        type_reasoning='where question',
        answer_reasoning='found at [0.78, -0.41, 0.0]',
        text='The sofa is in the living room',
        position=[0.78, -0.41, 0.0],
        orientation=0.0,
    )

    mock_pub.publish.assert_called_once()
    payload = mock_pub.publish.call_args[0][0]
    assert payload['type'] == 'position'
    assert payload['type_reasoning'] == 'where question'
    assert payload['answer_reasoning'] == 'found at [0.78, -0.41, 0.0]'
    assert payload['position'] == [0.78, -0.41, 0.0]
    assert payload['orientation'] == 0.0
    assert payload['text'] == 'The sofa is in the living room'
    assert payload['binary'] is None
    assert payload['time'] is None
    assert payload['duration'] is None


def test_submit_result_binary_question():
    mock_pub = MagicMock()
    _inject(ros_pub=mock_pub)

    from remembr_mcp.server import submit_result
    submit_result(
        type='binary',
        type_reasoning='yes/no question',
        answer_reasoning='found refrigerator in kitchen',
        text='Yes, there is a refrigerator in the kitchen',
        binary='yes',
    )

    payload = mock_pub.publish.call_args[0][0]
    assert payload['binary'] == 'yes'
    assert payload['position'] is None


def test_init_services_injects_dependencies():
    mock_search = MagicMock()
    mock_pub = MagicMock()

    import remembr_mcp.server as srv
    srv.init_services(mock_search, mock_pub)

    assert srv._search_service is mock_search
    assert srv._ros_publisher is mock_pub
