from typing import Literal

from mcp.server.fastmcp import FastMCP

from .ros_publisher import RosPublisher

mcp = FastMCP('remembr-mcp')

_search_service = None
_ros_publisher: RosPublisher | None = None


def init_services(search_service, ros_publisher: RosPublisher) -> None:
    """Inject initialized services. Called from __main__ before mcp.run()."""
    global _search_service, _ros_publisher
    _search_service = search_service
    _ros_publisher = ros_publisher


@mcp.tool()
def retrieve_from_text(query: str) -> str:
    """Search video memory by text description. Do NOT use for location or time queries."""
    results = _search_service.search_by_text(query)
    return _search_service.format_results(results, f'text: {query}')


@mcp.tool()
def retrieve_from_position(x: float, y: float, z: float) -> str:
    """Search video memory near an (x,y,z) position in meters."""
    results = _search_service.search_by_position((x, y, z))
    return _search_service.format_results(results, f'position: ({x},{y},{z})')


@mcp.tool()
def retrieve_from_time(time_str: str) -> str:
    """Search video memory near a specific time in H:M:S format (e.g. 08:02:03)."""
    results = _search_service.search_by_time(time_str)
    return _search_service.format_results(results, f'time: {time_str}')


@mcp.tool()
def submit_result(
    type: Literal['position', 'binary', 'time', 'text'],
    type_reasoning: str,
    answer_reasoning: str,
    text: str,
    binary: str | None = None,
    position: list[float] | None = None,
    orientation: float | None = None,
    time: float | None = None,
    duration: float | None = None,
) -> str:
    """Submit the final structured answer. MUST be called to complete a query."""
    result = {
        'type': type,
        'type_reasoning': type_reasoning,
        'answer_reasoning': answer_reasoning,
        'text': text,
        'binary': binary,
        'position': position,
        'orientation': orientation,
        'time': time,
        'duration': duration,
    }
    _ros_publisher.publish(result)
    return '완료'
