from typing import Protocol, TypedDict


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...


class SearchResult(TypedDict):
    """Search result from SearchService."""

    text: str
    position: list[float]
    orientation: float
    time: float
    distance: float


class SearchService(Protocol):
    """Protocol for search service interface."""

    def search_by_text(self, query: str, limit: int = 5) -> list[SearchResult]: ...
    def search_by_position(self, position: tuple, limit: int = 5) -> list[SearchResult]: ...
    def search_by_time(self, time_str: str, limit: int = 5) -> list[SearchResult]: ...
    def format_results(self, results: list[SearchResult], query_info: str = "") -> str: ...
