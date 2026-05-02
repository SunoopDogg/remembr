from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..models.protocols import SearchService
from ..utils.ranking_utils import rerank_by_novelty


class TextSearchInput(BaseModel):
    query: str = Field(
        description="Query for vector similarity search. Should be a descriptive phrase "
        "like 'a crowd gathering' or 'a green car driving down the road'."
    )


class PositionSearchInput(BaseModel):
    position: tuple = Field(
        description="Position query as (x,y,z) tuple. Example: (0.5, 0.2, 0.1)"
    )


class TimeSearchInput(BaseModel):
    time_str: str = Field(description="Time query in H:M:S format. Example: 08:02:03")


def create_search_tools(
    search_service: SearchService,
    novelty_weight: float = 0.0,
) -> list[StructuredTool]:
    """Create search tools bound to a SearchService instance."""

    def _format(results, query_info):
        if novelty_weight > 0.0:
            results = rerank_by_novelty(results, novelty_weight)
        return search_service.format_results(results, query_info)

    text_search_tool = StructuredTool.from_function(
        func=lambda query: _format(
            search_service.search_by_text(query), f"text: {query}"
        ),
        name="retrieve_from_text",
        description="Search video memory by text description",
        args_schema=TextSearchInput,
    )

    position_search_tool = StructuredTool.from_function(
        func=lambda position: _format(
            search_service.search_by_position(position),
            f"position: {position}",
        ),
        name="retrieve_from_position",
        description="Search video memory by (x,y,z) position",
        args_schema=PositionSearchInput,
    )

    time_search_tool = StructuredTool.from_function(
        func=lambda time_str: _format(
            search_service.search_by_time(time_str), f"time: {time_str}"
        ),
        name="retrieve_from_time",
        description="Search video memory by H:M:S time",
        args_schema=TimeSearchInput,
    )

    return [text_search_tool, position_search_tool, time_search_tool]
