from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from ..models.protocols import SearchService


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
) -> tuple[list[StructuredTool], list[dict]]:
    """Create search tools bound to a SearchService instance."""
    text_search_tool = StructuredTool.from_function(
        func=lambda query: search_service.format_results(
            search_service.search_by_text(query), f"text: {query}"
        ),
        name="retrieve_from_text",
        description="Search video memory by text description",
        args_schema=TextSearchInput,
    )

    position_search_tool = StructuredTool.from_function(
        func=lambda position: search_service.format_results(
            search_service.search_by_position(position), f"position: {position}"
        ),
        name="retrieve_from_position",
        description="Search video memory by (x,y,z) position",
        args_schema=PositionSearchInput,
    )

    time_search_tool = StructuredTool.from_function(
        func=lambda time_str: search_service.format_results(
            search_service.search_by_time(time_str), f"time: {time_str}"
        ),
        name="retrieve_from_time",
        description="Search video memory by H:M:S time",
        args_schema=TimeSearchInput,
    )

    tools = [text_search_tool, position_search_tool, time_search_tool]
    tool_definitions = [convert_to_openai_function(t) for t in tools]

    return tools, tool_definitions
