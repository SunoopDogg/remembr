import ast
import json
import os
from typing import Protocol

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import AgentOutput, AgentState
from ..utils import file_to_string, parse_json


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...


class SearchService(Protocol):
    """Protocol for search service interface."""

    def search_by_text(self, query: str, limit: int = 5) -> list[dict]: ...
    def search_by_position(self, position: tuple, limit: int = 5) -> list[dict]: ...
    def search_by_time(self, time_str: str, limit: int = 5) -> list[dict]: ...
    def format_results(self, results: list[dict], query_info: str = "") -> str: ...


class ReMEmbRAgent:
    """ReMEmbR Agent using Ollama LLM with LangGraph workflow."""

    def __init__(self, config: AgentConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        # Initialize LLM
        self.llm = ChatOllama(
            model=self.config.model,
            format="json",
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx
        )

        # Load prompts
        prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        self.agent_prompt = file_to_string(f'{prompt_dir}/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(f'{prompt_dir}/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(f'{prompt_dir}/agent_gen_system_prompt.txt')

        # Agent state
        self._reset_state()

        # Services
        self.search_service = None
        self.graph = None

        self.logger.info(f'ReMEmbRAgent initialized with model: {self.config.model}')

    def _reset_state(self) -> None:
        """Reset agent state for new query."""
        self.tool_history = "Previously used tools:\n"
        self.iteration_count = 0

    def set_search_service(self, search_service: SearchService) -> None:
        """Set search service and build agent graph."""
        self.search_service = search_service
        self._create_tools()
        self._build_graph()
        self.logger.info('Search service connected, agent graph built')

    def _create_tools(self) -> None:
        """Create retrieval tools using SearchService."""

        class TextSearchInput(BaseModel):
            query: str = Field(
                description="Query for vector similarity search. Should be a descriptive phrase "
                "like 'a crowd gathering' or 'a green car driving down the road'."
            )

        class PositionSearchInput(BaseModel):
            position: tuple = Field(
                description="Position query as (x,y,z) tuple with float values. "
                "Example: (0.5, 0.2, 0.1)"
            )

        class TimeSearchInput(BaseModel):
            time_str: str = Field(
                description="Time query in H:M:S format with leading zeros. "
                "Example: 08:02:03"
            )

        self.text_search_tool = StructuredTool.from_function(
            func=lambda query: self.search_service.format_results(
                self.search_service.search_by_text(query), f"text: {query}"
            ),
            name="retrieve_from_text",
            description="Search video memory by text description using vector similarity",
            args_schema=TextSearchInput
        )

        self.position_search_tool = StructuredTool.from_function(
            func=lambda position: self.search_service.format_results(
                self.search_service.search_by_position(position), f"position: {position}"
            ),
            name="retrieve_from_position",
            description="Search video memory by (x,y,z) position",
            args_schema=PositionSearchInput
        )

        self.time_search_tool = StructuredTool.from_function(
            func=lambda time_str: self.search_service.format_results(
                self.search_service.search_by_time(time_str), f"time: {time_str}"
            ),
            name="retrieve_from_time",
            description="Search video memory by H:M:S time",
            args_schema=TimeSearchInput
        )

        self.tools = [
            self.text_search_tool,
            self.position_search_tool,
            self.time_search_tool
        ]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tools]

    def _build_graph(self) -> None:
        """Build LangGraph StateGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("action", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "action", "end": "generate"}
        )
        workflow.add_edge("action", "agent")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue tool execution."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    def _agent_node(self, state: AgentState) -> dict:
        """Agent node - decides to use tools or generate response."""
        messages = state["messages"]

        if self.iteration_count < self.config.max_tool_calls:
            model = self.llm.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            model = self.llm
            prompt = self.agent_gen_only_prompt

        agent_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", self.tool_history),
            ("ai", prompt),
            ("human", "{question}"),
        ])

        chain = agent_prompt | model
        question = f"The question is: {messages[0]}"

        # Convert ToolMessages to AIMessages for chat history compatibility
        chat_history = [
            AIMessage(id=msg.id, content=msg.content) if isinstance(msg, ToolMessage) else msg
            for msg in messages
        ]

        response = chain.invoke({"question": question, "chat_history": chat_history})

        # Record tool usage for context
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    self.tool_history += f"Used {tool_call['name']}.\n"

        self.iteration_count += 1
        return {"messages": [response]}

    def _generate_node(self, state: AgentState) -> dict:
        """Generate final structured response."""
        messages = state["messages"]
        question = messages[0].content + "\n Please respond in the desired format."

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question': question})

        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", filled_prompt.text),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        chain = gen_prompt | self.llm
        response = chain.invoke({"question": question, "chat_history": messages[1:]})

        parsed = parse_json(response.content)
        response_data = self._extract_response(parsed)
        self._normalize_and_validate(response_data)

        # Store result for retrieval in query()
        self._last_result = response_data
        self._reset_state()

        return {"messages": [AIMessage(content=json.dumps(response_data))]}

    def _extract_response(self, parsed: dict) -> dict:
        """Extract response data from nested LLM output."""
        # Navigate through possible nesting: tool_input.response or response
        if "tool_input" in parsed:
            parsed = parsed["tool_input"]
        if "response" in parsed:
            return parsed["response"]
        return parsed

    def _normalize_and_validate(self, data: dict) -> None:
        """Normalize data types and validate required fields."""
        required_keys = ["time", "text", "binary", "position", "duration"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Parse string literals to Python objects
        if isinstance(data.get('position'), str):
            data['position'] = ast.literal_eval(data['position'])
        if isinstance(data.get('orientation'), str):
            data['orientation'] = ast.literal_eval(data['orientation'])

        if data['position'] is not None and len(data['position']) != 3:
            raise ValueError(f"Invalid position shape: {data['position']}")

    def query(self, question: str) -> AgentOutput:
        """Execute a query against the agent."""
        if self.graph is None:
            raise RuntimeError("Agent not initialized. Call set_search_service first.")

        self.logger.info(f'Processing query: "{question[:50]}..."')

        inputs = {"messages": [("user", question)]}
        self.graph.invoke(inputs)

        # Use stored result from _generate_node to avoid double JSON parsing
        result = AgentOutput.from_dict(self._last_result)
        self.logger.info(f'Query completed: {result.text[:50] if result.text else "N/A"}...')

        return result
