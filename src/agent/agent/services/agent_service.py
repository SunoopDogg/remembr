import os
from functools import lru_cache

from langchain_ollama import ChatOllama

from ..config import AgentConfig
from ..models import AgentOutput
from ..models.protocols import Logger, SearchService
from ..tools import create_search_tools
from ..graph import GraphNodes, build_agent_graph
from ..utils import file_to_string
from .functions_wrapper import FunctionsWrapper


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
    return {
        "agent": file_to_string(f'{prompt_dir}/agent_system_prompt.txt'),
        "generate": file_to_string(f'{prompt_dir}/generate_system_prompt.txt'),
        "agent_gen_only": file_to_string(f'{prompt_dir}/agent_gen_system_prompt.txt'),
    }


class ReMEmbRAgent:
    """ReMEmbR Agent using Ollama LLM with LangGraph workflow."""

    def __init__(self, config: AgentConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger

        # Initialize LLM with FunctionsWrapper for consistent tool calling.
        # FunctionsWrapper injects tool definitions into the system prompt
        # and parses JSON output, matching the reference remembr behavior.
        self.base_llm = ChatOllama(
            model=self.config.model,
            format="json",
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )
        self.llm = FunctionsWrapper(self.base_llm)

        # Load prompts (cached)
        self.prompts = _load_prompts()

        self.graph = None
        self._nodes = None

        self.logger.info(f'ReMEmbRAgent initialized with model: {self.config.model}')

    def set_search_service(self, search_service: SearchService) -> None:
        """Bind a search service and build the agent graph."""
        tools, tool_definitions = create_search_tools(search_service)
        self._nodes = GraphNodes(self.llm, self.config, self.prompts, tool_definitions, base_llm=self.base_llm)
        self.graph = build_agent_graph(self._nodes, tools)

        self.logger.info('Search service connected, agent graph built')

    def query(self, question: str) -> AgentOutput:
        """Execute a query against the agent."""
        if self.graph is None:
            raise RuntimeError("Agent not initialized. Call set_search_service first.")

        self.logger.info(f'Processing query: "{question[:50]}..."')

        inputs = {"messages": [("user", question)]}
        self.graph.invoke(inputs)

        result = AgentOutput.from_dict(self._nodes.last_parsed_result)
        self.logger.info(f'Query completed: {result.text[:50] if result.text else "N/A"}...')

        return result
