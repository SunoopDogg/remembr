import json
import os
from functools import lru_cache

from langchain_ollama import ChatOllama

from ..config import AgentConfig
from ..models import AgentOutput
from ..models.protocols import Logger, SearchService
from ..tools import create_search_tools
from ..graph import GraphNodes, build_agent_graph
from ..utils import file_to_string


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

        # Build ChatOllama with optional JSON format constraint.
        # The original code uses format="json" for non-command-r Ollama models
        # to force valid JSON output, which is critical for tool-call parsing.
        llm_kwargs = dict(
            model=self.config.model,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )
        if self.config.use_json_format:
            llm_kwargs["format"] = "json"

        base_llm = ChatOllama(**llm_kwargs)

        # Optionally wrap with FunctionsWrapper for models that don't support
        # native tool calling. FunctionsWrapper injects tool schemas into the
        # system prompt and parses JSON output, working with all Ollama models.
        if self.config.use_functions_wrapper:
            from .functions_wrapper import FunctionsWrapper
            self.llm = FunctionsWrapper(base_llm)
        else:
            self.llm = base_llm

        # Load prompts (cached)
        self.prompts = _load_prompts()

        self.graph = None
        self._nodes = None

        self.logger.info(f'ReMEmbRAgent initialized with model: {self.config.model}')

    def set_search_service(self, search_service: SearchService) -> None:
        """Bind a search service and build the agent graph."""
        tools = create_search_tools(search_service)
        self._nodes = GraphNodes(self.llm, self.config, self.prompts, tools)
        self.graph = build_agent_graph(self._nodes, tools)

        self.logger.info('Search service connected, agent graph built')

    def query(self, question: str) -> AgentOutput:
        """Execute a query against the agent.

        Uses the graph's return value directly instead of a side-channel,
        matching the original remembr implementation's approach.
        """
        if self.graph is None:
            raise RuntimeError("Agent not initialized. Call set_search_service first.")

        self.logger.info(f'Processing query: "{question[:50]}..."')

        inputs = {"messages": [("user", question)]}
        out = self.graph.invoke(inputs)

        # Parse result from graph output (last message contains JSON from
        # _generate_node_impl's json.dumps). This is more robust than
        # reading from the side-channel last_parsed_result.
        response = out['messages'][-1]
        raw_content = ''.join(response.content.splitlines())
        try:
            parsed = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Agent generate node returned non-JSON content: {raw_content!r}"
            ) from e
        result = AgentOutput.from_dict(parsed)

        self.logger.info(f'Query completed: {result.text[:50] if result.text else "N/A"}...')

        return result
