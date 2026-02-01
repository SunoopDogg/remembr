import ast
import json
import re
import sys
import time as time_module
import traceback

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from ..config import AgentConfig
from ..models import AgentState
from ..utils import parse_json

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

_ARGS_CLEANUP_PATTERN = re.compile(r"\{.*?\}")


def _retry_node_on_failure(state, func, logger=None, max_retries=MAX_RETRIES):
    """Retry a function on failure with backoff and max retries."""
    for attempt in range(max_retries):
        try:
            return func(state)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            if logger:
                logger(f"Node {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time_module.sleep(delay)
            else:
                raise


class GraphNodes:
    """Encapsulates LangGraph node logic and shared state."""

    def __init__(self, llm, config: AgentConfig, prompts: dict, tool_definitions: list, base_llm=None) -> None:
        self.llm = llm
        self.base_llm = base_llm or llm
        self.config = config
        self.prompts = prompts
        self.tool_definitions = tool_definitions
        self.reset_state()

    def reset_state(self) -> None:
        """Reset mutable state for a new query."""
        self._tool_history_items: list = []
        self.iteration_count = 0
        self.last_parsed_result: dict | None = None

    def _get_tool_history(self) -> str:
        """Get formatted tool history string."""
        if not self._tool_history_items:
            return "These are the tools I have previously used so far: \n"
        return "These are the tools I have previously used so far: \n" + "\n".join(self._tool_history_items)

    def should_continue(self, state: AgentState) -> str:
        """Determine whether to continue tool execution."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    def agent_node(self, state: AgentState) -> dict:
        """Agent node with retry - decides to use tools or generate response."""
        return _retry_node_on_failure(state, self._agent_node_impl)

    def _agent_node_impl(self, state: AgentState) -> dict:
        """Agent node implementation."""
        messages = state["messages"]

        if self.iteration_count < self.config.max_tool_calls:
            model = self.llm.bind_tools(tools=self.tool_definitions)
            prompt = self.prompts["agent"]
        else:
            model = self.llm
            prompt = self.prompts["agent_gen_only"]

        agent_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", self._get_tool_history()),
            ("ai", prompt),
            ("human", "{question}"),
        ])

        chain = agent_prompt | model
        question = f"The question is: {messages[0]}"

        chat_history = [
            AIMessage(id=msg.id, content=msg.content) if isinstance(msg, ToolMessage) else msg
            for msg in messages
        ]

        response = chain.invoke({"question": question, "chat_history": chat_history})

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = _ARGS_CLEANUP_PATTERN.sub("", str(tool_call['args']))
                    self._tool_history_items.append(
                        f"I previously used the {tool_call['name']} tool "
                        f"with the arguments: {args}."
                    )

        self.iteration_count += 1
        return {"messages": [response]}

    def generate_node(self, state: AgentState) -> dict:
        """Generate final structured response with retry."""
        return _retry_node_on_failure(state, self._generate_node_impl)

    def _generate_node_impl(self, state: AgentState) -> dict:
        """Generate node implementation."""
        messages = state["messages"]
        question = messages[0].content + "\n Please respond in the desired format."

        prompt = PromptTemplate(
            template=self.prompts["generate"],
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question': question})

        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", filled_prompt.text),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        chain = gen_prompt | self.base_llm
        response = chain.invoke({"question": question, "chat_history": messages[1:]})

        # Strip newlines before parsing, matching reference behavior
        raw_content = ''.join(response.content.splitlines())
        parsed = parse_json(raw_content)
        response_data = self._extract_response(parsed)
        self._normalize_and_validate(response_data)

        self.reset_state()
        self.last_parsed_result = response_data

        return {"messages": [AIMessage(content=json.dumps(response_data))]}

    def _extract_response(self, parsed: dict) -> dict:
        """Extract response data from nested LLM output."""
        if "tool_input" in parsed:
            parsed = parsed["tool_input"]
        if "response" in parsed and isinstance(parsed["response"], dict):
            return parsed["response"]
        return parsed

    def _normalize_and_validate(self, data: dict) -> None:
        """Normalize data types and validate required fields."""
        required_keys = ["time", "text", "binary", "position", "duration"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        if isinstance(data.get('position'), str):
            data['position'] = ast.literal_eval(data['position'])
        if isinstance(data.get('orientation'), str):
            data['orientation'] = ast.literal_eval(data['orientation'])

        if data['position'] is not None and len(data['position']) != 3:
            raise ValueError(f"Invalid position shape: {data['position']}")
