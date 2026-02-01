"""FunctionsWrapper for Ollama LLMs.

Wraps an LLM to inject tool definitions into the system prompt and parse
JSON output into LangChain ToolCall objects. This replicates the behavior
from the reference remembr project, ensuring consistent tool calling
across different Ollama models.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..utils.parser_utils import parse_json


DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""

DEFAULT_RESPONSE_FUNCTION = {
    "name": "__conversational_response",
    "description": (
        "Respond conversationally if no other tools should be called for a given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Conversational response to the user.",
            },
        },
        "required": ["response"],
    },
}


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and (
        issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


def convert_to_ollama_tool(tool: Any) -> Dict:
    """Convert a Pydantic model to an Ollama tool definition."""
    if _is_pydantic_class(tool):
        schema = tool.construct().schema()
        definition = {"name": schema["title"], "properties": schema["properties"]}
        if "required" in schema:
            definition["required"] = schema["required"]
        return definition
    raise ValueError(
        f"Cannot convert {tool} to an Ollama tool. Needs to be a Pydantic model."
    )


class FunctionsWrapper(BaseChatModel, BaseLanguageModel):
    """Wraps an LLM to provide tool-calling via system prompt injection.

    Instead of relying on native model tool-calling support, this injects
    tool definitions into the system prompt and parses the JSON output.
    """

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE
    llm: Any = None

    def __init__(self, llm) -> None:
        super().__init__()
        self.llm = llm

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(functions=tools, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]

        if len(functions) > 0 and _is_pydantic_class(functions[0]):
            functions = [convert_to_ollama_tool(fn) for fn in functions]

        functions.insert(0, DEFAULT_RESPONSE_FUNCTION)

        from langchain_core.prompts import SystemMessagePromptTemplate
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )

        response_message = self.llm.invoke([system_message] + messages)
        chat_generation_content = response_message.content

        if not isinstance(chat_generation_content, str):
            raise ValueError("FunctionsWrapper does not support non-string output.")

        try:
            parsed_chat_result = parse_json(chat_generation_content)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"Model did not respond with valid JSON. "
                f"Response: {chat_generation_content}"
            )

        # Unwrap if LLM wrapped the list inside a key like
        # {"reasoning": [...]}, {"tool_calls": [...]}, etc.
        if isinstance(parsed_chat_result, dict):
            for key in ("reasoning", "tool_calls", "tools"):
                if key in parsed_chat_result and isinstance(parsed_chat_result[key], list):
                    parsed_chat_result = parsed_chat_result[key]
                    break

        if not isinstance(parsed_chat_result, list):
            parsed_chat_result = [parsed_chat_result]

        # Separate real tool calls from conversational responses.
        # If the LLM emits both tool calls and __conversational_response
        # in the same list, prioritize tool calls so actual DB searches
        # execute rather than returning hallucinated coordinates.
        real_tool_items = []
        response_item = None
        for item in parsed_chat_result:
            name = item.get("tool")
            if name == DEFAULT_RESPONSE_FUNCTION["name"]:
                response_item = item
            else:
                real_tool_items.append(item)

        # Only use __conversational_response when it's the ONLY item
        if response_item is not None and len(real_tool_items) == 0:
            if "tool_input" in response_item:
                if isinstance(response_item['tool_input'], str):
                    response = response_item['tool_input']
                elif isinstance(response_item['tool_input'], dict) and "response" in response_item["tool_input"]:
                    response = response_item["tool_input"]["response"]
                else:
                    raise ValueError(
                        f"Failed to parse a response from output: "
                        f"{chat_generation_content}"
                    )
            elif "response" in response_item:
                response = response_item["response"]
            else:
                raise ValueError(
                    f"Failed to parse a response from output: "
                    f"{chat_generation_content}"
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=str(response))
                    )
                ]
            )

        # Process real tool calls only
        items_to_process = real_tool_items if real_tool_items else parsed_chat_result
        tool_calls = []

        for tool in items_to_process:
            called_tool_name = tool.get("tool")
            called_tool = next(
                (fn for fn in functions if fn["name"] == called_tool_name), None
            )

            if called_tool is None:
                raise ValueError(
                    f"Failed to parse a function call from output: "
                    f"{chat_generation_content}"
                )

            called_tool_arguments = tool.get("tool_input", {})

            tool_call = ToolCall(
                name=called_tool_name,
                args=called_tool_arguments,
                id=f"call_{str(uuid.uuid4()).replace('-', '')}",
            )
            tool_calls.append(tool_call)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="", tool_calls=tool_calls)
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"
