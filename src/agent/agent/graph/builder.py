from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..models import AgentState
from .nodes import GraphNodes


def build_agent_graph(nodes: GraphNodes, tools: list):
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("action", ToolNode(tools))
    workflow.add_node("generate", nodes.generate_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        nodes.should_continue,
        {"continue": "action", "end": "generate"},
    )
    workflow.add_edge("action", "agent")
    workflow.add_edge("generate", END)

    return workflow.compile()
