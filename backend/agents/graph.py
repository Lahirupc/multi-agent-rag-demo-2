import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import GraphState
from .supervisor import supervisor_node, route_after_supervisor
from .retrieval import retrieval_node
from .research import research_node, should_continue_research
from .response import response_node

logger = logging.getLogger(__name__)


def build_graph():
    """Build the LangGraph multi-agent graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("research", research_node)
    graph.add_node("response", response_node)

    # Set entry point
    graph.add_edge(START, "supervisor")

    # Add conditional edges from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retrieval": "retrieval",
            "research": "research",
            "response": "response",
        },
    )

    # Add edges
    graph.add_edge("retrieval", "response")

    # Add conditional edges from research
    graph.add_conditional_edges(
        "research",
        should_continue_research,
        {
            "research": "research",
            "response": "response",
        },
    )

    # Response leads to end
    graph.add_edge("response", END)

    # Compile with memory saver for checkpointing
    compiled_graph = graph.compile(checkpointer=MemorySaver())

    logger.info("LangGraph compiled successfully")
    return compiled_graph
