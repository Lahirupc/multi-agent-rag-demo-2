import logging
from langchain_core.messages import SystemMessage
from .state import GraphState, Route
from .llm import get_chat_model

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """You are a supervisor agent that routes user messages to the appropriate handler.

Analyze the user's latest message and determine which agent should handle it:
- "retrieval" - for factual questions that need vector search of documents
- "research" - for complex questions that need deep investigation/multiple searches
- "response" - for casual conversation, greetings, or simple questions that don't need retrieval

Respond with ONLY one word: retrieval, research, or response
No explanation, no punctuation, just the single word."""


async def supervisor_node(state: GraphState) -> dict:
    """Route the user message to the appropriate agent."""
    if not state["messages"]:
        return {"route": "response"}

    latest_message = state["messages"][-1]

    chat_model = get_chat_model()

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        latest_message,
    ]

    response = await chat_model.ainvoke(messages)
    route_text = response.content.strip().lower()

    # Parse the route
    if "retrieval" in route_text:
        route = "retrieval"
    elif "research" in route_text:
        route = "research"
    else:
        route = "response"

    logger.info(f"Supervisor routed to: {route}")
    return {"route": route}


def route_after_supervisor(state: GraphState) -> Route:
    """Determine the next node based on supervisor's routing decision."""
    route = state.get("route")
    if route in ("retrieval", "research", "response"):
        return route
    return "response"
