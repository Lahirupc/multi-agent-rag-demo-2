import logging
from langchain_core.messages import AIMessage, SystemMessage
from .state import GraphState
from .llm import get_chat_model

logger = logging.getLogger(__name__)


def _build_system_prompt(state: GraphState) -> str:
    """Build the system prompt with context from retrieved docs and findings."""
    retrieved_docs = state.get("retrieved_docs", [])
    research_findings = state.get("research_findings", [])
    route = state.get("route")

    has_context = bool(retrieved_docs or research_findings)

    if not has_context:
        if route in ("retrieval", "research"):
            # Retrieval/research was attempted but found nothing relevant.
            # Be explicit so the model doesn't fall back to fabricating an answer.
            return (
                "You are a helpful AI assistant. You searched the knowledge base for "
                "information relevant to the user's question, but found nothing relevant. "
                "Tell the user you don't have information on this topic in the available "
                "documents. Do not answer from your own general knowledge or make up "
                "information."
            )
        # No retrieval was attempted (casual conversation) - general assistant is fine.
        return "You are a helpful AI assistant."

    context_parts = []

    if retrieved_docs:
        context_parts.append("Context from documents:")
        for doc in retrieved_docs:
            source = doc.get("source", "unknown")
            content = doc.get("content", "")
            context_parts.append(f"\n[From {source}]:\n{content}")

    if research_findings:
        context_parts.append("\nResearch findings:")
        for i, finding in enumerate(research_findings, 1):
            context_parts.append(f"\n{i}. {finding}")

    context_text = "\n".join(context_parts)

    return f"""You are a helpful AI assistant. Use the following context to answer the user's question.

Context:
{context_text}

Only use the information in the context above to answer. If the context doesn't fully answer the question, tell the user what information is missing rather than guessing or using outside knowledge."""


async def response_node(state: GraphState) -> dict:
    """Generate the final response."""
    if not state["messages"]:
        return {"messages": []}

    chat_model = get_chat_model()

    system_prompt = _build_system_prompt(state)

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]

    response = await chat_model.ainvoke(messages)

    logger.info("Response generated")

    return {"messages": [response]}
