import logging
from langchain_core.messages import AIMessage, SystemMessage
from .state import GraphState
from .llm import get_chat_model

logger = logging.getLogger(__name__)


def _build_system_prompt(state: GraphState) -> str:
    """Build the system prompt with context from retrieved docs and findings."""
    retrieved_docs = state.get("retrieved_docs", [])
    research_findings = state.get("research_findings", [])

    context_parts = []
    
    if not retrieved_docs and not research_findings:
        context_parts.append("No context")


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

            If the context is "No context" then say "I don't have any information related to this."""


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
