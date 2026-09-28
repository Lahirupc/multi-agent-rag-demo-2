import logging
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from .state import GraphState
from .hybrid_retrieval import hybrid_search
from .llm import get_chat_model

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESEARCH_ITERATIONS = 3

RESEARCH_REFORMULATE_PROMPT = """You are a research assistant helping to investigate a complex question more deeply.

Given the user's original question and what you've already found, reformulate the query to search for different or more specific aspects.

Original question: {original_question}

Findings so far:
{findings}

Provide a new search query (just the query, no explanation) that explores a different angle or digs deeper."""


async def research_node(state: GraphState) -> dict:
    """Perform iterative research by reformulating and searching."""
    research_iterations = state.get("research_iterations", 0)
    research_findings = state.get("research_findings", [])

    if not state["messages"]:
        return {
            "research_findings": [],
            "research_iterations": research_iterations + 1,
        }

    # Use the current turn's question (rewritten to be standalone by memory_load),
    # not the first message of the whole conversation.
    user_message = state.get("standalone_question")
    if not user_message:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

    if not user_message:
        return {
            "research_findings": research_findings,
            "research_iterations": research_iterations + 1,
        }

    # Determine search query
    if research_iterations == 0:
        # First iteration: use original question
        search_query = user_message
    else:
        # Reformulate the query
        chat_model = get_chat_model()
        findings_text = "\n".join(state.get("research_findings", []))

        prompt = RESEARCH_REFORMULATE_PROMPT.format(
            original_question=user_message,
            findings=findings_text if findings_text else "None yet",
        )

        response = await chat_model.ainvoke([SystemMessage(content=prompt)])
        search_query = response.content.strip()

    # Search for documents via hybrid (dense + sparse) search
    results = await hybrid_search(search_query, k=4)

    # Add findings that aren't already in research_findings (copy rather than
    # mutate the existing list in place, since it may be shared with state)
    current_findings = list(state.get("research_findings", []))
    for doc in results:
        content = doc.page_content
        if content not in current_findings:
            current_findings.append(content)

    logger.info(f"Research iteration {research_iterations + 1}: found {len(results)} documents")

    return {
        "research_findings": current_findings,
        "research_iterations": research_iterations + 1,
        "research_query": search_query,
    }


def should_continue_research(state: GraphState) -> Literal["research", "response"]:
    """Determine if research should continue or move to response."""
    research_iterations = state.get("research_iterations", 0)
    max_iterations = state.get("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)

    if research_iterations < max_iterations:
        return "research"
    else:
        return "response"
