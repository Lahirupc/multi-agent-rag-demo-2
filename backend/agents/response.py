import logging
from langchain_core.messages import AIMessage, SystemMessage
from .memory import render_memory_block, render_previous_questions
from .state import GraphState
from .llm import get_chat_model

logger = logging.getLogger(__name__)


def _build_system_prompt(state: GraphState) -> str:
    """Build the system prompt with context from retrieved docs, findings, and memory."""
    retrieved_docs = state.get("retrieved_docs", [])
    research_findings = state.get("research_findings", [])
    route = state.get("route")

    context_parts = []

    needs_kb_context = route in ("retrieval", "research")
    if needs_kb_context and not retrieved_docs and not research_findings:
        context_parts.append("No knowledge-base context")

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

    context_text = "\n".join(context_parts) if context_parts else "No knowledge-base context"

    memory_block = render_memory_block(state.get("user_profile", {}), state.get("memory_context"))
    previous_questions = render_previous_questions(state.get("interactions", []))

    memory_section = ""
    if memory_block or previous_questions:
        memory_section = "\n\nConversation memory:\n" + "\n\n".join(
            part for part in (previous_questions, memory_block) if part
        )

    kb_instruction = (
        'If the knowledge-base context above is "No knowledge-base context" and the '
        "question requires document lookup, say \"I don't have any information related "
        "to this.\" For questions about the user or the conversation itself, answer from "
        "the conversation memory below instead."
        if needs_kb_context
        else ""
    )

    return f"""You are a helpful AI assistant. Use the following context to answer the user's question.

            Context:
                {context_text}
            {memory_section}

            {kb_instruction}"""


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
