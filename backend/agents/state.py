from typing import Annotated, Literal, NotRequired, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedDoc(TypedDict):
    content: str
    source: str


Route = Literal["retrieval", "research", "response"]


class Interaction(TypedDict):
    """A single past turn, kept for the conversation's lifetime."""

    turn: int
    question: str
    standalone_question: str
    answer: str
    route: Route | None


class MemoryContext(TypedDict):
    """What memory_load resolves for the current turn; read by later nodes."""

    standalone_question: str
    relevant_interactions: list[Interaction]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    route: Route | None
    retrieved_docs: list[RetrievedDoc]
    research_findings: list[str]
    research_query: str | None
    research_iterations: int
    max_research_iterations: int

    # Conversational memory (persisted across turns via the checkpointer).
    user_profile: NotRequired[dict[str, str]]
    interactions: NotRequired[list[Interaction]]
    turn_count: NotRequired[int]

    # Per-turn scratch fields set by memory_load, consumed by later nodes.
    standalone_question: NotRequired[str]
    memory_context: NotRequired[MemoryContext]
