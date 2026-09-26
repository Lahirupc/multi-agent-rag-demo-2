from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedDoc(TypedDict):
    content: str
    source: str


Route = Literal["retrieval", "research", "response"]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    route: Route | None
    retrieved_docs: list[RetrievedDoc]
    research_findings: list[str]
    research_query: str | None
    research_iterations: int
    max_research_iterations: int
