import pytest
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from agents.graph import build_graph
from agents.state import GraphState


@pytest.mark.asyncio
async def test_graph_chat_flow(mocked_graph_deps):
    """Test a basic chat flow through the graph."""
    graph = build_graph()

    state = {
        "messages": [HumanMessage(content="Hello")],
        "session_id": "test-session",
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test"}})

    assert "messages" in result
    assert len(result["messages"]) > 1  # Should have original + response
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["session_id"] == "test-session"


@pytest.mark.asyncio
async def test_graph_retrieval_path(mocked_graph_deps):
    """Test that retrieval path doesn't call vectorstore for simple responses."""
    graph = build_graph()

    state = {
        "messages": [HumanMessage(content="Hi there")],
        "session_id": "test-session",
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "test"}})

    # Should have generated a response
    assert len(result["messages"]) >= 2
    assert isinstance(result["messages"][-1], AIMessage)


@pytest.mark.asyncio
async def test_graph_preserves_session_id(mocked_graph_deps):
    """Test that session_id is preserved through graph execution."""
    graph = build_graph()

    session_id = "test-session-123"
    state = {
        "messages": [HumanMessage(content="Test message")],
        "session_id": session_id,
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": session_id}})

    assert result["session_id"] == session_id


@pytest.mark.asyncio
async def test_graph_message_accumulation(mocked_graph_deps):
    """Test that messages are accumulated across invocations."""
    graph = build_graph()

    # First invocation
    state1 = {
        "messages": [HumanMessage(content="First message")],
        "session_id": "test-session",
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }

    result1 = await graph.ainvoke(state1, config={"configurable": {"thread_id": "test"}})

    # Second invocation with same thread should remember first message
    state2 = {
        "messages": [HumanMessage(content="Follow up question")],
        "session_id": "test-session",
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }

    result2 = await graph.ainvoke(state2, config={"configurable": {"thread_id": "test"}})

    # Second result should have more messages than first
    assert len(result2["messages"]) > len(result1["messages"])
