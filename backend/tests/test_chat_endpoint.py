import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_graph():
    """Create a mock graph for testing."""
    mock_graph = AsyncMock()

    async def ainvoke_side_effect(state, config=None):
        # Simulate graph response
        return {
            **state,
            "messages": [
                *state["messages"],
                AIMessage(content="Test response"),
            ],
        }

    mock_graph.ainvoke = ainvoke_side_effect
    return mock_graph


@pytest.fixture
def test_client(mocked_graph_deps, mock_graph):
    """Create a test client with mocked dependencies."""
    # Mock the ingest function
    with patch("main.ingest_documents_if_needed", new_callable=AsyncMock):
        # Mock build_graph to return our mock graph
        with patch("main.build_graph", return_value=mock_graph):
            # Import after mocking
            from main import app

            # Ensure the app is initialized
            with TestClient(app) as client:
                # Manually set the graph on app state
                client.app.state.graph = mock_graph
                yield client


def test_empty_message_returns_400(test_client):
    """Test that empty message returns 400 status."""
    response = test_client.post("/chat", json={"message": ""})
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]


def test_whitespace_only_message_returns_400(test_client):
    """Test that whitespace-only message returns 400 status."""
    response = test_client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_missing_api_key_returns_500(test_client):
    """Test that missing API key returns 500 with specific detail."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
        # Clear and reset the environ
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ.pop("OPENROUTER_API_KEY", None)

        try:
            response = test_client.post("/chat", json={"message": "Hello"})
            assert response.status_code == 500
            assert "API key not configured" in response.json()["detail"]
        finally:
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key


def test_successful_chat_request(test_client):
    """Test successful chat request."""
    response = test_client.post(
        "/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "status" in data
    assert "session_id" in data
    assert data["status"] == "success"
    assert data["response"] == "Test response"


def test_chat_with_session_id(test_client):
    """Test that provided session_id is echoed back."""
    session_id = "custom-session-123"
    response = test_client.post(
        "/chat",
        json={"message": "Hello", "session_id": session_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id


def test_chat_generates_session_id_when_missing(test_client):
    """Test that session_id is generated when not provided."""
    response = test_client.post(
        "/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0
    # Verify it's a valid UUID format (rough check)
    assert len(data["session_id"].split("-")) == 5


def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(test_client):
    """Test root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data
