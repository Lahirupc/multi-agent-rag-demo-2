import pytest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.documents import Document


@pytest.fixture
def fake_chat_model():
    """Create a fake chat model that returns scripted responses."""
    responses = [
        AIMessage(content="response"),
        AIMessage(content="RESPONSE"),
        AIMessage(content="RETRIEVAL"),
        AIMessage(content="RESEARCH"),
    ]
    return FakeListChatModel(responses=responses * 10)


@pytest.fixture
def fake_vectorstore():
    """Create a fake vectorstore with predictable search results."""
    mock_vectorstore = MagicMock()

    # Mock synchronous search
    mock_docs = [
        Document(
            page_content="Vector databases store embeddings for fast similarity search.",
            metadata={"source": "test.md", "chunk": 0},
        ),
        Document(
            page_content="Pinecone is a managed vector database service.",
            metadata={"source": "test.md", "chunk": 1},
        ),
    ]
    mock_vectorstore.similarity_search.return_value = mock_docs

    # Mock asynchronous search
    async def async_similarity_search(query, k=4):
        return mock_docs[:k]

    mock_vectorstore.asimilarity_search = async_similarity_search

    # Mock add_documents
    mock_vectorstore.add_documents = MagicMock(return_value=None)

    return mock_vectorstore


@pytest.fixture
def mock_llm(fake_chat_model):
    """Mock the get_chat_model function."""
    with patch("agents.llm.get_chat_model", return_value=fake_chat_model):
        # Clear the cache
        import agents.llm
        agents.llm.get_chat_model.cache_clear()
        yield fake_chat_model
        agents.llm.get_chat_model.cache_clear()


@pytest.fixture
def mock_vectorstore_fn(fake_vectorstore):
    """Mock the get_vectorstore function."""
    with patch("agents.vectorstore.get_vectorstore", return_value=fake_vectorstore):
        # Clear the cache
        import agents.vectorstore
        agents.vectorstore.get_vectorstore.cache_clear()
        yield fake_vectorstore
        agents.vectorstore.get_vectorstore.cache_clear()


@pytest.fixture
def mocked_graph_deps(mock_llm, mock_vectorstore_fn):
    """Fixture that mocks both LLM and vectorstore dependencies."""
    return {"llm": mock_llm, "vectorstore": mock_vectorstore_fn}
