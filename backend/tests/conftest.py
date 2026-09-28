import json
import re
from typing import Any, Optional

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.outputs import ChatGeneration, ChatResult


class ContentAwareFakeChatModel(BaseChatModel):
    """A fake chat model whose reply depends on the prompt it's given.

    The real graph makes several different kinds of LLM calls in one request
    (supervisor routing, memory extraction, research reformulation, final
    response), so a fixed queue of canned responses breaks as soon as a new
    call is added anywhere in the chain. This instead inspects the prompt
    text and returns a response shaped like what that call expects.
    """

    @property
    def _llm_type(self) -> str:
        return "content-aware-fake"

    def _call_content(self, messages: list[BaseMessage]) -> str:
        text = " ".join(str(m.content) for m in messages).lower()

        if "respond with only a json object" in text:
            # memory_load's extraction call. The whole prompt (including the
            # "Latest message: ..." line) arrives as one SystemMessage, so
            # pull the actual latest user message out of it, then extract a
            # name if it states one, so multi-user/multi-turn tests are meaningful.
            full_prompt = messages[-1].content if messages else ""
            latest_match = re.search(r"Latest message:\s*(.*)", full_prompt)
            latest_raw = latest_match.group(1).strip() if latest_match else full_prompt
            facts = {}
            name_match = re.search(
                r"i(?:'m| am)\s+([A-Z][a-zA-Z]*)|my name is\s+([A-Z][a-zA-Z]*)",
                latest_raw,
                re.IGNORECASE,
            )
            if name_match:
                facts["name"] = name_match.group(1) or name_match.group(2)
            standalone = latest_raw
            return json.dumps({"standalone_question": standalone, "user_facts": facts})

        if "respond with only one word" in text:
            # supervisor routing call
            latest = messages[-1].content.lower() if messages else ""
            if any(word in latest for word in ("hi", "hello", "hey", "my name", "what's my name", "what did i")):
                return "response"
            if any(word in latest for word in ("investigate", "deeply", "comprehensive")):
                return "research"
            return "retrieval"

        if "reformulate the query" in text:
            return "reformulated search query"

        return "This is a generated response."

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self._call_content(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def fake_chat_model():
    """Create a fake chat model that answers based on the prompt content."""
    return ContentAwareFakeChatModel()


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

    # Mock asynchronous search with scores (used by hybrid dense retrieval)
    async def async_similarity_search_with_score(query, k=4):
        return [(doc, 1.0 - i * 0.1) for i, doc in enumerate(mock_docs[:k])]

    mock_vectorstore.asimilarity_search_with_score = async_similarity_search_with_score

    # Mock add_documents
    mock_vectorstore.add_documents = MagicMock(return_value=None)

    return mock_vectorstore


@pytest.fixture
def mock_llm(fake_chat_model):
    """Mock get_chat_model everywhere it's imported.

    Each agent module does `from .llm import get_chat_model`, binding its own
    reference at import time, so patching `agents.llm.get_chat_model` alone
    would not affect calls made from supervisor/research/response/memory.
    """
    import agents.llm

    targets = [
        "agents.llm.get_chat_model",
        "agents.supervisor.get_chat_model",
        "agents.research.get_chat_model",
        "agents.response.get_chat_model",
        "agents.memory.get_chat_model",
    ]
    patchers = [patch(target, return_value=fake_chat_model) for target in targets]
    for p in patchers:
        p.start()
    agents.llm.get_chat_model.cache_clear()
    try:
        yield fake_chat_model
    finally:
        for p in patchers:
            p.stop()
        agents.llm.get_chat_model.cache_clear()


@pytest.fixture
def mock_vectorstore_fn(fake_vectorstore):
    """Mock get_vectorstore everywhere it's imported.

    `agents.hybrid_retrieval` does `from .vectorstore import get_vectorstore`,
    binding its own reference at import time, so patching
    `agents.vectorstore.get_vectorstore` alone would not affect calls made
    from hybrid_retrieval.
    """
    import agents.vectorstore

    targets = [
        "agents.vectorstore.get_vectorstore",
        "agents.hybrid_retrieval.get_vectorstore",
    ]
    patchers = [patch(target, return_value=fake_vectorstore) for target in targets]
    for p in patchers:
        p.start()
    agents.vectorstore.get_vectorstore.cache_clear()
    try:
        yield fake_vectorstore
    finally:
        for p in patchers:
            p.stop()
        agents.vectorstore.get_vectorstore.cache_clear()


@pytest.fixture
def mocked_graph_deps(mock_llm, mock_vectorstore_fn):
    """Fixture that mocks both LLM and vectorstore dependencies."""
    return {"llm": mock_llm, "vectorstore": mock_vectorstore_fn}
