import logging
from langchain_core.messages import HumanMessage
from .state import GraphState, RetrievedDoc
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


async def retrieval_node(state: GraphState) -> dict:
    """Retrieve documents from the vector store."""
    if not state["messages"]:
        return {"retrieved_docs": []}

    latest_message = state["messages"][-1]
    query = latest_message.content if isinstance(latest_message, HumanMessage) else str(latest_message)

    vectorstore = get_vectorstore()

    # Perform similarity search
    results = await vectorstore.asimilarity_search(query, k=4)

    retrieved_docs = [
        RetrievedDoc(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
        )
        for doc in results
    ]

    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    return {"retrieved_docs": retrieved_docs}
