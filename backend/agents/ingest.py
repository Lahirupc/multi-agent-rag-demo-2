import logging
from .corpus import load_seed_documents_chunked
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


async def ingest_documents_if_needed() -> None:
    """Ingest seed documents from backend/data/ into Pinecone (idempotent)."""
    try:
        vectorstore = get_vectorstore()

        # Check if any documents already exist (simple heuristic)
        try:
            # Try a simple search - if it works, assume docs are already ingested
            results = vectorstore.similarity_search("meeting", k=1)
            if results:
                logger.info("Documents already ingested into Pinecone; skipping")
                return
        except Exception:
            # If search fails, proceed with ingestion
            pass

        all_documents = load_seed_documents_chunked()

        if all_documents:
            vectorstore.add_documents(all_documents)
            logger.info(f"Ingested {len(all_documents)} total chunks into Pinecone")
        else:
            logger.warning("No documents found to ingest")

    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise
