import os
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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

        if not DATA_DIR.exists():
            logger.warning(f"Data directory not found at {DATA_DIR}")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        all_documents = []
        for file_path in DATA_DIR.glob("*.md") | DATA_DIR.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.name,
                            "chunk": i,
                            "type": file_path.name.split("-"),
                        },
                    )
                    all_documents.append(doc)
                logger.info(f"Ingested {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Error ingesting {file_path.name}: {e}")

        if all_documents:
            vectorstore.add_documents(all_documents)
            logger.info(f"Ingested {len(all_documents)} total chunks into Pinecone")
        else:
            logger.warning("No documents found to ingest")

    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise
