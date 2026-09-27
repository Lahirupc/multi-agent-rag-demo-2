import logging
from itertools import chain
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_seed_documents_chunked() -> list[Document]:
    """Load and chunk the seed corpus from backend/data/.

    Shared by the dense ingestion pipeline and the sparse (BM25) index so both
    retrieval paths score the exact same chunks, keyed by (source, chunk).
    """
    if not DATA_DIR.exists():
        logger.warning(f"Data directory not found at {DATA_DIR}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []
    for file_path in chain(DATA_DIR.glob("*.md"), DATA_DIR.glob("*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.name,
                            "chunk": i,
                            "type": file_path.name.split("-"),
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")

    return documents
