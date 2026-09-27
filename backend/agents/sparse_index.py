import functools
import logging
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .corpus import load_seed_documents_chunked

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class SparseIndex:
    """Keyword (BM25) index over the seed corpus, used for sparse retrieval."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the top-k (document, BM25 score) pairs for the query."""
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, score) for doc, score in ranked[:k] if score > 0]


@functools.lru_cache(maxsize=1)
def get_sparse_index() -> SparseIndex:
    documents_chunked = load_seed_documents_chunked()
    logger.info(f"Built BM25 sparse index over {len(documents_chunked)} chunks")
    return SparseIndex(documents_chunked)
