import logging

from langchain_core.documents import Document

from .sparse_index import get_sparse_index
from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Relative weight given to each signal when combining normalized scores.
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
DocKey = tuple[str, int]

def _doc_key(doc: Document) -> DocKey:
    return (doc.metadata.get("source", "unknown"), doc.metadata.get("chunk", -1))


def _normalize_scores(scores: dict[DocKey, float]) -> dict[DocKey, float]:
    """Min-max normalize scores to [0, 1] so dense and sparse scales are comparable."""
    if not scores:
        return {}

    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


async def hybrid_search(query: str, k: int = 4) -> list[Document]:
    """Hybrid retrieval: combine dense (embedding) and sparse (BM25) search.

    Dense search finds semantically similar chunks via vector similarity;
    sparse search finds chunks sharing exact keywords/terms with the query.
    Each leg's scores are min-max normalized independently, then combined as
    a weighted sum (DENSE_WEIGHT * dense + SPARSE_WEIGHT * sparse) to rank
    the union of candidates from both legs.
    """
    
    # How many candidates each retrieval leg pulls before fusion; wider than the
    # final k so a strong match on one signal isn't dropped before ranking.
    FETCH_K = int(k * 2.5)
    
    dense_results = await get_vectorstore().asimilarity_search_with_score(query, k=FETCH_K)
    sparse_results = get_sparse_index().search(query, k=FETCH_K)

    dense_scores: dict[DocKey, float] = {}
    sparse_scores: dict[DocKey, float] = {}
    docs_by_key: dict[DocKey, Document] = {}

    for doc, score in dense_results:
        key = _doc_key(doc)
        dense_scores[key] = score
        docs_by_key[key] = doc

    for doc, score in sparse_results:
        key = _doc_key(doc)
        sparse_scores[key] = score
        docs_by_key.setdefault(key, doc) # set doc with the key only if it does not already exist

    dense_norm = _normalize_scores(dense_scores)
    sparse_norm = _normalize_scores(sparse_scores)

    combined_scores = {
        key: DENSE_WEIGHT * dense_norm.get(key, 0.0) + SPARSE_WEIGHT * sparse_norm.get(key, 0.0)
        for key in docs_by_key
    }

    ranked_keys = sorted(combined_scores, key=combined_scores.get, reverse=True)

    logger.info(
        f"Hybrid search: {len(dense_scores)} dense + {len(sparse_scores)} sparse "
        f"candidates -> {min(k, len(ranked_keys))} results"
    )

    return [docs_by_key[key] for key in ranked_keys[:k]]
