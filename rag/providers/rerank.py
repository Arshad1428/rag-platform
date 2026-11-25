"""Reranker: scores each candidate 0-1 for how well it answers the question (Gate 2 uses it).
Used with Qdrant. Azure AI Search brings its own semantic ranker, so this is skipped there.
Runs locally (GPU if available), fully offline once the model is installed."""
from functools import lru_cache

from rag.config import settings


@lru_cache
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model, max_length=512, device=settings.reranker_device or None)


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    # single-score rerankers return a 0-1 probability by default
    scores = _model().predict([(query, c["content"][:2000]) for c in chunks])
    for c, s in zip(chunks, scores):
        c["score"] = float(s)
    return sorted(chunks, key=lambda c: -c["score"])[:top_k]
