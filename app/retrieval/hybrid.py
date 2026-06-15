from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .base import RetrieverBase


class HybridRetriever(RetrieverBase):
    def __init__(self, vectorstore: Chroma, k: int = 5, rrf_k: int = 60):
        self._k = k
        self._rrf_k = rrf_k
        self._dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        result = vectorstore.get(include=["documents", "metadatas"])
        self._corpus: list[Document] = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(result["documents"], result["metadatas"], strict=False)
        ]
        tokenized = [doc.page_content.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def invoke(self, query: str) -> list[Document]:
        dense_docs = self._dense_retriever.invoke(query)

        if self._bm25 is None:
            return dense_docs

        scores = self._bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][: self._k]
        bm25_docs = [self._corpus[i] for i in top_indices]

        return self._rrf([dense_docs, bm25_docs])

    def _rrf(self, rankings: list[list[Document]]) -> list[Document]:
        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                key = hash(doc.page_content)
                if key not in scores:
                    scores[key] = 0.0
                    doc_map[key] = doc
                scores[key] += 1.0 / (self._rrf_k + rank)

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[: self._k]]
