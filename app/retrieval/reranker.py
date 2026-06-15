from langchain_core.documents import Document

from .base import RetrieverBase


class RerankerRetriever(RetrieverBase):
    def __init__(
        self,
        base_retriever: RetrieverBase,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):

        from sentence_transformers import CrossEncoder

        self._base = base_retriever
        self._cross_encoder = CrossEncoder(model_name)
        self._top_k = top_k

    def invoke(self, query: str) -> list[Document]:
        docs = self._base.invoke(query)
        if not docs:
            return docs

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, docs, strict=False), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self._top_k]]
