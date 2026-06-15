from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base import RetrieverBase


class DenseRetriever(RetrieverBase):
    def __init__(self, vectorstore: Chroma, k: int = 5):
        self._retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def invoke(self, query: str) -> list[Document]:
        return self._retriever.invoke(query)
