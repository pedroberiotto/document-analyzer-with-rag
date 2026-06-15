from abc import ABC, abstractmethod

from langchain_core.documents import Document


class RetrieverBase(ABC):
    @abstractmethod
    def invoke(self, query: str) -> list[Document]: ...
