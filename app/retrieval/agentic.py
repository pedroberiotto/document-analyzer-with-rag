from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .base import RetrieverBase


class _ContextAssessment(BaseModel):
    confidence: float = Field(
        description="Confidence 0-1 that the context contains enough information to answer the query."
    )
    reformulated_query: str = Field(
        description="A better reformulated query to use if confidence is low."
    )


class AgenticRetriever(RetrieverBase):
    def __init__(
        self,
        base_retriever: RetrieverBase,
        llm: BaseChatModel,
        confidence_threshold: float = 0.6,
        max_iterations: int = 3,
    ):
        self._base = base_retriever
        self._assessor = llm.with_structured_output(_ContextAssessment)
        self._threshold = confidence_threshold
        self._max_iterations = max_iterations

    def invoke(self, query: str) -> list[Document]:
        seen: set[int] = set()
        all_docs: list[Document] = []
        current_query = query

        for _ in range(self._max_iterations):
            docs = self._base.invoke(current_query)

            for doc in docs:
                key = hash(doc.page_content)
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

            assessment = self._assess(query, docs)
            if assessment.confidence >= self._threshold:
                break

            current_query = assessment.reformulated_query

        return all_docs

    def _assess(self, original_query: str, docs: list[Document]) -> _ContextAssessment:
        context_preview = "\n\n".join(d.page_content[:300] for d in docs)
        prompt = (
            f"Original query: {original_query}\n\n"
            f"Retrieved context:\n{context_preview}\n\n"
            "Does this context contain enough information to answer the query? "
            "Score confidence 0-1. If low, provide a better reformulated query."
        )
        return self._assessor.invoke(prompt)
