from enum import Enum

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings

from .agentic import AgenticRetriever
from .base import RetrieverBase
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .reranker import RerankerRetriever


class RetrieverStrategy(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    AGENTIC = "agentic"


class RerankerType(str, Enum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"


def build_retriever(
    vectorstore: Chroma,
    strategy: RetrieverStrategy = RetrieverStrategy.DENSE,
    reranker: RerankerType = RerankerType.NONE,
    llm: BaseChatModel | None = None,
    k: int | None = None,
) -> RetrieverBase:
    settings = get_settings()
    k = k if k is not None else settings.retrieval_k

    if strategy == RetrieverStrategy.DENSE:
        retriever: RetrieverBase = DenseRetriever(vectorstore, k=k)
    elif strategy == RetrieverStrategy.HYBRID:
        retriever = HybridRetriever(vectorstore, k=k, rrf_k=settings.rrf_k)
    elif strategy == RetrieverStrategy.AGENTIC:
        if llm is None:
            raise ValueError("llm is required for AgenticRetriever")
        retriever = AgenticRetriever(
            DenseRetriever(vectorstore, k=k),
            llm=llm,
            confidence_threshold=settings.agentic_confidence_threshold,
            max_iterations=settings.agentic_max_iterations,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if reranker == RerankerType.CROSS_ENCODER:
        retriever = RerankerRetriever(retriever, model_name=settings.reranker_model, top_k=k)

    return retriever
