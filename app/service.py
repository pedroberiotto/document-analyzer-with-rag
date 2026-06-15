import time
from dataclasses import dataclass

from langchain_chroma import Chroma

from app.config import get_settings
from app.extraction import build_chat_model, build_structured_llm, extract_fields
from app.ingestion import document_exists, index_document, load_vectorstore
from app.models import ExtractionResult, ExtractionSchema, RunTelemetry
from app.retrieval import RerankerType, RetrieverStrategy, build_retriever
from app.telemetry.tracker import ExtractionContext, get_tracker


@dataclass
class ExtractionOutcome:
    result: ExtractionResult
    telemetry: RunTelemetry


def _model_name(provider: str) -> str:
    s = get_settings()
    return s.ollama_model if provider.lower() == "ollama" else s.llm_model


class ExtractionService:
    def __init__(self) -> None:
        self._vectorstores: dict[str, Chroma] = {}

    def index_document(
        self,
        file_path: str,
        document_id: str,
        embedding_provider: str | None = None,
    ) -> None:
        self._vectorstores[document_id] = index_document(
            file_path, document_id, embedding_provider=embedding_provider
        )

    def has_document(self, document_id: str) -> bool:
        return document_id in self._vectorstores or document_exists(document_id)

    def _get_vectorstore(self, document_id: str, embedding_provider: str | None = None) -> Chroma:
        if document_id not in self._vectorstores:
            self._vectorstores[document_id] = load_vectorstore(
                document_id, embedding_provider=embedding_provider
            )
        return self._vectorstores[document_id]

    def extract(
        self,
        document_id: str,
        schema: ExtractionSchema,
        strategy: RetrieverStrategy = RetrieverStrategy.DENSE,
        reranker: RerankerType = RerankerType.NONE,
        k: int | None = None,
        llm_provider: str | None = None,
        embedding_provider: str | None = None,
    ) -> ExtractionOutcome:
        settings = get_settings()
        llm_provider = llm_provider or settings.llm_provider
        embedding_provider = embedding_provider or settings.embedding_provider

        vectorstore = self._get_vectorstore(document_id, embedding_provider)

        llm = build_chat_model(llm_provider)
        retriever = build_retriever(
            vectorstore,
            strategy=strategy,
            reranker=reranker,
            llm=llm if strategy == RetrieverStrategy.AGENTIC else None,
            k=k,
        )

        model = _model_name(llm_provider)
        ctx = ExtractionContext(
            strategy=strategy.value,
            reranker=reranker.value,
            model=model,
        )

        t0 = time.perf_counter()
        result = extract_fields(
            document_id, schema, retriever, ctx=ctx, structured_llm=build_structured_llm(llm)
        )
        latency = time.perf_counter() - t0

        records = get_tracker().pop_run_records(ctx.run_id)
        telemetry = RunTelemetry.from_records(
            run_id=ctx.run_id,
            model=model,
            strategy=strategy.value,
            reranker=reranker.value,
            records=records,
            latency_s=latency,
        )
        return ExtractionOutcome(result=result, telemetry=telemetry)


_service: ExtractionService | None = None


def get_service() -> ExtractionService:
    global _service
    if _service is None:
        _service = ExtractionService()
    return _service
