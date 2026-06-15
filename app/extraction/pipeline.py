import time

from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.config import get_settings
from app.errors import ExtractionError
from app.models import ExtractionResult, ExtractionSchema, FieldResult, SourceSpan
from app.retrieval import RetrieverBase
from app.telemetry.tracker import ExtractionContext, get_tracker

from .llm import FieldAnswer, get_prompt, get_structured_llm


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "?")
        parts.append(f"[Chunk {i} - page {page}]\n{d.page_content}")
    return "\n\n".join(parts)


def _build_sources(docs: list[Document]) -> list[SourceSpan]:
    sources: list[SourceSpan] = []
    for d in docs:
        normalized = " ".join(d.page_content.split())
        sources.append(SourceSpan(page=d.metadata.get("page"), text_snippet=normalized[:400]))
    return sources


def extract_fields(
    document_id: str,
    schema: ExtractionSchema,
    retriever: RetrieverBase,
    ctx: ExtractionContext | None = None,
    structured_llm: Runnable | None = None,
) -> ExtractionResult:
    ctx = ctx or ExtractionContext(persist=False)
    tracker = get_tracker()

    chain = (get_prompt() | (structured_llm or get_structured_llm())).with_retry(
        stop_after_attempt=get_settings().llm_max_retries,
        wait_exponential_jitter=True,
    )
    fields_results: list[FieldResult] = []
    doc_prompt_tokens = 0
    doc_completion_tokens = 0
    doc_start = time.perf_counter()

    for field in schema.fields:
        # Retrieval query is decoupled from the (detailed) extraction description:
        # a short natural-language query embeds far better against the document
        # than dumping the full field instructions into the search.
        query = field.retrieval_query or f"{field.name.replace('_', ' ')}: {field.description}"

        field_start = time.perf_counter()

        try:
            with get_openai_callback() as cb:
                docs: list[Document] = retriever.invoke(query)
                answer: FieldAnswer = chain.invoke(
                    {
                        "field_name": field.name,
                        "field_description": field.description,
                        "context": _format_context(docs),
                    }
                )
        except Exception as exc:  # noqa: BLE001 — surface a clean domain error
            raise ExtractionError(f"Extraction failed for field '{field.name}': {exc}") from exc
        field_latency = time.perf_counter() - field_start

        tracker.record_field(
            ctx,
            document_id,
            schema.name,
            field.name,
            predicted_value=answer.value,
            confidence=answer.confidence,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            latency_s=field_latency,
            n_sources=len(docs),
        )

        doc_prompt_tokens += cb.prompt_tokens
        doc_completion_tokens += cb.completion_tokens

        fields_results.append(
            FieldResult(
                name=field.name,
                value=answer.value,
                confidence=answer.confidence,
                sources=_build_sources(docs),
            )
        )

    tracker.record_document(
        ctx,
        document_id,
        schema.name,
        n_fields=len(fields_results),
        prompt_tokens=doc_prompt_tokens,
        completion_tokens=doc_completion_tokens,
        latency_s=time.perf_counter() - doc_start,
    )

    return ExtractionResult(
        document_id=document_id,
        schema_name=schema.name,
        fields=fields_results,
    )
