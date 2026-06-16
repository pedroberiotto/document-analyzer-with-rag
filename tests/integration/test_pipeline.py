import pytest
from langchain_core.runnables import RunnableLambda

from app.errors import ExtractionError
from app.extraction import pipeline
from app.extraction.llm import FieldAnswer
from app.models import ExtractionSchema
from app.telemetry.tracker import ExtractionContext, get_tracker


def _mock_structured_llm(value="MOCK", confidence=0.77):
    return lambda: RunnableLambda(lambda _prompt: FieldAnswer(value=value, confidence=confidence))


def test_extract_fields_happy_path(monkeypatch, fake_retriever):
    monkeypatch.setattr(pipeline, "get_structured_llm", _mock_structured_llm("ACME", 0.9))

    schema = ExtractionSchema(
        name="invoice",
        description="invoice fields",
        fields=[
            {"name": "issuer", "description": "issuer name"},
            {"name": "total", "description": "total amount", "type": "number"},
        ],
    )
    ctx = ExtractionContext(strategy="dense", reranker="none", model="gpt-4.1-mini", persist=False)

    result = pipeline.extract_fields("doc1", schema, fake_retriever, ctx=ctx)

    assert result.document_id == "doc1"
    assert [f.name for f in result.fields] == ["issuer", "total"]
    assert all(f.value == "ACME" and f.confidence == 0.9 for f in result.fields)

    assert result.fields[0].sources
    assert result.fields[0].sources[0].page == 0


def test_extract_fields_records_telemetry(monkeypatch, fake_retriever):
    monkeypatch.setattr(pipeline, "get_structured_llm", _mock_structured_llm())

    schema = ExtractionSchema(name="s", description="d", fields=[{"name": "a", "description": "x"}])
    ctx = ExtractionContext(run_id="test-run-telemetry", persist=False)

    pipeline.extract_fields("doc2", schema, fake_retriever, ctx=ctx)

    records = get_tracker().pop_run_records("test-run-telemetry")
    assert len(records) == 1
    assert records[0]["field_name"] == "a"
    assert "total_tokens" in records[0]
    assert "cost_usd" in records[0]


def test_extract_fields_wraps_llm_failure(monkeypatch, fake_retriever):

    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    from app.config import get_settings

    get_settings.cache_clear()

    def boom(_prompt):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(pipeline, "get_structured_llm", lambda: RunnableLambda(boom))

    schema = ExtractionSchema(name="s", description="d", fields=[{"name": "a", "description": "x"}])

    with pytest.raises(ExtractionError):
        pipeline.extract_fields("doc3", schema, fake_retriever)
