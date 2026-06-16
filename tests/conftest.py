import pytest
from langchain_core.documents import Document

from app.retrieval import RetrieverBase


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeRetriever(RetrieverBase):
    def __init__(self, docs: list[Document]):
        self._docs = docs

    def invoke(self, query: str) -> list[Document]:  # noqa: ARG002
        return self._docs


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever(
        [
            Document(page_content="ACME Corp issued invoice INV-2024-001.", metadata={"page": 0}),
            Document(page_content="Total amount due: 1250.00 USD.", metadata={"page": 1}),
        ]
    )
