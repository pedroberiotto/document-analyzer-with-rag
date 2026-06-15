from pathlib import Path

import structlog
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.errors import DocumentNotFoundError, InvalidDocumentError

log = structlog.get_logger(__name__)


def _embeddings(provider: str | None = None) -> Embeddings:
    s = get_settings()
    provider = (provider or s.embedding_provider).lower()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=s.ollama_embedding_model,
            base_url=s.ollama_base_url,
        )
    return OpenAIEmbeddings(model=s.embedding_model)


def _persist_dir(document_id: str) -> Path:
    return get_settings().chroma_dir / document_id


def _collection_name(document_id: str) -> str:
    return f"doc_{document_id}"


def _table_to_markdown(rows: list[list]) -> str:
    cleaned = [
        [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]
        for row in rows
        if row
    ]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = [
        "| " + " | ".join(cleaned[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in cleaned[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _extract_table_docs(pdf_path: Path) -> list[Document]:
    import pdfplumber

    table_docs: list[Document] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:  # noqa: BLE001 — skip pages pdfplumber can't parse
                continue
            for t_idx, table in enumerate(tables):
                md = _table_to_markdown(table)
                if not md:
                    continue
                table_docs.append(
                    Document(
                        page_content=f"[Table on page {page_idx}]\n{md}",
                        metadata={"page": page_idx, "source_type": "table", "table_index": t_idx},
                    )
                )
    return table_docs


def index_document(
    file_path: str,
    document_id: str,
    force: bool | None = None,
    embedding_provider: str | None = None,
) -> Chroma:
    settings = get_settings()
    force = settings.reindex_if_exists if force is None else force

    if not force and document_exists(document_id):
        log.info("index_cache_hit", document_id=document_id)
        return load_vectorstore(document_id, embedding_provider=embedding_provider)

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise InvalidDocumentError(f"File not found: {file_path}")

    try:
        docs = PyPDFLoader(str(pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        splits = splitter.split_documents(docs)

        n_tables = 0
        if settings.extract_tables:
            table_docs = _extract_table_docs(pdf_path)
            n_tables = len(table_docs)
            splits = splits + table_docs
    except Exception as exc:  # noqa: BLE001 — surface a clean domain error
        raise InvalidDocumentError(f"Could not read PDF '{file_path}': {exc}") from exc

    if not splits:
        raise InvalidDocumentError(f"No extractable text in PDF '{file_path}'.")

    log.info(
        "indexing_document",
        document_id=document_id,
        pages=len(docs),
        text_chunks=len(splits) - n_tables,
        table_chunks=n_tables,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return Chroma.from_documents(
        documents=splits,
        embedding=_embeddings(embedding_provider),
        persist_directory=str(_persist_dir(document_id)),
        collection_name=_collection_name(document_id),
    )


def load_vectorstore(document_id: str, embedding_provider: str | None = None) -> Chroma:
    persist_dir = _persist_dir(document_id)
    if not persist_dir.exists():
        raise DocumentNotFoundError(f"No indexed data for document_id={document_id}")

    return Chroma(
        embedding_function=_embeddings(embedding_provider),
        persist_directory=str(persist_dir),
        collection_name=_collection_name(document_id),
    )


def document_exists(document_id: str) -> bool:
    return _persist_dir(document_id).exists()


def list_documents() -> list[str]:
    chroma_dir = get_settings().chroma_dir
    if not chroma_dir.exists():
        return []
    return sorted(p.name for p in chroma_dir.iterdir() if p.is_dir())
