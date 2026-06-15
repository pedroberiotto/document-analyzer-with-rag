from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.errors import DocumentNotFoundError, ExtractionError, InvalidDocumentError
from app.models import ExtractionResponse, ExtractionSchema
from app.retrieval import RerankerType, RetrieverStrategy
from app.service import get_service
from app.telemetry import configure_logging

SCHEMAS: dict[str, ExtractionSchema] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    get_settings().ensure_dirs()
    yield


app = FastAPI(
    title="Document Analyser With RAG",
    description="Extract structured fields from PDFs using RAG (LangChain + Chroma + OpenAI).",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid4())
    file_path = get_settings().uploads_dir / f"{doc_id}.pdf"
    file_path.write_bytes(await file.read())

    try:
        get_service().index_document(str(file_path), document_id=doc_id)
    except InvalidDocumentError as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"document_id": doc_id, "filename": file.filename}


@app.get("/documents")
async def list_documents() -> list[str]:
    from app.ingestion import list_documents as _list

    return _list()


@app.post("/schemas")
async def create_schema(schema: ExtractionSchema):
    SCHEMAS[schema.name] = schema
    return {"schema_name": schema.name}


@app.get("/schemas")
async def list_schemas() -> list[str]:
    return sorted(SCHEMAS.keys())


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    document_id: str,
    schema_name: str,
    retriever_strategy: RetrieverStrategy = RetrieverStrategy.DENSE,
    reranker: RerankerType = RerankerType.NONE,
    k: int | None = None,
):
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise HTTPException(status_code=404, detail="schema not found.")

    try:
        outcome = get_service().extract(
            document_id=document_id,
            schema=schema,
            strategy=retriever_strategy,
            reranker=reranker,
            k=k,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document_id not found.") from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExtractionResponse(result=outcome.result, telemetry=outcome.telemetry)
