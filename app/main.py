from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.models.extraction_schema import ExtractionSchema
from app.models.extraction_result import ExtractionResult
from app.services.ingestion_langchain import (
    build_retriever_from_pdf,
    load_retriever_for_document,
)
from app.services.rag_langchain import extract_with_langchain


app = FastAPI(title="Document Analyser With RAG (LangChain + Chroma + OpenAI)")

BASE_PATH = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_PATH / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVERS: Dict[str, object] = {}
SCHEMAS: Dict[str, ExtractionSchema] = {}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, index it in Chroma and return a document_id.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for now.",
        )

    doc_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"

    content = await file.read()
    with file_path.open("wb") as f:
        f.write(content)

    retriever = build_retriever_from_pdf(str(file_path), document_id=doc_id)
    RETRIEVERS[doc_id] = retriever

    return {"document_id": doc_id, "filename": file.filename}


@app.post("/schemas", response_model=dict)
async def create_schema(schema: ExtractionSchema):
    """
    Register an extraction schema (fields + descriptions).
    """
    SCHEMAS[schema.name] = schema
    return {"schema_name": schema.name}


def _get_retriever(document_id: str):
    """
    Get a retriever for a given document_id, either from memory cache
    or by reloading from Chroma.
    """
    if document_id in RETRIEVERS:
        return RETRIEVERS[document_id]
    retriever = load_retriever_for_document(document_id)
    RETRIEVERS[document_id] = retriever
    return retriever


@app.post("/extract", response_model=ExtractionResult)
async def extract(document_id: str, schema_name: str):
    """
    Run the extraction (RAG) for a given document_id + schema_name.
    """
    retriever = _get_retriever(document_id)
    if retriever is None:
        raise HTTPException(
            status_code=404,
            detail="document_id not found.",
        )

    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail="schema not found.",
        )

    result = extract_with_langchain(
        document_id=document_id,
        schema=schema,
        retriever=retriever,
    )
    return result
