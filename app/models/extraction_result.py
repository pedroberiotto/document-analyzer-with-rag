from typing import Any

from pydantic import BaseModel


class SourceSpan(BaseModel):
    page: int | None = None
    text_snippet: str


class FieldResult(BaseModel):
    name: str
    value: Any | None
    confidence: float
    sources: list[SourceSpan] = []


class ExtractionResult(BaseModel):
    document_id: str
    schema_name: str
    fields: list[FieldResult]
