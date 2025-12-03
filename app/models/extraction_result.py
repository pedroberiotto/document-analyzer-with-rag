from pydantic import BaseModel
from typing import List, Optional, Any


class SourceSpan(BaseModel):
    """
    Represents a snippet of the document used as evidence for an answer.
    """
    page: Optional[int] = None
    text_snippet: str


class FieldResult(BaseModel):
    """
    Extraction result for a single field.
    """
    name: str
    value: Optional[Any]
    confidence: float
    sources: List[SourceSpan] = []


class ExtractionResult(BaseModel):
    """
    Full extraction result for a given document + schema.
    """
    document_id: str
    schema_name: str
    fields: List[FieldResult]
