from pydantic import BaseModel
from typing import List, Literal

FieldType = Literal["string", "number", "date", "boolean", "list"]


class ExtractionField(BaseModel):
    """
    Defines a single field to be extracted from the document.
    """
    name: str
    description: str
    type: FieldType = "string"
    required: bool = False


class ExtractionSchema(BaseModel):
    """
    Full extraction schema (multiple fields).
    """
    name: str
    description: str
    fields: List[ExtractionField]
