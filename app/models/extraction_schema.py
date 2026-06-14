from typing import Literal

from pydantic import BaseModel

FieldType = Literal["string", "number", "date", "boolean", "list"]


class ExtractionField(BaseModel):
    name: str
    description: str
    type: FieldType = "string"
    required: bool = False
    retrieval_query: str | None = None


class ExtractionSchema(BaseModel):
    name: str
    description: str
    fields: list[ExtractionField]
