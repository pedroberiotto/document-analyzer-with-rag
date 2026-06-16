import pytest
from pydantic import ValidationError

from app.models import ExtractionField, ExtractionSchema


def test_parse_valid_schema():
    schema = ExtractionSchema(
        name="invoice",
        description="invoice fields",
        fields=[
            {"name": "total", "description": "total amount", "type": "number", "required": True},
            {"name": "issuer", "description": "issuer name"},
        ],
    )
    assert schema.name == "invoice"
    assert len(schema.fields) == 2
    assert schema.fields[1].type == "string"
    assert schema.fields[1].required is False


def test_field_defaults():
    f = ExtractionField(name="x", description="y")
    assert f.type == "string"
    assert f.required is False


def test_invalid_field_type_rejected():
    with pytest.raises(ValidationError):
        ExtractionField(name="x", description="y", type="banana")


def test_missing_required_key_rejected():
    with pytest.raises(ValidationError):
        ExtractionSchema(name="x", description="y")
