from app.models import SourceSpan
from eval.metrics import (
    compute_faithfulness,
    compute_field_match,
    compute_retrieval_precision,
)


def test_field_match_string_normalized():
    assert compute_field_match("ACME  Corp.", "acme corp") == 1.0
    assert compute_field_match("Acme", "Globex") == 0.0


def test_field_match_numeric_tolerance():
    assert compute_field_match("1,250.00", "1250") == 1.0
    assert compute_field_match("100", "200") == 0.0


def test_field_match_none_ground_truth_is_skipped():

    assert compute_field_match("anything", None) == 1.0


def test_field_match_missing_prediction():
    assert compute_field_match(None, "expected") == 0.0


def test_retrieval_precision_found_in_sources():
    sources = [SourceSpan(page=1, text_snippet="The total is 1250.00 USD")]
    assert compute_retrieval_precision(sources, "1250.00") == 1.0
    assert compute_retrieval_precision(sources, "9999") == 0.0


def test_faithfulness_grounded_vs_hallucinated():
    sources = [SourceSpan(page=1, text_snippet="Issued by ACME Corp")]
    assert compute_faithfulness("ACME Corp", sources) == 1.0
    assert compute_faithfulness("Globex Inc", sources) == 0.0
