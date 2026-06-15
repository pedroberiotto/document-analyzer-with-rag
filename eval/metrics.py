import re
import unicodedata

from app.models import SourceSpan


def _normalize_str(value: str | None) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = re.sub(r"[^\d.,\-]", "", str(value))

    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def compute_field_match(
    predicted: str | None,
    ground_truth: str | None,
    numeric_tolerance: float = 0.01,
) -> float:
    if ground_truth is None:
        return 1.0

    if predicted is None:
        return 0.0

    pred_f = _to_float(predicted)
    gt_f = _to_float(ground_truth)
    if pred_f is not None and gt_f is not None:
        if gt_f == 0:
            return 1.0 if pred_f == 0 else 0.0
        return 1.0 if abs(pred_f - gt_f) / abs(gt_f) <= numeric_tolerance else 0.0

    return 1.0 if _normalize_str(predicted) == _normalize_str(ground_truth) else 0.0


def compute_retrieval_precision(
    sources: list[SourceSpan],
    ground_truth: str | None,
) -> float:
    if ground_truth is None or not sources:
        return 1.0

    gt_norm = _normalize_str(ground_truth)
    if not gt_norm:
        return 1.0

    for src in sources:
        if gt_norm in _normalize_str(src.text_snippet):
            return 1.0

    gt_f = _to_float(ground_truth)
    if gt_f is not None:
        gt_digits = re.sub(r"\D", "", str(ground_truth))
        for src in sources:
            if gt_digits and gt_digits in re.sub(r"\D", "", src.text_snippet):
                return 1.0

    return 0.0


def compute_faithfulness(
    predicted: str | None,
    sources: list[SourceSpan],
) -> float:
    if predicted is None or not sources:
        return 1.0

    pred_norm = _normalize_str(predicted)
    if not pred_norm:
        return 1.0

    combined = _normalize_str(" ".join(src.text_snippet for src in sources))

    if pred_norm in combined:
        return 1.0

    pred_f = _to_float(predicted)
    if pred_f is not None:
        pred_digits = re.sub(r"\D", "", str(predicted))
        if pred_digits and pred_digits in re.sub(r"\D", "", combined):
            return 1.0

    return 0.0
