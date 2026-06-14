import json
from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=1)
def _load_pricing() -> dict:
    with get_settings().pricing_file.open() as f:
        return json.load(f)


def model_prices(model: str) -> dict | None:
    return _load_pricing()["models"].get(model)


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    prices = model_prices(model)
    if prices is None:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * prices["input_per_1m"]
    output_cost = (completion_tokens / 1_000_000) * prices["output_per_1m"]
    return round(input_cost + output_cost, 8)


def list_models() -> list[str]:
    return list(_load_pricing()["models"].keys())
