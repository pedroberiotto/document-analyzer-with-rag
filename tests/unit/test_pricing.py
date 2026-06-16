from app.telemetry.pricing import compute_cost, model_prices


def test_known_model_cost():

    cost = compute_cost("gpt-4.1-mini", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == 0.40
    cost = compute_cost("gpt-4.1-mini", prompt_tokens=0, completion_tokens=1_000_000)
    assert cost == 1.60


def test_combined_cost():
    cost = compute_cost("gpt-4.1-mini", prompt_tokens=1000, completion_tokens=200)
    assert round(cost, 6) == round((1000 / 1e6) * 0.40 + (200 / 1e6) * 1.60, 6)


def test_local_model_is_free():
    assert compute_cost("llama3.2:3b", 5000, 2000) == 0.0


def test_unknown_model_returns_zero():
    assert compute_cost("does-not-exist", 1000, 1000) == 0.0


def test_model_prices_lookup():
    assert model_prices("gpt-4.1-mini") is not None
    assert model_prices("nope") is None
