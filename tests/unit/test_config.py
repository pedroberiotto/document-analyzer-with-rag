from app.config import get_settings


def test_defaults():
    s = get_settings()
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.retrieval_k == 8
    assert s.llm_max_retries == 3


def test_active_model_is_provider_aware(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    get_settings.cache_clear()
    s = get_settings()
    assert s.active_llm_model == s.ollama_model
    assert s.active_embedding_model == s.ollama_embedding_model


def test_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "9")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    get_settings.cache_clear()
    s = get_settings()
    assert s.retrieval_k == 9
    assert s.chunk_size == 500


def test_derived_paths_under_data_dir():
    s = get_settings()
    assert s.chroma_dir.parent == s.data_dir
    assert s.uploads_dir.parent == s.data_dir
    assert s.telemetry_db.parent == s.data_dir
