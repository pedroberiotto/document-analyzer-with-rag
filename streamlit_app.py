import json
import os
from uuid import uuid4

import streamlit as st

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from app.config import get_settings  # noqa: E402
from app.models import ExtractionSchema  # noqa: E402
from app.retrieval import RerankerType, RetrieverStrategy  # noqa: E402
from app.service import ExtractionOutcome, ExtractionService  # noqa: E402
from app.telemetry import configure_logging  # noqa: E402

configure_logging()
SETTINGS = get_settings()

PROVIDER_CHOICES = {
    "OpenAI": ["openai"],
    "Ollama (local)": ["ollama"],
    "Both — compare": ["openai", "ollama"],
}

DEFAULT_SCHEMA = {
    "name": "rag_paper",
    "description": "Core components of a RAG paper.",
    "fields": [
        {
            "name": "knowledge_source",
            "description": "Non-parametric memory source. Single name.",
            "type": "string",
            "required": True,
        },
        {
            "name": "generator_model",
            "description": "Pre-trained seq2seq generator. Name/acronym only.",
            "type": "string",
            "required": True,
        },
        {
            "name": "retriever",
            "description": "Neural retriever. Acronym only.",
            "type": "string",
            "required": True,
        },
        {
            "name": "search_method",
            "description": "Top-K search algorithm. Acronym only.",
            "type": "string",
            "required": True,
        },
    ],
}


@st.cache_resource
def get_service() -> ExtractionService:
    return ExtractionService()


def render_outcome(container, provider: str, outcome: ExtractionOutcome) -> None:
    result, tel = outcome.result, outcome.telemetry
    usage_by_field = {f.field_name: f for f in tel.fields}
    with container:
        st.markdown(f"#### {provider} · `{tel.model}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cost", f"${tel.cost_usd:.4f}")
        m2.metric("Tokens", f"{tel.total_tokens:,}")
        m3.metric("Latency", f"{tel.latency_s:.1f}s")
        st.divider()
        for field in result.fields:
            usage = usage_by_field.get(field.name)
            st.markdown(f"**`{field.name}`** → `{field.value}`")
            st.progress(
                min(max(field.confidence, 0.0), 1.0),
                text=f"confidence {field.confidence:.2f}",
            )
            if usage:
                st.caption(
                    f"💵 ${usage.cost_usd:.5f} · 🔤 {usage.total_tokens:,} tok · ⏱️ {usage.latency_s:.1f}s"
                )
            with st.expander(f"📚 Evidence ({len(field.sources)})"):
                for i, src in enumerate(field.sources, start=1):
                    page = src.page if src.page is not None else "unknown"
                    st.markdown(f"**Snippet {i}** (page: {page})")
                    st.caption(src.text_snippet)
            st.divider()


st.set_page_config(page_title="Document Analyser RAG", layout="wide")
st.title("📄 Document Analyser RAG")
st.caption("Extract structured fields from PDFs — LangChain · Chroma · OpenAI / Ollama")

with st.sidebar:
    st.header("🔌 Provider")
    provider_label = st.radio(
        "Model provider",
        list(PROVIDER_CHOICES),
        help="Pick a provider, or run both side by side to compare quality / cost / latency.",
    )
    providers = PROVIDER_CHOICES[provider_label]
    st.caption(f"OpenAI: `{SETTINGS.llm_model}` · Ollama: `{SETTINGS.ollama_model}`")
    if "ollama" in providers:
        st.caption("⚠️ Ollama must be running (`ollama serve`).")

    st.header("⚙️ Retrieval")
    _strategies = [s.value for s in RetrieverStrategy]
    _rerankers = [r.value for r in RerankerType]
    strategy = RetrieverStrategy(
        st.selectbox(
            "Strategy",
            _strategies,
            index=_strategies.index(SETTINGS.default_strategy)
            if SETTINGS.default_strategy in _strategies
            else 0,
            help="dense: vectors only · hybrid: BM25 + dense (RRF) · agentic: iterative re-querying",
        )
    )
    reranker = RerankerType(
        st.selectbox(
            "Reranker",
            _rerankers,
            help="cross_encoder re-ranks results (needs the 'reranker' extra installed)",
        )
    )
    k = st.slider("Chunks retrieved (k)", min_value=1, max_value=15, value=SETTINGS.retrieval_k)

uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])
st.markdown("#### 🔧 Extraction schema (JSON)")
schema_text = st.text_area(
    "Schema JSON",
    value=json.dumps(DEFAULT_SCHEMA, indent=2),
    height=240,
    label_visibility="collapsed",
)
run_button = st.button("▶️ Run extraction", type="primary")

if run_button:
    if not uploaded_file:
        st.error("Please upload a PDF first.")
        st.stop()
    try:
        schema = ExtractionSchema(**json.loads(schema_text))
    except Exception as exc:
        st.error(f"Invalid schema JSON: {exc}")
        st.stop()

    base_id = str(uuid4())
    file_path = SETTINGS.uploads_dir / f"{base_id}.pdf"
    file_path.write_bytes(uploaded_file.getbuffer())
    service = get_service()

    st.markdown("### ✅ Results")
    columns = st.columns(len(providers))

    for col, provider in zip(columns, providers, strict=True):
        label = "OpenAI" if provider == "openai" else "Ollama"
        # Provider-scoped document id so each embeds into its own collection.
        doc_id = f"{base_id}_{provider}"
        try:
            with st.spinner(f"[{label}] indexing…"):
                service.index_document(str(file_path), doc_id, embedding_provider=provider)
            with st.spinner(f"[{label}] extracting ({strategy.value}, k={k})…"):
                outcome = service.extract(
                    document_id=doc_id,
                    schema=schema,
                    strategy=strategy,
                    reranker=reranker,
                    k=k,
                    llm_provider=provider,
                    embedding_provider=provider,
                )
            render_outcome(col, label, outcome)
        except Exception as exc:
            col.error(f"**{label}** failed: {exc}")
