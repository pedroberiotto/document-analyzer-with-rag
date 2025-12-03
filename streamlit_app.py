import os
import json
from uuid import uuid4
from pathlib import Path

import streamlit as st

from app.models.extraction_schema import ExtractionSchema
from app.services.ingestion_langchain import build_retriever_from_pdf
from app.services.rag_langchain import extract_with_langchain

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

BASE_PATH = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_PATH / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Document Analyser RAG",
    layout="wide",
)

st.title("📄 Document Analyser RAG (OpenAI + LangChain + Chroma)")

st.markdown(
    """
This app lets you:

1. Upload a **PDF document**  
2. Provide a **JSON extraction schema** (which fields you want and what they mean)  
3. Run a **RAG pipeline** (OpenAI + LangChain + Chroma) to extract those fields  
4. See the **values** and the **source snippets** used as evidence
"""
)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    uploaded_file = st.file_uploader("📎 Upload a PDF", type=["pdf"])

    st.markdown("### 🔧 Extraction schema (JSON)")
    default_schema = {
        "name": "basic_invoice",
        "description": "Main fields from an invoice.",
        "fields": [
            {
                "name": "issuer_name",
                "description": "Name of the company issuing the invoice.",
                "type": "string",
                "required": True,
            },
            {
                "name": "invoice_number",
                "description": "Invoice number.",
                "type": "string",
                "required": True,
            },
            {
                "name": "issue_date",
                "description": "Invoice issue date.",
                "type": "date",
                "required": True,
            },
            {
                "name": "total_amount",
                "description": "Total amount of the invoice.",
                "type": "number",
                "required": True,
            },
        ],
    }

    schema_text = st.text_area(
        "Schema JSON",
        value=json.dumps(default_schema, indent=2),
        height=260,
    )

    run_button = st.button("▶️ Run extraction", type="primary")

with col_right:
    result_container = st.container()

if run_button:
    if not uploaded_file:
        st.error("Please upload a PDF first.")
    else:
        doc_id = str(uuid4())
        file_path = UPLOAD_DIR / f"{doc_id}.pdf"
        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            schema_dict = json.loads(schema_text)
            schema = ExtractionSchema(**schema_dict)
        except Exception as e:
            st.error(f"Invalid schema JSON: {e}")
        else:
            with st.spinner("Indexing document into Chroma..."):
                retriever = build_retriever_from_pdf(str(file_path), document_id=doc_id)

            with st.spinner("Running RAG field-by-field (OpenAI)..."):
                extraction_result = extract_with_langchain(
                    document_id=doc_id,
                    schema=schema,
                    retriever=retriever,
                )

            with result_container:
                st.markdown("## ✅ Extraction result")
                for field in extraction_result.fields:
                    st.markdown(
                        f"### 🏷️ `{field.name}` "
                        f"- confidence: **{field.confidence:.2f}**"
                    )
                    st.markdown(f"**Value:** `{field.value}`")

                    with st.expander("📚 Evidence"):
                        if not field.sources:
                            st.write("No sources returned.")
                        else:
                            for i, src in enumerate(field.sources, start=1):
                                st.markdown(
                                    f"**Snippet {i}** "
                                    f"(page: {src.page if src.page is not None else 'unknown'})"
                                )
                                st.caption(src.text_snippet)
                                st.markdown("---")
