# 📄 Document Analyser RAG

> A production-ready AI pipeline for extracting structured data from PDFs using RAG with OpenAI, LangChain, and Chroma.

![App](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![Backend](https://img.shields.io/badge/API-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-OpenAI-412991)
![RAG](https://img.shields.io/badge/Pattern-RAG-1f6feb)
![Vector%20Store](https://img.shields.io/badge/Vector%20Store-Chroma-00c853)
![Orchestration](https://img.shields.io/badge/Orchestration-LangChain-1a73e8)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## 🧠 RAG Architecture

High-level flow from PDF upload to structured output:

```mermaid
flowchart LR
    A[Upload PDF\n(Streamlit / API)] --> B[PDF Loader\n(PyPDFLoader)]
    B --> C[Text Splitter\n(RecursiveCharacterTextSplitter)]
    C --> D[Embeddings\n(OpenAI text-embedding-3-small)]
    D --> E[Vector Store\n(Chroma per document_id)]

    subgraph RAG Loop per Field
        F[Extraction Schema\n(JSON: fields + descriptions)]
        F --> G[Build Field Query]
        G --> H[Retriever\n(E.from Chroma)]
        H --> I[LLM + Prompt\n(ChatOpenAI gpt-4.1-mini)]
        I --> J[FieldAnswer\n(value, confidence, justification)]
    end

    E --> H
    J --> K[Aggregation\nExtractionResult]
    K --> L[UI Response\n(values + confidence + evidence)]
```

---

## ✨ Features

- 🔌 **Bring-your-own schema**  
  Define custom fields in JSON (name + description + type).

- 📚 **RAG over a single document**  
  Chunks the PDF, embeds with OpenAI, and uses a vector store (Chroma) to find the most relevant parts.

- 🧠 **LLM-powered field extraction**  
  Uses `gpt-4.1-mini` (via LangChain + OpenAI) with structured outputs (Pydantic).

- 🔍 **Traceable answers**  
  Every field comes with:
  - `confidence` (0–1)
  - `sources[]` (page + text snippet used as evidence)

- 🖥️ **Streamlit UI**  
  Simple interface to test schemas and documents without writing code.

- 🧪 **API-ready**  
  FastAPI backend exposes `/documents/upload`, `/schemas`, `/extract` for programmatic use.

---

## 📂 Project Structure

```bash
document-analyser-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app (optional, for REST API)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── extraction_result.py # Output models (ExtractionResult, FieldResult, SourceSpan)
│   │   └── extraction_schema.py # Input models (ExtractionSchema, ExtractionField)
│   └── services/
│       ├── __init__.py
│       ├── ingestion_langchain.py  # PDF loading, splitting, embeddings, Chroma retriever
│       └── rag_langchain.py        # Field-by-field RAG pipeline
├── data/
│   ├── uploads/                 # Uploaded PDFs (created at runtime)
│   └── chroma/                  # Chroma persistence per document_id (created at runtime)
├── streamlit_app.py             # Streamlit UI entrypoint
├── requirements.txt
├── .gitignore
└── README.md
