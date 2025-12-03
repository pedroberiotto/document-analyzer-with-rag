from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


BASE_PATH = Path(__file__).resolve().parent.parent.parent
CHROMA_BASE_DIR = BASE_PATH / "data" / "chroma"
CHROMA_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Returns the OpenAI embeddings object.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def build_retriever_from_pdf(file_path: str, document_id: str):
    """
    Reads a PDF, splits it into chunks, indexes in a Chroma vector store
    and returns a retriever for that document_id.
    Each document gets its own Chroma directory.
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(file_path)

    # 1) Load PDF
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(docs)

    # 3) OpenAI embeddings
    embeddings = _get_embeddings()
    persist_dir = CHROMA_BASE_DIR / document_id

    # 4) Create persistent Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"doc_{document_id}",
    )

    # 5) Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def load_retriever_for_document(document_id: str):
    """
    Reopens an existing Chroma collection from disk and returns a retriever.
    """
    embeddings = _get_embeddings()
    persist_dir = CHROMA_BASE_DIR / document_id

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Persist directory for document_id={document_id} not found: {persist_dir}"
        )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"doc_{document_id}",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever
