from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document


from app.models.extraction_schema import ExtractionSchema
from app.models.extraction_result import (
    ExtractionResult,
    FieldResult,
    SourceSpan,
)


class FieldAnswer(BaseModel):
    """
    Structured answer returned by the LLM for a single field.
    """
    value: str | None = Field(
        default=None,
        description="Extracted value for the requested field."
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score between 0 and 1 (0 = low, 1 = high)."
    )
    justification: str | None = Field(
        default=None,
        description="Short justification based on the document context."
    )


_llm: ChatOpenAI | None = None
_prompt: ChatPromptTemplate | None = None


def get_llm() -> ChatOpenAI:
    """
    Simple singleton for ChatOpenAI.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
        )
    return _llm


def get_structured_llm():
    """
    Wraps the base LLM to always return a FieldAnswer using structured outputs.
    """
    llm = get_llm()
    return llm.with_structured_output(FieldAnswer)


def get_prompt() -> ChatPromptTemplate:
    """
    Base prompt for field extraction.
    """
    global _prompt
    if _prompt is None:
        _prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant specialized in extracting specific fields "
                    "from documents. Use ONLY the provided context. "
                    "If you are not sure, return value = null and a low confidence."
                ),
                (
                    "user",
                    "Field: {field_name}\n"
                    "Field description: {field_description}\n\n"
                    "Document context:\n{context}\n\n"
                    "Return the field value, a confidence between 0 and 1, "
                    "and a short justification."
                ),
            ]
        )
    return _prompt


def _format_context(docs: List[Document]) -> str:
    """
    Formats the retrieved Documents into a single context string for the prompt.
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        page = d.metadata.get("page", "?")
        parts.append(f"[Chunk {i} - page {page}]\n{d.page_content}")
    return "\n\n".join(parts)


def _build_sources(docs: List[Document]) -> List[SourceSpan]:
    """
    Builds the list of evidence sources from the retrieved Documents.
    Normalizes whitespace so snippets don't show one word per line.
    """
    sources: List[SourceSpan] = []
    for d in docs:
        page = d.metadata.get("page")

        # Collapse all whitespace (spaces, newlines, tabs) into single spaces
        normalized_text = " ".join(d.page_content.split())

        # Limit snippet size so it doesn't get huge
        snippet = normalized_text[:400]

        sources.append(
            SourceSpan(
                page=page,
                text_snippet=snippet,
            )
        )
    return sources


def extract_with_langchain(
    document_id: str,
    schema: ExtractionSchema,
    retriever,
) -> ExtractionResult:
    """
    Runs the RAG pipeline field-by-field:
    - uses the retriever to get relevant context for each field;
    - calls the LLM with structured output;
    - returns an ExtractionResult.
    """
    structured_llm = get_structured_llm()
    prompt = get_prompt()

    chain = prompt | structured_llm
    fields_results: List[FieldResult] = []

    for field in schema.fields:
        # 1) Build a question to retrieve relevant context
        question = (
            f"Field: {field.name}. {field.description}. "
            "What is the value of this field in the document?"
        )

        docs: List[Document] = retriever.invoke(question)
        context = _format_context(docs)

        chain_input = {
            "field_name": field.name,
            "field_description": field.description,
            "context": context,
        }

        # 2) Call LLM with structured output -> FieldAnswer
        answer: FieldAnswer = chain.invoke(chain_input)

        # 3) Build evidence sources
        sources = _build_sources(docs)

        fields_results.append(
            FieldResult(
                name=field.name,
                value=answer.value,
                confidence=answer.confidence,
                sources=sources,
            )
        )

    return ExtractionResult(
        document_id=document_id,
        schema_name=schema.name,
        fields=fields_results,
    )
