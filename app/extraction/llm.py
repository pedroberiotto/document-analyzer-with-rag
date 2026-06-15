from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings


class FieldAnswer(BaseModel):
    value: str | None = Field(
        default=None,
        description="Extracted value for the requested field.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score between 0 and 1 (0 = low, 1 = high).",
    )
    justification: str | None = Field(
        default=None,
        description="Short justification based on the document context.",
    )


_llm: BaseChatModel | None = None
_prompt: ChatPromptTemplate | None = None


def build_chat_model(provider: str | None = None) -> BaseChatModel:
    s = get_settings()
    provider = (provider or s.llm_provider).lower()

    if provider == "openai":
        return ChatOpenAI(
            model=s.llm_model,
            temperature=s.llm_temperature,
            seed=s.llm_seed,
            timeout=s.llm_timeout,
            max_retries=s.llm_max_retries,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.ollama_model,
            temperature=s.llm_temperature,
            base_url=s.ollama_base_url,
            client_kwargs={"timeout": s.llm_timeout},
        )

    raise ValueError(f"Unknown llm_provider: {provider!r} (use openai|ollama)")


def build_structured_llm(llm: BaseChatModel) -> Runnable:
    return llm.with_structured_output(FieldAnswer)


def get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = build_chat_model()
    return _llm


def get_structured_llm() -> Runnable:
    return build_structured_llm(get_llm())


def get_prompt() -> ChatPromptTemplate:
    global _prompt
    if _prompt is None:
        _prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise information-extraction assistant. "
                    "Extract the requested field STRICTLY from the provided context, "
                    "following the field description exactly. "
                    "Return the value verbatim as it appears in the document — do not "
                    "paraphrase, translate, summarize or reformat it. "
                    "When the field description distinguishes between similar or related "
                    "entities, return the one that is asked for and not a related one. "
                    "If the value is not present in the provided context, return "
                    "value = null with a low confidence instead of guessing.",
                ),
                (
                    "user",
                    "Field to extract: {field_name}\n"
                    "Detailed instructions for this field: {field_description}\n\n"
                    "Document context (the only source you may use):\n{context}\n\n"
                    "Return the extracted value (verbatim, or null if absent), a "
                    "confidence between 0 and 1, and a short justification citing the "
                    "evidence from the context.",
                ),
            ]
        )
    return _prompt
