from pydantic import BaseModel


class FieldUsage(BaseModel):
    field_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    n_sources: int = 0


class RunTelemetry(BaseModel):
    run_id: str
    model: str
    strategy: str
    reranker: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    fields: list[FieldUsage] = []

    @classmethod
    def from_records(
        cls,
        *,
        run_id: str,
        model: str,
        strategy: str,
        reranker: str,
        records: list[dict],
        latency_s: float,
    ) -> "RunTelemetry":
        fields = [
            FieldUsage(
                field_name=r["field_name"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                total_tokens=r["total_tokens"],
                cost_usd=r["cost_usd"],
                latency_s=r["latency_s"],
                n_sources=r["n_sources"],
            )
            for r in records
        ]
        return cls(
            run_id=run_id,
            model=model,
            strategy=strategy,
            reranker=reranker,
            prompt_tokens=sum(f.prompt_tokens for f in fields),
            completion_tokens=sum(f.completion_tokens for f in fields),
            total_tokens=sum(f.total_tokens for f in fields),
            cost_usd=round(sum(f.cost_usd for f in fields), 6),
            latency_s=round(latency_s, 3),
            fields=fields,
        )


class ExtractionResponse(BaseModel):
    result: "ExtractionResultRef"
    telemetry: RunTelemetry


from app.models.extraction_result import ExtractionResult as ExtractionResultRef  # noqa: E402

ExtractionResponse.model_rebuild()
