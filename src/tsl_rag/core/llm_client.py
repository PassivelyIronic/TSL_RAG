from __future__ import annotations

from openai import AsyncOpenAI

from tsl_rag.core.settings import Settings


def get_llm_client(settings: Settings) -> AsyncOpenAI:
    """
    Zwraca klienta AsyncOpenAI wskazującego na:
    - Ollama local  (base_url=localhost:11434/v1)
    - OpenAI cloud  (base_url=domyślny)

    Ollama udostępnia endpoint /v1 kompatybilny z OpenAI SDK.
    Zmiana providera = 1 linia w .env, zero zmian w kodzie.
    """
    if settings.llm_provider == "ollama":
        return AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",  # Ollama ignoruje tę wartość, SDK wymaga niepustej
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY nie jest ustawiony")

    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    )


async def get_embedding(
    text: str,
    settings: Settings,
    client: AsyncOpenAI,
) -> list[float]:
    model = (
        settings.ollama_embed_model
        if settings.llm_provider == "ollama"
        else settings.openai_embedding_model
    )
    kwargs: dict = {"model": model, "input": text}
    if settings.llm_provider == "openai":
        kwargs["dimensions"] = settings.openai_embedding_dimensions

    response = await client.embeddings.create(**kwargs)
    return response.data[0].embedding


async def get_embeddings_batch(
    texts: list[str],
    settings: Settings,
    client: AsyncOpenAI,
    batch_size: int = 32,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        model = (
            settings.ollama_embed_model
            if settings.llm_provider == "ollama"
            else settings.openai_embedding_model
        )
        kwargs: dict = {"model": model, "input": batch}
        if settings.llm_provider == "openai":
            kwargs["dimensions"] = settings.openai_embedding_dimensions

        response = await client.embeddings.create(**kwargs)
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings
