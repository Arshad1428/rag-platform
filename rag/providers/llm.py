"""AI models (the inference layer). Profile decides where they physically run:
  ollama        on-prem GPU server (profile C)
  azure_openai  Azure OpenAI (profiles A and B; B = same, via private endpoint)
Every other file only calls chat_model(), vision_model(), embeddings() and structured()."""
from functools import lru_cache

from rag.config import settings


@lru_cache
def _azure_token():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    return get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


def _azure_chat(deployment: str, temperature: float = 0):
    from langchain_openai import AzureChatOpenAI
    kw = dict(azure_endpoint=settings.aoai_endpoint, azure_deployment=deployment,
              api_version=settings.aoai_api_version, azure_ad_token_provider=_azure_token(),
              max_retries=6)  # retries with back-off on rate limits (HTTP 429)
    if not deployment.startswith(("gpt-5", "o")):  # reasoning models only accept the default
        kw["temperature"] = temperature
    return AzureChatOpenAI(**kw)


def _ollama_chat(model: str, ctx: int = 16384):
    from langchain_ollama import ChatOllama
    return ChatOllama(base_url=settings.ollama_url, model=model, temperature=0, num_ctx=ctx,
                      reasoning=False)


def chat_model(small: bool = False):
    if settings.llm_provider == "ollama":
        return _ollama_chat(settings.ollama_small_model if small else settings.ollama_chat_model)
    if settings.llm_provider == "azure_openai":
        return _azure_chat(settings.aoai_small_deployment if small else settings.aoai_chat_deployment)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider}")


def vision_model():
    if settings.llm_provider == "ollama":
        return _ollama_chat(settings.ollama_vision_model, ctx=8192)
    return _azure_chat(settings.aoai_vision_deployment)


@lru_cache
def embeddings():
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=settings.ollama_url, model=settings.ollama_embed_model)
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(azure_endpoint=settings.aoai_endpoint,
                                 azure_deployment=settings.aoai_embed_deployment,
                                 api_version=settings.aoai_api_version,
                                 azure_ad_token_provider=_azure_token(),
                                 dimensions=settings.embed_dimensions, max_retries=6)


def structured(llm, schema):
    """Force the model to answer in a fixed shape (a Pydantic class)."""
    if settings.llm_provider == "ollama":
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)
