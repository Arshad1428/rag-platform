"""All settings in one place. Values come from environment variables or a .env file.

PROFILE picks sensible defaults for each deployment profile:
  A  connected cloud     Azure OpenAI + Document Intelligence + Azure AI Search + Blob
  B  restricted network  same services as A, reached through private endpoints only
  C  on-prem / air-gap   Ollama models + Docling + Qdrant + local disk, no internet

Every individual choice (LLM_PROVIDER, READER, INDEX, STORAGE, AUTH_MODE) can still be
overridden, so mixed setups are possible (e.g. profile A with Qdrant instead of AI Search).
"""
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILE_DEFAULTS = {
    "A": {"llm_provider": "azure_openai", "reader": "azure_di", "index": "azure_search",
          "storage": "azure_blob", "auth_mode": "entra_easyauth"},
    "B": {"llm_provider": "azure_openai", "reader": "azure_di", "index": "azure_search",
          "storage": "azure_blob", "auth_mode": "entra_easyauth"},
    "C": {"llm_provider": "ollama", "reader": "docling", "index": "qdrant",
          "storage": "local", "auth_mode": "demo"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---------------- deployment profile ----------------
    profile: str = "C"
    llm_provider: str = ""   # ollama | azure_openai
    reader: str = ""         # docling | azure_di
    index: str = ""          # qdrant | azure_search
    storage: str = ""        # local | azure_blob
    auth_mode: str = ""      # demo | entra_easyauth | oidc | none

    # ---------------- profile C: local models (Ollama) ----------------
    ollama_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3-vl:8b"      # answers (bigger = better if the GPU allows)
    ollama_small_model: str = "qwen3-vl:8b"     # gates, summaries
    ollama_vision_model: str = "qwen3-vl:8b"    # figure descriptions
    ollama_embed_model: str = "bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # local cross-encoder (used with Qdrant)
    reranker_device: str = ""                   # "" = auto (GPU if available), or "cpu" / "cuda"
    ocr_mode: str = "auto"                      # auto | on | off (Docling)

    # ---------------- profiles A/B: Azure ----------------
    # Profile B uses the same settings; the endpoints simply resolve to private endpoints.
    aoai_endpoint: str = ""                     # https://<name>.openai.azure.com/
    aoai_api_version: str = "2024-10-21"
    aoai_chat_deployment: str = "gpt-5.1"       # confirm what your region offers
    aoai_small_deployment: str = "gpt-5-mini"
    aoai_vision_deployment: str = "gpt-5.1"
    aoai_embed_deployment: str = "text-embedding-3-large"
    docintel_endpoint: str = ""
    search_endpoint: str = ""
    blob_account_url: str = ""
    blob_container: str = "rag-data"

    # ---------------- index ----------------
    qdrant_url: str = ""                        # "" = embedded (data/qdrant); else http://qdrant:6333
    qdrant_api_key: str = ""
    index_name: str = "rag-chunks"
    catalog_name: str = "rag-catalog"
    embed_dimensions: int = 1024                # text-embedding-3-large can be shortened; bge-m3 = 1024

    # ---------------- identity ----------------
    demo_users_file: Path = Path("deploy/demo_users.json")
    single_tenant: str = "default"              # tenant used with entra_easyauth (one org)
    oidc_issuer: str = ""                       # e.g. https://keycloak.local/realms/company
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_tenant_claim: str = "tenant"
    oidc_groups_claim: str = "groups"
    everyone_group: str = "all"                 # ACL value meaning "everyone in the tenant"

    # ---------------- data ----------------
    data_dir: Path = Path("data")
    soffice_path: str = "soffice"
    max_upload_mb: int = 200
    describe_images: bool = True
    vision_max_px: int = 1024                   # images are shrunk before the vision model
    min_figure_px: int = 120                    # smaller images (icons, logos) are skipped
    vision_concurrency: int = 2                 # figure descriptions in parallel

    # ---------------- answer quality ----------------
    top_k: int = 8
    candidates: int = 30
    min_relevance: float = 0.30                 # 0-1 for every index (Azure's 0-4 is rescaled)
    chunk_target_tokens: int = 600
    chunk_max_tokens: int = 900
    chunk_overlap_tokens: int = 80

    # ---------------- logging ----------------
    log_level: str = "INFO"
    # Operators see process logs + file names, never document text, questions or answers.

    @model_validator(mode="after")
    def _apply_profile(self):
        self.profile = self.profile.upper()
        defaults = PROFILE_DEFAULTS.get(self.profile)
        if defaults is None:
            raise ValueError("PROFILE must be A, B or C")
        for key, value in defaults.items():
            if not getattr(self, key):
                setattr(self, key, value)
        return self

    # folders (local storage + caches)
    @property
    def files_dir(self) -> Path: return self.data_dir / "files"
    @property
    def figures_dir(self) -> Path: return self.data_dir / "figures"
    @property
    def cache_dir(self) -> Path: return self.data_dir / "cache"
    @property
    def jobs_dir(self) -> Path: return self.data_dir / "jobs"


settings = Settings()
for _d in (settings.files_dir, settings.figures_dir, settings.cache_dir, settings.jobs_dir):
    _d.mkdir(parents=True, exist_ok=True)
