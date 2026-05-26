"""Environment and SSL configuration for Windows compatibility."""

import os
from pathlib import Path

import certifi
import truststore
from dotenv import load_dotenv

# Fix SSL on Windows Store Python (must run before google/grpc imports)
truststore.inject_into_ssl()

_ssl_bundle = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _ssl_bundle)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ssl_bundle)
os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", _ssl_bundle)
# Force Gemini SDK to use REST (avoids gRPC SSL handshake failures on Windows)
os.environ.setdefault("GOOGLE_API_USE_REST", "true")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _clean_key(value: str | None) -> str:
    """Strip quotes and whitespace from env values."""
    if not value:
        return ""
    api_key = value.strip()
    if (api_key.startswith('"') and api_key.endswith('"')) or (
        api_key.startswith("'") and api_key.endswith("'")
    ):
        api_key = api_key[1:-1].strip()
    return api_key


def load_environment() -> None:
    """Load .env from project root regardless of working directory."""
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_gemini_api_key() -> str:
    """Return stripped Google Gemini API key from environment."""
    load_environment()
    return _clean_key(os.getenv("GEMINI_API_KEY"))


def is_api_key_configured() -> bool:
    """Check if Gemini API key is present."""
    return bool(get_gemini_api_key())
