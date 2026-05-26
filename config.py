"""Application configuration."""

# Try models in order until one works (free-tier friendly first)
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 90.0
