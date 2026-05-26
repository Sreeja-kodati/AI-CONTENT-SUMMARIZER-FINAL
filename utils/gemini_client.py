"""Gemini REST API client (no gRPC — Windows SSL safe)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from config import GEMINI_API_BASE, GEMINI_MODELS, REQUEST_TIMEOUT
from utils.env_config import get_gemini_api_key, load_environment

load_environment()

_active_model: str | None = None


class GeminiAPIError(Exception):
    """Raised when all Gemini models fail."""

    def __init__(self, message: str, last_status: int | None = None):
        super().__init__(message)
        self.last_status = last_status


def get_active_model() -> str:
    """Return the model that last succeeded, or the first configured model."""
    return _active_model or GEMINI_MODELS[0]


def _parse_response(data: dict[str, Any]) -> str:
    """Extract text from Gemini generateContent response."""
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = (
            data.get("promptFeedback", {}).get("blockReason")
            or "No candidates returned"
        )
        raise GeminiAPIError(f"Content blocked or empty: {block_reason}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if p.get("text")]
    text = "".join(text_parts).strip()
    if not text:
        raise GeminiAPIError("Empty text in Gemini response.")
    return text


def _format_http_error(status: int, body: str) -> str:
    """Build a short user-facing error from HTTP response."""
    try:
        payload = json.loads(body)
        message = payload.get("error", {}).get("message", body)
    except json.JSONDecodeError:
        message = body[:300]
    msg_lower = message.lower()

    if status == 400 and "api key" in msg_lower:
        return (
            "Invalid GEMINI_API_KEY. Create a key at "
            "https://aistudio.google.com/apikey"
        )
    if status == 429 or "quota" in msg_lower:
        return (
            "Gemini free quota exceeded for this model. "
            "Wait 1 minute and retry, or enable billing in Google AI Studio."
        )
    if status in (401, 403):
        return "Gemini API access denied. Check your API key permissions."
    return f"Gemini API error ({status}): {message}"


def _should_try_next_model(status: int, body: str) -> bool:
    """Whether to attempt the next model in the fallback list."""
    if status == 404:
        return True
    if status == 429:
        return True
    body_lower = body.lower()
    if "not found" in body_lower and "model" in body_lower:
        return True
    return False


def generate_content(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
    max_output_tokens: int = 2000,
) -> str:
    """
    Call Gemini generateContent via REST with automatic model fallback.
    """
    global _active_model

    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiAPIError(
            "GEMINI_API_KEY not found. Add it to .env:\n"
            "GEMINI_API_KEY=your_key_here\n"
            "Get key: https://aistudio.google.com/apikey"
        )

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    headers = {"Content-Type": "application/json"}
    errors: list[str] = []
    last_status: int | None = None

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for model in GEMINI_MODELS:
            url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
            try:
                response = client.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    headers=headers,
                )
            except httpx.ConnectError as exc:
                raise GeminiAPIError(
                    f"Network/SSL error connecting to Gemini: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise GeminiAPIError("Gemini API request timed out.") from exc

            last_status = response.status_code

            if response.status_code == 200:
                _active_model = model
                return _parse_response(response.json())

            body = response.text
            errors.append(f"{model}: {_format_http_error(response.status_code, body)}")

            if _should_try_next_model(response.status_code, body):
                continue
            break

    raise GeminiAPIError(
        "All Gemini models failed:\n- " + "\n- ".join(errors),
        last_status=last_status,
    )
