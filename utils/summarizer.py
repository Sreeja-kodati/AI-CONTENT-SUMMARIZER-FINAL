"""Gemini-powered summarization and AI feature utilities."""

import re
from typing import Any

from config import GEMINI_MODELS
from utils.env_config import load_environment
from utils.gemini_client import GeminiAPIError, generate_content, get_active_model

load_environment()

MODEL = GEMINI_MODELS[0]  # default; runtime may switch via fallback
MAX_CHUNK_CHARS = 12000
OVERLAP_CHARS = 500

SUMMARY_STYLES = {
    "Short Summary": "Provide a concise summary in 3-5 sentences.",
    "Detailed Summary": "Provide a comprehensive detailed summary covering all major points.",
    "Bullet Points": "Provide the summary as clear bullet points.",
    "Key Highlights": "Extract and list the key highlights only.",
    "Executive Summary": "Provide a professional executive summary for decision makers.",
}

TONES = {
    "Professional": "Use a professional business tone.",
    "Academic": "Use a formal academic tone with precise language.",
    "Simple": "Use simple, easy-to-understand language.",
    "Technical": "Use a technical tone with domain-appropriate terminology.",
    "Casual": "Use a friendly, casual conversational tone.",
}


def _generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
    max_output_tokens: int = 2000,
) -> str:
    """Wrapper that converts GeminiAPIError to ValueError for the UI."""
    try:
        return generate_content(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except GeminiAPIError as exc:
        raise ValueError(str(exc)) from exc


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks for large documents."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]

        if end < len(text):
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind(" "))
            if last_break > max_chars // 2:
                chunk = chunk[:last_break]
                end = start + last_break

        chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else end

    return [c for c in chunks if c]


def summarize_text(
    text: str,
    summary_style: str = "Short Summary",
    tone: str = "Professional",
) -> str:
    """Summarize text using Gemini with chunking for large documents."""
    if not text or not text.strip():
        raise ValueError("No text provided for summarization.")

    style_instruction = SUMMARY_STYLES.get(summary_style, SUMMARY_STYLES["Short Summary"])
    tone_instruction = TONES.get(tone, TONES["Professional"])

    chunks = split_into_chunks(text)
    chunk_summaries = []

    system_prompt = (
        "You are an expert content summarizer. "
        f"{style_instruction} {tone_instruction} "
        "Preserve factual accuracy. Do not invent information."
    )

    for i, chunk in enumerate(chunks):
        user_prompt = (
            f"Summarize the following content (part {i + 1} of {len(chunks)}):\n\n{chunk}"
        )
        summary = _generate(system_prompt, user_prompt, temperature=0.4)
        chunk_summaries.append(summary)

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n".join(chunk_summaries)
    merge_prompt = (
        f"The following are partial summaries of a large document. "
        f"Merge them into one cohesive {summary_style.lower()} "
        f"using a {tone.lower()} tone:\n\n{combined}"
    )
    return _generate(system_prompt, merge_prompt, temperature=0.3, max_output_tokens=2500)


def extract_keywords(text: str, count: int = 10) -> list[str]:
    """Extract keywords from text using Gemini."""
    truncated = text[:8000]
    system_prompt = "You are an SEO and content analysis expert."
    user_prompt = (
        f"Extract exactly {count} important keywords from the text below. "
        "Return only a comma-separated list with no numbering or extra text.\n\n"
        f"{truncated}"
    )
    result = _generate(system_prompt, user_prompt, temperature=0.3, max_output_tokens=300)
    keywords = [k.strip() for k in result.split(",") if k.strip()]
    return keywords[:count]


def extract_topics(text: str, count: int = 5) -> list[str]:
    """Extract important topics from text."""
    truncated = text[:8000]
    system_prompt = "You are a content analyst."
    user_prompt = (
        f"Identify the top {count} important topics from the text below. "
        "Return only a bullet list with one topic per line.\n\n"
        f"{truncated}"
    )
    result = _generate(system_prompt, user_prompt, temperature=0.3, max_output_tokens=400)
    topics = []
    for line in result.split("\n"):
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if line:
            topics.append(line)
    return topics[:count]


def generate_title(text: str) -> str:
    """Generate a title for the content."""
    truncated = text[:4000]
    system_prompt = "You are a headline writer."
    user_prompt = (
        "Generate a single compelling title for the following content. "
        "Return only the title with no quotes or extra text.\n\n"
        f"{truncated}"
    )
    return _generate(system_prompt, user_prompt, temperature=0.6, max_output_tokens=60)


def generate_hashtags(text: str, count: int = 8) -> list[str]:
    """Generate social media hashtags."""
    truncated = text[:4000]
    system_prompt = "You are a social media strategist."
    user_prompt = (
        f"Generate {count} relevant hashtags for the content below. "
        "Return only hashtags separated by spaces. Include the # symbol.\n\n"
        f"{truncated}"
    )
    result = _generate(system_prompt, user_prompt, temperature=0.6, max_output_tokens=200)
    hashtags = re.findall(r"#\w+", result)
    if not hashtags:
        hashtags = [f"#{tag.strip().replace(' ', '')}" for tag in result.split() if tag.strip()]
    return hashtags[:count]


def run_full_ai_pipeline(
    text: str,
    summary_style: str,
    tone: str,
    include_keywords: bool = True,
    include_topics: bool = True,
    include_title: bool = True,
    include_hashtags: bool = True,
) -> dict[str, Any]:
    """Run complete AI pipeline: summarize + optional extras."""
    summary = summarize_text(text, summary_style, tone)
    result: dict[str, Any] = {
        "summary": summary,
        "model_used": get_active_model(),
    }

    if include_keywords:
        result["keywords"] = extract_keywords(text)
    if include_topics:
        result["topics"] = extract_topics(text)
    if include_title:
        result["title"] = generate_title(text)
    if include_hashtags:
        result["hashtags"] = generate_hashtags(text)

    return result
