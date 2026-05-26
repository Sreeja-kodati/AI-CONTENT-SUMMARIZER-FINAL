"""Analytics utilities for content analysis."""

import re
from typing import Any

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
    "happy", "positive", "success", "successful", "best", "better", "beautiful",
    "awesome", "brilliant", "outstanding", "perfect", "pleased", "enjoy",
    "helpful", "improve", "improved", "win", "winning", "benefit", "beneficial",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "worse", "hate",
    "sad", "negative", "fail", "failed", "failure", "problem", "difficult",
    "wrong", "broken", "disappointing", "disappointed", "angry", "fear",
    "risk", "risky", "loss", "lose", "crisis", "damage", "damaged",
}


def count_words(text: str) -> int:
    """Count words in text."""
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text))


def count_characters(text: str) -> int:
    """Count characters in text."""
    return len(text) if text else 0


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """Estimate reading time in minutes."""
    words = count_words(text)
    if words == 0:
        return 0.0
    return round(words / words_per_minute, 2)


def get_sentiment(text: str) -> dict[str, Any]:
    """Analyze sentiment using lightweight lexical scoring (no NLTK required)."""
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "label": "Neutral",
        }

    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if not words:
        return {"polarity": 0.0, "subjectivity": 0.0, "label": "Neutral"}

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_sentiment_words = pos_count + neg_count

    if total_sentiment_words == 0:
        polarity = 0.0
    else:
        polarity = round((pos_count - neg_count) / total_sentiment_words, 3)

    subjectivity = round(min(total_sentiment_words / len(words) * 5, 1.0), 3)

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "label": label,
    }


def compression_percentage(original_words: int, summary_words: int) -> float:
    """Calculate compression percentage."""
    if original_words == 0:
        return 0.0
    if summary_words >= original_words:
        return 0.0
    return round((1 - summary_words / original_words) * 100, 2)


def build_analytics(original_text: str, summary_text: str) -> dict[str, Any]:
    """Build complete analytics dashboard data."""
    original_words = count_words(original_text)
    summary_words = count_words(summary_text)
    original_sentiment = get_sentiment(original_text)
    summary_sentiment = get_sentiment(summary_text)

    return {
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "original_char_count": count_characters(original_text),
        "summary_char_count": count_characters(summary_text),
        "compression_percentage": compression_percentage(original_words, summary_words),
        "original_reading_time": estimate_reading_time(original_text),
        "summary_reading_time": estimate_reading_time(summary_text),
        "original_sentiment": original_sentiment,
        "summary_sentiment": summary_sentiment,
    }
