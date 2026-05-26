"""Web scraping utilities for URL content extraction."""

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url.strip())
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_article_text(html: str) -> str:
    """Extract main article text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    candidates = []

    article = soup.find("article")
    if article:
        candidates.append(article.get_text(separator=" ", strip=True))

    main = soup.find("main")
    if main:
        candidates.append(main.get_text(separator=" ", strip=True))

    for selector in [".article-body", ".post-content", ".entry-content", "#content", ".content"]:
        element = soup.select_one(selector)
        if element:
            candidates.append(element.get_text(separator=" ", strip=True))

    paragraphs = soup.find_all("p")
    if paragraphs:
        para_text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
        candidates.append(para_text)

    body = soup.find("body")
    if body:
        candidates.append(body.get_text(separator=" ", strip=True))

    candidates = [c for c in candidates if c and len(c) > 100]
    if not candidates:
        raise ValueError("Could not extract meaningful content from the URL.")

    best = max(candidates, key=len)
    return clean_extracted_text(best)


def fetch_url_content(url: str, timeout: int = 15) -> dict[str, str]:
    """Fetch and extract content from a URL."""
    if not is_valid_url(url):
        raise ValueError("Invalid URL. Please enter a valid http or https URL.")

    response = requests.get(url.strip(), headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise ValueError("URL does not point to an HTML page.")

    html = response.text
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    text = extract_article_text(html)
    if len(text) < 50:
        raise ValueError("Extracted content is too short. The page may be blocked or empty.")

    return {"title": title, "text": text, "url": url.strip()}
