"""
AI Content Summarizer - Streamlit Application
A production-ready AI-powered content summarization tool.
"""

# Initialize SSL certificates before any Google/network imports
import utils.env_config  # noqa: F401

from datetime import datetime
from pathlib import Path

import streamlit as st

from utils.env_config import is_api_key_configured, load_environment
from utils.analytics import build_analytics, count_characters, count_words, estimate_reading_time
from utils.export_utils import build_export_filename, export_to_docx, export_to_pdf, export_to_txt
from utils.file_handler import extract_text_from_upload
from utils.gemini_client import get_active_model
from utils.summarizer import SUMMARY_STYLES, TONES, run_full_ai_pipeline
from utils.web_scraper import fetch_url_content, is_valid_url

load_environment()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_CHAR_LIMIT = 100000


def load_css() -> None:
    """Load custom CSS stylesheet."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "source_text": "",
        "summary_text": "",
        "keywords": [],
        "topics": [],
        "hashtags": [],
        "generated_title": "",
        "analytics": {},
        "history": [],
        "last_input_method": "text",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_history(summary: str, style: str, tone: str, source_preview: str) -> None:
    """Add a summary entry to session history."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary[:500],
        "style": style,
        "tone": tone,
        "preview": source_preview[:120] + ("..." if len(source_preview) > 120 else ""),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


def render_header() -> None:
    """Render application header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>📝 AI Content Summarizer</h1>
            <p>Transform articles, documents, URLs, and text into intelligent summaries powered by AI</p>
            <span class="ai-badge">Powered by Google Gemini</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render sidebar and return configuration."""
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">✨ AI Summarizer</p>', unsafe_allow_html=True)
        st.markdown("---")

        theme = st.radio("Theme", ["Dark", "Light"], index=0, key="theme_select")

        st.markdown("---")
        st.subheader("⚙️ Summary Options")

        summary_style = st.selectbox(
            "Summary Style",
            list(SUMMARY_STYLES.keys()),
            index=0,
        )
        tone = st.selectbox("Tone", list(TONES.keys()), index=0)

        st.markdown("---")
        st.subheader("🧠 AI Features")
        include_keywords = st.checkbox("Keywords Extraction", value=False)
        include_topics = st.checkbox("Important Topics", value=False)
        include_title = st.checkbox("Title Generation", value=True)
        include_hashtags = st.checkbox("Hashtag Generation", value=False)
        st.caption("Extra AI features use more API quota.")

        st.markdown("---")
        st.subheader("📜 Recent Summaries")
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[:5]):
                with st.expander(f"#{i + 1} — {item['timestamp']}", expanded=False):
                    st.caption(f"Style: {item['style']} | Tone: {item['tone']}")
                    st.write(item["preview"])
                    if st.button("Load", key=f"load_hist_{i}"):
                        st.session_state.summary_text = item["summary"]
                        st.rerun()
        else:
            st.caption("No summaries yet.")

        if st.session_state.history and st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
        if is_api_key_configured():
            st.success("✅ Gemini API Key OK")
            try:
                st.caption(f"Model: {get_active_model()}")
            except Exception:
                st.caption("Model: auto-select on first run")
        else:
            st.error("❌ API Key Missing")
            st.caption("Add GEMINI_API_KEY to .env")

    return {
        "theme": theme,
        "summary_style": summary_style,
        "tone": tone,
        "include_keywords": include_keywords,
        "include_topics": include_topics,
        "include_title": include_title,
        "include_hashtags": include_hashtags,
    }


def apply_theme(theme: str) -> None:
    """Apply dark or light theme overrides."""
    if theme == "Light":
        st.markdown(
            """
            <style>
            .stApp { background-color: #f5f7ff; }
            .main .block-container { color: #1a1a2e; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0f0f1a; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def process_summarization(source_text: str, config: dict) -> None:
    """Run summarization pipeline with progress indicators."""
    if not source_text.strip():
        st.warning("Please provide content to summarize.")
        return

    if not is_api_key_configured():
        st.error("Gemini API key not configured. Please set GEMINI_API_KEY in your .env file.")
        return

    progress = st.progress(0, text="Initializing AI pipeline...")
    try:
        progress.progress(20, text="Analyzing content...")
        result = run_full_ai_pipeline(
            text=source_text,
            summary_style=config["summary_style"],
            tone=config["tone"],
            include_keywords=config["include_keywords"],
            include_topics=config["include_topics"],
            include_title=config["include_title"],
            include_hashtags=config["include_hashtags"],
        )

        progress.progress(80, text="Building analytics...")
        st.session_state.source_text = source_text
        st.session_state.summary_text = result.get("summary", "")
        st.session_state.keywords = result.get("keywords", [])
        st.session_state.topics = result.get("topics", [])
        st.session_state.hashtags = result.get("hashtags", [])
        st.session_state.generated_title = result.get("title", "")
        st.session_state.analytics = build_analytics(source_text, st.session_state.summary_text)

        add_to_history(
            st.session_state.summary_text,
            config["summary_style"],
            config["tone"],
            source_text,
        )

        progress.progress(100, text="Complete!")
        model = result.get("model_used", get_active_model())
        st.success(f"Summary generated successfully! (Model: {model})")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
    finally:
        progress.empty()


def tab_text_input(config: dict) -> str:
    """Text input tab."""
    st.subheader("📄 Text Input")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.source_text = ""
            st.rerun()

    text = st.text_area(
        "Paste your content here",
        value=st.session_state.get("source_text", ""),
        height=300,
        placeholder="Paste articles, paragraphs, YouTube transcripts, or any text here...",
        key="text_input_area",
    )

    char_count = count_characters(text)
    word_count = count_words(text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", f"{char_count:,}")
    c2.metric("Words", f"{word_count:,}")
    c3.metric("Est. Reading Time", f"{estimate_reading_time(text)} min")

    if char_count > MAX_CHAR_LIMIT:
        st.warning(f"Text exceeds {MAX_CHAR_LIMIT:,} character limit. It will be chunked automatically.")

    if st.button("🚀 Summarize Text", type="primary", use_container_width=True):
        process_summarization(text, config)

    return text


def tab_file_upload(config: dict) -> str:
    """File upload tab."""
    st.subheader("📁 File Upload")
    st.caption("Supported formats: TXT, PDF, DOCX")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx"],
        help="Upload a text document to extract and summarize",
    )

    extracted = ""
    if uploaded:
        with st.spinner("Extracting text from file..."):
            try:
                extracted = extract_text_from_upload(uploaded)
                st.success(f"Extracted {count_words(extracted):,} words from **{uploaded.name}**")
                st.text_area("Extracted Content Preview", extracted[:3000], height=200, disabled=True)
                if len(extracted) > 3000:
                    st.caption(f"... showing first 3,000 characters of {len(extracted):,} total")
            except Exception as exc:
                st.error(f"File extraction failed: {exc}")

    if extracted and st.button("🚀 Summarize File", type="primary", use_container_width=True):
        process_summarization(extracted, config)

    return extracted


def tab_url_input(config: dict) -> str:
    """URL summarization tab."""
    st.subheader("🌐 URL Summarization")
    url = st.text_input("Enter article URL", placeholder="https://example.com/article")

    extracted = ""
    if url:
        if not is_valid_url(url):
            st.warning("Please enter a valid URL starting with http:// or https://")
        elif st.button("🔍 Fetch & Summarize URL", type="primary", use_container_width=True):
            with st.spinner("Fetching content from URL..."):
                try:
                    result = fetch_url_content(url)
                    extracted = result["text"]
                    st.session_state.source_text = extracted
                    st.success(f"Fetched: **{result['title']}** ({count_words(extracted):,} words)")
                    st.text_area("Extracted Content Preview", extracted[:3000], height=200, disabled=True)
                    process_summarization(extracted, config)
                except Exception as exc:
                    st.error(f"URL fetch failed: {exc}")

    return extracted


def render_results(config: dict) -> None:
    """Render summary results, analytics, and export."""
    if not st.session_state.summary_text:
        st.info("👆 Provide content and click Summarize to see results here.")
        return

    st.markdown("---")
    st.subheader("✨ Generated Summary")

    if st.session_state.generated_title:
        st.markdown(f"### {st.session_state.generated_title}")

    col_copy, col_space = st.columns([1, 5])
    with col_copy:
        st.code(st.session_state.summary_text, language=None)

    st.markdown(
        f'<div class="summary-box">{st.session_state.summary_text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # Keywords, topics, hashtags
    if st.session_state.keywords or st.session_state.topics or st.session_state.hashtags:
        st.subheader("🏷️ AI Insights")
        i1, i2, i3 = st.columns(3)

        with i1:
            if st.session_state.keywords:
                st.markdown("**Keywords**")
                tags = " ".join(
                    f'<span class="tag-pill">{kw}</span>' for kw in st.session_state.keywords
                )
                st.markdown(tags, unsafe_allow_html=True)

        with i2:
            if st.session_state.topics:
                st.markdown("**Important Topics**")
                for topic in st.session_state.topics:
                    st.markdown(f"- {topic}")

        with i3:
            if st.session_state.hashtags:
                st.markdown("**Hashtags**")
                st.write(" ".join(st.session_state.hashtags))

    # Analytics dashboard
    if st.session_state.analytics:
        st.subheader("📊 Analytics Dashboard")
        a = st.session_state.analytics

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Original Words", f"{a['original_word_count']:,}")
        m2.metric("Summary Words", f"{a['summary_word_count']:,}")
        m3.metric("Compression", f"{a['compression_percentage']}%")
        m4.metric("Original Read Time", f"{a['original_reading_time']} min")
        m5.metric("Summary Read Time", f"{a['summary_reading_time']} min")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric(
                "Original Sentiment",
                a["original_sentiment"]["label"],
                f"Polarity: {a['original_sentiment']['polarity']}",
            )
        with s2:
            st.metric(
                "Summary Sentiment",
                a["summary_sentiment"]["label"],
                f"Polarity: {a['summary_sentiment']['polarity']}",
            )
        with s3:
            st.metric("Character Count", f"{a['original_char_count']:,} → {a['summary_char_count']:,}")

    # Export section
    st.subheader("📥 Export Options")
    export_title = st.session_state.generated_title or "AI_Summary"
    summary = st.session_state.summary_text

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            label="⬇️ Download TXT",
            data=export_to_txt(summary, export_title),
            file_name=build_export_filename(export_title, "txt"),
            mime="text/plain",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            label="⬇️ Download PDF",
            data=export_to_pdf(summary, export_title),
            file_name=build_export_filename(export_title, "pdf"),
            mime="application/pdf",
            use_container_width=True,
        )
    with e3:
        st.download_button(
            label="⬇️ Download DOCX",
            data=export_to_docx(summary, export_title),
            file_name=build_export_filename(export_title, "docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


def render_footer() -> None:
    """Render application footer."""
    st.markdown(
        """
        <div class="footer-section">
            <p>AI Content Summarizer &copy; 2026 | Built with Streamlit & Google Gemini</p>
            <p>Made with ❤️ for intelligent content processing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main application entry point."""
    load_environment()
    load_css()
    init_session_state()
    render_header()

    config = render_sidebar()
    apply_theme(config["theme"])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 Text Input", "📁 File Upload", "🌐 URL", "📋 Results & Export"]
    )

    with tab1:
        tab_text_input(config)

    with tab2:
        tab_file_upload(config)

    with tab3:
        tab_url_input(config)

    with tab4:
        render_results(config)

    render_footer()


if __name__ == "__main__":
    main()
