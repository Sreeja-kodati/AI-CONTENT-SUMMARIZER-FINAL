"""Step-by-step Gemini API diagnostic."""

from utils.env_config import get_gemini_api_key, load_environment
from utils.gemini_client import GeminiAPIError, generate_content, get_active_model

load_environment()
api_key = get_gemini_api_key()

print("=" * 50)
print("STEP 1: Check .env")
print("=" * 50)
print(f"API key loaded : {bool(api_key)}")
print(f"Key prefix     : {api_key[:8]}..." if api_key else "MISSING")

if not api_key:
    print("\nFAIL: Add GEMINI_API_KEY to .env")
    raise SystemExit(1)

print("\n" + "=" * 50)
print("STEP 2: Test Gemini REST API (SSL + models)")
print("=" * 50)

try:
    reply = generate_content(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with exactly: API WORKING",
        temperature=0.1,
        max_output_tokens=20,
    )
    print("PASS: API WORKING")
    print(f"Model used: {get_active_model()}")
    print(f"Response  : {reply}")
except GeminiAPIError as e:
    print("FAIL:", e)
    raise SystemExit(1)

print("\n" + "=" * 50)
print("STEP 3: Test summarization")
print("=" * 50)

from utils.summarizer import summarize_text

sample = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning enables computers to learn from data. "
    "Natural language processing helps machines understand human language."
)
summary = summarize_text(sample, "Short Summary", "Simple")
print("PASS: Summary generated")
print(summary[:200])

print("\nAll tests passed. Run: python -m streamlit run app.py")
