"""
config/settings.py

System configuration loader. Loads environment variables from the .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditKG/1.0")

# LLM Providers
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Preferred provider fallback
LLM_PROVIDER = os.getenv("LLM_PROVIDER")

# Ingestion Filter Settings
RAG_KEYWORDS_FILTER = os.getenv("RAG_KEYWORDS_FILTER", "true").lower() == "true"
MIN_COMMENTS_FOR_RELEVANCE = int(os.getenv("MIN_COMMENTS_FOR_RELEVANCE", "2"))

# Databases
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7689")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://127.0.0.1:8080")


def validate_settings(use_real_reddit: bool = False, require_llm: bool = False):
    """Validate active credentials for the configuration profile."""
    has_anthropic = ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("your_")
    has_gemini = GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_")
    has_mistral = MISTRAL_API_KEY and not MISTRAL_API_KEY.startswith("your_")

    if require_llm and not (has_anthropic or has_gemini or has_mistral):
        raise EnvironmentError(
            "Missing active LLM provider credentials. Set one of these keys in your .env file:\n"
            "  1. ANTHROPIC_API_KEY\n"
            "  2. GEMINI_API_KEY\n"
            "  3. MISTRAL_API_KEY"
        )

    if use_real_reddit:
        reddit_missing = []
        if not REDDIT_CLIENT_ID or REDDIT_CLIENT_ID.startswith("your_"):
            reddit_missing.append("REDDIT_CLIENT_ID")
        if not REDDIT_CLIENT_SECRET or REDDIT_CLIENT_SECRET.startswith("your_"):
            reddit_missing.append("REDDIT_CLIENT_SECRET")
        
        if reddit_missing:
            raise EnvironmentError(
                f"Missing Reddit API credentials: {', '.join(reddit_missing)}\n"
                "Reddit credentials are required for live scraping (--real)."
            )

