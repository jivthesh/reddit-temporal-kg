"""
config/constants.py

Hard-coded system configuration parameters.
"""

# Subreddits configuration
TARGET_SUBREDDITS = [
    "MachineLearning",
    "LanguageModels",
    "artificial",
]

POSTS_PER_SUBREDDIT = 200
MAX_COMMENTS_PER_POST = 50

# Text chunking settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Embedding model settings (Runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# LLM model configuration
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_MAX_TOKENS = 1500

# Retrieval limits and fusion parameters
TOP_K_GRAPH = 30
TOP_K_VECTOR = 50
TOP_K_FINAL = 10
RRF_K = 60

# Weaviate storage schema name
WEAVIATE_CLASS = "RedditChunk"
