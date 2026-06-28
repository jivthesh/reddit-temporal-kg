"""
src/storage/chunking.py

Splits post bodies and nested comment trees into overlapping text chunks.
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.constants import CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import clean_text

logger = logging.getLogger("reddit-kg")


def chunk_post(post: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Splits a single post and its comment objects into document chunk dicts."""
    chunks = []
    splitter = _get_splitter()

    post_text = f"{post['title']}\n\n{post.get('text', '')}".strip()

    if post_text:
        post_text = clean_text(post_text)
        text_chunks = splitter.split_text(post_text)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"post_{post['id']}_chunk_{i}",
                "text": chunk_text,
                "source_post_id": post["id"],
                "source_comment_id": None,
                "source_type": "post",
                "author": post.get("author", "[deleted]"),
                "subreddit": post.get("subreddit", ""),
                "sentiment": post.get("sentiment", 0.5),
                "created_at": post.get("created_at", 0),
                "created_iso": post.get("created_iso", ""),
                "time_period": post.get("time_period", ""),
                "recency_label": post.get("recency_label", ""),
                "topics": post.get("topics", []),
                "score": post.get("score", 0),
            })

    for comment in post.get("comments", []):
        comment_text = clean_text(comment.get("text", ""))

        if len(comment_text) < 30:
            continue

        comment_chunks = splitter.split_text(comment_text)

        for i, chunk_text in enumerate(comment_chunks):
            chunks.append({
                "chunk_id": f"comment_{comment['id']}_chunk_{i}",
                "text": chunk_text,
                "source_post_id": post["id"],
                "source_comment_id": comment["id"],
                "source_type": "comment",
                "author": comment.get("author", "[deleted]"),
                "subreddit": post.get("subreddit", ""),
                "sentiment": post.get("sentiment", 0.5),
                "created_at": comment.get("created_at", 0),
                "created_iso": comment.get("created_iso", ""),
                "time_period": comment.get("time_period", ""),
                "recency_label": post.get("recency_label", "old"),
                "topics": post.get("topics", []),
                "score": comment.get("score", 0),
            })

    return chunks


def chunk_all_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits a list of posts and returns all chunks as a flat list."""
    all_chunks = []

    for i, post in enumerate(posts):
        post_chunks = chunk_post(post)
        all_chunks.extend(post_chunks)

        if (i + 1) % 50 == 0:
            logger.info(f"Chunking: {i+1}/{len(posts)} posts, {len(all_chunks)} chunks so far")

    logger.info(f"Chunking complete: {len(posts)} posts → {len(all_chunks)} chunks")
    return all_chunks


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Builds the text splitter using config parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
