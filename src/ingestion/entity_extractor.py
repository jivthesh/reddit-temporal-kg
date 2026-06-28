"""
src/ingestion/entity_extractor.py

Uses LLM client to parse key topics, entities, and sentiment from raw Reddit submissions.
"""

import json
import logging
import time
from typing import Dict, Any, List

from config.constants import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
from utils.helpers import truncate
from src.llm.llm_client import LLMClient

logger = logging.getLogger("reddit-kg")

llm_client = LLMClient()


def _extract_json_block(raw_text: str) -> str:
    raw = raw_text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return raw


def extract_entities_from_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """Runs extraction on a single post, updating topics, entities, and sentiment."""
    combined_text = _build_text_for_analysis(post)

    try:
        result = _call_llm_for_extraction(combined_text)
    except Exception as e:
        logger.warning(f"Entity extraction failed for post {post['id']}: {e}")
        result = _default_extraction()

    post["topics"] = result.get("topics", [])
    post["entities"] = result.get("entities", [])
    post["sentiment"] = result.get("sentiment", 0.5)

    return post


def _build_text_for_analysis(post: Dict[str, Any]) -> str:
    parts = [f"TITLE: {post['title']}"]

    if post.get("text"):
        parts.append(f"POST BODY: {truncate(post['text'], 500)}")

    top_comments = post.get("comments", [])[:3]
    if top_comments:
        comment_texts = [truncate(c["text"], 200) for c in top_comments if c.get("text")]
        parts.append("TOP COMMENTS:\n" + "\n---\n".join(comment_texts))

    return "\n\n".join(parts)


def _call_llm_for_extraction(text: str) -> Dict[str, Any]:
    prompt = f"""Analyze the following Reddit content and extract structured information.

Respond ONLY with a valid JSON object, no explanation text before or after.

Reddit content:
{text}

Return this exact JSON structure:
{{
  "topics": ["topic1", "topic2"],
  "entities": [
    {{"text": "entity name", "type": "TOOL|PERSON|CONCEPT|COMPANY|PAPER"}}
  ],
  "sentiment": 0.7
}}

Rules:
- topics: 1-5 broad subject areas (e.g. "RAG", "AI Safety", "LLMs", "Open Source")
- entities: specific named things mentioned (tools like "LangChain", people, papers, companies)
- sentiment: float from 0.0 (very negative/critical) to 1.0 (very positive/enthusiastic)
- Return ONLY the JSON object, nothing else"""

    response = llm_client.generate_completion(prompt, max_tokens=500)
    return json.loads(_extract_json_block(response))


def _default_extraction() -> Dict[str, Any]:
    return {"topics": [], "entities": [], "sentiment": 0.5}


def batch_extract_entities(
    posts: List[Dict[str, Any]], delay_between_calls: float = 0.5
) -> List[Dict[str, Any]]:
    """Runs entity extraction sequentially over a collection of posts."""
    enriched_posts = []
    total = len(posts)
    skipped_count = 0

    for i, post in enumerate(posts):
        has_metadata = (
            post.get("topics") and
            post.get("entities") and
            "sentiment" in post
        )
        if has_metadata:
            enriched_posts.append(post)
            skipped_count += 1
            continue

        logger.info(f"Extracting entities: post {i+1}/{total} (id={post['id']})")
        enriched_posts.append(extract_entities_from_post(post))

        if i < total - 1:
            time.sleep(delay_between_calls)

    if skipped_count > 0:
        logger.info(f"Skipped extraction for {skipped_count}/{total} posts (already had metadata)")
    logger.info(f"Entity extraction complete: {len(enriched_posts)} posts processed")
    return enriched_posts


def extract_sentiment(text: str) -> float:
    """Computes a general sentiment float rating between 0.0 and 1.0."""
    prompt = f"""Analyze the sentiment of this technical Reddit post.
Consider context: negative /= problem-focused, positive /= no criticism.

Text: "{text[:500]}"

Respond with ONLY a number between 0.0 and 1.0:
- 0.0-0.3: Highly critical/negative
- 0.3-0.5: Skeptical/cautious
- 0.5-0.7: Balanced/mixed
- 0.7-1.0: Positive/optimistic

Number:"""
    try:
        response = llm_client.generate_completion(prompt, max_tokens=10)
        return max(0.0, min(1.0, float(response.strip())))
    except Exception:
        return 0.5


class EntityExtractor:
    """Helper wrapper class for extraction routines."""

    def __init__(self):
        self.llm_client = llm_client

    def extract_sentiment(self, text: str) -> float:
        return extract_sentiment(text)

    async def extract_sentiment_async(self, text: str) -> float:
        return extract_sentiment(text)
