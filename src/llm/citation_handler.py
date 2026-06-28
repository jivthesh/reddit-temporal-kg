"""
src/llm/citation_handler.py

Handles formatting of source citation indices [1], [2], and generating the bibliography block.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger("reddit-kg")


def format_sources_section(results: List[Dict[str, Any]]) -> str:
    """Builds a formatted Sources section listing matching document references."""
    if not results:
        return "Sources: None available"

    lines = ["Sources:"]

    for i, result in enumerate(results, 1):
        author = result.get("author", "unknown")
        subreddit = result.get("subreddit", "unknown")
        time_period = result.get("time_period", "")
        rrf_score = result.get("rrf_score", 0.0)
        source_type = result.get("source", "unknown")
        text = result.get("text", "")

        author_label = f"u/{author}" if author != "[deleted]" else "[deleted]"
        subreddit_label = f"r/{subreddit}" if subreddit else ""

        text_preview = text[:120].strip()
        if len(text) > 120:
            text_preview += "..."

        meta_parts = []
        if time_period:
            meta_parts.append(time_period)
        if source_type:
            meta_parts.append(f"from {source_type}")
        meta_parts.append(f"RRF: {rrf_score:.4f}")

        header = f"[{i}] {author_label}"
        if subreddit_label:
            header += f" in {subreddit_label}"
        if meta_parts:
            header += f" ({', '.join(meta_parts)})"

        lines.append(header)
        lines.append(f'    "{text_preview}"')
        lines.append("")

    return "\n".join(lines)


def inject_citation_numbers(answer_text: str, results: List[Dict[str, Any]]) -> str:
    """Appends formatted bibliography list to the generated response."""
    has_citations = bool(re.search(r"\[\d+\]", answer_text))

    if has_citations:
        logger.debug("Answer already contains citation markers")
    else:
        logger.debug("No citation markers found in answer — sources will be appended")

    sources = format_sources_section(results)
    return f"{answer_text.strip()}\n\n{sources}"


def build_context_for_llm(results: List[Dict[str, Any]]) -> str:
    """Formats list of retrieved documents into a clean context block."""
    context_parts = []

    for i, result in enumerate(results, 1):
        author = result.get("author", "unknown")
        subreddit = result.get("subreddit", "unknown")
        time_period = result.get("time_period", "")
        sentiment = result.get("sentiment", 0.5)
        rrf_score = result.get("rrf_score", 0.0)
        text = result.get("text", "")

        text_excerpt = text[:300].strip()
        if len(text) > 300:
            text_excerpt += "..."

        context_parts.append(
            f"[{i}] u/{author} in r/{subreddit} ({time_period}):\n"
            f"    \"{text_excerpt}\"\n"
            f"    Sentiment: {sentiment:.2f} | RRF Score: {rrf_score:.4f}"
        )

    return "\n\n".join(context_parts)
