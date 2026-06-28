"""
src/retrieval/vector_retriever.py

Executes semantic searches on Weaviate based on query metadata.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from src.storage.vector_client import VectorClient
from src.ingestion.temporal_parser import parse_time_range_from_query
from config.constants import TOP_K_VECTOR

logger = logging.getLogger("reddit-kg")


class VectorRetriever:
    """
    Translates query understanding metadata into vector search parameters for Weaviate.
    """

    def __init__(self, vector_client: VectorClient):
        self.vector = vector_client

    def retrieve(
        self, query_info: Dict[str, Any], anchor_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Constructs and executes a semantic search query based on parsed intent."""
        search_query = self._build_search_query(query_info)

        time_range_label = query_info.get("time_range")
        time_range = parse_time_range_from_query(time_range_label, anchor_ts=anchor_ts)

        logger.info(f"Vector retrieval: query='{search_query}', time_range={time_range}")

        raw_results = self.vector.semantic_search(
            query=search_query,
            time_range=time_range,
            limit=TOP_K_VECTOR,
        )

        for result in raw_results:
            result.setdefault("source", "vector")

        logger.info(f"Vector retrieval found {len(raw_results)} chunks")
        return raw_results

    def _build_search_query(self, query_info: Dict[str, Any]) -> str:
        """Assembles a search string from parsed topic and keyword tokens."""
        parts = []

        topic = query_info.get("main_topic", "")
        if topic:
            parts.append(topic)

        intent = query_info.get("intent", "")
        if intent and intent != "general":
            parts.append(intent)

        keywords = query_info.get("keywords", [])
        for kw in keywords[:3]:
            if kw.lower() not in topic.lower():
                parts.append(kw)

        if not parts:
            return query_info.get("original_question", "")

        return " ".join(parts)
