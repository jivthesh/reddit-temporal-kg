"""
src/storage/vector_client.py

Handles embedding generation and search operations in the Weaviate vector database.
"""

import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import weaviate
from sentence_transformers import SentenceTransformer

from config.settings import WEAVIATE_URL
from config.constants import (
    WEAVIATE_CLASS,
    EMBEDDING_MODEL,
    TOP_K_VECTOR,
)

logger = logging.getLogger("reddit-kg")


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    logger.info(f"Loading local embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def _build_time_filter(start_ts: int, end_ts: int) -> Dict[str, Any]:
    return {
        "operator": "And",
        "operands": [
            {
                "path": ["created_at"],
                "operator": "GreaterThanEqual",
                "valueInt": start_ts,
            },
            {
                "path": ["created_at"],
                "operator": "LessThanEqual",
                "valueInt": end_ts,
            },
        ],
    }


class VectorClient:
    """
    Client wrapper for Weaviate vector database operations.
    """

    def __init__(self):
        self.client = weaviate.Client(WEAVIATE_URL)

        if not self.client.is_ready():
            raise ConnectionError(f"Weaviate is not ready at {WEAVIATE_URL}.")

        logger.info(f"VectorClient connected to Weaviate at {WEAVIATE_URL}")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Creates Weaviate schema class if not already existing."""
        existing = self.client.schema.get()
        existing_classes = [c["class"] for c in existing.get("classes", [])]

        if WEAVIATE_CLASS in existing_classes:
            return

        schema = {
            "class": WEAVIATE_CLASS,
            "description": "Chunks of Reddit posts and comments with embeddings",
            "vectorizer": "none",
            "properties": [
                {"name": "chunk_id",          "dataType": ["text"]},
                {"name": "text",              "dataType": ["text"]},
                {"name": "source_post_id",    "dataType": ["text"]},
                {"name": "source_comment_id", "dataType": ["text"]},
                {"name": "source_type",       "dataType": ["text"]},
                {"name": "author",            "dataType": ["text"]},
                {"name": "subreddit",         "dataType": ["text"]},
                {"name": "sentiment",         "dataType": ["number"]},
                {"name": "created_at",        "dataType": ["int"]},
                {"name": "created_iso",       "dataType": ["text"]},
                {"name": "time_period",       "dataType": ["text"]},
                {"name": "recency_label",     "dataType": ["text"]},
                {"name": "score",             "dataType": ["int"]},
                {"name": "topics",            "dataType": ["text"]},
            ],
        }

        self.client.schema.create_class(schema)
        logger.info(f"Weaviate class '{WEAVIATE_CLASS}' created")

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding vector for a given string using local model."""
        text = text[:2000]
        model = _get_embedding_model()
        vector = model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def store_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> None:
        """Stores a list of documents in Weaviate using batch insertion."""
        total = len(chunks)
        logger.info(f"Storing {total} chunks in Weaviate (batch_size={batch_size})")

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start: batch_start + batch_size]

            with self.client.batch as batch_writer:
                batch_writer.batch_size = batch_size

                for chunk in batch:
                    try:
                        vector = self.embed_text(chunk["text"])
                    except Exception as e:
                        logger.warning(f"Embedding failed for chunk {chunk['chunk_id']}: {e}")
                        continue

                    properties = {
                        "chunk_id":          chunk["chunk_id"],
                        "text":              chunk["text"],
                        "source_post_id":    chunk.get("source_post_id", ""),
                        "source_comment_id": chunk.get("source_comment_id") or "",
                        "source_type":       chunk.get("source_type", "post"),
                        "author":            chunk.get("author", "[deleted]"),
                        "subreddit":         chunk.get("subreddit", ""),
                        "sentiment":         float(chunk.get("sentiment", 0.5)),
                        "created_at":        int(chunk.get("created_at", 0)),
                        "created_iso":       chunk.get("created_iso", ""),
                        "time_period":       chunk.get("time_period", ""),
                        "recency_label":     chunk.get("recency_label", ""),
                        "score":             int(chunk.get("score", 0)),
                        "topics":            ",".join(chunk.get("topics", [])),
                    }

                    batch_writer.add_data_object(
                        data_object=properties,
                        class_name=WEAVIATE_CLASS,
                        vector=vector,
                    )

            progress = min(batch_start + batch_size, total)
            logger.info(f"Vector storage: {progress}/{total} chunks stored")

    def semantic_search(
        self,
        query: str,
        time_range: Optional[Tuple[int, int]] = None,
        limit: int = TOP_K_VECTOR,
    ) -> List[Dict[str, Any]]:
        """Executes semantic search with optional metadata time filter."""
        query_vector = self.embed_text(query)

        weaviate_query = (
            self.client.query
            .get(WEAVIATE_CLASS, [
                "chunk_id", "text", "source_post_id", "author",
                "subreddit", "sentiment", "created_at", "created_iso",
                "time_period", "topics", "score",
            ])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .with_additional(["certainty", "distance"])
        )

        if time_range:
            start_ts, end_ts = time_range
            weaviate_query = weaviate_query.with_where(_build_time_filter(start_ts, end_ts))

        result = weaviate_query.do()
        raw_results = result.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])

        formatted = []
        for item in raw_results:
            additional = item.get("_additional", {})
            formatted.append({
                "id": item.get("chunk_id", ""),
                "text": item.get("text", ""),
                "source_post_id": item.get("source_post_id", ""),
                "author": item.get("author", ""),
                "subreddit": item.get("subreddit", ""),
                "sentiment": item.get("sentiment", 0.5),
                "created_at": item.get("created_at", 0),
                "created_iso": item.get("created_iso", ""),
                "time_period": item.get("time_period", ""),
                "topics": item.get("topics", "").split(",") if item.get("topics") else [],
                "score": item.get("score", 0),
                "similarity_score": additional.get("certainty", 0.0),
                "source": "vector",
            })

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Returns the count of vectors stored in the Weaviate class."""
        result = (
            self.client.query
            .aggregate(WEAVIATE_CLASS)
            .with_meta_count()
            .do()
        )
        count = (
            result.get("data", {})
            .get("Aggregate", {})
            .get(WEAVIATE_CLASS, [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )
        return {"total_chunks": count}
