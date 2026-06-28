"""
src/retrieval/graph_retriever.py

Retrieves relevant graph nodes from Neo4j based on parsed query intents.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from src.storage.graph_client import GraphClient
from src.ingestion.temporal_parser import parse_time_range_from_query
from config.constants import TOP_K_GRAPH

logger = logging.getLogger("reddit-kg")


class GraphRetriever:
    """
    Executes graph queries on Neo4j matching routed question intents.
    """

    def __init__(self, graph_client: GraphClient):
        self.graph = graph_client

    def _get_data_max_ts(self) -> Optional[int]:
        """Returns the maximum created_at timestamp in the Neo4j dataset."""
        try:
            with self.graph.driver.session() as session:
                result = session.run("MATCH (p:Post) RETURN max(p.created_at) AS max_ts")
                record = result.single()
                return record["max_ts"] if record else None
        except Exception:
            return None

    def retrieve(self, query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Maps classification intent to appropriate graph query traversal method."""
        intent = query_info.get("intent", "general")
        topic = query_info.get("main_topic", "")
        time_range_label = query_info.get("time_range")

        anchor_ts = self._get_data_max_ts()
        time_range = parse_time_range_from_query(time_range_label, anchor_ts=anchor_ts)

        logger.info(f"Graph retrieval: topic='{topic}', intent='{intent}', time_range={time_range}")

        if intent == "people":
            return self._retrieve_users(topic)
        elif intent in ("trend", "sentiment"):
            return self._retrieve_for_sentiment(topic, time_range)
        else:
            return self._retrieve_posts(topic, time_range)

    def _retrieve_posts(
        self, topic: str, time_range: Optional[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        raw_results = self.graph.find_posts_by_topic(
            topic=topic,
            time_range=time_range,
            limit=TOP_K_GRAPH,
        )

        results = []
        for post in raw_results:
            results.append({
                "id": post.get("id", ""),
                "text": f"{post.get('title', '')} {post.get('text', '')}".strip(),
                "title": post.get("title", ""),
                "author": post.get("author", ""),
                "subreddit": post.get("subreddit", ""),
                "sentiment": post.get("sentiment", 0.5),
                "created_at": post.get("created_at", 0),
                "created_iso": post.get("created_iso", ""),
                "time_period": post.get("time_period", ""),
                "score": post.get("score", 0),
                "source": "graph",
                "similarity_score": 0.0,
            })

        logger.info(f"Graph retrieval found {len(results)} posts for topic '{topic}'")
        return results

    def _retrieve_for_sentiment(
        self, topic: str, time_range: Optional[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        posts = self._retrieve_posts(topic, time_range)
        trend_data = self.graph.get_sentiment_over_time(topic, time_range)

        if trend_data:
            trend_text = _format_trend_as_text(trend_data, topic)
            posts.insert(0, {
                "id": f"trend_summary_{topic}",
                "text": trend_text,
                "title": f"Sentiment trend for {topic}",
                "author": "system",
                "subreddit": "aggregate",
                "sentiment": 0.5,
                "created_at": 0,
                "created_iso": "",
                "time_period": "aggregated",
                "score": 999,
                "source": "graph_aggregate",
                "similarity_score": 0.0,
            })

        return posts

    def _retrieve_users(self, topic: str) -> List[Dict[str, Any]]:
        raw_users = self.graph.find_influential_users(topic, limit=10)

        results = []
        for user in raw_users:
            results.append({
                "id": f"user_{user.get('username', '')}",
                "text": (
                    f"User @{user.get('username')} has made {user.get('post_count')} "
                    f"posts about {topic} with average sentiment "
                    f"{user.get('avg_sentiment', 0.5):.2f}"
                ),
                "title": f"Influential user: {user.get('username')}",
                "author": user.get("username", ""),
                "subreddit": "aggregate",
                "sentiment": float(user.get("avg_sentiment", 0.5)),
                "created_at": 0,
                "created_iso": "",
                "time_period": "",
                "score": int(user.get("post_count", 0)),
                "source": "graph_user",
                "similarity_score": 0.0,
            })

        logger.info(f"Graph retrieval found {len(results)} users for topic '{topic}'")
        return results


def _format_trend_as_text(trend_data: List[Dict], topic: str) -> str:
    lines = [f"Sentiment trend for '{topic}' over time:"]

    for i, month_data in enumerate(trend_data, 1):
        avg_sent = month_data.get("avg_sentiment", 0.5)
        count = month_data.get("post_count", 0)

        if i > 1:
            prev = trend_data[i - 2].get("avg_sentiment", 0.5)
            direction = "↑" if avg_sent > prev else "↓" if avg_sent < prev else "→"
        else:
            direction = ""

        lines.append(f"  Period {i}: avg sentiment {avg_sent:.2f} {direction} ({count} posts)")

    return "\n".join(lines)
