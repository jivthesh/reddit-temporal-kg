"""
src/storage/graph_client.py

Manages all write and read queries targeting the Neo4j temporal graph store.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger("reddit-kg")


class GraphClient:
    """
    Client wrapper for executing Cypher queries and managing Neo4j connections.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info("GraphClient connected to Neo4j")

    def close(self):
        self.driver.close()

    def store_post(self, post: Dict[str, Any]) -> None:
        """Stores a post, its author, topics, entities, and comments in Neo4j."""
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Post {id: $id})
                SET p.title      = $title,
                    p.text       = $text,
                    p.sentiment  = $sentiment,
                    p.created_at = $created_at,
                    p.created_iso = $created_iso,
                    p.time_period = $time_period,
                    p.recency_label = $recency_label,
                    p.score      = $score,
                    p.subreddit  = $subreddit,
                    p.url        = $url
            """, {
                "id": post["id"],
                "title": post["title"],
                "text": post.get("text", ""),
                "sentiment": post.get("sentiment", 0.5),
                "created_at": post["created_at"],
                "created_iso": post.get("created_iso", ""),
                "time_period": post.get("time_period", ""),
                "recency_label": post.get("recency_label", ""),
                "score": post.get("score", 0),
                "subreddit": post["subreddit"],
                "url": post.get("url", ""),
            })

            if post.get("author") and post["author"] != "[deleted]":
                session.run("""
                    MERGE (u:User {name: $author})
                    WITH u
                    MATCH (p:Post {id: $post_id})
                    MERGE (u)-[:POSTED]->(p)
                """, {"author": post["author"], "post_id": post["id"]})

            for topic in post.get("topics", []):
                session.run("""
                    MERGE (t:Topic {name: $topic})
                    WITH t
                    MATCH (p:Post {id: $post_id})
                    MERGE (p)-[:DISCUSSES]->(t)
                """, {"topic": topic, "post_id": post["id"]})

            for entity in post.get("entities", []):
                session.run("""
                    MERGE (e:Entity {text: $text, type: $type})
                    WITH e
                    MATCH (p:Post {id: $post_id})
                    MERGE (p)-[:MENTIONS]->(e)
                """, {
                    "text": entity.get("text", ""),
                    "type": entity.get("type", "CONCEPT"),
                    "post_id": post["id"],
                })

        for comment in post.get("comments", []):
            self.store_comment(comment, post["id"])

    def store_comment(self, comment: Dict[str, Any], post_id: str) -> None:
        """Stores a single comment node and replies relationships."""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Comment {id: $id})
                SET c.text        = $text,
                    c.author      = $author,
                    c.created_at  = $created_at,
                    c.created_iso = $created_iso,
                    c.time_period = $time_period,
                    c.score       = $score
            """, {
                "id": comment["id"],
                "text": comment.get("text", ""),
                "author": comment.get("author", "[deleted]"),
                "created_at": comment.get("created_at", 0),
                "created_iso": comment.get("created_iso", ""),
                "time_period": comment.get("time_period", ""),
                "score": comment.get("score", 0),
            })

            session.run("""
                MATCH (c:Comment {id: $comment_id})
                MATCH (p:Post {id: $post_id})
                MERGE (c)-[:REPLIED_TO]->(p)
            """, {"comment_id": comment["id"], "post_id": post_id})

            author = comment.get("author", "[deleted]")
            if author and author != "[deleted]":
                session.run("""
                    MERGE (u:User {name: $author})
                    WITH u
                    MATCH (p:Post {id: $post_id})
                    MERGE (u)-[:COMMENTED_ON]->(p)
                """, {"author": author, "post_id": post_id})

    def find_posts_by_topic(
        self,
        topic: str,
        time_range: Optional[Tuple[int, int]] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Queries graph database for posts tagged under specific topic keywords."""
        with self.driver.session() as session:
            if time_range:
                result = session.run("""
                    MATCH (p:Post)-[:DISCUSSES]->(t:Topic)
                    WHERE (
                        toLower(t.name) CONTAINS toLower($topic)
                        OR toLower($topic) CONTAINS toLower(t.name)
                        OR (toLower($topic) CONTAINS 'rag' AND toLower(t.name) CONTAINS 'rag')
                        OR (toLower($topic) CONTAINS 'retrieval' AND toLower(t.name) CONTAINS 'retrieval')
                    )
                      AND p.created_at >= $start_ts
                      AND p.created_at <= $end_ts
                    RETURN p.id AS id,
                           p.title AS title,
                           p.text AS text,
                           p.sentiment AS sentiment,
                           p.created_at AS created_at,
                           p.created_iso AS created_iso,
                           p.time_period AS time_period,
                           p.subreddit AS subreddit,
                           p.score AS score
                    ORDER BY p.created_at DESC
                    LIMIT $limit
                """, {
                    "topic": topic,
                    "start_ts": time_range[0],
                    "end_ts": time_range[1],
                    "limit": limit,
                })
            else:
                result = session.run("""
                    MATCH (p:Post)-[:DISCUSSES]->(t:Topic)
                    WHERE (
                        toLower(t.name) CONTAINS toLower($topic)
                        OR toLower($topic) CONTAINS toLower(t.name)
                        OR (toLower($topic) CONTAINS 'rag' AND toLower(t.name) CONTAINS 'rag')
                        OR (toLower($topic) CONTAINS 'retrieval' AND toLower(t.name) CONTAINS 'retrieval')
                    )
                    RETURN p.id AS id,
                           p.title AS title,
                           p.text AS text,
                           p.sentiment AS sentiment,
                           p.created_at AS created_at,
                           p.created_iso AS created_iso,
                           p.time_period AS time_period,
                           p.subreddit AS subreddit,
                           p.score AS score
                    ORDER BY p.score DESC
                    LIMIT $limit
                """, {"topic": topic, "limit": limit})

            return [dict(record) for record in result]

    def find_influential_users(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Queries for top posting authors on a target topic."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[:POSTED]->(p:Post)-[:DISCUSSES]->(t:Topic)
                WHERE (toLower(t.name) CONTAINS toLower($topic) OR toLower($topic) CONTAINS toLower(t.name))
                RETURN u.name AS username,
                       COUNT(p) AS post_count,
                       AVG(p.sentiment) AS avg_sentiment
                ORDER BY post_count DESC
                LIMIT $limit
            """, {"topic": topic, "limit": limit})

            return [dict(record) for record in result]

    def get_sentiment_over_time(
        self, topic: str, time_range: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """Returns monthly grouped sentiment metrics for a topic."""
        seconds_per_month = 2_592_000

        with self.driver.session() as session:
            if time_range:
                result = session.run("""
                    MATCH (p:Post)-[:DISCUSSES]->(t:Topic)
                    WHERE (
                        toLower(t.name) CONTAINS toLower($topic)
                        OR toLower($topic) CONTAINS toLower(t.name)
                        OR (toLower($topic) CONTAINS 'rag' AND toLower(t.name) CONTAINS 'rag')
                        OR (toLower($topic) CONTAINS 'retrieval' AND toLower(t.name) CONTAINS 'retrieval')
                    )
                      AND p.created_at >= $start_ts
                      AND p.created_at <= $end_ts
                    RETURN
                        toInteger(p.created_at / $seconds_per_month) AS month_bucket,
                        AVG(p.sentiment) AS avg_sentiment,
                        COUNT(p) AS post_count
                    ORDER BY month_bucket ASC
                """, {
                    "topic": topic,
                    "start_ts": time_range[0],
                    "end_ts": time_range[1],
                    "seconds_per_month": seconds_per_month,
                })
            else:
                result = session.run("""
                    MATCH (p:Post)-[:DISCUSSES]->(t:Topic)
                    WHERE (
                        toLower(t.name) CONTAINS toLower($topic)
                        OR toLower($topic) CONTAINS toLower(t.name)
                        OR (toLower($topic) CONTAINS 'rag' AND toLower(t.name) CONTAINS 'rag')
                        OR (toLower($topic) CONTAINS 'retrieval' AND toLower(t.name) CONTAINS 'retrieval')
                    )
                    RETURN
                        toInteger(p.created_at / $seconds_per_month) AS month_bucket,
                        AVG(p.sentiment) AS avg_sentiment,
                        COUNT(p) AS post_count
                    ORDER BY month_bucket ASC
                """, {
                    "topic": topic,
                    "seconds_per_month": seconds_per_month,
                })

            return [dict(record) for record in result]

    def get_stats(self) -> Dict[str, int]:
        """Returns counts for graph node types."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Post) WITH COUNT(p) AS posts
                MATCH (c:Comment) WITH posts, COUNT(c) AS comments
                MATCH (u:User) WITH posts, comments, COUNT(u) AS users
                MATCH (t:Topic) WITH posts, comments, users, COUNT(t) AS topics
                MATCH (e:Entity) WITH posts, comments, users, topics, COUNT(e) AS entities
                RETURN posts, comments, users, topics, entities
            """)
            record = result.single()
            return dict(record) if record else {}

    def batch_store_posts(self, posts: List[Dict[str, Any]]) -> None:
        """Stores a batch list of post dicts inside the graph."""
        total = len(posts)
        for i, post in enumerate(posts):
            self.store_post(post)
            if (i + 1) % 10 == 0:
                logger.info(f"Graph storage: {i+1}/{total} posts stored")

        logger.info(f"All {total} posts stored in Neo4j")
