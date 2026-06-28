"""
src/storage/graph_schema.py

Sets up unique constraints and indexes on Neo4j schema nodes.
"""

import logging
from neo4j import GraphDatabase

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger("reddit-kg")


def create_schema(driver):
    """Creates Neo4j constraints and indices for fast query lookups."""
    with driver.session() as session:
        logger.info("Creating Neo4j constraints and indexes...")

        session.run("""
            CREATE CONSTRAINT post_id IF NOT EXISTS
            FOR (p:Post) REQUIRE p.id IS UNIQUE
        """)

        session.run("""
            CREATE CONSTRAINT comment_id IF NOT EXISTS
            FOR (c:Comment) REQUIRE c.id IS UNIQUE
        """)

        session.run("""
            CREATE CONSTRAINT user_name IF NOT EXISTS
            FOR (u:User) REQUIRE u.name IS UNIQUE
        """)

        session.run("""
            CREATE CONSTRAINT topic_name IF NOT EXISTS
            FOR (t:Topic) REQUIRE t.name IS UNIQUE
        """)

        session.run("""
            CREATE INDEX entity_text_type IF NOT EXISTS
            FOR (e:Entity) ON (e.text, e.type)
        """)

        session.run("""
            CREATE INDEX post_created_at IF NOT EXISTS
            FOR (p:Post) ON (p.created_at)
        """)

        session.run("""
            CREATE INDEX post_subreddit IF NOT EXISTS
            FOR (p:Post) ON (p.subreddit)
        """)

        session.run("""
            CREATE INDEX post_sentiment IF NOT EXISTS
            FOR (p:Post) ON (p.sentiment)
        """)

        session.run("""
            CREATE INDEX comment_created_at IF NOT EXISTS
            FOR (c:Comment) ON (c.created_at)
        """)

        logger.info("Neo4j schema created successfully")


def initialize_graph_database():
    """Initializes GraphClient driver and setups database indexes."""
    logger.info(f"Connecting to Neo4j at {NEO4J_URI}")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    driver.verify_connectivity()
    logger.info("Neo4j connection verified")

    create_schema(driver)
    return driver


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    driver = initialize_graph_database()
    driver.close()
    print("Schema setup complete!")
