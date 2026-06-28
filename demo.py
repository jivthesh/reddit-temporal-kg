"""
demo.py

Command-line interface to execute the Reddit Temporal Knowledge Graph pipeline and run evaluation queries.
"""

import asyncio
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

from config.constants import TARGET_SUBREDDITS
from src.ingestion.data_loader import DataLoader
from src.ingestion.entity_extractor import batch_extract_entities
from src.ingestion.temporal_parser import batch_enrich_timestamps
from src.storage.graph_schema import create_schema
from src.storage.graph_client import GraphClient
from src.storage.vector_client import VectorClient
from src.storage.chunking import chunk_all_posts
from src.retrieval.hybrid_fusion import HybridFusion
from src.llm.answer_generator import AnswerGenerator
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, validate_settings

DEMO_QUERIES = [
    {
        "label": "Semantic (Vector-dominant)",
        "description": "Topical question — finds semantically similar chunks",
        "question": "What are the main concerns and criticisms about RAG systems?",
    },
    {
        "label": "Relational (Graph-dominant)",
        "description": "User/influence question — traverses user→post→topic graph",
        "question": "Who are the most active voices discussing AI safety on Reddit?",
    },
    {
        "label": "Hybrid (Both DBs)",
        "description": "Topic + sentiment — needs both semantic and graph data",
        "question": "How have community discussions about open-source LLMs shifted recently?",
    },
    {
        "label": "Temporal Analysis",
        "description": "Time-based sentiment change — graph provides trend data",
        "question": "How has sentiment around RAG changed in the last 6 months?",
    },
]


async def run_ingestion(limit_per_subreddit: int = 50, use_real_reddit: bool = False, use_huggingface: bool = True):
    """Executes the ingestion pipeline: data loading, entity extraction, timestamp parsing, and storage."""
    if use_real_reddit:
        loader_mode = "live"
    elif use_huggingface:
        loader_mode = "huggingface"
    else:
        loader_mode = "synthetic"

    print("\n" + "=" * 70)
    print("STEP 1: ACQUIRING REDDIT DATA")
    print(f"       Mode: {loader_mode.upper()}")
    print("=" * 70)

    loader = DataLoader(mode=loader_mode)
    if loader_mode == "huggingface":
        total_limit = limit_per_subreddit * len(TARGET_SUBREDDITS)
        posts = loader.load_rag_posts(
            limit=total_limit,
            time_range=("2025-04-01", "2026-06-25")
        )
        print(f"✓ Loaded {len(posts)} posts from Q2 2025 to Q2 2026")
    else:
        posts = await loader.load_posts(
            subreddits=TARGET_SUBREDDITS,
            limit=limit_per_subreddit,
        )
        print(f"✓ Loaded {len(posts)} posts from {len(TARGET_SUBREDDITS)} subreddits")

    if posts:
        sources = set(p.get('source', 'unknown') for p in posts)
        print(f"  Data source(s): {', '.join(sources)}")

    print("\n" + "=" * 70)
    print("STEP 2: EXTRACTING ENTITIES")
    print("=" * 70)

    posts = batch_extract_entities(posts, delay_between_calls=0.3)
    print(f"✓ Extracted entities from {len(posts)} posts")

    print("\n" + "=" * 70)
    print("STEP 3: ENRICHING TIMESTAMPS")
    print("=" * 70)

    posts = batch_enrich_timestamps(posts)
    print(f"✓ Timestamps enriched for {len(posts)} posts")

    print("\n" + "=" * 70)
    print("STEP 4: STORING IN NEO4J (GRAPH DB)")
    print("=" * 70)

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    create_schema(neo4j_driver)
    neo4j_driver.close()

    graph_client = GraphClient()
    graph_client.batch_store_posts(posts)
    stats = graph_client.get_stats()
    print(f"✓ Neo4j now has: {stats}")
    graph_client.close()

    print("\n" + "=" * 70)
    print("STEP 5: CHUNKING + STORING IN WEAVIATE (VECTOR DB)")
    print("=" * 70)

    chunks = chunk_all_posts(posts)
    print(f"✓ Created {len(chunks)} chunks from {len(posts)} posts")

    vector_client = VectorClient()
    vector_client.store_chunks(chunks, batch_size=25)
    vector_stats = vector_client.get_stats()
    print(f"✓ Weaviate now has: {vector_stats}")

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE ✓")
    print("=" * 70)


def run_queries():
    """Runs the demo evaluation queries against the stored data."""
    print("\n" + "=" * 70)
    print("INITIALIZING RETRIEVAL SYSTEM")
    print("=" * 70)

    graph_client = GraphClient()
    vector_client = VectorClient()
    fusion = HybridFusion(graph_client, vector_client)
    answer_gen = AnswerGenerator()

    print("✓ Connected to Neo4j and Weaviate")
    print(f"\nRunning {len(DEMO_QUERIES)} demo queries...\n")

    try:
        for i, demo in enumerate(DEMO_QUERIES, 1):
            _run_single_query(i, demo, fusion, answer_gen)
    finally:
        graph_client.close()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE ✓")
    print("=" * 70)


def _run_single_query(
    index: int,
    demo: dict,
    fusion: HybridFusion,
    answer_gen: AnswerGenerator,
) -> None:
    print("\n" + "─" * 70)
    print(f"[Query {index}/4] {demo['label']}")
    print(f"Description: {demo['description']}")
    print(f"Question: {demo['question']}")
    print("─" * 70)

    try:
        retrieval_output = fusion.retrieve(demo["question"])
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return

    results = retrieval_output.get("results", [])
    query_info = retrieval_output.get("query_info", {})

    print(f"\n[Retrieval Summary]")
    print(f"  Query type:  {query_info.get('query_type', 'unknown')}")
    print(f"  Main topic:  {query_info.get('main_topic', 'unknown')}")
    print(f"  Time range:  {query_info.get('time_range', 'none')}")
    print(f"  Intent:      {query_info.get('intent', 'unknown')}")
    print(f"  Graph hits:  {retrieval_output.get('graph_count', 0)}")
    print(f"  Vector hits: {retrieval_output.get('vector_count', 0)}")
    print(f"  Fused top-K: {len(results)}")

    graph_results = retrieval_output.get("graph_results", [])
    if graph_results:
        print("\n[Top 3 Graph-Only Results]")
        for j, result in enumerate(graph_results[:3], 1):
            source_tag = result.get('source', 'graph')
            meta = ""
            if source_tag == "graph":
                meta = f"u/{result.get('author', '?')} in r/{result.get('subreddit', '?')} ({result.get('time_period', '?')}) Score={result.get('score', 0)}"
            elif source_tag == "graph_user":
                meta = f"User @{result.get('author', '?')} (Post Count: {result.get('score', 0)})"
            elif source_tag == "graph_aggregate":
                meta = f"Aggregate Trend summary for {query_info.get('main_topic', 'unknown')}"
            print(f"  {j}. [{source_tag}] {meta}")
            text_preview = result.get("text", "")[:100].replace('\n', ' ')
            print(f"     \"{text_preview}...\"")
    else:
        print("\n[Graph-Only Results] None")

    vector_results = retrieval_output.get("vector_results", [])
    if vector_results:
        print("\n[Top 3 Vector-Only Results]")
        for j, result in enumerate(vector_results[:3], 1):
            print(
                f"  {j}. [vector] u/{result.get('author', '?')} "
                f"in r/{result.get('subreddit', '?')} "
                f"({result.get('time_period', '?')}) "
                f"Similarity={result.get('similarity_score', 0.0):.4f}"
            )
            text_preview = result.get("text", "")[:100].replace('\n', ' ')
            print(f"     \"{text_preview}...\"")
    else:
        print("\n[Vector-Only Results] None")

    if results:
        print("\n[Top 3 Fused Results]")
        for j, result in enumerate(results[:3], 1):
            print(
                f"  {j}. [{result.get('source', '?')}] "
                f"u/{result.get('author', '?')} "
                f"in r/{result.get('subreddit', '?')} "
                f"({result.get('time_period', '?')}) "
                f"RRF={result.get('rrf_score', 0):.4f}"
            )
            text_preview = result.get("text", "")[:100].replace('\n', ' ')
            print(f"     \"{text_preview}...\"")
    else:
        print("\n[Fused Results] None")

    print("\n[Generated Answer]")
    try:
        answer = answer_gen.generate(demo["question"], retrieval_output)
        print(answer)
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Reddit Temporal Knowledge Graph CLI Demo",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the data ingestion pipeline",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Execute query demonstration queries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of posts per subreddit limit",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Ingest via Live Reddit API credentials",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        default=True,
        help="Ingest via Hugging Face archived datasets",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ingest via pre-packaged offline synthetic data",
    )

    args = parser.parse_args()

    try:
        validate_settings(use_real_reddit=args.real)
    except EnvironmentError as e:
        print(f"\n❌ Configuration error:\n{e}\n")
        sys.exit(1)

    if args.real:
        loader_mode_flag = dict(use_real_reddit=True, use_huggingface=False)
    elif args.synthetic:
        loader_mode_flag = dict(use_real_reddit=False, use_huggingface=False)
    else:
        loader_mode_flag = dict(use_real_reddit=False, use_huggingface=True)

    run_both = not args.ingest and not args.query

    if args.ingest or run_both:
        asyncio.run(
            run_ingestion(
                limit_per_subreddit=args.limit,
                **loader_mode_flag,
            )
        )

    if args.query or run_both:
        run_queries()


if __name__ == "__main__":
    main()
