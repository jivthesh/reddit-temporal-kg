"""
src/retrieval/hybrid_fusion.py

Orchestrates hybrid retrieval by merging results from Neo4j (graph) and Weaviate (vector) databases using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Dict, Any, Optional

from src.retrieval.query_router import QueryRouter
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.vector_retriever import VectorRetriever
from config.constants import RRF_K, TOP_K_FINAL

logger = logging.getLogger("reddit-kg")


class HybridFusion:
    """
    Manages parallel queries to the graph and vector backends and merges outputs.
    """

    def __init__(self, graph_client, vector_client):
        self.router = QueryRouter()
        self.graph_retriever = GraphRetriever(graph_client)
        self.vector_retriever = VectorRetriever(vector_client)

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K_FINAL,
        query_type_override: str = "hybrid",
        time_range_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes query routing, parallel retrieval, and RRF ranking.
        """
        logger.info(f"Retrieving for query: '{question}'")
        query_info = self.router.route(question)

        query_type = query_type_override if query_type_override != "hybrid" else query_info.get("query_type", "hybrid")
        logger.info(f"Retrieval mode: {query_type}")

        if time_range_override is not None:
            query_info["time_range"] = time_range_override

        graph_results = []
        vector_results = []

        anchor_ts = self.graph_retriever._get_data_max_ts()

        if query_type in ("graph", "hybrid"):
            graph_results = self.graph_retriever.retrieve(query_info)

        if query_type in ("vector", "hybrid"):
            vector_results = self.vector_retriever.retrieve(query_info, anchor_ts=anchor_ts)

        logger.info(f"Raw hits: {len(graph_results)} graph, {len(vector_results)} vector")

        fused_results = self._rrf_fusion(graph_results, vector_results, top_k=top_k)
        top_results = self._hydrate_results(fused_results, graph_results, vector_results)

        return {
            "results": top_results,
            "query_info": query_info,
            "graph_count": len(graph_results),
            "vector_count": len(vector_results),
            "graph_results": graph_results,
            "vector_results": vector_results,
        }

    def _rrf_fusion(
        self,
        graph_results: List[Dict],
        vector_results: List[Dict],
        k: int = RRF_K,
        top_k: int = TOP_K_FINAL,
    ) -> List[tuple]:
        """
        Calculates Reciprocal Rank Fusion scores across both ranked result lists.
        """
        scores: Dict[str, float] = {}

        for rank, result in enumerate(graph_results):
            item_id = result.get("id", "")
            if item_id:
                scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank + 1))

        for rank, result in enumerate(vector_results):
            item_id = result.get("id", "")
            if item_id:
                scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank + 1))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _hydrate_results(
        self,
        fused: List[tuple],
        graph_results: List[Dict],
        vector_results: List[Dict],
    ) -> List[Dict]:
        """
        Hydrates fused document IDs back into full result dictionaries with metadata.
        """
        graph_by_id = {r["id"]: r for r in graph_results if r.get("id")}
        vector_by_id = {r["id"]: r for r in vector_results if r.get("id")}

        hydrated = []
        for item_id, rrf_score in fused:
            result = graph_by_id.get(item_id) or vector_by_id.get(item_id)
            if result:
                enriched = {**result, "rrf_score": round(rrf_score, 6)}
                hydrated.append(enriched)

        return hydrated
