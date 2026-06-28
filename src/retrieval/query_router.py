"""
src/retrieval/query_router.py

Parses and classifies user queries to determine the optimal retrieval strategy (graph, vector, or hybrid).
"""

import json
import logging
from typing import Dict, Any, Optional

from config.constants import CLAUDE_MODEL
from src.llm.prompts import QUERY_CLASSIFICATION_PROMPT, QUERY_UNDERSTANDING_PROMPT
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


class QueryRouter:
    """
    Classifies query retrieval types and parses structured semantic intents.
    """

    def classify_query(self, question: str) -> str:
        """Classifies the query into 'graph', 'vector', or 'hybrid' modes."""
        prompt = QUERY_CLASSIFICATION_PROMPT.format(question=question)

        try:
            response = llm_client.generate_completion(prompt, max_tokens=10)
            cleaned = response.strip().lower()

            if "graph" in cleaned:
                classification = "graph"
            elif "vector" in cleaned:
                classification = "vector"
            elif "hybrid" in cleaned:
                classification = "hybrid"
            else:
                logger.warning(f"Unexpected classification response: '{response.strip()}', defaulting to 'hybrid'")
                classification = "hybrid"

            logger.info(f"Query classified as: {classification}")
            return classification

        except Exception as e:
            logger.error(f"Query classification failed: {e}. Defaulting to 'hybrid'")
            return "hybrid"

    def understand_query(self, question: str) -> Dict[str, Any]:
        """Extracts key topic, time limits, and classification intents."""
        prompt = QUERY_UNDERSTANDING_PROMPT.format(question=question)

        try:
            response = llm_client.generate_completion(prompt, max_tokens=300)
            raw = _extract_json_block(response)
            parsed = json.loads(raw)
            if "main_topic" in parsed and isinstance(parsed["main_topic"], str):
                parsed["main_topic"] = parsed["main_topic"].replace("_", " ").replace("-", " ").strip()

            logger.info(
                f"Query understood: topic='{parsed.get('main_topic')}', "
                f"time='{parsed.get('time_range')}', intent='{parsed.get('intent')}'"
            )
            return parsed

        except Exception as e:
            logger.error(f"Query understanding failed: {e}. Using fallback.")
            keywords = question.lower().split()[:5]
            return {
                "main_topic": keywords[0] if keywords else "general",
                "time_range": None,
                "intent": "general",
                "keywords": keywords,
            }

    def route(self, question: str) -> Dict[str, Any]:
        """Orchestrates query classification and intent extraction."""
        query_type = self.classify_query(question)
        understanding = self.understand_query(question)

        return {
            "query_type": query_type,
            "original_question": question,
            **understanding,
        }
