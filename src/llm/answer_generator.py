"""
src/llm/answer_generator.py

Synthesizes final answers with source citations using the retrieved context from hybrid databases.
"""

import logging
from typing import Dict, Any, List

from config.constants import CLAUDE_MAX_TOKENS
from src.llm.prompts import ANSWER_SYNTHESIS_PROMPT
from src.llm.citation_handler import build_context_for_llm, inject_citation_numbers
from src.llm.llm_client import LLMClient

logger = logging.getLogger("reddit-kg")

llm_client = LLMClient()


class AnswerGenerator:
    """
    Generates structured answers using LLM providers, incorporating inline sources and citations.
    """

    def generate(self, question: str, retrieval_output: Dict[str, Any], append_sources: bool = True) -> str:
        """Synthesizes final answer from retrieval outputs."""
        results = retrieval_output.get("results", [])
        query_info = retrieval_output.get("query_info", {})

        if not results:
            return (
                f"I couldn't find relevant Reddit discussions to answer: '{question}'\n\n"
                "This might be because:\n"
                "- The topic hasn't been ingested yet\n"
                "- The time range is too narrow\n"
                "- Try rephrasing with different keywords"
            )

        context = build_context_for_llm(results)
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            context=context,
        )

        logger.info(f"Generating answer for: '{question}' (using {len(results)} context items)")

        try:
            raw_answer = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}\n\n{context}"

        if append_sources:
            final_answer = inject_citation_numbers(raw_answer, results)
        else:
            final_answer = raw_answer

        return final_answer

    def _call_llm(self, prompt: str) -> str:
        return llm_client.generate_completion(prompt, max_tokens=CLAUDE_MAX_TOKENS)

    def generate_summary(self, results: List[Dict[str, Any]], topic: str) -> str:
        """Generates a brief summary from search results without bibliography listings."""
        if not results:
            return f"No data found for topic: {topic}"

        short_context = build_context_for_llm(results[:5])
        prompt = (
            f"Summarize in 2-3 sentences what Reddit users are saying about '{topic}' "
            f"based on these posts:\n\n{short_context}\n\nSummary:"
        )

        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Could not generate summary: {e}"
