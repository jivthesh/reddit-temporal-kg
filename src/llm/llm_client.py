"""
src/llm/llm_client.py

Provides a unified interface to various LLM APIs (Gemini, Claude, Mistral, Groq) with a local deterministic fallback.
"""

import os
import json
import logging
import re
import httpx
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from config.settings import ANTHROPIC_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, LLM_PROVIDER

logger = logging.getLogger("reddit-kg")


def is_retryable_http_error(exception: Exception) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in {429, 502, 503, 504}
    return False


def _key_is_valid(key: Optional[str]) -> bool:
    return bool(key and not key.startswith("your_"))


def _resolve_provider(
    preference: str,
    has_gemini: bool,
    has_mistral: bool,
    has_anthropic: bool,
) -> str:
    if preference in ("gemini", "mistral", "anthropic"):
        if preference == "gemini" and has_gemini:
            return "gemini"
        if preference == "mistral" and has_mistral:
            return "mistral"
        if preference == "anthropic" and has_anthropic:
            return "anthropic"
        logger.warning(
            f"LLM_PROVIDER set to '{preference}' but key is missing. Using auto-detection."
        )

    if has_gemini:
        return "gemini"
    if has_mistral:
        return "mistral"
    if has_anthropic:
        return "anthropic"
    return "local"


class LLMClient:
    """
    Unified client for text completions across LLM APIs, including local heuristics fallback.
    """

    def __init__(self):
        provider_preference = (LLM_PROVIDER or "").strip().lower()
        
        has_gemini = _key_is_valid(GEMINI_API_KEY)
        has_mistral = _key_is_valid(MISTRAL_API_KEY)
        has_anthropic = _key_is_valid(ANTHROPIC_API_KEY)

        self.provider = _resolve_provider(
            provider_preference,
            has_gemini,
            has_mistral,
            has_anthropic
        )
        logger.info(f"LLMClient initialized using provider: {self.provider.upper()}")

    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        """Dispatches prompting completion to selected provider or local fallback."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "mistral":
            return self._call_mistral(prompt, max_tokens)
        elif self.provider == "anthropic":
            return self._call_claude(prompt, max_tokens)
        return self._fallback_completion(prompt, max_tokens)

    def _fallback_completion(self, prompt: str, max_tokens: int) -> str:
        prompt_lower = prompt.lower()

        if "return this exact json structure" in prompt_lower or '"topics"' in prompt_lower:
            topics = ["RAG", "Open Source", "AI Safety"]
            entities = [{"text": "Reddit Community", "type": "CONCEPT"}]
            if "rag" in prompt_lower:
                topics = ["RAG", "Retrieval", "Vector Search"]
            elif "safety" in prompt_lower:
                topics = ["AI Safety", "Alignment", "Risk"]
            elif "llm" in prompt_lower or "open source" in prompt_lower:
                topics = ["Open Source LLMs", "Model Deployment", "Community Discussion"]
            return json.dumps({
                "topics": topics,
                "entities": entities,
                "sentiment": 0.55,
            })

        if "respond with only one word" in prompt_lower or "classifying a user's question" in prompt_lower:
            question = prompt.split("Question:", 1)[-1].strip().splitlines()[0] if "Question:" in prompt else prompt
            q_lower = question.lower()
            if any(token in q_lower for token in ["who", "user", "author", "influential", "active", "people"]):
                return "graph"
            if any(token in q_lower for token in ["how has", "trend", "changed", "sentiment", "evolved", "shift", "over time"]):
                return "hybrid"
            return "vector"

        if '"main_topic"' in prompt_lower or "parse this question" in prompt_lower:
            question = prompt.split("Question:", 1)[-1].strip().splitlines()[0] if "Question:" in prompt else prompt
            words = [w for w in re.findall(r"[A-Za-z0-9]+", question) if len(w) > 2][:5]
            main_topic = words[0] if words else "general"
            time_range = "null"
            if any(token in question.lower() for token in ["last 6 months", "6 months", "past 6 months"]):
                time_range = "6_months"
            elif any(token in question.lower() for token in ["last year", "year"]):
                time_range = "1_year"
            intent = "general"
            if any(token in question.lower() for token in ["sentiment", "positive", "negative"]):
                intent = "sentiment"
            elif any(token in question.lower() for token in ["trend", "changed", "shift", "evolved"]):
                intent = "trend"
            elif any(token in question.lower() for token in ["who", "user", "author"]):
                intent = "people"
            return json.dumps({
                "main_topic": main_topic,
                "time_range": time_range,
                "intent": intent,
                "keywords": words,
            })

        return (
            "I could not use an external LLM here, so I am relying on the retrieved "
            "Reddit evidence and local heuristics to answer the request."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_http_error),
        reraise=True
    )
    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            try:
                return data["contents"][0]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse Gemini response: {data}. Error: {e}")
                raise ValueError("Unexpected response structure from Gemini API")

    def _call_claude(self, prompt: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_http_error),
        reraise=True
    )
    def _call_mistral(self, prompt: str, max_tokens: int) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "open-mistral-7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse Mistral response: {data}. Error: {e}")
                raise ValueError("Unexpected response structure from Mistral API")
