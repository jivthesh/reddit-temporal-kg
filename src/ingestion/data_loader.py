"""
src/ingestion/data_loader.py

Unified data loader that handles streaming, live scraping, or synthetic fallback data.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.ingestion.synthetic_scraper import SyntheticDataScraper

logger = logging.getLogger("reddit-kg")

VALID_MODES = ("huggingface", "live", "synthetic")


def _resolve_time_range_strings(time_range: Optional[Any]) -> Tuple[str, str]:
    if time_range is None:
        time_range = {"start": "2025-04-01", "end": "2026-06-25"}

    if isinstance(time_range, dict):
        start_str = time_range.get("start", "2025-04-01")
        end_str = time_range.get("end", "2026-06-25")
    elif isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        start_str = time_range[0]
        end_str = time_range[1]
    else:
        start_str = "2025-04-01"
        end_str = "2026-06-25"
    return start_str, end_str


class DataLoader:
    """
    Main loader interface for Reddit posts.
    """

    def __init__(self, mode: str = "huggingface", use_real_reddit: Optional[bool] = None):
        if use_real_reddit is not None:
            mode = "live" if use_real_reddit else "synthetic"
            logger.warning(
                f"[DataLoader] use_real_reddit={use_real_reddit} is deprecated. Use mode='{mode}' instead."
            )

        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")

        self.mode = mode
        self._hf_loader = None
        self._live_scraper = None

        if self.mode == "live":
            from src.ingestion.reddit_scraper import RedditScraper
            self._live_scraper = RedditScraper()

        logger.info(f"[DataLoader] Initialised in mode='{self.mode}'")

    async def load_posts(self, subreddits: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch posts based on configured loading mode."""
        if self.mode == "huggingface":
            return self._load_huggingface(subreddits, limit)
        if self.mode == "live":
            return await self._load_live(subreddits, limit)
        return self._load_synthetic(subreddits, limit)

    def load_rag_posts(self, limit: int = 500, time_range: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Loads historical posts for RAG over specified range."""
        start_str, end_str = _resolve_time_range_strings(time_range)

        from utils.helpers import iso_to_ts
        try:
            start_ts = iso_to_ts(start_str)
            end_ts = iso_to_ts(end_str)
        except Exception:
            start_ts = 1743465600  # 2025-04-01
            end_ts = 1782396120    # 2026-06-25

        from src.ingestion.huggingface_loader import HuggingFaceRedditLoader, TARGET_SUBREDDITS

        if self._hf_loader is None:
            self._hf_loader = HuggingFaceRedditLoader()

        limit_per_sub = max(10, limit // len(TARGET_SUBREDDITS))
        raw_posts = self._hf_loader.load_posts(
            subreddits=TARGET_SUBREDDITS,
            limit=limit_per_sub,
            keyword_filter=True
        )

        filtered = [
            p for p in raw_posts
            if start_ts <= p.get("created_at", 0) <= end_ts
        ]

        logger.info(f"[DataLoader] load_rag_posts: {len(filtered)} posts in range {start_str} → {end_str}")
        return filtered[:limit]

    def _load_huggingface(self, subreddits: List[str], limit: int) -> List[Dict[str, Any]]:
        from src.ingestion.huggingface_loader import HuggingFaceRedditLoader

        if self._hf_loader is None:
            self._hf_loader = HuggingFaceRedditLoader()

        logger.info(f"[DataLoader] HuggingFace: subreddits={subreddits}, limit={limit}")
        posts = self._hf_loader.load_posts(subreddits=subreddits, limit=limit)
        logger.info(f"[DataLoader] HuggingFace returned {len(posts)} posts.")
        return posts

    async def _load_live(self, subreddits: List[str], limit: int) -> List[Dict[str, Any]]:
        logger.info(f"[DataLoader] Live Reddit: subreddits={subreddits}, limit={limit}")
        if self._live_scraper is None:
            from src.ingestion.reddit_scraper import RedditScraper
            self._live_scraper = RedditScraper()
        try:
            return await self._live_scraper.scrape_multiple_subreddits(
                subreddits=subreddits, limit=limit
            )
        finally:
            await self._live_scraper.close()

    def _load_synthetic(self, subreddits: List[str], limit: int) -> List[Dict[str, Any]]:
        logger.info("[DataLoader] Loading synthetic posts…")
        filepath = "synthetic_reddit_data.json"

        if not os.path.exists(filepath):
            logger.info(f"[DataLoader] '{filepath}' not found — generating mock data…")
            generator = SyntheticDataScraper()
            posts = generator.generate_all(150)
            generator.save_to_json(posts, filepath)
        else:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    posts = json.load(f)
                logger.info(f"[DataLoader] Loaded {len(posts)} posts from '{filepath}'")
            except Exception as e:
                logger.error(f"[DataLoader] Failed to read '{filepath}': {e}. Regenerating…")
                generator = SyntheticDataScraper()
                posts = generator.generate_all(150)
                generator.save_to_json(posts, filepath)

        filtered_posts: List[Dict[str, Any]] = []
        for sub in subreddits:
            sub_posts = [p for p in posts if p.get("subreddit", "").lower() == sub.lower()]
            filtered_posts.extend(sub_posts[:limit])
            logger.info(f"[DataLoader]   r/{sub}: {min(len(sub_posts), limit)} posts")

        logger.info(f"[DataLoader] Synthetic total: {len(filtered_posts)} posts returned")
        return filtered_posts
