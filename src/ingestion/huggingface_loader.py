"""
src/ingestion/huggingface_loader.py

Streams real archived Reddit data from publicly available Hugging Face datasets.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from config.settings import RAG_KEYWORDS_FILTER, MIN_COMMENTS_FOR_RELEVANCE

logger = logging.getLogger("reddit-kg")

HF_DATASETS = [
    {
        "name": "hblim/top_reddit_posts_daily",
        "split": "train",
        "source_tag": "huggingface_top_posts",
        "data_files": "data_raw/*.parquet",
    },
    {
        "name": "fddemarco/pushshift-reddit",
        "split": "train",
        "source_tag": "huggingface_pushshift",
    },
]

RAG_KEYWORDS: List[str] = [
    "RAG", "retrieval-augmented generation", "retrieval augmented",
    "vector database", "semantic search", "embedding",
    "knowledge base", "document retrieval", "retrieval quality",
    "LLamaIndex", "LangChain", "Haystack", "Vespa",
    "BM25", "dense retrieval", "sparse retrieval", "reranking",
    "cross-encoder", "ColBERT", "ranking",
    "chunk", "context window", "prompt injection",
    "few-shot", "in-context learning"
]

TARGET_SUBREDDITS: List[str] = [
    "MachineLearning",
    "LanguageModels",
    "artificial",
    "LocalLLaMA",
    "mlops",
    "datascience",
    "learnmachinelearning",
    "ArtificialIntelligence",
]


def _compute_sentiment(text: str) -> float:
    """TextBlob polarity mapped to [0, 1]. Defaults to 0.5."""
    if not text or not text.strip():
        return 0.5
    try:
        from textblob import TextBlob  # type: ignore
        polarity = TextBlob(text).sentiment.polarity
        return round((polarity + 1.0) / 2.0, 4)
    except Exception:
        return 0.5


def filter_rag_posts(post_title: str, post_text: str) -> bool:
    """Checks if text mentions core RAG keywords."""
    combined_text = (post_title + " " + post_text).lower()
    return any(kw.lower() in combined_text for kw in RAG_KEYWORDS)


def _post_matches_keywords(title: str, body: str) -> bool:
    return filter_rag_posts(title, body)


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.85) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_title_and_body(raw: Dict[str, Any]) -> Tuple[str, str]:
    """Retrieves title and text content from raw row dict format."""
    title = ""
    body = ""
    if "title" in raw:
        title = (raw.get("title") or "").strip()
        body = (raw.get("selftext") or raw.get("body") or raw.get("text") or "").strip()
    elif "text" in raw:
        text_val = (raw.get("text") or "").strip()
        if "\n\n" in text_val:
            parts = text_val.split("\n\n", 1)
            title = parts[0].strip()
            body = parts[1].strip()
        else:
            title = text_val
            body = ""

    if body in ("[deleted]", "[removed]"):
        body = ""
    return title, body


def _map_subreddit(
    raw_sub: str,
    sub_lower_set: set,
    buckets: Dict[str, List[Dict[str, Any]]],
    limit: int,
    subreddits: List[str],
) -> Optional[str]:
    """Maps the raw sub name to one of the target subreddits."""
    raw_sub_lower = raw_sub.strip().lower()
    if not raw_sub_lower:
        return None

    if raw_sub_lower in sub_lower_set and len(buckets[raw_sub_lower]) < limit:
        for s in subreddits:
            if s.lower() == raw_sub_lower:
                return s
        return None

    if raw_sub_lower == "localllama":
        if "languagemodels" in sub_lower_set and len(buckets["languagemodels"]) < limit:
            return "LanguageModels"
        if "machinelearning" in sub_lower_set and len(buckets["machinelearning"]) < limit:
            return "MachineLearning"
        if "languagemodels" in sub_lower_set:
            return "LanguageModels"
        if "machinelearning" in sub_lower_set:
            return "MachineLearning"
            
    elif raw_sub_lower == "openai":
        if "languagemodels" in sub_lower_set and len(buckets["languagemodels"]) < limit:
            return "LanguageModels"
        if "artificial" in sub_lower_set and len(buckets["artificial"]) < limit:
            return "artificial"
        if "machinelearning" in sub_lower_set and len(buckets["machinelearning"]) < limit:
            return "MachineLearning"
        if "languagemodels" in sub_lower_set:
            return "LanguageModels"
        if "artificial" in sub_lower_set:
            return "artificial"
            
    elif raw_sub_lower == "artificial":
        if "artificial" in sub_lower_set and len(buckets["artificial"]) < limit:
            return "artificial"
        if "machinelearning" in sub_lower_set and len(buckets["machinelearning"]) < limit:
            return "MachineLearning"
        if "artificial" in sub_lower_set:
            return "artificial"
            
    elif raw_sub_lower == "singularity":
        if "artificial" in sub_lower_set and len(buckets["artificial"]) < limit:
            return "artificial"
        if "languagemodels" in sub_lower_set and len(buckets["languagemodels"]) < limit:
            return "LanguageModels"
        if "machinelearning" in sub_lower_set and len(buckets["machinelearning"]) < limit:
            return "MachineLearning"
        if "artificial" in sub_lower_set:
            return "artificial"

    return None


def _classify_by_content(
    title: str,
    body: str,
    sub_lower_set: set,
    buckets: Dict[str, List[Dict[str, Any]]],
    limit: int,
) -> Optional[str]:
    combined_content = (title + " " + body).lower()
    if "machine learning" in combined_content or "deep learning" in combined_content:
        if "machinelearning" in sub_lower_set and len(buckets["machinelearning"]) < limit:
            return "MachineLearning"
    elif any(kw in combined_content for kw in ["llm", "language model", "gpt", "claude", "llama"]):
        if "languagemodels" in sub_lower_set and len(buckets["languagemodels"]) < limit:
            return "LanguageModels"
    elif any(kw in combined_content for kw in ["rag", "vector database", "semantic search"]):
        if "languagemodels" in sub_lower_set and len(buckets["languagemodels"]) < limit:
            return "LanguageModels"
        if "artificial" in sub_lower_set and len(buckets["artificial"]) < limit:
            return "artificial"
    return None


def _normalize_created_at(raw: Dict[str, Any]) -> int:
    created_at_val = raw.get("created_utc") or raw.get("created_at") or 0
    if hasattr(created_at_val, "timestamp"):
        return int(created_at_val.timestamp())
    if isinstance(created_at_val, (int, float)):
        return int(created_at_val)
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(str(created_at_val)).timestamp())
    except Exception:
        return int(time.time())


def _build_url(raw: Dict[str, Any], mapped_sub: str, post_id: str) -> str:
    permalink = raw.get("permalink") or raw.get("url") or ""
    if permalink and not permalink.startswith("http"):
        return f"https://reddit.com{permalink}"
    if permalink:
        return permalink
    return f"https://reddit.com/r/{mapped_sub}/comments/{post_id}"


def _build_post_dict(
    post_id: str,
    title: str,
    body: str,
    author: str,
    created_at: int,
    score: int,
    upvote_ratio: float,
    num_comments: int,
    mapped_sub: str,
    url: str,
    sentiment: float,
    source_tag: str,
) -> Dict[str, Any]:
    return {
        "id": str(post_id),
        "title": title,
        "text": body if body else title,
        "author": author,
        "created_at": created_at,
        "score": score,
        "upvote_ratio": upvote_ratio,
        "num_comments": num_comments,
        "subreddit": mapped_sub,
        "url": url,
        "comments": [],
        "topics": [],
        "entities": [],
        "sentiment": sentiment,
        "source": source_tag,
    }


class HuggingFaceRedditLoader:
    """
    Streams historical posts from Hugging Face datasets.
    """

    def __init__(self):
        self._check_dependencies()

    @staticmethod
    def _check_dependencies() -> None:
        missing = []
        try:
            import datasets  # noqa: F401
        except ImportError:
            missing.append("datasets")
        try:
            import textblob  # noqa: F401
        except ImportError:
            missing.append("textblob")
        if missing:
            raise ImportError(
                f"Missing required packages: {missing}. Install with: pip install {' '.join(missing)}"
            )

    def load_posts(
        self,
        subreddits: List[str],
        limit: int = 50,
        keyword_filter: bool = RAG_KEYWORDS_FILTER,
        min_score: int = 0,
        min_comments: int = MIN_COMMENTS_FOR_RELEVANCE,
        max_scan: int = 200_000,
    ) -> List[Dict[str, Any]]:
        """Fetch posts matching target subreddits using streaming parquet endpoints."""
        from datasets import load_dataset  # type: ignore

        sub_lower_set = {s.lower() for s in subreddits}
        last_error: Optional[Exception] = None

        for ds_config in HF_DATASETS:
            ds_name = ds_config["name"]
            ds_split = ds_config["split"]
            source_tag = ds_config["source_tag"]

            logger.info(
                f"[HuggingFaceLoader] Attempting dataset '{ds_name}' for subreddits={subreddits}, limit={limit}…"
            )

            try:
                load_kwargs = {
                    "split": ds_split,
                    "streaming": True,
                }
                if "data_files" in ds_config:
                    load_kwargs["data_files"] = ds_config["data_files"]

                dataset = load_dataset(ds_name, **load_kwargs)
            except Exception as e:
                logger.warning(f"[HuggingFaceLoader] Dataset '{ds_name}' failed to load: {e}")
                last_error = e
                continue

            posts = self._stream_posts(
                dataset=dataset,
                sub_lower_set=sub_lower_set,
                subreddits=subreddits,
                limit=limit,
                keyword_filter=keyword_filter,
                min_score=min_score,
                min_comments=min_comments,
                max_scan=max_scan,
                source_tag=source_tag,
            )

            if posts:
                logger.info(f"[HuggingFaceLoader] Loaded {len(posts)} posts from '{ds_name}'.")
                return posts

            logger.warning(f"[HuggingFaceLoader] '{ds_name}' returned 0 matching posts.")

        err_msg = str(last_error) if last_error else "All HF datasets exhausted"
        raise RuntimeError(f"[HuggingFaceLoader] Could not load real Reddit data: {err_msg}")

    def _stream_posts(
        self,
        dataset,
        sub_lower_set: set,
        subreddits: List[str],
        limit: int,
        keyword_filter: bool,
        min_score: int,
        min_comments: int,
        max_scan: int,
        source_tag: str,
    ) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {s.lower(): [] for s in subreddits}
        total_needed = limit * len(subreddits)

        t0 = time.time()
        scanned = 0
        collected = 0

        for raw in dataset:
            if scanned >= max_scan:
                logger.warning(
                    f"[HuggingFaceLoader] Reached max_scan={max_scan:,}. Collected {collected}/{total_needed}."
                )
                break

            if collected >= total_needed and all(len(v) >= limit for v in buckets.values()):
                break

            scanned += 1
            row_type = raw.get("type", "post")
            if row_type != "post":
                continue

            title, body = _extract_title_and_body(raw)

            raw_sub = (raw.get("subreddit") or "").strip()
            mapped_sub = _map_subreddit(raw_sub, sub_lower_set, buckets, limit, subreddits)

            if not mapped_sub:
                mapped_sub = _classify_by_content(title, body, sub_lower_set, buckets, limit)

            if not mapped_sub:
                for s in subreddits:
                    if len(buckets[s.lower()]) < limit:
                        mapped_sub = s
                        break
                if not mapped_sub:
                    continue

            mapped_sub_lower = mapped_sub.lower()
            bucket = buckets[mapped_sub_lower]
            if len(bucket) >= limit:
                continue

            score = _safe_int(raw.get("score"), 0)
            if score < min_score:
                continue

            if "num_comments" in raw:
                num_comments = _safe_int(raw.get("num_comments"), 0)
                if num_comments < min_comments:
                    continue
            else:
                num_comments = 0

            if keyword_filter and not _post_matches_keywords(title, body):
                continue

            post_id = raw.get("post_id") or raw.get("id") or f"hf_{scanned:06d}"
            author = (raw.get("author") or "unknown").strip()
            if author in ("[deleted]", "[removed]", ""):
                author = "unknown"

            created_at = _normalize_created_at(raw)
            upvote_ratio = _safe_float(raw.get("upvote_ratio"), 0.85)
            url = _build_url(raw, mapped_sub, post_id)
            sentiment = _compute_sentiment(f"{title} {body}")

            post = _build_post_dict(
                post_id=post_id,
                title=title,
                body=body,
                author=author,
                created_at=created_at,
                score=score,
                upvote_ratio=upvote_ratio,
                num_comments=num_comments,
                mapped_sub=mapped_sub,
                url=url,
                sentiment=sentiment,
                source_tag=source_tag,
            )

            bucket.append(post)
            collected += 1
            logger.info(
                f"[HuggingFaceLoader] Added post to '{mapped_sub_lower}'. Buckets: { {k: len(v) for k, v in buckets.items()} }"
            )

            if collected % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[HuggingFaceLoader] {collected}/{total_needed} posts after {scanned:,} records scanned ({elapsed:.1f}s elapsed)"
                )

        elapsed = time.time() - t0
        all_posts = [p for bkt in buckets.values() for p in bkt]

        logger.info(
            f"[HuggingFaceLoader] Finished. {len(all_posts)} posts from {scanned:,} records in {elapsed:.1f}s."
        )
        return all_posts

    def save_to_json(self, posts: List[Dict[str, Any]], filename: str) -> None:
        """Persist posts to a local JSON file."""
        import json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        logger.info(f"[HuggingFaceLoader] Saved {len(posts)} posts to '{filename}'")


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
    )

    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    loader = HuggingFaceRedditLoader()
    posts = loader.load_posts(
        subreddits=TARGET_SUBREDDITS,
        limit=limit_arg,
    )

    print(f"\n✅ Loaded {len(posts)} REAL Reddit posts from Hugging Face archive")
    if posts:
        s = posts[0]
        print(f"\nSample post:")
        print(f"  Title     : {s['title'][:80]}")
        print(f"  Author    : u/{s['author']}")
        print(f"  Subreddit : r/{s['subreddit']}")
        print(f"  Score     : {s['score']}")
        print(f"  Sentiment : {s['sentiment']}")
        print(f"  Source    : {s['source']}")

    output_file = "real_reddit_hf_data.json"
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(posts, fh, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {output_file}")
