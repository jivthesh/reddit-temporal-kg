"""
src/ingestion/temporal_parser.py

Converts raw Reddit timestamps into standardized ISO strings, quarters, and recency categories.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

from utils.helpers import ts_to_iso, get_quarter

logger = logging.getLogger("reddit-kg")

SECONDS_IN_DAY = 86_400
SECONDS_IN_WEEK = SECONDS_IN_DAY * 7
SECONDS_IN_MONTH = SECONDS_IN_DAY * 30
SECONDS_IN_6_MONTHS = SECONDS_IN_MONTH * 6
SECONDS_IN_YEAR = SECONDS_IN_DAY * 365


def enrich_post_timestamps(post: Dict[str, Any]) -> Dict[str, Any]:
    """Decorates posts and comment nodes in-place with ISO strings and quarter labels."""
    ts = post.get("created_at", 0)

    post["created_iso"] = ts_to_iso(ts)
    post["time_period"] = get_quarter(ts)
    post["recency_label"] = _classify_recency(ts)

    for comment in post.get("comments", []):
        comment["created_iso"] = ts_to_iso(comment.get("created_at", 0))
        comment["time_period"] = get_quarter(comment.get("created_at", 0))

    return post


def _classify_recency(unix_ts: int) -> str:
    """Categorizes elapsed age intervals relative to the current timestamp."""
    now = int(datetime.now(tz=timezone.utc).timestamp())
    age_seconds = now - unix_ts

    if age_seconds < SECONDS_IN_WEEK:
        return "very_recent"
    elif age_seconds < SECONDS_IN_MONTH:
        return "recent"
    elif age_seconds < SECONDS_IN_MONTH * 3:
        return "3_months_ago"
    elif age_seconds < SECONDS_IN_6_MONTHS:
        return "6_months_ago"
    elif age_seconds < SECONDS_IN_YEAR:
        return "1_year_ago"
    else:
        return "old"


def parse_time_range_from_query(
    time_range, anchor_ts: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Converts a query's target range input into (start_ts, end_ts) Unix timestamps.

    anchor_ts: if provided, relative labels like '6_months' are resolved relative
    to this timestamp instead of wall-clock now. Pass the dataset's max created_at
    so that temporal queries work against the actual data window.
    """
    if time_range is None:
        return None

    if isinstance(time_range, str) and time_range.lower().strip() in ("null", "none"):
        return None

    ref_ts = anchor_ts if anchor_ts is not None else int(datetime.now(tz=timezone.utc).timestamp())

    if isinstance(time_range, str):
        return _relative_to_absolute(time_range, ref_ts)

    if isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        start, end = time_range

        if isinstance(start, int):
            return (start, end)

        from utils.helpers import iso_to_ts
        return (iso_to_ts(str(start)), iso_to_ts(str(end)))

    logger.warning(f"Unrecognized time_range format: {time_range}. No filter applied.")
    return None


def _relative_to_absolute(label: str, now_ts: int) -> Tuple[int, int]:
    label = label.lower().strip()

    offsets = {
        "1_week":    SECONDS_IN_WEEK,
        "1_month":   SECONDS_IN_MONTH,
        "3_months":  SECONDS_IN_MONTH * 3,
        "6_months":  SECONDS_IN_6_MONTHS,
        "1_year":    SECONDS_IN_YEAR,
    }

    offset = offsets.get(label)
    if offset is None:
        logger.warning(f"Unknown time label '{label}', defaulting to 1 year")
        offset = SECONDS_IN_YEAR

    return (now_ts - offset, now_ts)


def batch_enrich_timestamps(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies timestamp enrichment to a batch collection of post objects."""
    enriched = [enrich_post_timestamps(post) for post in posts]
    logger.info(f"Timestamp enrichment done: {len(enriched)} posts")
    return enriched
