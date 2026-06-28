"""
utils/helpers.py

General utility functions used across components.
"""

import logging
import time
from datetime import datetime, timezone


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configures system-wide console logging format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("reddit-kg")


def ts_to_iso(unix_timestamp: int) -> str:
    """Converts a Unix timestamp integer to an ISO 8601 string."""
    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_ts(iso_string: str) -> int:
    """Converts an ISO 8601 string back to a Unix timestamp integer."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(iso_string, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue

    raise ValueError(f"Cannot parse date string: {iso_string}")


def get_quarter(unix_timestamp: int) -> str:
    """Maps a Unix timestamp to its corresponding calendar quarter."""
    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    quarter = (dt.month - 1) // 3 + 1
    return f"Q{quarter}-{dt.year}"


def sleep_with_log(seconds: float, reason: str = "rate limit"):
    """Pauses thread execution and writes a warning log."""
    logger = logging.getLogger("reddit-kg")
    logger.warning(f"Sleeping {seconds}s due to {reason}...")
    time.sleep(seconds)


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncates string to maximum character count."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def clean_text(text: str) -> str:
    """Standardizes whitespaces and strips padding from raw text."""
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()
