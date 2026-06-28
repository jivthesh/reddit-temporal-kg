"""
src/ingestion/reddit_scraper.py

Scrapes Reddit posts and comment trees concurrently using AsyncPRAW.
"""

import asyncio
import logging
import asyncpraw
from typing import List, Dict, Any

from config.settings import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from config.constants import MAX_COMMENTS_PER_POST, POSTS_PER_SUBREDDIT, TARGET_SUBREDDITS
from utils.helpers import clean_text

logger = logging.getLogger("reddit-kg")


class RedditScraper:
    """
    Scraper implementation wrapping AsyncPRAW client.
    """

    def __init__(self):
        self.reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            read_only=True,
        )
        logger.info("Reddit client initialized (read-only mode)")

    async def scrape_subreddit(
        self, subreddit_name: str, limit: int = POSTS_PER_SUBREDDIT
    ) -> List[Dict[str, Any]]:
        """Scrapes hot submissions and comments for a specific subreddit."""
        logger.info(f"Scraping r/{subreddit_name} (limit={limit})")
        posts = []

        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            async for submission in subreddit.hot(limit=limit):
                post_data = await self._extract_post(submission, subreddit_name)
                posts.append(post_data)

                if len(posts) % 50 == 0:
                    logger.info(f"  r/{subreddit_name}: fetched {len(posts)} posts so far")

        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {e}")

        logger.info(f"Done scraping r/{subreddit_name}: {len(posts)} posts")
        return posts

    async def _extract_post(self, submission, subreddit_name: str) -> Dict[str, Any]:
        """Maps submission object attributes into canonical post schema dict."""
        author_name = "[deleted]"
        if submission.author:
            try:
                author_name = submission.author.name
            except Exception:
                author_name = "[deleted]"

        post_data = {
            "id": submission.id,
            "title": submission.title,
            "text": clean_text(submission.selftext or ""),
            "author": author_name,
            "created_at": int(submission.created_utc),
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "subreddit": subreddit_name,
            "url": submission.url,
            "comments": [],
        }

        post_data["comments"] = await self._extract_comments(submission)
        return post_data

    async def _extract_comments(self, submission) -> List[Dict[str, Any]]:
        """Reconstructs top comments list from submission comment tree."""
        comments = []

        try:
            await submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:MAX_COMMENTS_PER_POST]:
                if not comment.body or comment.body in ("[deleted]", "[removed]"):
                    continue

                author_name = "[deleted]"
                if comment.author:
                    try:
                        author_name = comment.author.name
                    except Exception:
                        author_name = "[deleted]"

                comments.append({
                    "id": comment.id,
                    "text": clean_text(comment.body),
                    "author": author_name,
                    "created_at": int(comment.created_utc),
                    "score": comment.score,
                    "parent_id": comment.parent_id,
                })

        except Exception as e:
            logger.warning(f"Could not load comments for post: {e}")

        return comments

    async def scrape_multiple_subreddits(
        self, subreddits: List[str] = None, limit: int = POSTS_PER_SUBREDDIT
    ) -> List[Dict[str, Any]]:
        """Orchestrates concurrent scrapes across multiple subreddits."""
        subreddits = subreddits or TARGET_SUBREDDITS
        logger.info(f"Starting parallel scrape of {len(subreddits)} subreddits")

        tasks = [
            self.scrape_subreddit(sub, limit=limit)
            for sub in subreddits
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Subreddit {subreddits[i]} failed: {result}")
            else:
                all_posts.extend(result)

        logger.info(f"Total posts scraped: {len(all_posts)}")
        return all_posts

    async def close(self):
        await self.reddit.close()
