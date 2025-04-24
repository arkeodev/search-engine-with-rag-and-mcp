"""Search module for web search capabilities."""

import asyncio
import os
from typing import Any, List, Tuple, cast

import requests  # type: ignore
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from exa_py import Exa
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.documents import Document
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Import the logger
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("search")

# Load .env variables
load_dotenv(override=True)

# Set FireCrawl API key if available
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")

# Initialize the Exa client if available
exa_api_key = os.getenv("EXA_API_KEY", "")
exa = Exa(api_key=exa_api_key) if exa_api_key else None

# Default search config
websearch_config = {"parameters": {"default_num_results": 5, "include_domains": []}}

# Constants for web content fetching
MAX_RETRIES = 3
FIRECRAWL_TIMEOUT = 30  # seconds
# Rate limiting settings - FireCrawl free tier allows around 20-25 requests per minute
# Set a conservative limit to avoid rate limit errors
RATE_LIMIT_REQUESTS_PER_MINUTE = 10

# Initialize limiter: 10 requests per 60 seconds
firecrawl_limiter = AsyncLimiter(
    max_rate=RATE_LIMIT_REQUESTS_PER_MINUTE, time_period=60
)


async def search_web(query: str, num_results: int = 0) -> Tuple[str, list]:
    """Search the web using Exa API and return both formatted results and raw results."""
    if not exa:
        return "Exa API client not initialized. Check your API key.", []

    search_args = {
        "num_results": num_results
        or websearch_config["parameters"]["default_num_results"]
    }

    search_results = exa.search_and_contents(
        query, summary={"query": "Main points and key takeaways"}, **search_args
    )

    formatted_results = format_search_results(search_results)
    return formatted_results, search_results.results


def format_search_results(search_results: Any) -> str:
    """Format search results into a readable markdown string."""
    if not hasattr(search_results, "results") or not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"

    for idx, result in enumerate(search_results.results, 1):
        title = (
            result.title if hasattr(result, "title") and result.title else "Untitled"
        )
        url = result.url if hasattr(result, "url") else ""

        published_date = ""
        if hasattr(result, "published_date") and result.published_date:
            published_date = f" (Published: {result.published_date})"

        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"

        if hasattr(result, "summary") and result.summary:
            markdown_results += f">**Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"

    return markdown_results


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, Exception)),
)
async def fetch_with_firecrawl(url: str) -> List[Document]:
    """Fetch web content with FireCrawl, using tenacity for retries."""
    async with firecrawl_limiter:
        try:
            # Create FireCrawlLoader instance
            loader = FireCrawlLoader(
                url=url,
                mode="scrape",
                api_key=firecrawl_api_key,
            )

            # Load documents with timeout protection
            documents = await asyncio.wait_for(
                loader.aload(), timeout=FIRECRAWL_TIMEOUT
            )

            # Return results if documents retrieved successfully
            if documents:
                return cast(List[Document], documents)

            # No documents found
            logger.info(f"No documents retrieved from {url}")
            return []

        except requests.exceptions.HTTPError as e:  # type: ignore
            # Handle specific HTTP errors that should not be retried
            if "Website Not Supported" in str(e):
                logger.info(f"Website not supported by FireCrawl: {url}")
                return [
                    Document(
                        page_content=f"Content from {url} could not be retrieved: Website not supported by FireCrawl.",
                        metadata={"source": url, "error": "Website not supported"},
                    )
                ]
            elif "Unrecognized key" in str(e) or "unrecognized_keys" in str(e):
                logger.error(f"API compatibility error with FireCrawl: {e}")
                return [
                    Document(
                        page_content=f"FireCrawl API compatibility error: {e}",
                        metadata={"source": url, "error": "API compatibility error"},
                    )
                ]

            # For other HTTP errors, let tenacity handle the retry
            logger.info(f"HTTP error retrieving content from {url}: {str(e)}")
            raise
