"""Search module for web search capabilities."""

import asyncio
import os
import time
from typing import Any, List, Tuple, cast

import requests
from dotenv import load_dotenv
from exa_py import Exa
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.documents import Document

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
REQUEST_DELAY_SECONDS = 60.0 / RATE_LIMIT_REQUESTS_PER_MINUTE


class RateLimiter:
    """Rate limiter class to control the frequency of API requests."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter."""
        self.delay = 60.0 / requests_per_minute
        self.last_request_time = 0.0  # Use float for time values

    async def wait(self) -> None:
        """Wait as needed to comply with rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            logger.info(
                f"Rate limiting: waiting {wait_time:.2f} seconds before next request"
            )
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()


# Initialize rate limiter
firecrawl_rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS_PER_MINUTE)


async def search_web(query: str, num_results: int = 0) -> Tuple[str, list]:
    """Search the web using Exa API and return both formatted results and raw results."""
    try:
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
    except Exception as e:
        return f"An error occurred while searching with Exa: {e}", []


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


async def get_web_content(url: str) -> List[Document]:
    """Get web content and convert to document list."""
    # Apply rate limiting before each request
    await firecrawl_rate_limiter.wait()

    for attempt in range(MAX_RETRIES):
        try:
            # Create FireCrawlLoader instance compatible with version 1.7.0
            loader = FireCrawlLoader(
                url=url,
                mode="scrape",
                api_key=firecrawl_api_key,  # Explicitly pass API key for v1.7.0
            )

            # Use timeout protection
            documents = await asyncio.wait_for(
                loader.aload(), timeout=FIRECRAWL_TIMEOUT
            )

            # Return results if documents retrieved successfully
            if documents and len(documents) > 0:
                return cast(List[Document], documents)

            # Retry if no documents but no exception
            logger.info(
                f"No documents retrieved from {url} (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)  # Wait 1 second before retrying
                continue

        except requests.exceptions.HTTPError as e:
            if "Website Not Supported" in str(e):
                # Create a minimal document with error info
                logger.info(f"Website not supported by FireCrawl: {url}")
                content = f"Content from {url} could not be retrieved: Website not supported by FireCrawl."
                return [
                    Document(
                        page_content=content,
                        metadata={"source": url, "error": "Website not supported"},
                    )
                ]
            elif "Unrecognized key" in str(e) or "unrecognized_keys" in str(e):
                # Log API compatibility error
                logger.error(f"API compatibility error with FireCrawl: {e}")
                content = f"FireCrawl API compatibility error: {e}"
                return [
                    Document(
                        page_content=content,
                        metadata={"source": url, "error": "API compatibility error"},
                    )
                ]
            else:
                logger.info(
                    f"HTTP error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})"
                )

            if attempt < MAX_RETRIES - 1:
                # Wait before retry, apply longer backoff for rate limit errors
                if "429" in str(e) or "Rate limit" in str(e):
                    wait_time = min(30, 2**attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(1)
                continue

            raise

        except Exception as e:
            logger.info(
                f"Error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue

            raise

    # Return empty list if all retries failed
    return []
