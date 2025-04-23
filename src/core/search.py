"""Search module for web search capabilities."""

import asyncio
import os
from typing import Any, Tuple

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.documents import Document

from src.core.logger import logger

# Load .env variables
load_dotenv(override=True)

# Initialize the Exa client if available
try:
    from exa_py import Exa

    exa_api_key = os.getenv("EXA_API_KEY", "")
    exa = Exa(api_key=exa_api_key) if exa_api_key else None
except ImportError:
    exa = None
    print("Warning: exa-py not installed. Web search functionality will be limited.")

# Set FireCrawl API key if available
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
if firecrawl_api_key:
    os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

# Default search config
websearch_config = {"parameters": {"default_num_results": 5, "include_domains": []}}

# Constants for web content fetching
MAX_RETRIES = 3
FIRECRAWL_TIMEOUT = 30  # seconds


async def search_web(query: str, num_results: int = 0) -> Tuple[str, list]:
    """Search the web using Exa API and return both formatted and raw results."""
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
    """
    Format search results into a readable markdown string.

    Args:
        search_results: Search results from Exa API

    Returns:
        Formatted markdown string
    """
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


async def get_web_content(url: str) -> Any:
    """Get web content and convert to document list."""
    for attempt in range(MAX_RETRIES):
        try:
            # Create FireCrawlLoader instance
            loader = FireCrawlLoader(url=url, mode="scrape")

            # Use timeout protection
            documents = await asyncio.wait_for(
                loader.aload(), timeout=FIRECRAWL_TIMEOUT
            )

            # Return results if documents retrieved successfully
            if documents and len(documents) > 0:
                return documents

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
            else:
                logger.info(
                    f"HTTP error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})"
                )

            if attempt < MAX_RETRIES - 1:
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
