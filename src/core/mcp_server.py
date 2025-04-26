"""MCP server module for providing web search and RAG capabilities."""

import asyncio

from fastmcp import FastMCP

from src.core import rag, search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("mcp_server")


class MCPServer:
    """MCP server for web search and RAG capabilities."""

    def __init__(
        self,
        name: str = "web_search",
        version: str = "1.0.0",
        description: str = "Web search and RAG capabilities",
    ):
        """Initialize the MCP server."""
        self.mcp = FastMCP(name=name, version=version, description=description)
        self._register_tools()

    def _register_tools(self) -> None:
        """Register tools with the MCP server."""

        @self.mcp.tool()
        async def search_web_tool(query: str, max_urls: int = 3) -> str:
            """Search the web and process results using RAG."""
            logger.info(f"Searching web for query: {query}")

            # Get search results
            formatted_results, raw_results = await search.search_web(query)
            if not raw_results:
                return "No search results found."

            # Extract and validate URLs using list comprehension
            urls = [
                result.url for result in raw_results if getattr(result, "url", None)
            ]
            if not urls:
                return "No valid URLs found in search results."

            # Apply URL limit
            urls = urls[:max_urls]
            logger.info(f"Processing {len(urls)} URLs")

            try:
                # Process with RAG
                vectorstore = await rag.create_rag(urls)
                rag_result = await rag.search_rag(
                    query, vectorstore, return_source_documents=True
                )

                # Safely unpack result
                rag_response = (
                    rag_result[0] if isinstance(rag_result, tuple) else rag_result
                )
                rag_docs = rag_result[1] if isinstance(rag_result, tuple) else []

                # Build result string using f-strings
                result_parts = [formatted_results, "### RAG Results:", rag_response]

                if rag_docs:
                    result_parts.extend(
                        [
                            "### Sources:",
                            "\n---\n".join(doc.page_content for doc in rag_docs),
                        ]
                    )

                return "\n\n".join(result_parts)

            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                return f"{formatted_results}\n\nRAG processing failed: {str(e)}"

        @self.mcp.tool()
        async def get_web_content_tool(url: str) -> str:
            """Get the content of a webpage."""
            try:
                logger.info(f"Fetching content from URL with rate limiting: {url}")
                documents = await asyncio.wait_for(
                    search.fetch_with_firecrawl(url), timeout=15
                )

                if documents:
                    return "\n\n".join([doc.page_content for doc in documents])

                return "Unable to retrieve web content."
            except asyncio.TimeoutError:
                return "Timeout occurred while fetching web content. Please try again later."
            except Exception as e:
                return f"An error occurred while fetching web content: {str(e)}"

    def start(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the MCP server."""
        logger.info(f"Starting MCP server on {host}:{port}")
        self.mcp.run(host=host, port=port, transport="sse")
