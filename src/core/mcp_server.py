"""MCP server module for providing web search and RAG capabilities."""
import asyncio
import os
from typing import Any, Dict, Optional

from src.core import rag, search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("mcp_server")

try:
    from fastmcp import FastMCP
    mcp_available = True
except ImportError:
    mcp_available = False
    logger.warning("FastMCP not available. MCP server cannot be started.")

class MCPServer:
    """MCP server for web search and RAG capabilities."""
    
    def __init__(
        self, 
        name: str = "web_search", 
        version: str = "1.0.0", 
        description: str = "Web search and RAG capabilities"
    ):
        """Initialize the MCP server."""
        if not mcp_available:
            raise ImportError("FastMCP is not installed. Please install it with 'pip install fastmcp'.")
            
        self.mcp = FastMCP(
            name=name,
            version=version,
            description=description
        )
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self) -> None:
        """Register tools with the MCP server."""
        
        @self.mcp.tool()
        async def search_web_tool(query: str) -> str:
            """
            Search the web for the given query and return results.
            
            Args:
                query: Search query
                
            Returns:
                Formatted search results
            """
            logger.info(f"Searching web for query: {query}")
            formatted_results, raw_results = await search.search_web(query)
            
            if not raw_results:
                return "No search results found."
                
            urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
            
            if not urls:
                return "No valid URLs found in search results."
                
            try:
                vectorstore = await rag.create_rag(urls)
                rag_results = await rag.search_rag(query, vectorstore)
                
                # Include both search results and RAG results
                full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
                full_results += '\n---\n'.join(doc.page_content for doc in rag_results)
                
                return full_results
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                return f"{formatted_results}\n\nRAG processing failed: {str(e)}"
        
        @self.mcp.tool()
        async def get_web_content_tool(url: str) -> str:
            """
            Get the content of a webpage.
            
            Args:
                url: URL to fetch content from
                
            Returns:
                Webpage content
            """
            try:
                documents = await asyncio.wait_for(search.get_web_content(url), timeout=15)
                
                if documents:
                    return '\n\n'.join([doc.page_content for doc in documents])
                    
                return "Unable to retrieve web content."
                
            except asyncio.TimeoutError:
                return "Timeout occurred while fetching web content. Please try again later."
                
            except Exception as e:
                return f"An error occurred while fetching web content: {str(e)}"
    
    def start(self, host: str = "localhost", port: int = 8000) -> None:
        """
        Start the MCP server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        logger.info(f"Starting MCP server on {host}:{port}")
        self.mcp.run(host=host, port=port, transport='sse')
        
def get_tools() -> Dict[str, Any]:
    """
    Get the MCP tools for direct use without server.
    
    Returns:
        Dictionary of tools
    """
    if not mcp_available:
        logger.warning("FastMCP not available. Returning empty tools dictionary.")
        return {}
        
    tools: Dict[str, Any] = {}
    
    async def search_web_tool(query: str) -> str:
        """Search the web and return results."""
        logger.info(f"Searching web for query: {query}")
        formatted_results, raw_results = await search.search_web(query)
        
        if not raw_results:
            return "No search results found."
            
        urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
        
        if not urls:
            return "No valid URLs found in search results."
            
        try:
            vectorstore = await rag.create_rag(urls)
            rag_results = await rag.search_rag(query, vectorstore)
            
            # Include both search results and RAG results
            full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
            full_results += '\n---\n'.join(doc.page_content for doc in rag_results)
            
            return full_results
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            return f"{formatted_results}\n\nRAG processing failed: {str(e)}"
    
    async def get_web_content_tool(url: str) -> str:
        """Get webpage content."""
        try:
            documents = await asyncio.wait_for(search.get_web_content(url), timeout=15)
            
            if documents:
                return '\n\n'.join([doc.page_content for doc in documents])
                
            return "Unable to retrieve web content."
            
        except asyncio.TimeoutError:
            return "Timeout occurred while fetching web content. Please try again later."
            
        except Exception as e:
            return f"An error occurred while fetching web content: {str(e)}"
    
    # Add tools to the dictionary
    tools["search_web"] = search_web_tool
    tools["get_web_content"] = get_web_content_tool
    
    return tools

def main() -> None:
    """Run the MCP server."""
    host = os.getenv("MCP_HOST", "localhost")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    server = MCPServer()
    server.start(host=host, port=port)

if __name__ == "__main__":
    main() 