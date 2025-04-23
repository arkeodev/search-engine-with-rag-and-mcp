"""Main module for the search engine with RAG and MCP capabilities."""
import argparse
import asyncio
import sys
from typing import Optional

from src.utils.env import load_env
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("main")

async def run_agent(query: Optional[str] = None) -> int:
    """
    Run the search agent.
    
    Args:
        query: Optional search query. If not provided, will prompt the user.
        
    Returns:
        Exit code
    """
    try:
        from src.core.agent import SearchAgent
        
        agent = SearchAgent()
        
        if not query:
            query = input("Enter search query: ")
            
        logger.info(f"Running agent with query: {query}")
        result = await agent.query(query)
        
        print("\n=== Agent Response ===")
        print(result["output"])
        
        # Print intermediate steps if available
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print("\n=== Intermediate Steps ===")
            for step in result["intermediate_steps"]:
                print(f"- Tool: {step[0].tool}")
                print(f"- Input: {step[0].tool_input}")
                print(f"- Output: {step[1]}")
                print()
                
        return 0
        
    except ImportError:
        logger.error("LangChain not available. Cannot run agent.")
        return 1
        
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return 1

def run_server(host: str = "localhost", port: int = 8000) -> int:
    """
    Run the MCP server.
    
    Args:
        host: Server host
        port: Server port
        
    Returns:
        Exit code
    """
    try:
        from src.core.mcp_server import MCPServer
        
        logger.info(f"Starting MCP server on {host}:{port}")
        server = MCPServer()
        server.start(host=host, port=port)
        
        # Since server.start is now blocking, we need to return a value
        # after the server is stopped (this code won't be reached during normal operation)
        return 0
        
    except ImportError:
        logger.error("FastMCP not available. Cannot run server.")
        return 1
        
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return 1

async def run_direct_search(query: Optional[str] = None) -> int:
    """
    Run a direct search without using the agent.
    
    Args:
        query: Optional search query. If not provided, will prompt the user.
        
    Returns:
        Exit code
    """
    try:
        from src.core import search, rag
        
        if not query:
            query = input("Enter search query: ")
            
        logger.info(f"Running direct search with query: {query}")
        formatted_results, raw_results = await search.search_web(query)
        
        if not raw_results:
            print("No search results found.")
            return 0
            
        print("\n=== Search Results ===")
        print(formatted_results)
        
        urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
        
        if not urls:
            print("No valid URLs found in search results.")
            return 0
            
        try:
            print("Processing RAG...")
            vectorstore = await rag.create_rag(urls)
            rag_results = await rag.search_rag(query, vectorstore)
            
            print("\n=== RAG Results ===")
            for doc in rag_results:
                print(f"\n---\n{doc.page_content}")
                
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            print(f"RAG processing failed: {e}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error in direct search: {e}")
        return 1

async def run_async_main(args: argparse.Namespace) -> int:
    """
    Run the application based on the command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code or special code -999 to indicate server should be run directly
    """
    # Return a special code for server mode to avoid nested event loops
    if args.server:
        return -999  # Special code to indicate server should be run directly
    # Agent and direct search are still async
    elif args.agent:
        return await run_agent(args.query)
    else:
        return await run_direct_search(args.query)

def main() -> int:
    """
    Run the main process.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="Search Engine with RAG and MCP")
    parser.add_argument(
        "--env-file", 
        help="Path to .env file",
        default=None
    )
    parser.add_argument(
        "--log-level", 
        help="Logging level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--server",
        help="Run as MCP server",
        action="store_true"
    )
    parser.add_argument(
        "--agent",
        help="Use LangChain agent",
        action="store_true"
    )
    parser.add_argument(
        "--host",
        help="Server host",
        default="localhost"
    )
    parser.add_argument(
        "--port",
        help="Server port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "query",
        help="Search query",
        nargs="*",
        default=None
    )
    
    args = parser.parse_args()
    
    # Convert query list to string
    if args.query:
        args.query = " ".join(args.query)
    else:
        args.query = None
    
    # Load environment variables
    try:
        load_env(args.env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return 1
    
    logger.info("Starting search engine")
    
    # Run the async part first
    result = asyncio.run(run_async_main(args))
    
    # Check for special return code
    if result == -999:
        # Run server directly without asyncio.run to avoid nested event loops
        return run_server(args.host, args.port)
    
    return result

if __name__ == "__main__":
    sys.exit(main()) 