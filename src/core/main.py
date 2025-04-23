"""Main module for the search engine with RAG and MCP capabilities."""

import asyncio
import sys

import typer
from typing_extensions import Annotated, Optional

from src.core import rag, search
from src.core.agent import SearchAgent
from src.core.mcp_server import MCPServer
from src.utils.env import load_env
from src.utils.help_texts import APP_DESCRIPTION, DETAILED_HELP, OPTION_HELP
from src.utils.logger import LogLevel, get_logger

# Initialize logger
logger = get_logger("main")

# Create typer app with rich help text
app = typer.Typer(
    name="search-engine",
    help=APP_DESCRIPTION,
    rich_markup_mode="rich",
)


async def run_agent(
    query: Optional[str] = None, use_ollama: bool = True, use_openai: bool = True
) -> int:
    """Run the search agent."""
    agent = SearchAgent(use_ollama=use_ollama, use_openai=use_openai)

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


def run_server(host: str = "localhost", port: int = 8000) -> int:
    """Run the MCP server."""
    logger.info(f"Starting MCP server on {host}:{port}")
    server = MCPServer()
    server.start(host=host, port=port)

    # Since server.start is now blocking, we need to return a value
    # after the server is stopped (this code won't be reached during normal operation)
    return 0


async def run_direct_search(query: Optional[str] = None) -> int:
    """Run a direct search without using the agent."""
    if not query:
        query = input("Enter search query: ")

    logger.info(f"Running direct search with query: {query}")
    formatted_results, raw_results = await search.search_web(query)

    if not raw_results:
        print("No search results found.")
        return 0

    print("\n=== Search Results ===")
    print(formatted_results)

    urls = [
        result.url for result in raw_results if hasattr(result, "url") and result.url
    ]

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


async def run_async_main(
    server: bool = False,
    agent: bool = False,
    host: str = "localhost",
    port: int = 8000,
    query: Optional[str] = None,
) -> int:
    """Run the application based on the command-line arguments."""
    # Return a special code for server mode to avoid nested event loops
    if server:
        return -999  # Special code to indicate server should be run directly
    # Agent and direct search are still async
    elif agent:
        return await run_agent(query)
    else:
        return await run_direct_search(query)


@app.command("server", help=DETAILED_HELP["server"])
def server_command(
    env_file: Annotated[
        Optional[str],
        typer.Option(
            "--env-file", help=OPTION_HELP["env_file"], rich_help_panel="Configuration"
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help=OPTION_HELP["log_level"],
            rich_help_panel="Configuration",
        ),
    ] = LogLevel.INFO,
    host: Annotated[
        str,
        typer.Option(
            "--host", help=OPTION_HELP["host"], rich_help_panel="Server Options"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port", help=OPTION_HELP["port"], rich_help_panel="Server Options"
        ),
    ] = 8000,
) -> None:
    """Run as MCP server."""
    # Load environment variables
    try:
        load_env(env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    logger.info("Starting search engine in server mode")
    exit_code = run_server(host=host, port=port)
    sys.exit(exit_code)


@app.command("agent", help=DETAILED_HELP["agent"])
def agent_command(
    query: Annotated[Optional[str], typer.Argument(help=OPTION_HELP["query"])] = None,
    env_file: Annotated[
        Optional[str],
        typer.Option(
            "--env-file", help=OPTION_HELP["env_file"], rich_help_panel="Configuration"
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help=OPTION_HELP["log_level"],
            rich_help_panel="Configuration",
        ),
    ] = LogLevel.INFO,
    use_ollama: Annotated[
        bool,
        typer.Option(
            "--use-ollama",
            help="Whether to use Ollama for the agent",
            rich_help_panel="LLM Options",
        ),
    ] = True,
    use_openai: Annotated[
        bool,
        typer.Option(
            "--use-openai",
            help="Whether to use OpenAI as a fallback if Ollama is not available",
            rich_help_panel="LLM Options",
        ),
    ] = True,
) -> None:
    """Run with LangChain agent."""
    # Load environment variables
    try:
        load_env(env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    logger.info("Starting search engine with LangChain agent")
    exit_code = asyncio.run(
        run_agent(query, use_ollama=use_ollama, use_openai=use_openai)
    )
    sys.exit(exit_code)


@app.command("search", help=DETAILED_HELP["search"])
def search_command(
    query: Annotated[Optional[str], typer.Argument(help=OPTION_HELP["query"])] = None,
    env_file: Annotated[
        Optional[str],
        typer.Option(
            "--env-file", help=OPTION_HELP["env_file"], rich_help_panel="Configuration"
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help=OPTION_HELP["log_level"],
            rich_help_panel="Configuration",
        ),
    ] = LogLevel.INFO,
    use_ollama: Annotated[
        bool,
        typer.Option(
            "--use-ollama",
            help="Whether to use Ollama for search processing",
            rich_help_panel="LLM Options",
        ),
    ] = True,
    use_openai: Annotated[
        bool,
        typer.Option(
            "--use-openai",
            help="Whether to use OpenAI as a fallback if Ollama is not available",
            rich_help_panel="LLM Options",
        ),
    ] = True,
) -> None:
    """Run direct search."""
    # Load environment variables
    try:
        load_env(env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    logger.info("Starting search engine in direct search mode")
    exit_code = asyncio.run(run_direct_search(query))
    sys.exit(exit_code)


if __name__ == "__main__":
    app()
