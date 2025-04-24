"""Main module for the search engine with RAG and MCP capabilities."""

import asyncio
import sys

import typer
from typing_extensions import Annotated, Optional

from src.core import rag, search
from src.core.agent import LLMType, query_agent
from src.core.mcp_server import MCPServer
from src.utils.env import load_env
from src.utils.help_texts import APP_DESCRIPTION, DETAILED_HELP, OPTION_HELP
from src.utils.logger import LogLevel, configure_all_loggers, get_logger

# Initialize logger
logger = get_logger("main")

# Create typer app with rich help text
app = typer.Typer(
    name="search-engine",
    help=APP_DESCRIPTION,
    rich_markup_mode="rich",
)


async def run_agent(
    query: Optional[str] = None, llm_type: LLMType = LLMType.OLLAMA
) -> int:
    """Run the search agent."""
    if not query:
        query = input("Enter search query: ")

    logger.info(f"Running agent with query: {query} using {llm_type.value}")

    # First perform a search to generate the URLs for RAG
    formatted_results, raw_results = await search.search_web(query)

    if not raw_results:
        print("No search results found.")
        return 0

    urls = [
        result.url for result in raw_results if hasattr(result, "url") and result.url
    ]

    if not urls:
        print("No valid URLs found in search results.")
        return 0

    print("Building RAG knowledge base...")
    print(f"Processing {len(urls)} URLs sequentially with rate limiting...")

    # Create vectorstore from search results
    vectorstore = await rag.create_rag(urls)

    # Run the agent query with the prepared vectorstore
    result = await query_agent(query, vectorstore, llm_type)

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

    return 0


async def run_direct_search(
    query: Optional[str] = None, llm_type: LLMType = LLMType.OLLAMA
) -> int:
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

    print("Processing RAG...")
    print(
        f"Note: Processing {len(urls)} URLs sequentially with rate limiting to avoid API limits."
    )
    print("This may take some time. Please be patient...")

    # Create vectorstore from search results
    vectorstore = await rag.create_rag(urls)

    # Search RAG with LLM processing
    logger.info(f"Using {llm_type.value} to process RAG results")
    rag_response = await rag.search_rag(
        query, vectorstore, llm_type=llm_type, return_source_documents=False
    )

    # Unpack response - it's a tuple (answer, docs) when return_source_documents is True
    rag_answer = rag_response[0]
    rag_docs = rag_response[1]

    print("\n=== Source Documents ===")
    for i, doc in enumerate(rag_docs):
        print(f"\n--- Document {i+1} ---")
        print(
            doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content
        )
    print("\n=== AI-Generated Answer ===")
    print(rag_answer)

    return 0


async def run_async_main(
    server: bool = False,
    agent: bool = False,
    host: str = "localhost",
    port: int = 8000,
    query: Optional[str] = None,
    llm_type: LLMType = LLMType.OLLAMA,
) -> int:
    """Run the application based on the command-line arguments."""
    # Return a special code for server mode to avoid nested event loops
    if server:
        return -999  # Special code to indicate server should be run directly
    # Agent and direct search are still async
    elif agent:
        return await run_agent(query, llm_type)
    else:
        return await run_direct_search(query, llm_type)


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
    # Configure all loggers first
    configure_all_loggers(log_level)

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
    llm: Annotated[
        str,
        typer.Option(
            "--llm",
            help="LLM to use (ollama or openai)",
            rich_help_panel="LLM Options",
        ),
    ] = "ollama",
) -> None:
    """Run with LangChain agent."""
    # Configure all loggers first
    configure_all_loggers(log_level)

    # Load environment variables
    try:
        load_env(env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    # Determine LLM type
    try:
        llm_type = LLMType(llm.lower())
    except ValueError:
        logger.error(f"Invalid LLM type: {llm}. Using Ollama as default.")
        llm_type = LLMType.OLLAMA

    logger.info(f"Starting search engine with LangChain agent using {llm_type.value}")
    exit_code = asyncio.run(run_agent(query, llm_type=llm_type))
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
    llm: Annotated[
        str,
        typer.Option(
            "--llm",
            help="LLM to use for processing RAG results (ollama or openai)",
            rich_help_panel="LLM Options",
        ),
    ] = "ollama",
) -> None:
    """Run direct search."""
    # Configure all loggers first
    configure_all_loggers(log_level)

    # Load environment variables
    try:
        load_env(env_file)
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        sys.exit(1)

    # Determine LLM type
    try:
        llm_type = LLMType(llm.lower())
    except ValueError:
        logger.error(f"Invalid LLM type: {llm}. Using Ollama as default.")
        llm_type = LLMType.OLLAMA

    logger.info("Starting search engine in direct search mode")
    exit_code = asyncio.run(run_direct_search(query, llm_type))
    sys.exit(exit_code)


if __name__ == "__main__":
    # Let Typer handle everything
    app()
