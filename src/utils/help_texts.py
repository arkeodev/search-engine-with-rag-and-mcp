"""Help texts for CLI commands and options."""

# General help
APP_DESCRIPTION = """
Search Engine with RAG and MCP capabilities.

This application provides:
1. Web search functionality with RAG enhancement
2. MCP server for integration with LLMs
3. LangChain agent capabilities
"""

# Command help texts
COMMAND_HELP = {
    "server": "Run as an MCP server to provide search and RAG capabilities to LLMs",
    "agent": "Run the LangChain agent for search with reasoning capabilities",
    "search": "Run a direct search without using an agent",
}

# Option help texts
OPTION_HELP = {
    "env_file": "Path to a .env file containing env variables for configuration",
    "log_level": "Set the logging level for application output",
    "host": "Host address to bind the server to (default: localhost)",
    "port": "Port to bind the server to (default: 8000)",
    "query": "Search query to process",
}

# Detailed descriptions
DETAILED_HELP = {
    "server": """
    Start the Model Context Protocol (MCP) server, which exposes web search and
    RAG capabilities to Large Language Models like Claude.

    The server provides two main tools:
    - search_web_tool: Searches the web for a query and enhances results with RAG
    - get_web_content_tool: Fetches and processes content from a specific URL

    You can connect it from Claude Desktop or any other MCP-compatible client.
    """,
    "agent": """
    Start a LangChain agent that can search the web and apply reasoning to the results.

    The agent uses a structured reasoning approach:
    1. Analyzes your query
    2. Determines what information is needed
    3. Searches the web for relevant information
    4. Applies RAG to improve the information quality
    5. Reasons about the results to provide a comprehensive answer

    This mode is ideal for complex questions requiring both search and reasoning.
    """,
    "search": """
    Run a direct search without using an agent or server.

    This mode:
    1. Searches the web for your query
    2. Retrieves relevant results
    3. Applies RAG enhancement to improve the results
    4. Prints the results directly to the console

    This is the simplest and most direct way to use the search functionality.
    """,
    "env_file": """
    Specify a path to a .env file containing environment variables for configuration.

    The application looks for variables like:
    - API_KEYS: For search services
    - MCP_HOST: Host for MCP server
    - MCP_PORT: Port for MCP server
    - LOG_LEVEL: Default logging level
    - RAG_CONFIG: Configuration for RAG processing

    If not specified, the application look for a .env file in the current directory.
    """,
    "host": """
    Host address to bind the server to.

    Use:
    - 'localhost' or '127.0.0.1' to make the server accessible only on this machine
    - '0.0.0.0' to make the server accessible from other machines on the network

    Default: localhost
    """,
    "port": """
    Port number to bind the server to.

    Choose a port that is not already in use by another service.
    Common ports: 8000, 8080, 9000

    Default: 8000
    """,
}
