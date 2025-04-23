# Using the Search Engine with Claude Desktop

This guide explains how to set up your RAG-enhanced search engine with Claude Desktop for a powerful AI assistant experience.

## Setup Instructions

### 1. Configure Claude Desktop

To use this MCP server with Claude Desktop, you need to configure it in Claude's configuration file:

1. Open or create Claude Desktop's configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add your server to the `mcpServers` section of the config file:

```json
{
  "mcpServers": {
    "search_engine": {
      "command": "/bin/sh",
      "args": [
        "-c", 
        "cd /Users/kenanagyel/Projects/search-engine-with-rag-and-mcp && poetry run python -m src.core.main --server"
      ]
    }
  }
}
```

**Note:** Replace the `cwd` path with the absolute path to your project directory.

3. Save the file and restart Claude Desktop for the changes to take effect.

### 2. Using the Search Engine in Claude

Once configured, you can use the search engine directly in Claude Desktop:

1. Start a new chat with Claude
2. Look for the tools icon (hammer) in the interface
3. When you ask Claude about recent news or information, it can use your search engine to retrieve up-to-date information

## Example Prompts

Here are some example prompts to try:

- "What are the latest developments in quantum computing in 2025?"
- "Can you find information about the most recent climate change legislation?"
- "I need to learn about advancements in renewable energy this year. Can you search for that?"
- "Find information about the current state of AI regulation worldwide."

## What Happens Behind the Scenes

When you ask Claude a question requiring recent information:

1. Claude recognizes it needs external data and calls your search engine
2. Your MCP server searches the web for relevant information
3. The server applies RAG (Retrieval Augmented Generation) to enhance the results
4. Claude receives the enriched information and crafts a comprehensive response

This setup allows Claude to access current information beyond its training data cutoff.

### Alternative Configuration Using Shell Script

I've created a shell script to make this easier. Here's how to use it:

1. Make sure the script is executable (run `chmod +x run_mcp_server.sh` if needed)
2. Configure Claude Desktop to use the script:

```json
{
  "mcpServers": {
    "search_engine": {
      "command": "/Users/kenanagyel/Projects/search-engine-with-rag-and-mcp/run_mcp_server.sh",
      "args": []
    }
  }
}
```

This is often more reliable as it handles directory changes correctly.

### Simplified Python Direct Runner (Recommended)

I've created a direct Python runner script to avoid Poetry path issues. This is the most reliable option:

1. Ensure Python is installed and available in Claude Desktop's environment
2. Configure Claude Desktop to use the Python script directly:

```json
{
  "mcpServers": {
    "search_engine": {
      "command": "python3",
      "args": [
        "/Users/kenanagyel/Projects/search-engine-with-rag-and-mcp/run_server.py"
      ]
    }
  }
}
```

This bypasses Poetry's environment management and runs the server directly. 