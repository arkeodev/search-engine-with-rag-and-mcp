# Environment Variables Reference

Create a `.env` file in the project root with these environment variables to configure the application.

## API Keys and External Services

```
# Exa API for web search (required for search functionality)
EXA_API_KEY=your_exa_api_key_here

# FireCrawl API for web content crawling (required for RAG functionality)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# OpenAI API for embeddings and LLM (alternative to Ollama)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional: custom endpoint
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, etc.
```

## Ollama Configuration (Local LLM and Embeddings)

```
# Ollama configuration for local LLM and embeddings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest  # or any other model you have downloaded
```

## MCP Server Configuration

```
# MCP server configuration
MCP_HOST=localhost
MCP_PORT=8000
```

## Logging Configuration

```
# Logging level
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Example .env file

Here's an example of a complete `.env` file:

```
# API Keys
EXA_API_KEY=exa-abc123def456
FIRECRAWL_API_KEY=fca-xyz789abc123

# OpenAI
OPENAI_API_KEY=sk-abc123def456
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_MODEL=gpt-3.5-turbo

# Ollama 
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest

# MCP Server
MCP_HOST=localhost
MCP_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## Using the Environment Variables

The application will automatically load these variables when you run it. You can also specify a custom `.env` file path with the `--env-file` command line option. 