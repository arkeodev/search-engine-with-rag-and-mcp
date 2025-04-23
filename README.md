# Search Engine with RAG and MCP

A powerful search engine that combines LangChain, Model Context Protocol (MCP), Retrieval-Augmented Generation (RAG), and Ollama to create an agentic AI system capable of searching the web, retrieving information, and providing relevant answers.

## Features

- Web search capabilities using the Exa API
- Web content retrieval using FireCrawl
- RAG (Retrieval-Augmented Generation) for more relevant information extraction
- MCP (Model Context Protocol) server for standardized tool invocation
- Support for both local LLMs via Ollama and cloud-based LLMs via OpenAI
- Flexible architecture supporting direct search, agent-based search, or server mode
- Comprehensive error handling and graceful fallbacks
- Python 3.13+ with type hints
- Asynchronous processing for efficient web operations

## Architecture

This project integrates several key components:

1. **Search Module**: Uses Exa API to search the web and FireCrawl to retrieve content
2. **RAG Module**: Embeds documents, chunks them, and stores them in a FAISS vector store
3. **MCP Server**: Provides a standardized protocol for tool invocation
4. **Agent**: LangChain-based agent that uses the search and RAG capabilities

## Project Structure

```
search-engine-with-rag-and-mcp/
├── LICENSE              # MIT License
├── README.md            # Project documentation
├── data/                # Data directories
├── docs/                # Documentation
│   └── env_template.md  # Environment variables documentation
├── logs/                # Log files directory (auto-created)
├── src/                 # Main package (source code)
│   ├── __init__.py      
│   ├── core/            # Core functionality
│   │   ├── __init__.py
│   │   ├── main.py      # Main entry point
│   │   ├── search.py    # Web search module
│   │   ├── rag.py       # RAG implementation
│   │   ├── agent.py     # LangChain agent
│   │   └── mcp_server.py # MCP server implementation
│   └── utils/           # Utility modules
│       ├── __init__.py
│       ├── env.py       # Environment variable loading
│       └── logger.py    # Logging configuration
├── pyproject.toml       # Poetry configuration
├── requirements.txt     # Project dependencies
└── tests/               # Test directory
```

## Getting Started

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation) (optional, for development)
- API keys for Exa and FireCrawl
- (Optional) Ollama installed locally
- (Optional) OpenAI API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/search-engine-with-rag-and-mcp.git
cd search-engine-with-rag-and-mcp
```

2. Install dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using poetry
poetry install
```

3. Create a `.env` file (use docs/env_template.md as a reference)

### Usage

The application has three main modes of operation:

#### 1. Direct Search Mode (Default)

```bash
# Using pip
python -m src.core.main "your search query"

# Or using poetry
poetry run python -m src.core.main "your search query"
```

#### 2. Agent Mode

```bash
python -m src.core.main --agent "your search query"
```

#### 3. MCP Server Mode

```bash
python -m src.core.main --server
```

You can also specify custom host and port:

```bash
python -m src.core.main --server --host 0.0.0.0 --port 8080
```

### Using Ollama (Optional)

To use Ollama for local embeddings and LLM capabilities:

1. Install Ollama: https://ollama.ai/
2. Pull a model:
```bash
ollama pull mistral:latest
```
3. Set the appropriate environment variables in your `.env` file:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest
```

## Development

This project follows these best practices:

- **Code formatting**: Black and isort for consistent code style
- **Type checking**: mypy for static type checking
- **Linting**: flake8 for code quality
- **Testing**: pytest for unit and integration tests
- **Environment Management**: python-dotenv for managing environment variables
- **Logging**: Structured logging to both console and file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for the standardized tool invocation
- [Ollama](https://ollama.ai/) for local LLM capabilities
- [Exa](https://exa.ai/) for web search capabilities
- [FireCrawl](https://firecrawl.dev/) for web content retrieval 