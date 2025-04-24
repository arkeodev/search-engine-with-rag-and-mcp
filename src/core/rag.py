"""RAG module for vector storage and retrieval."""

import os
from typing import Any, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Import the search module
from src.core import search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("rag")


async def create_rag(links: List[str]) -> FAISS:
    """Create a RAG system from a list of web links."""
    # Try to use Ollama embeddings if available
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    logger.info("Using Ollama embeddings")
    documents = []
    # Process URLs sequentially to avoid rate limits
    logger.info(f"Processing {len(links)} URLs sequentially to avoid rate limits")
    for i, url in enumerate(links):
        try:
            # Get content from URL with rate limiting already applied in the search module
            logger.info(f"Processing URL {i+1}/{len(links)}: {url}")
            result = await search.fetch_with_firecrawl(url)
            documents.extend(result)
            logger.info(f"Retrieved {len(result)} documents from {url}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            # Continue with other URLs even if one fails
            continue

    # Process only if we have documents
    if not documents:
        logger.warning("No documents retrieved from any URL")
        raise ValueError("No documents retrieved from any URL")

    # Text chunking processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    split_documents = text_splitter.split_documents(documents)
    logger.info(
        f"Created {len(split_documents)} document chunks from {len(documents)} documents"
    )

    # Create vector store
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    return vectorstore


async def search_rag(query: str, vectorstore: FAISS, k: int = 3) -> Any:
    """Search the RAG system for relevant documents."""
    try:
        documents = vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents from RAG")
        return documents
    except Exception as e:
        logger.error(f"Error in search_rag: {str(e)}")
        return []
