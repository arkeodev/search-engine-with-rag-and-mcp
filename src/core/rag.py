"""RAG module for vector storage and retrieval."""

import os
from typing import Any, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import the search module
from src.core import search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("rag")


async def create_rag(links: List[str]) -> FAISS:
    """
    Create a RAG system from a list of web links.

    Args:
        links: List of URLs to process

    Returns:
        FAISS: Vector store object
    """
    try:
        # Try to use Ollama embeddings if available
        try:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
            logger.info("Using Ollama embeddings")
        except (ImportError, Exception) as e:
            # Fall back to OpenAI embeddings if Ollama is not available
            logger.warning(f"Failed to initialize Ollama embeddings: {e}")
            try:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(
                    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_API_BASE"),
                    chunk_size=64,
                )
                logger.info("Using OpenAI embeddings")
            except (ImportError, Exception) as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise ValueError("No embedding model available")

        documents = []
        # Process URLs sequentially to avoid rate limits
        logger.info(f"Processing {len(links)} URLs sequentially to avoid rate limits")
        for i, url in enumerate(links):
            try:
                # Get content from URL with rate limiting already applied in the search module
                logger.info(f"Processing URL {i+1}/{len(links)}: {url}")
                result = await search.get_web_content(url)
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
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Error in create_rag: {str(e)}")
        raise


async def create_rag_from_documents(documents: List[Document]) -> FAISS:
    """Create a RAG system directly from a list of documents."""
    try:
        # Try to use Ollama embeddings if available
        try:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
            logger.info("Using Ollama embeddings")
        except (ImportError, Exception) as e:
            # Fall back to OpenAI embeddings if Ollama is not available
            logger.warning(f"Failed to initialize Ollama embeddings: {e}")
            try:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(
                    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("OPENAI_API_BASE"),
                    chunk_size=64,
                )
                logger.info("Using OpenAI embeddings")
            except (ImportError, Exception) as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise ValueError("No embedding model available")

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
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Error in create_rag_from_documents: {str(e)}")
        raise


async def search_rag(query: str, vectorstore: FAISS, k: int = 3) -> Any:
    """
    Search the RAG system for relevant documents.

    Args:
        query: Search query
        vectorstore: FAISS vector store
        k: Number of documents to retrieve

    Returns:
        List of Document objects
    """
    try:
        documents = vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents from RAG")
        return documents
    except Exception as e:
        logger.error(f"Error in search_rag: {str(e)}")
        return []
