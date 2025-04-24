"""RAG module for vector storage and retrieval."""

import os
import re
from typing import Any, List

from langchain.schema import Document
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
        # Get original documents
        documents = vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents from RAG")

        # Clean up the content of each document to remove HTML, website navigation and other cruft
        cleaned_docs = []
        for doc in documents:
            # Clean the content
            content = doc.page_content

            # Remove HTML tags
            content = re.sub(r"<[^>]+>", " ", content)

            # Remove URLs
            content = re.sub(r"https?://\S+", "", content)

            # Remove navigation text patterns
            content = re.sub(
                r"Kategori:|Görüş Bildir|Paylaş|Onedio Üyesi|İçeriğin Devamı Aşağıda|Rekl?am",
                "",
                content,
            )

            # Remove common website navigation elements
            content = re.sub(r"Yorumlar (ve Emojiler )?Aşağıda", "", content)
            content = re.sub(r"Tüm içerikleri", "", content)
            content = re.sub(r"\[iframe\].*?\[/iframe\]", "", content)
            content = re.sub(r"\[\.\.\.\]", "", content)

            # Remove social media buttons and icons
            content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

            # Remove multiple spaces, newlines and special characters
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"[\t\n\r]+", "\n", content)

            # Create a new Document with the cleaned content
            cleaned_doc = Document(page_content=content.strip(), metadata=doc.metadata)
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs
    except Exception as e:
        logger.error(f"Error in search_rag: {str(e)}")
        return []
