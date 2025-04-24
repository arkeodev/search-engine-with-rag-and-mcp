"""RAG module for vector storage and retrieval."""

from typing import Any, List, Optional

from boilerpy3 import extractors
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from readability import Document as ReadableDoc
from trafilatura import extract

# Import the search module
from src.core import search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("rag")


async def clean_html(content: str, metadata: dict) -> Optional[Document]:
    """Clean HTML content with multiple fallback methods for better extraction quality."""
    # Method 1: Trafilatura with optimized settings
    cleaned_text = extract(
        content, include_comments=False, favor_precision=False, include_tables=True
    )

    if not cleaned_text:
        logger.info("Trafilatura failed, trying BoilerPy3...")
        # Method 2: BoilerPy3
        try:
            cleaned_text = extractors.ArticleExtractor().get_content(content)
        except Exception:
            logger.warning("BoilerPy3 extraction failed!")

    if not cleaned_text:
        logger.info("BoilerPy3 failed, trying readability-lxml...")
        # Method 3: readability-lxml
        try:
            doc = ReadableDoc(content)
            cleaned_text = doc.summary(html_partial=False)
        except Exception:
            logger.warning("readability-lxml extraction failed!")

    # If we got content from any method
    if cleaned_text:
        logger.info(f"Content extracted successfully, length: {len(cleaned_text)}")
        return Document(page_content=cleaned_text.strip(), metadata=metadata)

    return None


async def create_rag(links: List[str]) -> FAISS:
    """Create a RAG system from a list of web links."""
    if not links:
        raise ValueError("No links provided to create RAG system")

    # Use a multilingual model that has good Turkish support
    logger.info("Using HuggingFace multilingual embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    documents = []
    logger.info(f"Processing {len(links)} URLs sequentially to avoid rate limits")

    for i, url in enumerate(links, 1):
        logger.info(f"Processing URL {i}/{len(links)}: {url}")

        # Get content from URL with rate limiting already applied in the search module
        raw_docs = await search.fetch_with_firecrawl(url)

        # Process retrieved documents
        url_documents = []
        for raw_doc in raw_docs:
            cleaned_doc = await clean_html(raw_doc.page_content, raw_doc.metadata)
            if cleaned_doc:
                url_documents.append(cleaned_doc)

        # Log results for this URL
        if url_documents:
            documents.extend(url_documents)
            logger.info(
                f"Retrieved and cleaned {len(url_documents)} documents from {url}"
            )
        else:
            logger.warning(f"No documents retrieved from {url}")

    # Check if we have any documents after processing all URLs
    if not documents:
        logger.warning("No documents retrieved from any URL")
        raise ValueError("No documents retrieved from any URL")

    # Text chunking with smaller chunks for more precise results
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    split_documents = text_splitter.split_documents(documents)
    logger.info(
        f"Created {len(split_documents)} document chunks from {len(documents)} documents"
    )

    # Create and return vector store
    return FAISS.from_documents(documents=split_documents, embedding=embeddings)


async def search_rag(query: str, vectorstore: FAISS, k: int = 10) -> Any:
    """Search the RAG system for relevant documents."""
    try:
        documents = vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(documents)} documents from RAG")

        # Take the top 3 most relevant documents after processing
        top_documents = documents[:3] if len(documents) > 3 else documents

        return top_documents
    except Exception as e:
        logger.error(f"Error in search_rag: {str(e)}")
        return []
