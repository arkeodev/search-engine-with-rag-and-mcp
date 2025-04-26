"""RAG module for vector storage and retrieval."""

from typing import Any, List, Optional, Tuple, Union

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_huggingface import HuggingFaceEmbeddings
from readability import Document as ReadableDoc
from trafilatura import extract

# Import the search module
from src.core import search
from src.core.llm_utils import LLMType, generate_response_from_docs
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
        logger.debug("Trafilatura failed, trying readability-lxml...")
        # Method 3: readability-lxml
        try:
            doc = ReadableDoc(content)
            cleaned_text = doc.summary(html_partial=False)
        except Exception as e:
            logger.debug(f"readability-lxml extraction failed with error: {e}")

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


async def search_rag(
    query: str,
    vectorstore: Any,
    llm_type: LLMType = LLMType.OLLAMA,
    num_documents: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
    return_source_documents: bool = False,
) -> Union[str, Tuple[str, List[Document]]]:
    """Search the vectorstore using RAG and generate a response."""
    # Get documents by using maximal marginal relevance to ensure diversity
    docs = await get_relevant_documents(
        query, vectorstore, num_documents, fetch_k, lambda_mult
    )

    if not docs:
        logger.warning("No relevant documents found for query: %s", query)
        if return_source_documents:
            return "No relevant information found.", []
        return "No relevant information found."

    # Log the documents that will be used for generation
    logger.info(f"Using {len(docs)} documents for generating response:")
    for i, doc in enumerate(docs):
        snippet = (
            doc.page_content[:150] + "..."
            if len(doc.page_content) > 150
            else doc.page_content
        )
        logger.info(f"Document {i+1}: {snippet}")

    # Generate a response using the selected LLM
    response = await generate_response_from_docs(query, docs, llm_type)

    if return_source_documents:
        return response, docs
    return response


async def get_relevant_documents(
    query: str,
    vectorstore: Any,
    num_documents: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
) -> List[Document]:
    """Retrieve relevant documents using maximal marginal relevance."""
    try:
        # Get document embeddings
        embeddings = vectorstore.embeddings
        if not embeddings:
            logger.error("No embeddings found in vectorstore")
            return []

        # Get the embedding for the query
        query_embedding = embeddings.embed_query(query)

        # Try direct similarity search first
        try:
            # First try to get more relevant documents with higher k
            docs_and_scores = vectorstore.similarity_search_with_score_by_vector(
                query_embedding, k=fetch_k
            )

            # Log the search scores to help with debugging
            for i, (doc, score) in enumerate(docs_and_scores[:5]):
                # Get a snippet of the document content
                snippet = (
                    doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                logger.info(f"Document {i+1} score: {score}, content: {snippet}")

            docs = [doc for doc, _ in docs_and_scores]
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Could not use similarity_search_with_score_by_vector, falling back to basic search: {str(e)}"
            )
            # Fallback to regular similarity search
            docs = vectorstore.similarity_search(query, k=fetch_k)

        # Apply maximal marginal relevance to get diverse results
        if len(docs) > num_documents:
            # Get the document embeddings - use embed_query instead of embed_document
            doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]

            # Convert embeddings to numpy arrays
            import numpy as np

            query_embedding_array = np.array(query_embedding)
            doc_embeddings_array = np.array(doc_embeddings)

            # Use MMR to select a diverse subset
            mmr_indices = maximal_marginal_relevance(
                query_embedding_array,
                doc_embeddings_array,
                k=num_documents,
                lambda_mult=lambda_mult,
            )

            # Select the documents using the MMR indices
            selected_docs = [docs[i] for i in mmr_indices]
            return selected_docs
        else:
            return docs

    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}")
        return []
