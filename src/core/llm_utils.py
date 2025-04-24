"""Utilities for working with language models."""

import os
from enum import Enum
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("llm_utils")


class LLMType(Enum):
    """Enum for LLM types."""

    OLLAMA = "ollama"
    OPENAI = "openai"


def create_llm(llm_type: LLMType) -> Optional[BaseChatModel]:
    """Create the appropriate LLM based on the type."""
    if llm_type == LLMType.OLLAMA:
        # Use local Ollama
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
        llm = ChatOllama(
            model=ollama_model,
            temperature=0.1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        logger.info(f"Using Ollama LLM: {ollama_model}")
        return llm

    elif llm_type == LLMType.OPENAI:
        # Use OpenAI
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(
            model=openai_model,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_api_base=os.getenv("OPENAI_API_BASE", None),
        )
        logger.info(f"Using OpenAI LLM: {openai_model}")
        return llm

    else:
        logger.error(f"Unsupported LLM type: {llm_type}")
        return None


async def generate_response_from_docs(
    query: str, documents: list, llm_type: LLMType
) -> str:
    """Generate a response from documents using an LLM."""
    # Get the LLM
    llm = create_llm(llm_type)
    if not llm:
        logger.error(f"Failed to initialize LLM of type {llm_type}")
        # Fallback to just returning document text
        return "\n\n".join([doc.page_content for doc in documents])

    # Prepare the context from the retrieved documents
    context = "\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
    )

    # Create a system message with instructions
    system_message = SystemMessage(
        content="""You are a helpful assistant that provides accurate information based on the given context.

Your task is to answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information to answer fully, acknowledge this limitation.
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and factual, and cite specific parts of the context when relevant.
Format your answer to be easily readable."""
    )

    # Create a human message with the context and query
    human_message = HumanMessage(
        content=f"""CONTEXT:
{context}

QUESTION:
{query}

Please provide an accurate answer based only on the information in the context."""
    )

    # Create a simple chat model chain with the messages
    try:
        messages = [system_message, human_message]
        response = await llm.ainvoke(messages)
        logger.info("Generated response using LLM")
        return str(response.content)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"
