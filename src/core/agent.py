"""Agent module for interacting with search and RAG capabilities."""

import asyncio
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.core.rag import search_rag
from src.core.search import search_web
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("agent")


class LLMType(Enum):
    """Enum for LLM types."""

    OLLAMA = "ollama"
    OPENAI = "openai"


def setup_tools(vectorstore: Any) -> List[Tool]:
    """Set up tools for the agent."""
    tools = [
        Tool(
            name="search_web",
            func=lambda query: asyncio.run(search_web(query))[
                0
            ],  # Get formatted results
            coroutine=lambda query: _get_formatted_results(
                query
            ),  # Use helper function to extract first element
            description="Search the web for information on a given query.",
        ),
        Tool(
            name="rag_search",
            func=lambda query: asyncio.run(search_rag(query, vectorstore)),
            coroutine=search_rag,
            description="Search the RAG system for more relevant information.",
        ),
    ]
    return tools


async def _get_formatted_results(query: str) -> str:
    """Get only the formatted results from search_web."""
    formatted_results, _ = await search_web(query)
    return formatted_results


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


def create_agent_executor(
    llm_type: LLMType, vectorstore: Any
) -> Optional[AgentExecutor]:
    """Create an agent executor with the specified LLM type and tools."""
    tools = setup_tools(vectorstore)

    if not tools:
        logger.warning("Cannot create agent executor due to missing tools.")
        return None

    llm = create_llm(llm_type)
    if not llm:
        logger.error("Failed to initialize LLM.")
        return None

    # Define the system prompt
    system_prompt = """You are a helpful AI assistant. \
    You can search the web and provide information. \

    You have access to the following tools:
    {tools}

    When deciding which action to take, think step‑by‑step.
    If a tool is useful, call it with the required input.
    Otherwise, answer directly.

    Always cite your sources when providing information."""

    tool_system_message = SystemMessage(
        content="The tools you have access to are: {tool_names}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            tool_system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Choose the appropriate agent constructor depending on LLM capabilities
    if isinstance(llm, ChatOllama):
        # ChatOllama does not support OpenAI‑style function calling,
        # so fall back to a standard ReAct agent
        # First prepare the tool_names and tools variables for the prompt
        tool_names = ", ".join([tool.name for tool in tools])
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

        # Now create the agent with the prepared variables
        prompt = prompt.partial(tool_names=tool_names, tools=tool_strings)

        # Use the proper format for ReAct agents that expects agent_scratchpad as messages
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    else:
        # OpenAI chat models support function calling
        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    # Create the agent executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,  # Add this to handle parsing errors
    )

    return agent_executor


async def query_agent(
    query: str,
    vectorstore: Any,
    llm_type: LLMType = LLMType.OLLAMA,
    chat_history: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Process a user query through the agent."""
    chat_history = chat_history or []

    # Create the agent executor
    agent_executor = create_agent_executor(llm_type, vectorstore)
    if not agent_executor:
        return {"output": "Failed to initialize agent.", "intermediate_steps": []}

    # Input data with only the required fields
    input_data = {"input": query, "chat_history": chat_history}

    # Execute the agent
    result = agent_executor.invoke(input_data)

    # Ensure we're returning a Dict[str, Any]
    if not isinstance(result, dict):
        return {"output": str(result), "intermediate_steps": []}

    return result
