"""Agent module for interacting with search and RAG capabilities."""

import asyncio
from typing import Any, Dict, List, Optional

from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from src.core.llm_utils import LLMType, create_llm
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("agent")


def setup_tools(vectorstore: Any) -> List[Tool]:
    """Set up tools for the agent."""
    # Import here to avoid circular imports
    from src.core.rag import search_rag
    from src.core.search import search_web

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
            func=lambda query: asyncio.run(
                search_rag(query, vectorstore, llm_type=LLMType.OLLAMA)
            ),
            coroutine=lambda query, vectorstore=vectorstore: search_rag(
                query, vectorstore, llm_type=LLMType.OLLAMA
            ),
            description="Search the RAG system with advanced processing to find and synthesize relevant information.",
        ),
    ]
    return tools


async def _get_formatted_results(query: str) -> str:
    """Get only the formatted results from search_web."""
    # Import here to avoid circular imports
    from src.core.search import search_web

    formatted_results, _ = await search_web(query)
    return formatted_results


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

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            SystemMessage(content="The tools you have access to are: {tool_names}"),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # Choose the appropriate agent constructor depending on LLM capabilities
    if isinstance(llm, ChatOllama):
        # ChatOllama does not support OpenAI-style function calling,
        # so fall back to the default ReAct agent prompt
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

    # Pass only the fields expected by the prompt template.
    input_data = {
        "input": query,
        "chat_history": chat_history,
        "agent_scratchpad": [],  # must be a list, not a string
    }

    # Execute the agent
    result = agent_executor.invoke(input_data)

    # Ensure we're returning a Dict[str, Any]
    if not isinstance(result, dict):
        return {"output": str(result), "intermediate_steps": []}

    return result
