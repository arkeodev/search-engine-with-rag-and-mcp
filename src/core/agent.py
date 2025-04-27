"""Agent module for interacting with search and RAG capabilities."""

from typing import Any, Dict, List, Optional

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from src.core.llm_utils import LLMType, create_llm
from src.core.rag import search_rag
from src.core.search import search_web
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("agent")


def setup_tools(vectorstore: Any) -> List[Tool]:
    """Set up tools for the agent."""
    tools = [
        Tool(
            name="search_web",
            func=None,
            coroutine=lambda query: _get_formatted_results(query),
            description="Search the web for information on a given query.",
        ),
        Tool(
            name="rag_search",
            func=None,
            coroutine=lambda query, vectorstore=vectorstore: search_rag(
                query, vectorstore, llm_type=LLMType.OLLAMA
            ),
            description="Search the RAG system with advanced processing to find and synthesize relevant information.",
        ),
    ]
    return tools


async def _get_formatted_results(query: str) -> str:
    """Get only the formatted results from search_web."""
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

    # Choose the appropriate agent constructor depending on LLM capabilities
    if isinstance(llm, ChatOllama):
        # Load the official ReAct template
        react_prompt = hub.pull("hwchase17/react")

        # Prepend custom system instructions and force proper Action/Action Input syntax
        custom_instructions = """
You are a helpful AI assistant that can search the web and provide information.
Always cite your sources when providing information.
Think step-by-step about each request before taking action.

Use this format exactly:
Thought: your reasoning here
Action: one of [{tool_names}]
Action Input: the raw input for the tool (no quotes or parentheses)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
Thought: I now know the final answer
Final Answer: your answer to the question
  """

        # Combine custom instructions with the reAct hub template
        full_prompt_template = PromptTemplate.from_template(
            custom_instructions + "\n\n" + react_prompt.template
        )
        # Fill in tool descriptions and names
        prompt = full_prompt_template.partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            tool_names=", ".join([tool.name for tool in tools]),
        )
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    else:
        # For OpenAI models, use function calling format
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a helpful AI assistant that can search the web and provide information.
Always cite your sources when providing information.
Think step-by-step about each request before taking action.

You have access to the following tools:
{tools}"""
                ),
                HumanMessage(content="{input}"),
            ]
        ).partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        )
        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=chat_prompt)

    # Create the agent executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,  # Add reasonable limit to iterations
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

    # Pass only the fields expected by the prompt template
    input_data = {
        "input": query,
        "chat_history": chat_history,
        "agent_scratchpad": "",  # Initialize as empty string for text-based prompts
    }

    # Execute the agent asynchronously to avoid nested event loop errors
    result = await agent_executor.ainvoke(input_data)

    # Ensure we're returning a Dict[str, Any]
    if not isinstance(result, dict):
        return {"output": str(result), "intermediate_steps": []}

    return result
