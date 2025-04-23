"""Agent module for interacting with search and RAG capabilities."""

import asyncio
import os
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
from langchain_openai import ChatOpenAI

from src.core import rag, search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("agent")


class SearchAgent:
    """Agent for searching and retrieving information."""

    def __init__(self, use_ollama: bool = True, use_openai: bool = True) -> None:
        """Initialize the search agent."""
        self.use_ollama = use_ollama
        self.use_openai = use_openai
        self.tools: list[Tool] = []
        self._setup_tools()
        self.agent_executor = self._create_agent_executor()

    async def search_web(self, query: str) -> str:
        """Search the web for information on a given query.

        Args:
            query (str): The search query

        Returns:
            str: Formatted search results
        """
        logger.info(f"Agent searching web for: {query}")
        formatted_results, raw_results = await search.search_web(query)

        if not raw_results:
            return "No search results found."

        urls = [
            result.url
            for result in raw_results
            if hasattr(result, "url") and result.url
        ]

        if not urls:
            return "No valid URLs found in search results."

        return formatted_results

    async def rag_search(self, query: str) -> str:
        """Search the web and use RAG to provide more relevant information.

        Args:
            query (str): The search query

        Returns:
            str: Relevant information from RAG processing
        """
        logger.info(f"Agent performing RAG search for: {query}")
        formatted_results, raw_results = await search.search_web(query)

        if not raw_results:
            return "No search results found."

        urls = [
            result.url
            for result in raw_results
            if hasattr(result, "url") and result.url
        ]

        if not urls:
            return "No valid URLs found in search results."

        try:
            vectorstore = await rag.create_rag(urls)
            rag_results = await rag.search_rag(query, vectorstore)

            if not rag_results:
                return "No relevant information found in RAG processing."

            result = "\n\n".join(
                [
                    f"From {doc.metadata.get('source', 'unknown source')}:\n{doc.page_content}"
                    for doc in rag_results
                ]
            )
            return result
        except Exception as e:
            logger.error(f"Error in RAG processing: {e}")
            return f"RAG processing failed: {str(e)}"

    async def query(
        self, query: str, chat_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Process a user query through the agent."""
        if not self.agent_executor:
            logger.warning(
                "Agent executor not available. Falling back to direct search."
            )
            formatted_results, raw_results = await search.search_web(query)

            if raw_results:
                urls = [
                    result.url
                    for result in raw_results
                    if hasattr(result, "url") and result.url
                ]

                if urls:
                    try:
                        vectorstore = await rag.create_rag(urls)
                        rag_results = await rag.search_rag(query, vectorstore)

                        result = {
                            "output": f"{formatted_results}\n\n### RAG Results:\n\n"
                            + "\n---\n".join(doc.page_content for doc in rag_results),
                            "intermediate_steps": [],
                        }
                        return result
                    except Exception as e:
                        logger.error(f"Error in RAG processing: {e}")

            return {
                "output": (
                    formatted_results if raw_results else "No search results found."
                ),
                "intermediate_steps": [],
            }

        chat_history = chat_history or []

        result = self.agent_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )

        return result

    def _setup_tools(self) -> None:
        # Wrap the search helpers as LangChain Tool objects
        self.tools = [
            Tool(
                name="search_web",
                func=lambda query: asyncio.run(self.search_web(query)),
                coroutine=self.search_web,
                description="Search the web for information on a given query.",
            ),
            Tool(
                name="rag_search",
                func=lambda query: asyncio.run(self.rag_search(query)),
                coroutine=self.rag_search,
                description="Search the web and use RAG for more relevant information.",
            ),
        ]

    def _create_agent_executor(self) -> Optional[Any]:
        """Create the agent executor with the appropriate LLM and tools."""
        if not self.tools:
            logger.warning(
                "Cannot create agent executor due to missing dependencies or tools."
            )
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

        # Initialize the LLM
        if self.use_ollama:
            try:
                # Try to use local Ollama first
                ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
                llm = ChatOllama(
                    model=ollama_model,
                    temperature=0.1,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                )
                logger.info(f"Using Ollama LLM: {ollama_model}")
            except Exception as e:
                logger.warning(
                    f"Could not initialize Ollama: {e}. Falling back to OpenAI."
                )
                # Fall back to OpenAI if Ollama fails and OpenAI is enabled
                if self.use_openai:
                    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                    llm = ChatOpenAI(
                        model=openai_model,
                        temperature=0.1,
                        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                        openai_api_base=os.getenv("OPENAI_API_BASE", None),
                    )
                    logger.info(f"Using OpenAI LLM: {openai_model}")
                else:
                    logger.error(
                        "Ollama failed and OpenAI is disabled. Cannot initialize LLM."
                    )
                    return None
        elif self.use_openai:
            # Use OpenAI
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            llm = ChatOpenAI(
                model=openai_model,
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_api_base=os.getenv("OPENAI_API_BASE", None),
            )
            logger.info(f"Using OpenAI LLM: {openai_model}")
        else:
            logger.error("Both Ollama and OpenAI are disabled. Cannot initialize LLM.")
            return None

        # Choose the appropriate agent constructor depending on LLM capabilities
        if isinstance(llm, ChatOllama):
            # ChatOllama does not support OpenAI‑style function calling,
            # so fall back to a standard ReAct agent
            agent = create_react_agent(llm=llm, tools=self.tools, prompt=prompt)
        else:
            # OpenAI chat models support function calling
            agent = create_openai_functions_agent(
                llm=llm, tools=self.tools, prompt=prompt
            )

        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True
        )

        return agent_executor
