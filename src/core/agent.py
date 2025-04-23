"""Agent module for interacting with search and RAG capabilities."""
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from src.core import rag, search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("agent")

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain.tools.render import format_tool_to_openai_function
    langchain_available = True
except ImportError:
    langchain_available = False
    logger.warning("LangChain not available. Agent capabilities will be limited.")

try:
    from langchain_ollama import ChatOllama
    ollama_available = True
except ImportError:
    ollama_available = False
    logger.warning("Ollama not available. Falling back to OpenAI if available.")

class SearchAgent:
    """Agent for searching and retrieving information."""
    
    def __init__(self) -> None:
        """Initialize the search agent."""
        self.tools = []
        self._setup_tools()
        self.agent_executor = self._create_agent_executor()
        
    def _setup_tools(self) -> None:
        """Set up the tools for the agent."""
        if not langchain_available:
            logger.warning("LangChain not available. Cannot set up tools.")
            return
            
        async def search_web(query: str) -> str:
            """
            Search the web for information on a given query.
            
            Args:
                query: The search query
                
            Returns:
                Formatted search results
            """
            logger.info(f"Agent searching web for: {query}")
            formatted_results, raw_results = await search.search_web(query)
            
            if not raw_results:
                return "No search results found."
                
            urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
            
            if not urls:
                return "No valid URLs found in search results."
                
            return formatted_results
        
        async def rag_search(query: str) -> str:
            """
            Search the web and use RAG to provide more relevant information.
            
            Args:
                query: The search query
                
            Returns:
                Relevant information from RAG processing
            """
            logger.info(f"Agent performing RAG search for: {query}")
            formatted_results, raw_results = await search.search_web(query)
            
            if not raw_results:
                return "No search results found."
                
            urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
            
            if not urls:
                return "No valid URLs found in search results."
                
            try:
                vectorstore = await rag.create_rag(urls)
                rag_results = await rag.search_rag(query, vectorstore)
                
                if not rag_results:
                    return "No relevant information found in RAG processing."
                    
                result = "\n\n".join([f"From {doc.metadata.get('source', 'unknown source')}:\n{doc.page_content}" 
                                    for doc in rag_results])
                return result
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                return f"RAG processing failed: {str(e)}"
                
        self.tools = [
            {
                "func": search_web,
                "name": "search_web",
                "description": "Search the web for information on a given query."
            },
            {
                "func": rag_search,
                "name": "rag_search",
                "description": "Search the web and use RAG to provide more relevant information."
            }
        ]
    
    def _create_agent_executor(self) -> Optional[Any]:
        """Create the agent executor with the appropriate LLM and tools."""
        if not langchain_available or not self.tools:
            logger.warning("Cannot create agent executor due to missing dependencies or tools.")
            return None
            
        # Define the system prompt
        system_prompt = """You are a helpful AI assistant that can search the web and provide information.
        
        To answer questions, you should use the tools available to you to gather information. 
        
        When using the search_web tool, you'll get a summary of search results.
        When using the rag_search tool, you'll get more detailed and relevant information extracted from web pages.
        
        Always cite your sources when providing information.
        """
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Initialize the LLM
        if ollama_available:
            try:
                # Try to use local Ollama first
                ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
                llm = ChatOllama(
                    model=ollama_model,
                    temperature=0.1,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
                logger.info(f"Using Ollama LLM: {ollama_model}")
            except Exception as e:
                logger.warning(f"Could not initialize Ollama: {e}. Falling back to OpenAI.")
                # Fall back to OpenAI if Ollama fails
                openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                llm = ChatOpenAI(
                    model=openai_model,
                    temperature=0.1,
                    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                    openai_api_base=os.getenv("OPENAI_API_BASE", None)
                )
                logger.info(f"Using OpenAI LLM: {openai_model}")
        else:
            # Use OpenAI
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            llm = ChatOpenAI(
                model=openai_model,
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_api_base=os.getenv("OPENAI_API_BASE", None)
            )
            logger.info(f"Using OpenAI LLM: {openai_model}")
            
        # Format tools for OpenAI function calling
        openai_tools = [format_tool_to_openai_function(t) for t in [
            tool["func"] for tool in self.tools
        ]]
        
        # Create the agent
        agent = create_openai_functions_agent(
            llm=llm,
            tools=[tool["func"] for tool in self.tools],
            prompt=prompt
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[tool["func"] for tool in self.tools],
            verbose=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
        
    async def query(self, query: str, chat_history: List[Any] = None) -> Dict[str, Any]:
        """
        Process a user query through the agent.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            Dictionary with response and intermediate steps
        """
        if not self.agent_executor:
            logger.warning("Agent executor not available. Falling back to direct search.")
            formatted_results, raw_results = await search.search_web(query)
            
            if raw_results:
                urls = [result.url for result in raw_results if hasattr(result, 'url') and result.url]
                
                if urls:
                    try:
                        vectorstore = await rag.create_rag(urls)
                        rag_results = await rag.search_rag(query, vectorstore)
                        
                        result = {
                            "output": f"{formatted_results}\n\n### RAG Results:\n\n" + 
                                     '\n---\n'.join(doc.page_content for doc in rag_results),
                            "intermediate_steps": []
                        }
                        return result
                    except Exception as e:
                        logger.error(f"Error in RAG processing: {e}")
                
            return {
                "output": formatted_results if raw_results else "No search results found.",
                "intermediate_steps": []
            }
        
        chat_history = chat_history or []
        
        # Sync wrapper for the async search tools
        def _sync_wrapper(async_func: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Any]:
            def wrapped(*args, **kwargs):
                return asyncio.run(async_func(*args, **kwargs))
            return wrapped
        
        # Create sync versions of the tools
        sync_tools = []
        for tool in self.tools:
            sync_func = _sync_wrapper(tool["func"])
            sync_func.__name__ = tool["name"]
            sync_func.__doc__ = tool["func"].__doc__
            sync_tools.append(sync_func)
        
        # Update the agent executor with sync tools
        self.agent_executor.tools = sync_tools
        
        # Run the agent
        result = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        return result

async def main() -> None:
    """Run the agent with command-line input."""
    agent = SearchAgent()
    
    # Get query from command line or input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter search query: ")
        
    print(f"Searching for: {query}")
    
    try:
        result = await agent.query(query)
        print("\n=== Agent Response ===")
        print(result["output"])
        
        # Print intermediate steps if available
        if "intermediate_steps" in result and result["intermediate_steps"]:
            print("\n=== Intermediate Steps ===")
            for step in result["intermediate_steps"]:
                print(f"- Tool: {step[0].tool}")
                print(f"- Input: {step[0].tool_input}")
                print(f"- Output: {step[1]}")
                print()
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 