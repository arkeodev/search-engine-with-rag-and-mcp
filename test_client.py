"""Simple test client for MCP server."""

import asyncio

from fastmcp import Client


async def test_mcp_server() -> None:
    """Test the MCP server by calling some tools."""
    print("Connecting to MCP server...")
    async with Client("http://localhost:8000") as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call search_web_tool
        search_query = "latest advancements in AI 2025"
        print(f"\nSearching web for: {search_query}")
        result = await client.call_tool("search_web_tool", {"query": search_query})

        # Print the result
        if result and result.content:
            for content in result.content:
                if content.type == "text":
                    print("\nSearch Results:")
                    print("---------------")
                    print(content.text)

        # Call get_web_content_tool
        url = "https://example.com"
        print(f"\nGetting content from: {url}")
        result = await client.call_tool("get_web_content_tool", {"url": url})

        # Print the result
        if result and result.content:
            for content in result.content:
                if content.type == "text":
                    print("\nWeb Content:")
                    print("------------")
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
