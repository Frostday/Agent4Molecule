from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, List
import asyncio

class MCPClient:

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None
        self._client_context = None
        self._session_context = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Connect to the MCP server"""
        self._client_context = stdio_client(self.server_params)
        self.read, self.write = await self._client_context.__aenter__()
        self._session_context = ClientSession(self.read, self.write)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
                self._session_context = None
                self.session = None
        except Exception as e:
            print(f"Error closing session: {e}")
        
        try:
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
                self._client_context = None
        except Exception as e:
            print(f"Error closing client: {e}")

    async def get_available_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        return tools

    def call_tool(self, tool_name: str) -> Any:
        """Return a callable that invokes the tool"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            try:
                response = await self.session.call_tool(tool_name, arguments=kwargs)
                return response.content[0].text
            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                raise

        return callable