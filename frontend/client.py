from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, List

class MCPClient:

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        return tools

    def call_tool(self, tool_name: str) -> Any:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable