from google import genai
from google.genai.types import GenerateContentConfig

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm_agent_name = "gemini"
        self.model_version = "gemini-1.5-flash"


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")


     
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        
        await self.session.initialize()

        
        # List available tools
        response = await self.session.list_tools()

        tools = response.tools
      

    async def process_query(self, query: str) -> str:
        """Process a query using external LLM and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        available_tools = [
                genai.types.Tool(
                    function_declarations=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additionalProperties", "$schema"]
                            },
                        }
                    ]
                )
                for tool in response.tools
            ]

        # response = self._client.models.generate_content(
        # model="gemini-2.0-flash", contents= messages[0]['content'],  
        # config= GenerateContentConfig(
        #     tools=available_tools

        # ),)
        response = self._client.models.generate_content(
        model="gemini-2.0-flash", contents=  genai.types.Content(
  role='user',
  parts=[genai.types.Part.from_text(text=query)]
),  
        config= GenerateContentConfig(
            tools=available_tools

        ),)
        


        # Process response and handle tool calls
        # final_text = []
        if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call

                result = await self.session.call_tool(
                    function_call.name, arguments=dict(function_call.args)
                )

                return result.content[0].text



    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())