import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import sys
import json

from google import genai
from google.genai.types import GenerateContentConfig, Content, Part

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompts import SYSTEM_MESSAGE

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = "gemini-2.0-flash"

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
            env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        try:
            await self.session.initialize()
        except Exception as e:
            print("Initialization failed:", e)
            raise

        # List available tools
        response = await self.session.list_tools()
        for tool in response.tools:
            print(f"\nTool Name: {tool.name}")
            print(f"Description: {tool.description}")
        
        tools = response.tools

    async def process_query(self, query: str) -> str:
        messages = [
            genai.types.Content( 
                role='user',
                parts=[
                    genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
                ]
            )
        ]

        # Get available tools from MCP
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

        final_response_parts = []

        while True:
            response = self._client.models.generate_content(
                model=self.llm,
                contents=messages,
                config=GenerateContentConfig(
                    tools=available_tools
                )
            )
           
            # Gather all parts of the response
            assistant_content = []
            tool_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "type") and part.type == "text":
                    assistant_content.append(part)
                    final_response_parts.append(part.text)
                else:
                    tool_calls.append(part)
                if not part.function_call:
                    break
            
            #  Add assistant message to history
            messages.append(
                {
                    'role': 'model', 
                    'parts' : [{
                        "functionCall": {
                            "name": part.function_call.name,
                            "args": part.function_call.args
                        }
                    }]
                }
            )
            
            if not tool_calls[0].function_call:
                break

            # Execute each tool call and append results
            ans = None
            function_responses = []
            for tool_call in tool_calls:
                tool_name = tool_call.function_call.name
                tool_args = tool_call.function_call.args
                tool_use_id = tool_call.function_call.id
                print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
                tool_result = await self.session.call_tool(tool_name, tool_args)
                messages.append(
                    {
                        'role': 'function',
                        'parts': [
                            {
                                'functionResponse': {
                                    'name': tool_name, 
                                    'response': {'output': tool_result.content[0].text}
                                }
                            }
                        ]
                    }
                )
                if tool_result: 
                    ans = tool_result.content[0].text
        return ans        
        return "\n".join(final_response_parts)

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
    asyncio.run(main())
