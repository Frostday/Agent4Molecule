from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import os

class MCPClient:
    def __init__(self, server_path="mcp_agent/server.py"):
        self.server_path = server_path
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self):
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_path],
            env=os.environ.copy(),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

    async def close(self):
        await self.exit_stack.aclose()

    async def process_query_stream(self, query: str):
        """
        Example streaming interface:
        Just streams tool names matching the query.
        Replace with real tool execution later.
        """
        tools = await self.session.list_tools()
        for tool in tools.tools:
            if query.lower() in tool.name.lower():
                yield f"Found tool: {tool.name}"
        yield f"Query '{query}' finished."


# import asyncio
# from typing import Optional, AsyncGenerator
# from contextlib import AsyncExitStack
# import os
# import json

# from google import genai
# from google.genai.types import GenerateContentConfig, Content, Part
# from .agent_utils import content_to_dict
# from .prompts import SYSTEM_MESSAGE

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client


# class MCPClient:
#     def __init__(self):
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#         self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#         self.llm = "gemini-2.0-flash"

#     async def connect_to_server(self, server_script_path: str):
#         """Connect to MCP server via stdio"""
#         server_params = StdioServerParameters(
#             command="python",
#             args=[server_script_path],
#             env=os.environ.copy()
#         )
#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
#         await self.session.initialize()
#         response = await self.session.list_tools()
#         print(f"Connected to MCP server. Tools: {[t.name for t in response.tools]}")

#     async def process_query_stream(self, query: str) -> AsyncGenerator[dict, None]:
#         """Stream reasoning, intermediate tool calls, and final answer"""
#         messages = [
#             genai.types.Content( 
#                 role='user',
#                 parts=[
#                     genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
#                 ]
#             )
#         ]

#         # List tools
#         response = await self.session.list_tools()
#         available_tools = [
#             genai.types.Tool(
#                 function_declarations=[
#                     {
#                         "name": t.name,
#                         "description": t.description,
#                         "parameters": {
#                             k: v
#                             for k, v in t.inputSchema.items()
#                             if k not in ["additionalProperties", "$schema"]
#                         },
#                     }
#                 ]
#             )
#             for t in response.tools
#         ]

#         while True:
#             resp = self._client.models.generate_content(
#                 model=self.llm,
#                 contents=messages,
#                 config=GenerateContentConfig(tools=available_tools)
#             )
#             messages.append(resp.candidates[0].content)

#             # Stream reasoning and model text
#             for part in resp.candidates[0].content.parts:
#                 if part.text:
#                     yield {"type": "model_text", "text": part.text}
#                 if part.function_call:
#                     tool_name = part.function_call.name
#                     tool_args = dict(part.function_call.args)
#                     yield {"type": "reasoning", "text": f"Calling tool {tool_name} with args {tool_args}"}
#                     tool_result = await self.session.call_tool(tool_name, tool_args)
#                     messages.append(
#                         Content(
#                             role="tool",
#                             parts=[Part.from_function_response(name=tool_name, response={"result": str(tool_result.content[0].text)})]
#                         )
#                     )

#             # Stop if no tool calls
#             if all(part.function_call is None for part in resp.candidates[0].content.parts):
#                 break

#         # Yield final answer
#         final_text = " ".join([part.text for m in messages for part in m.parts if part.text])
#         yield {"type": "final", "text": final_text}

#     async def cleanup(self):
#         await self.exit_stack.aclose()







# import asyncio
# from typing import Optional, AsyncGenerator
# from contextlib import AsyncExitStack
# import os
# import json

# from google import genai
# from google.genai.types import GenerateContentConfig, Content, Part
# from .agent_utils import content_to_dict
# from .prompts import SYSTEM_MESSAGE

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client


# class MCPClient:
#     def __init__(self):
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#         self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#         self.llm = "gemini-2.0-flash"

#     async def connect_to_server(self, server_script_path: str):
#         """Connect to MCP server"""
#         is_python = server_script_path.endswith(".py")
#         if not is_python:
#             raise ValueError("Currently only Python servers supported")

#         server_params = StdioServerParameters(
#             command="python",
#             args=[server_script_path],
#             env=os.environ.copy()
#         )
#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

#         await self.session.initialize()
#         response = await self.session.list_tools()
#         print(f"MCP Connected. Tools: {[t.name for t in response.tools]}")

#     async def process_query_stream(self, query: str) -> AsyncGenerator[dict, None]:
#         print("start",query)
#         """Stream reasoning, model text, and final answer"""
#         messages = [
#             genai.types.Content( 
#                 role='user',
#                 parts=[
#                     genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
#                 ]
#             )
#         ]
#         # print("messages",messages)
#         # List tools
#         print('before')
#         response = await self.session.list_tools()
#         print("down here")
#         available_tools = [
#             genai.types.Tool(
#                 function_declarations=[
#                     {
#                         "name": t.name,
#                         "description": t.description,
#                         "parameters": {
#                             k: v
#                             for k, v in t.inputSchema.items()
#                             if k not in ["additionalProperties", "$schema"]
#                         },
#                     }
#                 ]
#             )
#             for t in response.tools
#         ]

#         while True:
#             print("in looop")
#             # Generate content from LLM
#             resp = self._client.models.generate_content(
#                 model=self.llm,
#                 contents=messages,
#                 config=GenerateContentConfig(tools=available_tools)
#             )
#             print(resp)
#             messages.append(resp.candidates[0].content)

#             # Yield reasoning or model text incrementally
#             for part in resp.candidates[0].content.parts:
#                 if part.text:
#                     yield {"type": "model_text", "text": part.text}
#                 if part.function_call:
#                     tool_name = part.function_call.name
#                     tool_args = dict(part.function_call.args)
#                     yield {"type": "reasoning", "text": f"Calling tool {tool_name} with args {tool_args}"}
#                     tool_result = await self.session.call_tool(tool_name, tool_args)
#                     messages.append(
#                         Content(
#                             role="tool",
#                             parts=[
#                                 Part.from_function_response(
#                                     name=tool_name,
#                                     response={"result": str(tool_result.content[0].text)}
#                                 )
#                             ]
#                         )
#                     )

#             # Stop if no function calls pending
#             if all(part.function_call is None for part in resp.candidates[0].content.parts):
#                 break

#         # Yield final answer
#         final_text = " ".join([part.text for m in messages for part in m.parts if part.text])
#         print(final_text)
#         yield {"type": "final", "text": final_text}

#     async def cleanup(self):
#         """Close MCP session"""
#         await self.exit_stack.aclose()






# import asyncio
# from typing import Optional
# from contextlib import AsyncExitStack
# import os
# import sys
# import json

# from google import genai
# from google.genai.types import GenerateContentConfig, Content, Part
# from .agent_utils import content_to_dict

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from .prompts import SYSTEM_MESSAGE

# class MCPClient:
#     def __init__(self):
#         self.session: Optional[ClientSession] = None
#         self.exit_stack = AsyncExitStack()
#         self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#         self.llm = "gemini-2.0-flash"

#     async def connect_to_server(self, server_script_path: str):
#         """Connect to an MCP server
#         Args:
#             server_script_path: Path to the server script (.py or .js)
#         """
#         is_python = server_script_path.endswith('.py')
#         is_js = server_script_path.endswith('.js')
#         if not (is_python or is_js):
#             raise ValueError("Server script must be a .py or .js file")

#         command = "python" if is_python else "node"
#         server_params = StdioServerParameters(
#             command=command,
#             args=[server_script_path],
#             env=os.environ.copy()
#         )

#         stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

#         self.stdio, self.write = stdio_transport
#         self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

#         try:
#             await self.session.initialize()
#         except Exception as e:
#             print("Initialization failed:", e)
#             raise

#         # List available tools
#         response = await self.session.list_tools()
#         for tool in response.tools:
#             print(f"\nTool Name: {tool.name}")
#             print(f"Description: {tool.description}")
        
#         tools = response.tools

#     async def process_query(self, query: str) -> str:
#         messages = [
#             genai.types.Content( 
#                 role='user',
#                 parts=[
#                     genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
#                 ]
#             )
#         ]

#         # Get available tools from MCP
#         response = await self.session.list_tools()
#         available_tools = [
#             genai.types.Tool(
#                 function_declarations=[
#                     {
#                         "name": tool.name,
#                         "description": tool.description,
#                         "parameters": {
#                             k: v
#                             for k, v in tool.inputSchema.items()
#                             if k not in ["additionalProperties", "$schema"]
#                         },
#                     }
#                 ]
#             )
#             for tool in response.tools
#         ]

#         while True:
#             response = self._client.models.generate_content(
#                 model=self.llm,
#                 contents=messages,
#                 config=GenerateContentConfig(
#                     tools=available_tools
#                 )
#             )
#             messages.append(response.candidates[0].content)

#             os.makedirs("outputs", exist_ok=True)
#             with open("outputs/chat_history_temp.json", "w") as f:
#                 json.dump([content_to_dict(m) for m in messages], f, indent=2)

#             flag = True
#             for part in response.candidates[0].content.parts:
#                 if part.function_call:
#                     tool_id = part.function_call.id
#                     tool_name = part.function_call.name
#                     tool_args = part.function_call.args
#                     print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
#                     tool_result = await self.session.call_tool(tool_name, tool_args)
#                     flag = False
#                     messages.append(
#                         genai.types.Content(
#                             role='tool',
#                             parts=[
#                                 genai.types.Part.from_function_response(
#                                     name=tool_name,
#                                     response={"result": str(tool_result.content[0].text)}
#                                 )
#                             ]
#                         )
#                     )
#             if flag:
#                 break
#             # print(messages)
        
#         os.makedirs("outputs", exist_ok=True)
#         with open("outputs/chat_history.json", "w") as f:
#             json.dump([content_to_dict(m) for m in messages], f, indent=2)
#         os.system("rm -f outputs/chat_history_temp.json")
        
#         return messages
    

#     # inside MCPClient

# async def process_query_stream(self, query: str):
#     messages = [
#         genai.types.Content(
#             role='user',
#             parts=[
#                 genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=query)),
#             ]
#         )
#     ]

#     response = await self.session.list_tools()
#     available_tools = [
#         genai.types.Tool(
#             function_declarations=[
#                 {
#                     "name": tool.name,
#                     "description": tool.description,
#                     "parameters": {
#                         k: v
#                         for k, v in tool.inputSchema.items()
#                         if k not in ["additionalProperties", "$schema"]
#                     },
#                 }
#             ]
#         )
#         for tool in response.tools
#     ]

#     while True:
#         response = self._client.models.generate_content(
#             model=self.llm,
#             contents=messages,
#             config=GenerateContentConfig(tools=available_tools)
#         )

#         # Yield partial model text (if available)
#         for part in response.candidates[0].content.parts:
#             if part.text:
#                 yield {"type": "model_text", "text": part.text}

#         messages.append(response.candidates[0].content)

#         flag = True
#         for part in response.candidates[0].content.parts:
#             if part.function_call:
#                 tool_name = part.function_call.name
#                 tool_args = part.function_call.args
#                 yield {"type": "reasoning", "text": f"🔧 Calling tool **{tool_name}** with args: {tool_args}"}

#                 tool_result = await self.session.call_tool(tool_name, tool_args)

#                 yield {"type": "reasoning", "text": f"📥 Tool result: {tool_result.content[0].text}"}

#                 flag = False
#                 messages.append(
#                     genai.types.Content(
#                         role='tool',
#                         parts=[
#                             genai.types.Part.from_function_response(
#                                 name=tool_name,
#                                 response={"result": str(tool_result.content[0].text)}
#                             )
#                         ]
#                     )
#                 )
#         if flag:
#             break

#         final_answer = messages[-1].parts[0].text if messages and messages[-1].parts else ""
#         yield {"type": "final", "text": final_answer}


#     async def chat_loop(self):
#         """Run an interactive chat loop"""
#         print("\nMCP Client Started!")
#         print("Type your queries or 'quit' to exit.")
        
#         while True:
#             try:
#                 query = input("\nQuery: ").strip()
#                 # query = "Design an enzyme using the given data: - enzyme family = \"4.6.1\" - Motif sequence = \"DIG\" - Coordinates for the motif sequence (X1, Y1, Z1, X2, Y2, Z2) = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0] - Indices of the the motif sequence = [0, 1, 4] - PDB file = \"5cxl.A\" - EC4 file = \"4.6.1.1\" - Substrate file = \"CHEBI_57540.sdf\" - Recommended length = 20"
                
#                 if query.lower() == 'quit':
#                     break
                
#                 message_history = await self.process_query(query)
#                 print(message_history)
                    
#             except Exception as e:
#                 print(f"\nError: {str(e)}")
    
#     async def cleanup(self):
#         """Clean up resources"""
#         await self.exit_stack.aclose()

# async def main():
#     if len(sys.argv) < 2:
#         print("Usage: python client.py <path_to_server_script>")
#         sys.exit(1)
        
#     client = MCPClient()
#     try:
#         await client.connect_to_server(sys.argv[1])
#         await client.chat_loop()
#     finally:
#         await client.cleanup()

# if __name__ == "__main__":
#     asyncio.run(main())