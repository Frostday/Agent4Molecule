import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import os
import sys
import json

from google import genai
from google.genai.types import GenerateContentConfig, Content, Part
from agent_utils import content_to_dict

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

        while True:
            response = self._client.models.generate_content(
                model=self.llm,
                contents=messages,
                config=GenerateContentConfig(
                    tools=available_tools
                )
            )
            messages.append(response.candidates[0].content)

            os.makedirs("outputs", exist_ok=True)
            with open("outputs/chat_history_temp.json", "w") as f:
                json.dump([content_to_dict(m) for m in messages], f, indent=2)

            flag = True
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_id = part.function_call.id
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args
                    print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    flag = False
                    messages.append(
                        genai.types.Content(
                            role='tool',
                            parts=[
                                genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": str(tool_result.content[0].text)}
                                )
                            ]
                        )
                    )
            if flag:
                break
            # print(messages)
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/chat_history.json", "w") as f:
            json.dump([content_to_dict(m) for m in messages], f, indent=2)
        # os.system("rm -f outputs/chat_history_temp.json")
        
        return messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        # query = input("\nQuery: ").strip()
        # query = "Design an enzyme that functions as an adenylate-processing protein, acting like a cyclase to transform ATP into 3’,5’-cyclic AMP while releasing pyrophosphate. The enzyme should resemble known adenylylcyclases in structure and activity, and be capable of catalyzing the formation of cyclic AMP as a signaling molecule."
        # query = "Design a heme binding protein using the given data: \\n- Input PDB with protein and ligand: \"/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/input/7o2g_HBA.pdb\"\\n- Ligand name: \"HBA\"\\n- Parameters file: \"/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA.params\"\\n- CST file: \"/ocean/projects/cis240137p/dgarg2/github/heme_binder_diffusion/theozyme/HBA/HBA_CYS_UPO.cst\"\\n- ligand atoms that should be excluded from clashchecking because they are flexible: \"O1 O2 O3 O4 C5 C10\"\\n- ligand atoms that need to be more exposed and the required SASA for those atoms: \"C45 C46 C47\" and SASA should be 10.0\\n- amino acids should be excluded from consideration when generating protein sequences: \"CM\"\\n- ligand atom used for aligning the rotamers: \"N1\", \"N2\", \"N3\", \"N4\"\\nHere are some properties you should try to obtain:\\n- SASA <= 0.3\\n- RMSD <= 5\\n- LDDT >= 80\\n- Terminal residue limit < 15\\n- Radius of gyration limit for protein compactness <= 30\\n- all_cst <= 1.5\\n- CMS per atom >= 3.0\\n- CYS atom is A15".replace("\\n", "\n")
        query = "Use ClustalW2 to mind enzyme motifs. Enzyme fasta and reference chain protein are provided as default parameters. Use the default parameters for all other options if not provided."

        message_history = await self.process_query(query)
        # print(message_history)
    
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