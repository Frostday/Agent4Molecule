from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st
import ui
from client import MCPClient
import os
from mcp import StdioServerParameters
from google import genai
st.set_page_config(layout="wide", page_title="Streamlit Client for an MCP server")

st.title("Streamlit Client for an MCP server")


async def main():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # server_params = st.selectbox("Choose your MCP server", options=server_list.keys(), index=None)
    st.session_state.server_params = StdioServerParameters(
            command="python",
            args=["mcp_agent_v1/server.py"],
            env=os.environ.copy(),
        )
    # if server_params is not None:
    #     st.session_state.server_params = server_list[server_params]
    if not 'tools' in st.session_state:
        st.session_state.tools = []
    if "server_params" in st.session_state:
        async with MCPClient(
            st.session_state.server_params) as mcp_client:
            mcp_tools = await mcp_client.get_available_tools()

            st.session_state.tools = {
                tool.name: { "google_syntax": genai.types.Tool(
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
                ),
                "callable": mcp_client.call_tool(tool.name),

               
            }
             for tool in mcp_tools.tools
            }
       
            # st.session_state.tools = {
            #     tool.name: {
            #         "name": tool.name,
            #         "callable": mcp_client.call_tool(tool.name),
            #         "schema": {
            #             "type": "function",
            #             "function": {
            #                 "name": tool.name,
            #                 "description": tool.description,
            #                 "parameters": tool.inputSchema,
            #             },
            #         },
            #     }
            #     for tool in mcp_tools.tools
            # }
            # Available Tools
            if "tools" in st.session_state and st.session_state['tools'] is not None and len(
                    st.session_state['tools']) > 0:
                with st.sidebar:
                    st.subheader("Available Tools")
                    with st.expander("Tool List", expanded=False):
                        
                        for t in st.session_state.tools:
                            # fd = t.function_declarations[0]
                            # print(fd.parameters)
                            st.markdown(f"- *{t}*: {st.session_state.tools[t]}")

            await ui.ui()
    else:
        await ui.ui()


if __name__ == "__main__":
    asyncio.run(main())