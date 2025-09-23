from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st
import ui
from client import MCPClient
import os
from mcp import StdioServerParameters
from google import genai
import chat_history_utils
import py3Dmol
st.set_page_config(layout="wide", page_title="Chat page")

st.title("Agent4Molecule")


async def main():
    # for key in list(st.session_state.keys()):
    #     del st.session_state[key]
    # server_params = st.selectbox("Choose your MCP server", options=server_list.keys(), index=None)
            # ---- Sidebar: Conversation list ----
    st.sidebar.title("Chat History")

# Load chat index
    chat_index = chat_history_utils.load_index()  # list of dicts: {"id":..., "title":..., "last_updated":...}

# Convert to dict for easy lookup
    conv_titles = {conv["title"]: conv["id"] for conv in chat_index}

# Selectbox to choose conversation
    selected_title = st.sidebar.selectbox("Select a conversation", options=["-- New Chat --"] + list(conv_titles.keys()))

# ---- Main: Load conversation if selected ----
    if selected_title != "-- New Chat --":
        conv_id = conv_titles[selected_title]
        messages = chat_history_utils.load_conversation(conv_id)  # returns list of messages
    # Store in session_state for UI loop
        st.session_state["conv_id"] = conv_id
        st.session_state["messages"] = []
        for msg in messages:
        # Convert loaded dicts into genai.types.Content for your UI loop
            content = msg["content"]
            role = msg["role"]
            if isinstance(content, dict) and content.get("visualize"):
        # Render PDB
                with st.chat_message(role):
                    view = py3Dmol.view(width=800, height=600)
                    view.addModel(content["pdb_content"], "pdb")
                    view.setStyle({}, {"sphere": {"radius": 0.3, "color":"lightblue"}})
                    view.zoomTo()
                    st.components.v1.html(view._make_html(), height=600, width=800)
            else:
        # Normal text message
             with st.chat_message(role):
                st.markdown(str(content))
            #     st.session_state["messages"].append(
            #     genai.types.Content(
            #     role=msg["role"],
            #     parts=[genai.types.Part.from_text(text=str(content))]
            #     )
            # )
 
            
# st.session_state.messages.append(  genai.types.Content( 
#                 role='user',
#                 parts=[
#                     genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=prompt)),
#                 ]
#             ))
    else:
    # New chat: clear session_state
       if "conv_id" not in st.session_state:
            from datetime import datetime
            d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["conv_id"] = chat_history_utils.create_conversation(f"EnzyGen Input Build - {d}")
            st.session_state["messages"] = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conv_id" not in st.session_state:
        from datetime import datetime
        d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conv_id = chat_history_utils.create_conversation(
            f"EnzyGen Input Build - {d}"
        )
    st.session_state.server_params = StdioServerParameters(
            command="python",
            args=["mcp_agent_v1/enzygen_server.py"],
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
                                if k not in ["additional_properties", "$schema"]
                            },
                            
                        }
                    ]
                ),
                "callable": mcp_client.call_tool(tool.name),
                "description": tool.description,
                "parameters": {
                                k: v
                                for k, v in tool.inputSchema.items()
                                if k not in ["additional_properties", "$schema"]
                            },

               
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
            # if "tools" in st.session_state and st.session_state['tools'] is not None and len(
            #         st.session_state['tools']) > 0:
            #     with st.sidebar:
            #         st.subheader("Available Tools")
            #         with st.expander("Tool List", expanded=False):
                        
            #             for t in st.session_state.tools:
            #                 # fd = t.function_declarations[0]
            #                 # print(fd.parameters)
            #                 with st.expander(f"- *{t}*"):
            #                     st.markdown("{}".format(st.session_state.tools[t]["description"]))
            #                     st.markdown("Parameters: {}".format(st.session_state.tools[t]["parameters"]))


            await ui.ui()
    else:
        await ui.ui()


if __name__ == "__main__":
    asyncio.run(main())