
from dotenv import load_dotenv
import ast
load_dotenv()
import molviewspec as mvs
import asyncio
import streamlit as st
import ui
from client import MCPClient
import os,json
from mcp import StdioServerParameters
from google import genai
from chat_history_utils import ChatHistory
from datetime import datetime
from io import StringIO
from message_utils import render_message
from state_utils import ChatState
st.set_page_config(layout="wide", page_title="Chat page")


USER_DIR="/ocean/projects/cis240137p/ksubram4/Agent4Molecule/user_history"

st.title("Agent4Molecule")


def clean_schema_for_gemini(schema):
    """Recursively remove fields that Gemini API doesn't accept"""
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            # Skip fields that Gemini doesn't accept
            if k in ["additional_properties", "additionalProperties", "$schema"]:
                continue
            cleaned[k] = clean_schema_for_gemini(v)
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema_for_gemini(item) for item in schema]
    else:
        return schema


def initialize_chat_dir():
    #parent directory
    conv_id = f"conv_{datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S')}"

    st.session_state['chat_dir'] = USER_DIR + "/" + conv_id
    print(" chat making dir ",USER_DIR + "/" + conv_id)
    os.makedirs(USER_DIR + "/" + conv_id, exist_ok = True)
    output_dir = USER_DIR + "/" + conv_id + "/files"
    os.makedirs(output_dir, exist_ok = True)
 

    return conv_id, output_dir


    

async def main():

    os.makedirs(USER_DIR, exist_ok=True)
  
    st.sidebar.title("Chat History")
    import chat_history_utils
    ChatHistory = chat_history_utils.ChatHistory(USER_DIR)

# Load chat index
    chat_index = ChatHistory.load_index()  # list of dicts: {"id":..., "title":..., "last_updated":...}

# Convert to dict for easy lookup
    conv_titles = {conv["title"]: conv["id"] for conv in chat_index}
  
# Selectbox to choose conversation
    selected_title = st.sidebar.selectbox(
        "Select a conversation", 
        options=["-- New Chat --"] + list(conv_titles.keys()),
        index=0
    )
    previous_title = st.session_state.get("selected_title")
    st.session_state["selected_title"] = selected_title
    selection_changed = selected_title != previous_title

    st.markdown("-------") 


    # render_visualization_panel()

  
# ---- Main: Load conversation if selected ----
# Check if selection changed and update session state accordingly

    if selected_title != "-- New Chat --":
        conv_id = conv_titles[selected_title]

    # Only reload if selection changed
        if selection_changed or st.session_state.get("conv_id") != conv_id:
            messages = ChatHistory.load_conversation(conv_id)
            state = ChatState(USER_DIR)
            st.session_state["chat_state"] = state
            st.session_state["conv_id"] = conv_id
            st.session_state["messages"] = messages


    # Delete button
        if st.sidebar.button("🗑️ Delete this chat", type="primary"):
            ChatHistory.delete_conversation(conv_id)
            st.sidebar.success(f"Deleted chat: {selected_title}")
            if st.session_state.get("conv_id") == conv_id:
                for k in ["conv_id", "messages", "chat_state"]:
                    st.session_state.pop(k, None)
            st.rerun()

    else:
    # Only initialize a new chat once when selection changes to "-- New Chat --"
        if selection_changed or "conv_id" not in st.session_state:
            st.session_state.conv_id, st.session_state.output_dir = initialize_chat_dir()
            ChatHistory.init_history(st.session_state.conv_id)
            st.session_state.messages = []

    st.session_state.server_params = StdioServerParameters(
            command="python",
            args=["mcp_agent/enzygen_server.py"],
            env=os.environ.copy(),
        )

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
                            "parameters": clean_schema_for_gemini(tool.inputSchema),
                            
                        }
                    ]
                ),
                "callable": mcp_client.call_tool(tool.name),
                "description": tool.description,
                "parameters": clean_schema_for_gemini(tool.inputSchema),

               
            }
             for tool in mcp_tools.tools
            }

            await ui.ui()
    else:
        await ui.ui()


if __name__ == "__main__":
    asyncio.run(main())
