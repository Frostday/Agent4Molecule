# streamlit_app/app.py


# streamlit_app/app.py
import streamlit as st
import asyncio
from mcp_agent.client import MCPClient

if "client" not in st.session_state:
    st.session_state.client = MCPClient()
    asyncio.run(st.session_state.client.connect())

st.title("MCP Chat")

user_input = st.text_input("Enter query")

if user_input:
    placeholder = st.empty()

    async def run_stream():
        async for event in st.session_state.client.process_query_stream(user_input):
            placeholder.markdown(event)

    asyncio.run(run_stream())





# import streamlit as st
# import asyncio
# import nest_asyncio
# from mcp_agent.client import MCPClient


# # Create a new event loop for this thread and set it
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

# # Patch asyncio so nested loops work
# nest_asyncio.apply()
# st.set_page_config(page_title="MCP Agent UI", layout="wide")
# st.title("🤖 MCP Agent Chat with Streaming")

# # Initialize MCP client once per session
# if "client" not in st.session_state:
#     st.session_state.client = MCPClient()
#     # Use the current loop to initialize
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(
#         st.session_state.client.connect_to_server("mcp_agent/server.py")
#     )

# user_input = st.chat_input("Type your message...")

# if user_input:
#     st.chat_message("user").write(user_input)

#     with st.chat_message("assistant"):
#         reasoning_box = st.empty()
#         final_box = st.empty()

#         reasoning_log = []
#         final_text = ""

       
#         async def run_stream():
#             async for event in st.session_state.client.process_query_stream(user_input):
#                 if event["type"] == "reasoning":
#                     reasoning_log.append(event["text"])
#                     reasoning_box.markdown("\n".join(reasoning_log))
#                 elif event["type"] == "model_text":
#                     final_text += event["text"]
#                     final_box.markdown(final_text)
#                 elif event["type"] == "final":
#                     final_box.markdown(f"**Final Answer:** {event['text']}")

#         # Run the streaming generator using the existing loop
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(run_stream())



# import streamlit as st
# from mcp_agent.client_interface import MCPFrontend

# st.set_page_config(page_title="MCP Agent UI", layout="wide")
# st.title("🤖 MCP Agent Chat")

# # Initialize client once
# if "mcp" not in st.session_state:
#     # Change path to your actual MCP server file
#     st.session_state.mcp = MCPFrontend(server_script_path="mcp_agent/server.py")
#     st.session_state.history = []

# # Chat input
# user_input = st.chat_input("Type your message...")

# if user_input:
#     st.session_state.history.append({"role": "user", "content": user_input})

#     with st.spinner("Thinking..."):
#         response = st.session_state.mcp.query(user_input)

#     st.session_state.history.append({"role": "assistant", "content": response})

# # Render history
# for msg in st.session_state.history:
    # if msg["role"] == "user":
    #     st.chat_message("user").write(msg["content"])
    # else:
    #     st.chat_message("assistant").write(msg["content"])
