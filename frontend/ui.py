# agent_ui.py
import json
import os
import re
from typing import List, Dict, Any
import py3Dmol
import streamlit as st
from dotenv import load_dotenv
from google import genai
import base64
from google.genai.types import GenerateContentConfig
from message_utils import render_message
from mcp_agent.prompts import SYSTEM_MESSAGE
import chat_history_utils
from datetime import datetime
from state_utils import ChatState
import ast, re

load_dotenv()

USER_DIR = "/ocean/projects/cis240137p/ksubram4/Agent4Molecule/user_history"


def convert_role(msg):
    role = msg.get("role")

    if role == "assistant":
        return "model"
    if role == "tool":
        return "model"            # tools are treated as model output
    if role == "system":
        return "user"             # inject system prompts as user messages
    return "user"  

def initialize_chat_dir():
    conv_id = f"conv_{datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S')}"
    st.session_state['chat_dir'] = USER_DIR + "/" + conv_id
    os.makedirs(USER_DIR + "/" + conv_id, exist_ok=True)
    output_dir = USER_DIR + "/" + conv_id + "/files"
    os.makedirs(output_dir, exist_ok=True)
    return conv_id, output_dir


async def ui():
    """Top-level Streamlit UI coroutine."""
    gpt_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    ChatHistory = chat_history_utils.ChatHistory(USER_DIR)

    # Initialize chat_state / conv_id / output_dir
    chat_state = st.session_state.get("chat_state")
    conv_id = st.session_state.get("conv_id")
    if chat_state is None:
        if conv_id is None:
            conv_id, output_dir = initialize_chat_dir()
            st.session_state["conv_id"] = conv_id
            st.session_state["output_dir"] = output_dir
        ChatHistory.init_history(st.session_state.conv_id)
        chat_state = ChatState(USER_DIR)
        st.session_state["chat_state"] = chat_state

    # Load messages if present (they are saved as structured content dicts)
    if "messages" not in st.session_state:
        conv_file = os.path.join(USER_DIR, st.session_state.conv_id, f"{st.session_state.conv_id}.json")
        if os.path.exists(conv_file):
            with open(conv_file, "r") as f:
                st.session_state.messages = json.load(f)
        else:
            st.session_state.messages = []

    # Render saved messages in the UI
    for msg in st.session_state["messages"]:
        # msg is expected to be a dict with 'role' and 'content' fields
        # render_message should be able to accept role and content (adapt if needed)
        print(msg)
        render_message(msg.get("role", "assistant"), msg.get("content"))

    # Build state_messages (genai Content objects) from stored messages
    # We will store messages in st.session_state as dicts. When constructing LLM input,
    # we convert them back to genai.types.Content objects using from_dict().
    state_messages: List[genai.types.Content] = []
    for saved in st.session_state["messages"]:
        try:
            # If saved["content"] is already a dict representing the Content object, reconstruct it.
            content_dict = saved.get("content")
            if isinstance(content_dict, dict):
                # genai.types.Content.from_dict may or may not exist depending on SDK version.
                # If not available, rebuild via the constructor.
                try:
                    content_obj = genai.types.Content.from_dict(content_dict)
                except Exception:
                    # Fallback: build parts manually
                    parts = []
                    for part in content_dict.get("parts", []):
                        # part may have 'text' or 'function_call' or 'function_response' fields
                        if "text" in part:
                            parts.append(genai.types.Part.from_text(text=part["text"]))
                        elif part.get("function_call"):
                            fc = part["function_call"]
                            # The SDK may provide a way to construct Part from function_call directly;
                            # otherwise include as a function_call field wrapper
                            parts.append(genai.types.Part.from_function_call(name=fc.get("name"), arguments=fc.get("arguments")))
                        elif part.get("function_response"):
                            fr = part["function_response"]
                            parts.append(genai.types.Part.from_function_response(name=fr.get("name"), response=fr.get("response")))
                        else:
                            # generic
                            parts.append(genai.types.Part.from_text(text=json.dumps(part)))
                    content_obj = genai.types.Content(role=saved.get("role", "assistant"), parts=parts)
                state_messages.append(content_obj)
            else:
                # content is a string: wrap it
                content_obj = genai.types.Content(role=saved.get("role", "assistant"),
                                                  parts=[genai.types.Part.from_text(text=str(content_dict))])
                state_messages.append(content_obj)
        except Exception as e:
            # If anything goes wrong, fall back to a simple text part
            content_obj = genai.types.Content(role=saved.get("role", "assistant"),
                                              parts=[genai.types.Part.from_text(text=str(saved.get("content")))])
            state_messages.append(content_obj)

    # UI input
    if prompt := st.chat_input("Ask your query here", accept_file="multiple"):
        # Construct the system + user content
        print("prompt.text",prompt.text)
        system_text = SYSTEM_MESSAGE.format(
            query=prompt.text,
            execution_history=st.session_state['chat_state'].getState(st.session_state.conv_id),
            output_dir=st.session_state.output_dir
        )
        user_content = genai.types.Content(
            role='user',
            parts=[genai.types.Part.from_text(text=system_text)]
        )
        state_messages.append(user_content)
        ChatHistory.save_message(st.session_state.conv_id, "user", prompt.text)
        user_dict = {"role": "user", "content": prompt.text}
        if "messages" in st.session_state:
            st.session_state["messages"].append(user_dict)
        else:
            st.session_state["messages"] = [user_dict]

        # Save user message to UI and history
        with st.chat_message("user"):
            st.markdown(prompt.text)
        current_task_id = st.session_state['chat_state'].create_new_task(st.session_state.conv_id, prompt.text)
        print("saving ", prompt.text)
       

        # Call agent loop
        with st.spinner("Thinking..."):
            response_text, messages_back = await agent_loop(st.session_state.tools, gpt_client, state_messages,
                                                            current_task_id, ChatHistory)
            st.session_state.messages = messages_back
            # with st.chat_message("assistant"):
            #     st.markdown(response_text)
            #ChatHistory.save_message(st.session_state.conv_id, "assistant", response_text)


async def agent_loop(tools: Dict[str, Dict[str, Any]], llm_client: genai.Client,
                     messages: List[genai.types.Content],
                     current_task_id: str, ChatHistory) :

    available_tools = [tools[t]['google_syntax'] for t in tools]
    # messages_for_llm = list(messages)  # copy
    messages_for_llm = []
    final_tool_result = ""
    tool_result = ""

    messages_for_llm = []
    for m in messages:
        if isinstance(m, genai.types.Content):
            messages_for_llm.append(m)
        elif isinstance(m, dict):
            try:
                messages_for_llm.append(genai.types.Content(**m))
            except Exception as e:
                print(f"Failed to convert message: {m}, error: {e}")
                messages_for_llm.append(genai.types.Content(
                    role="user",
                    parts=[genai.types.Part.from_text(text="Previous message (malformed)")]
                ))
        else:
            messages_for_llm.append(genai.types.Content(
                role="user",
                parts=[genai.types.Part.from_text(text="str(m)")]
            ))


    print("messages for llm")
    for m in messages_for_llm: print(m)
    while True:
        # converted_messages = []

        # for m in messages_for_llm:
        #     # Convert message to dict
        #     if hasattr(m, "to_dict"):
        #         d = m.to_dict()
        #     elif isinstance(m, dict):
        #         d = m
        #     else:
        #         d = {"role": "user", "parts": [{"text": str(m)}]}

        #     # STRICT ROLE MAPPING: only "user" or "model"
        #     role = d.get("role", "user")
        #     if role in ["assistant", "tool"]:
        #         role = "model"
        #     elif role not in ["user", "model"]:
        #         role = "user"

        #     parts = d.get("parts", [])
        #     if not parts:
        #         parts = [{"text": ""}]

        #     converted_messages.append({"role": role, "parts": parts})

        # Call the LLM
        try:
            first_response = llm_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=messages_for_llm,
                config=GenerateContentConfig(tools=available_tools)
            )
        except Exception as e:
            error_msg = f"LLM call failed: {e}"
            with st.chat_message("assistant"):
                st.markdown(f"Error: {error_msg}")
            return error_msg, [m.to_dict() if hasattr(m, "to_dict") else {"role": "user", "content": str(m)} for m in messages_for_llm]

        candidate = first_response.candidates[0]
        assistant_content: genai.types.Content = candidate.content
        print("first_response", first_response)
        print("==========")
        # Convert assistant content to serializable dict
        try:
            assistant_dict = assistant_content.to_dict()
        except Exception:
            assistant_dict = {
                "role": "model",
                "content": " ".join([part.text for part in assistant_content.parts if getattr(part, "text", None)])
            }

        # Append to messages_for_llm
        messages_for_llm.append(assistant_content)
        flag = True
        # Save to session state and chat history
        # st.session_state["messages"].append({"role": "model", "content": assistant_dict})
        # ChatHistory.save_message(st.session_state.conv_id, "assistant", assistant_dict)
        st.session_state["messages"].append(assistant_dict)  # ✅ Not {"role": "model", "content": assistant_dict}
        ChatHistory.save_message(st.session_state.conv_id, "assistant", assistant_dict)
        st.session_state['chat_state'].add_agent_response(st.session_state['conv_id'], current_task_id, assistant_dict)

        # Render UI
        with st.chat_message("assistant"):
            print("assistant content",assistant_content)
            texts = [p.text for p in assistant_content.parts if getattr(p, "text", None)]
            if texts:
                st.markdown("\n\n".join(texts))
            else:
                st.markdown("Processing...")


        print("oat agent")

        for part in assistant_content.parts:
            fc = getattr(part, 'function_call', None)
            if fc is not None:  # ✅ Proper check
       
                print('in tool loop')
                print("==========")
                # fc = part.function_call
                tool_name = fc.name
                tool_id = fc.id
                try:
                    tool_args = json.loads(fc.args) if isinstance(fc.args, str) else dict(fc.args)
                except Exception:
                    tool_args = {}

                with st.chat_message("assistant"):
                    tool_text = f"""Using tool:```{tool_name}({tool_args})```to answer this question."""
                    st.markdown(tool_text)
                    ChatHistory.save_message(st.session_state.conv_id, "assistant",tool_text)
                
                st.session_state['chat_state'].add_tool_call(st.session_state['conv_id'], current_task_id,tool_name,tool_args,tool_id)
                
                print("tool name", tool_name)
                print("tool args", tool_args)
                tool_result = await tools[tool_name]["callable"](**tool_args)
                print("tool_result",tool_result)
                tool_result = json.loads(tool_result)
                flag = False

                ChatHistory.save_message(st.session_state.conv_id, "tool", tool_result)
                st.session_state['chat_state'].add_tool_response(st.session_state['conv_id'],current_task_id,tool_name,tool_result)
                tool_response_content = genai.types.Content(
                            role='model',
                            parts=[
                                genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": str(tool_result)}
                                )
                            ]
                        )
                    
                messages_for_llm.append(tool_response_content)
                tool_response_dict = tool_response_dict = {
                                                                "role": "model",
                                                                "parts": [
                                                                    {
                                                                        "function_response": {
                                                                            "name": tool_name,
                                                                            "response": tool_result
                                                                        }
                                                                    }
                                                                ]
                                                            }
                st.session_state["messages"].append(tool_response_dict) 
        

            #     st.session_state["messages"].append({
            #         "role": "model",
            #         "content": tool_response_content.to_dict()
            #  })
                # st.session_state["messages"].append({
                #     "role": "model",
                #     "content": {"role": "model", "content": {"parts": [{"function_response": {"name": tool_name, "response": tool_result}}]}}
                # })
                render_message("tool", tool_result)
            
            else:
                tool_result = part.text

        
        if flag: 
            final_tool_result = tool_result
            break
    
    return final_tool_result, messages
        
        
        
            # # Record tool call
            # st.session_state['chat_state'].add_tool_call(st.session_state['conv_id'], current_task_id, tool_name,
            #                                              tool_args, getattr(fc, "id", None))
            # tool_callable = tools.get(tool_name, {}).get("callable")

            # if tool_callable is None:
            #     tool_result_obj = {"error": f"No tool named {tool_name} registered."}
            # else:
            #     try:
            #         maybe_coro = tool_callable(**tool_args)
            #         tool_result_raw = await maybe_coro if hasattr(maybe_coro, "__await__") else maybe_coro
            #         if isinstance(tool_result_raw, str):
            #             try:
            #                 tool_result_obj = json.loads(tool_result_raw)
            #             except:
            #                 tool_result_obj = {"result": tool_result_raw}
            #         elif isinstance(tool_result_raw, dict):
            #             tool_result_obj = tool_result_raw
            #         else:
            #             tool_result_obj = {"result": str(tool_result_raw)}
            #     except Exception as e:
            #         tool_result_obj = {"error": str(e)}

            # # Save tool output
            # st.session_state["messages"].append({
            #     "role": "model",
            #     "content": {"role": "model", "content": {"parts": [{"function_response": {"name": tool_name, "response": tool_result_obj}}]}}
            # })
            # ChatHistory.save_message(st.session_state.conv_id, "tool", tool_result_obj)
            # render_message("tool", tool_result_obj)




