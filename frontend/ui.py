
import json
import os
from typing import List
import py3Dmol

import streamlit as st
from dotenv import load_dotenv
# from openai import AsyncOpenAI
# from openai.lib.azure import AsyncAzureOpenAI
from google import genai
import base64
from google.genai.types import GenerateContentConfig

from mcp_agent_v1.prompts import SYSTEM_MESSAGE
import chat_history_utils
from datetime import datetime
load_dotenv()




async def ui():
    gpt_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # if "conv_id" not in st.session_state:
    #     d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # st.session_state.conv_id = chat_history_utils.create_conversation(
    #     f"EnzyGen Input Build - {d}"
    # )

    

    # conv_id = st.session_state.conv_id
    # conv_id = st.session_state.get("conv_id")
    # if conv_id is None:
    #     d = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     conv_id = chat_history_utils.create_conversation(
    #     f"EnzyGen Input Build - {d}"
    # )
    #     st.session_state.conv_id = conv_id
    #     st.session_state.messages = []  # 
    #     print("creating new conv_id",st.session_state.conv_id )

   
    # print("conv id", st.session_state.conv_id)


    if "messages" not in st.session_state:
        conv_file = os.path.join(chat_history_utils.HISTORY_DIR, f"{st.session_state.conv_id}.json")
        if os.path.exists(conv_file):
            with open(conv_file, "r") as f:
                st.session_state.messages = json.load(f)
        else:
            st.session_state.messages = []

    for message in st.session_state.messages:
        # skip tool calls
 
        continue_flag = False
        for part in message.parts:
            if part.function_call: 
                continue_flag = True
                break
        # if message["role"] == "tool" or 'tool_calls' in message:
        if continue_flag: continue
        # else:
        # with st.chat_message(message["role"]):
        final_text = " ".join([part.text for part in message.parts if part.text])
        with st.chat_message(message.role):
            st.markdown(final_text)

    if prompt := st.chat_input("Ask your query here"):
        st.session_state.messages.append(  genai.types.Content( 
                role='user',
                parts=[
                    genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=prompt)),
                ]
            ))
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history_utils.save_message(st.session_state.conv_id, "user", prompt)
        with st.spinner("Thinking..."):
     
            response, messages = await agent_loop(st.session_state.tools, gpt_client, st.session_state.messages)
            st.session_state.messages = messages
            with st.chat_message("assistant"):
                st.markdown(response)
                chat_history_utils.save_message(st.session_state.conv_id, "assistant",response)


async def agent_loop(tools: dict, llm_client, messages):


   
    available_tools = [tools[t]['google_syntax'] for t in tools]

    final_tool_result = ""
    tool_result = ""
    while True:
        first_response = llm_client.models.generate_content(
            model="gemini-2.0-flash", contents= messages,  
        config= GenerateContentConfig(
                tools= available_tools
            ),)
    
        # print(first_response.candidates[0].content)
        messages.append(first_response.candidates[0].content)
        flag = True
        for part in first_response.candidates[0].content.parts:
            if part.function_call:
           
                tool_id = part.function_call.id
                tool_name = part.function_call.name
                tool_args = part.function_call.args
                # print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
                tool_result = await tools[tool_name]["callable"](**tool_args)
                tool_result = json.loads(tool_result)
                # print("tool_result",tool_result)
        
                flag = False
                messages.append(
                        genai.types.Content(
                            role='tool',
                            parts=[
                                genai.types.Part.from_function_response(
                                    name=tool_name,
                                    response={"result": str(tool_result)}
                                )
                            ]
                        )
                    )
            

                with st.chat_message("assistant"):
                    tool_text = f"""Using tool:```{tool_name}({tool_args})```to answer this question."""
                    st.markdown(tool_text)
                    chat_history_utils.save_message(st.session_state.conv_id, "assistant",tool_text)


                # print('new', type(tool_result))
                with st.chat_message("assistant"):
                    if tool_result["visualize"] == True:
                            chat_history_utils.save_message(
                            st.session_state.conv_id, 
                            "tool", 
                            {"visualize": True, "pdb_content": tool_result["pdb_content"]}
                             )
                            view = py3Dmol.view(width=800, height=600)
                            view.addModel(tool_result["pdb_content"], "pdb")
                            view.setStyle({}, {"sphere": {"radius": 0.3, "color":"lightblue"}})
                            view.zoomTo()

#                           # Render directly without stmol
                            st.components.v1.html(view._make_html(), height=600, width=800)
                        
                    else:
                     st.markdown("Tool response: {}".format(tool_result))
                     chat_history_utils.save_message(st.session_state.conv_id, "tool", tool_result)

            else:
                tool_result = part.text    
    
        if flag: 
            final_tool_result = tool_result
            break
    
    return final_tool_result, messages
    
    # print(first_response.candidates[0].content)

    # stop_reason = (
    #     "tool_calls"
    #     if first_response.candidates[0].content.function_call is not None
    #     else first_response.candidates[0].finish_reason
    # )

    # # response.candidates[0].function_call
    # if stop_reason == "tool_calls":
    #     for tool_call in first_response.choices[0].message.tool_calls:
    #         arguments = (
    #             json.loads(tool_call.function.arguments)
    #             if isinstance(tool_call.function.arguments, str)
    #             else tool_call.function.arguments
    #         )
    #         tool_result = await tools[tool_call.function.name]["callable"](**arguments)
    #         messages.append(
    #             first_response.choices[0].message.to_dict()
    #         )
    #         messages.append(
    #             {
    #                 "role": "tool",
    #                 "tool_call_id": tool_call.id,
    #                 "nsame": tool_call.function.name,
    #                 "content": json.dumps(tool_result),
    #             }
    #         )

    #         with st.chat_message("assistant"):
    #             st.markdown(f"""Using tool:```{tool_call.function.name}({arguments})```to answer this question.""")

    #         # with st.chat_message("assistant"):
    #         #     st.markdown(f"Tool response: {tool_result}")

    #         messages.append(
    #             {
    #                 "role": "assistant",
    #                 "content": f"""Using tool:```{tool_call.function.name}({arguments})```to answer this question."""
    #             }
    #         )

    #     new_response = await llm_client.chat.completions.create(
    #         model="gpt-40-mini",
    #         messages=messages,
    #     )

    # elif stop_reason == "stop":
    #     new_response = first_response

    # else:
    #     raise ValueError(f"Unknown stop reason: {stop_reason}")

    # messages.append(
    #     {"role": "assistant", "content": new_response.choices[0].message.content}
    # )

    # return new_response.choices[0].message.content, messages
