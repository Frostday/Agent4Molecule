
import json
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
# from openai import AsyncOpenAI
# from openai.lib.azure import AsyncAzureOpenAI
from google import genai
from google.genai.types import GenerateContentConfig

from mcp_agent_v1.prompts import SYSTEM_MESSAGE
load_dotenv()

# gpt_client = AsyncOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )

gpt_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



async def ui():
    if "messages" not in st.session_state:
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
        st.markdown(final_text)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append(  genai.types.Content( 
                role='user',
                parts=[
                    genai.types.Part.from_text(text=SYSTEM_MESSAGE.format(query=prompt)),
                ]
            ))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
     
            response, messages = await agent_loop(st.session_state.tools, gpt_client, st.session_state.messages)
            st.session_state.messages = messages
            with st.chat_message("assistant"):
                st.markdown(response)


async def agent_loop(tools: dict, llm_client, messages):

    # response = await st.session.list_tools()
    # available_tools = [
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
    #             for tool in st.session_state.tools
    #         ]
    
    # print("tools",available_tools)
    available_tools = [t['google_syntax'] for t in tools]
    first_response = llm_client.models.generate_content(
        model="gemini-2.0-flash", contents= messages,  
        config= GenerateContentConfig(
            tools= available_tools
        ),)
    

    messages.append(first_response.candidates[0].content)
    print("down here",first_response)
    for part in first_response.candidates[0].content.parts:
        if part.function_call:
            print(part)
            tool_id = part.function_call.id
            tool_name = part.function_call.name
            tool_args = part.function_call.args
            print(f"\n[Calling tool {tool_name} with args: {tool_args}]")
            tool_result = await tools[tool_name]["callable"](tool_args)
            print("tool_result",tool_result)
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
