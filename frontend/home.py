import streamlit as st
import requests
from typing import List, Dict

API_URL = "http://localhost:8080/chat"

def main():
    try:
        st.set_page_config(
            page_title="Chat Interface",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def call_chat_api(messages: List[Dict]) -> str:
        """Call the chat API with the conversation history."""
        try:
            headers = {"Content-Type": "application/json"}
            
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages
            ]
            
            payload = {"messages": formatted_messages}
            
            # Increase timeout
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                # Handle API response format
                if "response" in result:
                    return result["response"]
                else:
                    return str(result)
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    with st.sidebar:
        st.header("💬 Chat Controls")
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.title("💬 Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = call_chat_api(st.session_state.messages)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    if len(st.session_state.messages) == 0:
        st.divider()
        # st.markdown("""
        # ### How to use:
        # **1. Start the FastAPI server:**
        # ```bash
        # uvicorn test_api_server:app --port 8080 --reload
        # ```
        # **2. Type a message above and press Enter**
                
        # ### Test the API manually:
        # ```bash
        # curl -X POST http://localhost:8080/chat \\
        #      -H "Content-Type: application/json" \\
        #      -d '{"messages": [{"role": "user", "content": "world"}]}'
        # ```
        # """)

if __name__ == "__main__":
    main()

# streamlit run home.py
