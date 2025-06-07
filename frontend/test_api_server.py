from pydantic import BaseModel
from typing import List, Dict
import time
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Simple Chat API", version="1.0.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Simple chat endpoint
    """
    history = ' | '.join([msg.role + " " + msg.content for msg in request.messages])
    time.sleep(2)
    return ChatResponse(response=f"{history}")

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

# uvicorn test_api_server:app --port 8080 --reload
