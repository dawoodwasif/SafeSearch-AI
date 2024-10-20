from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import uvicorn
from v1 import v1
from v2 import v2
from chatv1 import *
from typing import Optional

app = FastAPI()

@app.get("/Search/pro")
async def v1_chat(prompt: str, model: str = "claude"):
    if model not in v1.AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not supported. Choose from {v1.AVAILABLE_MODELS}.")
    ai = v1(model=model)
    def response_generator():
        for chunk in ai.chat(prompt):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/v2/search")
async def v2_chat(prompt: str):
    ai = v2()
    def response_generator():
        for chunk in ai.chat(prompt, stream=True):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/chatv1")
async def chat_endpoint_get(
    user_prompt: str = Query(..., description="User's prompt"),
    system_prompt: Optional[str] = Query("You are a helpful AI assistant.", description="System prompt to set AI behavior")
):
    ai = CHATv1()
    def generate():
        for chunk in ai.chat(user_prompt, system_prompt):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/chatv1")
async def chat_endpoint_post(request: ChatRequest):
    ai = CHATv1()
    def generate():
        for chunk in ai.chat(request.user_prompt, request.system_prompt):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)