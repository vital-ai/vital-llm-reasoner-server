from fastapi import FastAPI
from typing import Dict

app = FastAPI()

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hello, VLLM!"}

@app.post("/test")
async def test_endpoint(request: Dict[str, str]) -> Dict[str, str]:
    message = request.get("message", "")
    return {"reply": f"You said: {message}"}
