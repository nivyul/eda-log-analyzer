# api/index.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your graph runner from the adjusted module
from eda_log_analyzer_tool_agents import run_graph

app = FastAPI(title="EDA Log Analyzer API")

# (Optional) loosen CORS for local testing; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with your domain(s)
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

@app.post("/api/run")
async def run(request: Request):
    """
    Body (example):
    {
      "messages": [
        {"role": "user", "content": "analyze"},
        {"role": "assistant", "content": "…"}
      ],
      "thread_id": "optional-stable-id"
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages = body.get("messages", [])
    thread_id = body.get("thread_id")

    if not isinstance(messages, list):
        raise HTTPException(400, "'messages' must be a list of {role, content} items")

    try:
        reply = run_graph(messages, thread_id=thread_id)
        return JSONResponse({"reply": reply})
    except Exception as e:
        # Surface error text for debugging; consider logging instead in prod
        raise HTTPException(500, f"Graph error: {e}")
