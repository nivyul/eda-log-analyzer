# api/index.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# adjust import to match your repo structure
from app.eda_log_analyzer_tool_agents import run_graph

app = FastAPI(title="EDA Log Analyzer API")

# (optional) CORS if you’ll call from another domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

@app.post("/run")
async def run(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages = body.get("messages", [])
    thread_id = body.get("thread_id")
    if not isinstance(messages, list):
        raise HTTPException(400, "'messages' must be a list")

    try:
        reply = run_graph(messages, thread_id=thread_id)
        return JSONResponse({"reply": reply})
    except Exception as e:
        raise HTTPException(500, f"Graph error: {e}")
