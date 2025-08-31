# ===============================
# eda_log_analyzer_tool_agents.py (Qdrant + Vercel-friendly)
# ===============================
# Key changes vs your original:
# - Replaced FAISS-on-disk with Qdrant Cloud (managed, TLS, API key)
# - Removed local indexing & file I/O; you will ingest externally (one-time job)
# - Replaced SqliteSaver with MemorySaver (serverless-safe)
# - Kept your LangGraph multi-agent flow and retrieve_tool API intact
# - Designed to be imported by api/index.py for Vercel serverless

import os, json, uuid, re
from typing import Literal, TypedDict, List, Annotated, Optional, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# NEW: Qdrant vector store
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as QdrantVS

# ─────────── Config ───────────
load_dotenv()
ROUTER_TAIL_MSGS = int(os.getenv("ROUTER_TAIL_MSGS", "6"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# These are the logical sources used in prompts & metadata during ingestion
STAGE_ORDER: List[str] = ["placement_log"]
StageKey = Literal["placement_log", "qor_summary", "timing_rpt"]
DEFAULT_HINT = "errors or warnings"

# ─────────── Thresholds ───────────
WNS_THRESHOLD  = -0.2   # trigger if WNS > -0.2
CONG_THRESHOLD = 0.01   # trigger if Congestion > 1%

# ─────────── State ───────────
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    mode: Literal["chat", "analyze"]
    current_stage: StageKey
    retrieval_hint: str
    steps: int
    needs_more: bool
    prior_answers: Dict[str, str]
    analysis_done: bool
    came_from_analyze: bool
    collected_findings: Dict[str, List[str]]  # stage -> list of findings
    gonogo_flags: Dict[str, bool]            # {"wns_exceeds": bool, "congestion_exceeds": bool}
    gonogo_values: Dict[str, float]          # parsed numeric values

# ─────────── Qdrant Setup (managed, secure) ───────────
QDRANT_URL = os.environ.get("QDRANT_URL")            # e.g. https://<cluster>.cloud.qdrant.io
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "eda_logs")

vector_store: Optional[QdrantVS] = None

if QDRANT_URL and QDRANT_API_KEY:
    try:
        _qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, https=True)
        # Bind to existing collection (must be pre-created & populated by an external ingest job)
        vector_store = QdrantVS.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print(f"[qdrant] bound to collection '{QDRANT_COLLECTION}' at {QDRANT_URL}")
    except Exception as e:
        print(f"[qdrant] WARNING: could not bind to existing collection: {e}")
        vector_store = None
else:
    print("[qdrant] ENV not set: QDRANT_URL/QDRANT_API_KEY; vector_store=None")

# ─────────── Utils ───────────

def _tail_msgs(msgs: List[AnyMessage], n: int) -> List[AnyMessage]:
    try:
        n = max(1, int(n))
    except Exception:
        n = 6
    return msgs[-n:]


def _conv(msgs: List[AnyMessage], limit: int = 10) -> str:
    return "\n".join(
        f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}" for m in msgs[-limit:]
    )


def _latest_user(msgs: List[AnyMessage]) -> str:
    return next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")


def _parse_json_str(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[\w-]*\n|\n```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

# ─────────── Tool: Retriever ───────────

@tool
def retrieve_tool(query: str, source_filter: Optional[str] = None, k: int = 8) -> str:
    """Search the Qdrant index and return concatenated text chunks.
    Expects that documents were ingested with metadata {"source": one of
    'placement_log'|'qor_summary'|'timing_rpt'}.
    """
    print(f"[retrieve_tool] query={query!r}, source_filter={source_filter!r}, k={k}")

    if not vector_store:
        reply_text = "(index empty)"
        print("[retrieve_tool] reply:", reply_text)
        return reply_text

    k = max(1, int(k))
    # Use MMR to diversify
    docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=max(32, k * 4))
    if source_filter:
        docs = [d for d in docs if d.metadata.get("source") == source_filter]

    reply_text = "\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs[:k]
    ) or "(no results)"

    preview_len = 1500
    preview = reply_text[:preview_len] + ("…" if len(reply_text) > preview_len else "")
    print(f"[retrieve_tool] reply ({len(reply_text)} chars): {preview}")

    return reply_text

# ─────────── Tool-driven agents (ReAct) ───────────

MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
ROUTER_MAX_STEPS = int(os.getenv("ROUTER_MAX_STEPS", "3"))
CHATBOT_MAX_STEPS = int(os.getenv("CHATBOT_MAX_STEPS", "4"))

log_analyze_system = (
    "You are Log Findings Agent. Goal: collect ALL unique warnings and errors from the placement log. "
    "Use `retrieve_tool` with source_filter='placement_log'. Iterate with multiple calls (vary queries like "
    "'errors or warnings', 'violation', 'fail', 'transition exceeds', 'overflow'). "
    f"STRICT STEP BUDGET: at most {MAX_AGENT_STEPS} steps/tool-calls in total. "
    "Stop when nothing new is found OR the budget is reached. "
    'Return ONLY JSON: {"warnings":[], "errors":[]}'
)

gonogo_system = (
    "You are Go/No-Go Agent. Use `retrieve_tool` with source_filter='qor_summary' to read the QoR summary "
    "(JSON or text). Extract numeric WNS (Timing) and overall Congestion. "
    "Iterate with multiple calls (vary queries like 'WNS', 'Timing', 'Congestion'). "
    f"STRICT STEP BUDGET: at most {MAX_AGENT_STEPS} steps/tool-calls in total. "
    'Return ONLY JSON: {"wns": number|null, "congestion": number|null, "congestion_units": "ratio"|"percent"}'
)

deep_timing_system = (
    "You are Timing Root Cause Agent. Use `retrieve_tool` with source_filter='timing_rpt' to mine the timing report. "
    "Given failing criteria, identify likely root causes and key next-step checks. "
    f"STRICT STEP BUDGET: at most {MAX_AGENT_STEPS} steps/tool-calls in total. "
    'Return ONLY JSON: {"root_causes":[], "checks":[]}'
)

router_system = (
    "You are a router. Read the conversation and decide the route. "
    'Return ONLY JSON: {"route": "general_analysis"|"log_qa"|"chatbot"}. '
    "Use 'general_analysis' when the user asks to run a full analysis of logs. "
    "Use 'log_qa' when the user asks a specific question about log contents/metrics (e.g., WNS, congestion, errors). "
    "Otherwise use 'chatbot'. No extra text."
)
router_agent = create_react_agent(llm, tools=[], state_modifier=router_system)

chatbot_system = (
    "You are a helpful chat assistant for EDA workflows. Answer conversationally and succinctly."
)
chatbot_agent = create_react_agent(llm, tools=[], state_modifier=chatbot_system)

log_agent    = create_react_agent(llm, tools=[retrieve_tool], state_modifier=log_analyze_system)

gonogo_agent = create_react_agent(llm, tools=[retrieve_tool], state_modifier=gonogo_system)

deep_agent   = create_react_agent(llm, tools=[retrieve_tool], state_modifier=deep_timing_system)

log_qa_system = (
    "You are a Log Q&A Agent. Answer the user's question ONLY using the provided project artifacts. "
    "You may call `retrieve_tool` with one or more source_filter values among: "
    "'placement_log' (compile_place log), 'qor_summary' (QoR JSON), 'timing_rpt' (timing report). "
    "Choose sources based on the question (e.g., warnings/errors → placement_log; WNS/Congestion → qor_summary; "
    "root-cause timing paths/slack/DRCs → timing_rpt). Keep answers concise and factual. "
    "If the answer is not in the logs, say so briefly."
    f"STRICT STEP BUDGET: at most {CHATBOT_MAX_STEPS} steps."
)
log_qa_agent = create_react_agent(llm, tools=[retrieve_tool], state_modifier=log_qa_system)


def _last_ai_text(result) -> str:
    try:
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                return m.content
    except Exception:
        pass
    try:
        return result.content  # type: ignore[attr-defined]
    except Exception:
        return ""

# ─────────── Prompts (chat router & summary) ───────────

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a backend log assistant. Detect if the user wants to analyze logs.\n"
            'Return JSON only. No explanation, no markdown. Format: {"needs_log_analyzer": true|false, "reply"?: "..."}'
        ),
    ),
    ("human", "Conversation so far:\n{conversation}\n\nUser:\n{user_msg}"),
])
chat_chain: Runnable = chat_prompt | llm

final_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Consolidate findings across agents into a clear, actionable report."),
    ("human", "Collected findings: {collected}\nGo/No-Go flags: {flags}\nValues: {values}\nDeep timing: {deep}"),
])
final_summary_chain: Runnable = final_summary_prompt | llm

# ─────────── Nodes ───────────

def chat_router(state: AgentState) -> str:
    try:
        classification_hint = (
            "Classify the latest user turn. Return ONLY JSON: "
            '{"route": "general_analysis"|"log_qa"|"chatbot"}.'
        )
        result = router_agent.invoke(
            {"messages": _tail_msgs(state["messages"], ROUTER_TAIL_MSGS) + [HumanMessage(content=classification_hint)]},
            config={"recursion_limit": 3},
        )
        data = _parse_json_str(_last_ai_text(result)) or {}
        route = (data.get("route") or "").strip().lower()
        if route == "general_analysis":
            return "go_log_agent"
        if route == "log_qa":
            return "log_qa"
        return "chatbot_reply"
    except Exception:
        pass

    # Fallback heuristics
    try:
        last = _latest_user(state["messages"]).lower()
    except Exception:
        last = ""
    qa_triggers = ("wns", "congestion", "overflow", "error", "warning", "violation", "slack", "timing", "drc")
    ga_triggers = ("analyz", "scan", "check", "run", "go", "start", "logs", "full analysis")
    if any(t in last for t in qa_triggers):
        return "log_qa"
    if any(t in last for t in ga_triggers):
        return "go_log_agent"
    return "chatbot_reply"


def log_qa_agent_node(state: AgentState) -> AgentState:
    try:
        res = log_qa_agent.invoke(
            {"messages": _tail_msgs(state["messages"], ROUTER_TAIL_MSGS)},
            config={"recursion_limit": MAX_AGENT_STEPS},
        )
        text = _last_ai_text(res) or "No answer could be derived from the available logs."
    except Exception as e:
        text = f"Log Q&A agent error: {e}"
    return {**state, "messages": state["messages"] + [AIMessage(content=text)]}


def log_analyze_agent_node(state: AgentState) -> AgentState:
    instruction = (
        "Collect all warnings and errors from the placement log. Ensure items are unique and concise. "
        "When no new findings appear, stop and return only the JSON."
    )
    result = log_agent.invoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )
    data = _parse_json_str(_last_ai_text(result)) or {}
    warnings = sorted(set(data.get("warnings", [])))
    errors   = sorted(set(data.get("errors", [])))
    findings = dict(state.get("collected_findings", {}))
    findings["placement_log"] = warnings + errors
    note = f"Collected via tool-agent: {len(warnings)} warnings, {len(errors)} errors."
    return {**state,
            "collected_findings": findings,
            "messages": state["messages"] + [AIMessage(content=note)]}


def gonogo_analyze_agent_node(state: AgentState) -> AgentState:
    instruction = (
        "Read the QoR via retrieve_tool(source_filter='qor_summary'). "
        'Return only JSON with numeric WNS and Congestion: {"wns": number|null, "congestion": number|null, '
        '"congestion_units": "ratio"|"percent"}.'
    )
    result = gonogo_agent.invoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )
    data = _parse_json_str(_last_ai_text(result)) or {}
    wns = data.get("wns", None)
    cong = data.get("congestion", None)
    units = (data.get("congestion_units") or "").lower().strip()

    cong_val = None
    if cong is not None:
        try:
            cong_val = float(cong)
            if units in {"percent", "%"}:
                cong_val /= 100.0
        except Exception:
            cong_val = None

    flags = {
        "wns_exceeds": (wns is not None and float(wns) > WNS_THRESHOLD),
        "congestion_exceeds": (cong_val is not None and cong_val > CONG_THRESHOLD),
    }
    values = {
        "WNS": (None if wns is None else float(wns)),
        "Congestion": cong_val,
        "Congestion_percent": (None if cong_val is None else float(cong_val) * 100.0),
    }

    note = f"Go/No-Go (tool-agent): WNS={values['WNS']}, Congestion={values['Congestion']} → flags={flags}"
    return {**state,
            "gonogo_flags": flags,
            "gonogo_values": values,
            "messages": state["messages"] + [AIMessage(content=note)]}


def deep_timing_analyze_agent_node(state: AgentState) -> AgentState:
    flags = state.get("gonogo_flags", {})
    criteria = []
    if flags.get("wns_exceeds"):        criteria.append("WNS > -0.2")
    if flags.get("congestion_exceeds"): criteria.append("Congestion > 1%")
    if not criteria:
        return state

    instruction = (
        "Using retrieve_tool(source_filter='timing_rpt'), mine the timing report for root causes "
        f"of these failing criteria: {', '.join(criteria)}. "
        'Return only JSON: {"root_causes":[], "checks":[]}. '
    )
    result = deep_agent.invoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={"recursion_limit": MAX_AGENT_STEPS},
    )
    parsed = _parse_json_str(_last_ai_text(result)) or {"root_causes": [], "checks": []}

    prior = dict(state.get("prior_answers", {}))
    prior["timing_rpt_root_causes"] = json.dumps(parsed, ensure_ascii=False)

    msg = f"Deep timing (tool-agent): {parsed}"
    return {**state,
            "prior_answers": prior,
            "messages": state["messages"] + [AIMessage(content=msg)]}


def chatbot_reply_node(state: AgentState) -> AgentState:
    try:
        res = chatbot_agent.invoke(
            {"messages": _tail_msgs(state["messages"], ROUTER_TAIL_MSGS)},
            config={"recursion_limit": CHATBOT_MAX_STEPS},
        )
        text = _last_ai_text(res) or "How can I help?"
    except Exception:
        text = "How can I help?"
    return {**state, "messages": state["messages"] + [AIMessage(content=text)]}


def summarize_node(state: AgentState):
    collected = state.get("collected_findings", {})
    flags     = state.get("gonogo_flags", {})
    values    = state.get("gonogo_values", {})
    deep      = state.get("prior_answers", {}).get("timing_rpt_root_causes", "{}")

    wns_val    = values.get("WNS", None)
    cong_ratio = values.get("Congestion", None)
    cong_pct   = values.get("Congestion_percent", None)

    def _fmt(x, digits=4):
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "N/A"

    metrics_lines = [
        "Go/No-Go Metrics:",
        f"- WNS (Timing): {_fmt(wns_val, 4)}" if wns_val is not None else "- WNS (Timing): N/A",
        (f"- Congestion: {_fmt(cong_pct, 2)}% ({_fmt(cong_ratio, 4)})"
         if cong_ratio is not None or cong_pct is not None else "- Congestion: N/A"),
        f"- Thresholds: WNS > {WNS_THRESHOLD}, Congestion > {CONG_THRESHOLD*100:.2f}%",
    ]
    metrics_block = "\n".join(metrics_lines)

    llm_summary = final_summary_chain.invoke({
        "collected": json.dumps(collected, ensure_ascii=False),
        "flags": json.dumps(flags),
        "values": json.dumps(values),
        "deep": deep,
    }).content.strip()

    full_text = f"{metrics_block}\n\n{llm_summary}"

    return {**state,
            "messages": state["messages"] + [AIMessage(content=f"**Final Answer**\n{full_text}")],
            "analysis_done": True, "mode": "chat"}

# ─────────── Graph ───────────

graph = StateGraph(AgentState)

graph.add_node("start", lambda state: state)

graph.add_node("chatbot_reply", chatbot_reply_node)

graph.add_node("log_analyze_agent", log_analyze_agent_node)

graph.add_node("gonogo_analyze_agent", gonogo_analyze_agent_node)

graph.add_node("deep_timing_analyze_agent", deep_timing_analyze_agent_node)

graph.add_node("summarize", summarize_node)

# Router

graph.set_entry_point("start")

graph.add_node("log_qa_agent", log_qa_agent_node)

graph.add_conditional_edges("start", chat_router, {
    "go_log_agent": "log_analyze_agent",
    "log_qa": "log_qa_agent",
    "chatbot_reply": "chatbot_reply",
})

graph.add_edge("log_analyze_agent", "gonogo_analyze_agent")


def _gonogo_branch(state: AgentState) -> str:
    flags = state.get("gonogo_flags", {})
    if flags.get("wns_exceeds") or flags.get("congestion_exceeds"):
        return "need_deep"
    return "summarize"


graph.add_conditional_edges("gonogo_analyze_agent", _gonogo_branch, {
    "need_deep": "deep_timing_analyze_agent",
    "summarize": "summarize",
})

graph.add_edge("deep_timing_analyze_agent", "summarize")

graph.add_edge("chatbot_reply", END)

graph.add_edge("summarize", END)

# Serverless-safe checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---- Public helper for FastAPI/Lambda ----
def run_graph(messages_list, thread_id: str = None):
    """Run the LangGraph app with a simple OpenAI-style messages list and return the assistant text."""
    msgs: List[AnyMessage] = []
    for m in (messages_list or []):
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))

    payload: AgentState = {
        "messages": msgs,
        "prior_answers": {},
        "current_stage": "placement_log",
        "retrieval_hint": DEFAULT_HINT,
        "steps": 0,
        "mode": "chat",
        "needs_more": False,
        "analysis_done": False,
        "came_from_analyze": False,
        "collected_findings": {},
        "gonogo_flags": {},
        "gonogo_values": {},
    }

    state = app.invoke(payload, config={"configurable": {"thread_id": thread_id or f"thread-{uuid.uuid4()}"}})
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m.content
    return "(no reply)"


# ===============================
# api/index.py (FastAPI entry for Vercel)
# ===============================
# Place this file under /api/index.py in your repo so Vercel deploys it as a Python serverless function.

if False:
    # This block is never executed; it just helps some editors with module resolution when viewing a single file.
    from fastapi import FastAPI, HTTPException, Request  # type: ignore

# Create a real FastAPI app only when imported by Vercel (separate module in your repo):
# --- In your repo, create a new file 'api/index.py' with the following content ---
#
# from fastapi import FastAPI, HTTPException, Request
# from eda_log_analyzer_tool_agents import run_graph
#
# app = FastAPI()
#
# @app.post("/api/run")
# async def run(request: Request):
#     body = await request.json()
#     msgs = body.get("messages", [])
#     try:
#         reply = run_graph(msgs)
#         return {"reply": reply}
#     except Exception as e:
#         raise HTTPException(500, str(e))
#
# ------------------------------------------------------
# requirements.txt (top-level)
# ------------------------------------------------------
# fastapi==0.115.0
# httpx==0.27.0
# qdrant-client==1.12.0
# langchain-community==0.2.16
# langchain-text-splitters==0.2.4
# langgraph==0.2.21
# python-dotenv==1.0.1
# langchain-openai==0.2.3
#
# (And whatever versions you already use for your model stack)
#
# ------------------------------------------------------
# Environment variables (Vercel Project Settings)
# ------------------------------------------------------
# QDRANT_URL=https://<cluster-id>.cloud.qdrant.io
# QDRANT_API_KEY=***
# QDRANT_COLLECTION=eda_logs
# OPENAI_API_KEY=***
# ROUTER_TAIL_MSGS=6
# AGENT_MAX_STEPS=10
#
# ------------------------------------------------------
# Ingestion (run outside Vercel)
# ------------------------------------------------------
# Use your existing local/CI job to embed and upload documents to Qdrant with metadata 'source' in
# {'placement_log'|'qor_summary'|'timing_rpt'}. Once populated, Vercel backend will query securely.
