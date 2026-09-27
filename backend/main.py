import os
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from agents.ingest import ingest_documents_if_needed
from agents.graph import build_graph
from agents.activity import build_activity_events

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    status: str
    session_id: str


def _build_initial_state(message: str, session_id: str) -> dict:
    """Build the initial state dict for graph invocation."""
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": int(os.getenv("MAX_RESEARCH_ITERATIONS", "3")),
    }


async def _stream_chat_events(graph, initial_state: dict, session_id: str):
    """Stream activity events from graph execution as NDJSON lines."""
    final_content = ""
    try:
        async for update in graph.astream(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            for node_name, payload in update.items():
                for event in build_activity_events(node_name, payload):
                    yield json.dumps(event) + "\n"
                if node_name == "response":
                    messages = payload.get("messages") or []
                    if messages:
                        final_content = messages[-1].content
        yield json.dumps({"type": "final", "content": final_content, "session_id": session_id}) + "\n"
    except Exception as e:
        logger.error(f"Error during streaming chat: {e}")
        yield json.dumps({"type": "error", "detail": "Error processing request"}) + "\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: ingesting documents into Pinecone (idempotent)...")
    try:
        await ingest_documents_if_needed()
        logger.info("Document ingestion complete")
    except Exception as e:
        logger.warning(f"Document ingestion encountered an issue: {e}")

    logger.info("Building LangGraph...")
    app.state.graph = build_graph()
    logger.info("Graph built; application ready to serve")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Multi-Agent RAG API",
    description="FastAPI backend for RAG chat application with multi-agent orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    # Validation before entering try block (so HTTPException propagates)
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        raise HTTPException(status_code=500, detail="API key not configured")

    session_id = chat_request.session_id or str(uuid.uuid4())

    try:
        initial_state = _build_initial_state(chat_request.message, session_id)
        result = await request.app.state.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )

        return ChatResponse(
            response=result["messages"][-1].content,
            status="success",
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")


@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    """Stream chat events as NDJSON for real-time activity panel updates."""
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        raise HTTPException(status_code=500, detail="API key not configured")

    session_id = chat_request.session_id or str(uuid.uuid4())
    initial_state = _build_initial_state(chat_request.message, session_id)

    return StreamingResponse(
        _stream_chat_events(request.app.state.graph, initial_state, session_id),
        media_type="application/x-ndjson",
    )


@app.get("/")
async def root():
    return {
        "message": "Multi-Agent RAG API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
