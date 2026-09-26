import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent RAG API",
    description="FastAPI backend for RAG chat application",
    version="0.1.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if not request.message or request.message.strip() == "":
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set")
            raise HTTPException(status_code=500, detail="API key not configured")

        model = ChatOpenRouter(
            model="liquid/lfm-2.5-2.6b:free",
            openrouter_api_key=api_key,
            temperature=0.7
        )

        message = HumanMessage(content=request.message)
        response = model.invoke([message])

        return ChatResponse(
            response=response.content,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent RAG API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
