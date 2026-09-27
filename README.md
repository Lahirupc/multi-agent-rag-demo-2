# Multi-Agent RAG Demo

A monorepo containing a frontend and backend for a multi-agent Retrieval-Augmented Generation (RAG) chat application.

## Project Structure

This project is organized as a monorepo with two main parts:

### Frontend
- **Framework**: Streamlit
- **Description**: A simple chat interface for streaming input and output
- **Features**: Real-time message streaming for user interactions

### Backend
- **Framework**: FastAPI
- **Package Manager**: uv
- **Description**: REST API that processes chat requests using LangChain and LLMs
- **Key Endpoint**:
  - `POST /chat` - Processes chat requests by passing them to an LLM via LangChain and returns the response

## Architecture

The frontend communicates with the backend's `/chat` endpoint to process user messages through an LLM, enabling a seamless chat experience with streaming capabilities.

### Agent Architecture

The backend orchestrates a LangGraph multi-agent workflow (`backend/agents/graph.py`) behind the `/chat` and `/chat/stream` endpoints:

```mermaid
flowchart TD
    START(["START"]) --> Supervisor

    Supervisor["🧭 Supervisor Node
(agents/supervisor.py)
LLM classifies intent"]

    Supervisor -- "route = retrieval
(factual question)" --> Retrieval
    Supervisor -- "route = research
(complex question)" --> Research
    Supervisor -- "route = response
(casual / simple)" --> Response

    Retrieval["📚 Retrieval Node
(agents/retrieval.py)
single hybrid search, k=4"]
    Retrieval --> Response

    Research["🔬 Research Node
(agents/research.py)
reformulate query + hybrid search
accumulate findings"]
    Research -- "iterations < max_research_iterations
reformulate & search again" --> Research
    Research -- "iterations >= max_research_iterations" --> Response

    Hybrid[["⚙️ Hybrid Search
(agents/hybrid_retrieval.py)
dense (Pinecone embeddings) +
sparse (BM25), weighted fusion"]]
    Retrieval -.uses.-> Hybrid
    Research -.uses.-> Hybrid

    Response["✍️ Response Node
(agents/response.py)
builds context prompt from
retrieved_docs / research_findings
LLM generates final answer"]
    Response --> END(["END"])

    LLM[("🤖 Chat Model
(agents/llm.py)
OpenRouter via LangChain")]
    Supervisor -.calls.-> LLM
    Research -.calls.-> LLM
    Response -.calls.-> LLM

    Checkpoint[("💾 MemorySaver Checkpointer
keyed by session_id / thread_id")]
    Checkpoint -.persists state for.-> Supervisor
```

- **Supervisor**: an LLM call classifies the latest user message into `retrieval`, `research`, or `response`.
- **Retrieval**: runs one hybrid (dense + BM25) search and flows straight to `response`.
- **Research**: iteratively reformulates the query and searches again, accumulating findings until `research_iterations` reaches `max_research_iterations` (default 3), then flows to `response`.
- **Response**: builds a context-aware system prompt from any retrieved docs/findings and generates the final answer via the chat model.
- State is checkpointed per `session_id` via LangGraph's `MemorySaver`, enabling multi-turn conversations.

## Getting Started

These steps will get both the backend API and the frontend chat UI running locally. Run each in its own terminal window, starting with the backend.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended for the backend), or plain `pip`
- API keys for the services used by the backend (OpenRouter, Pinecone, and optionally LangSmith)

### 1. Backend (FastAPI)

```powershell
cd backend
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Or with plain `pip`:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```powershell
copy .env.example .env
```

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=multi-agent-rag-demo
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=openai/text-embedding-3-small
CHAT_MODEL=deepseek/deepseek-v4.1-flash
MAX_RESEARCH_ITERATIONS=3
```

(`LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` are optional and only needed if you want LangSmith tracing.)

Start the server:

```powershell
python main.py
```

Or with uvicorn directly:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or with fastapi for dev server:

```powershell
uv run fastapi dev
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`. Confirm it's running by checking `http://localhost:8000/health`.

### 2. Frontend (Streamlit)

Open a second terminal:

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

```
BACKEND_URL=http://localhost:8000
REQUEST_TIMEOUT=90
```

Start the app:

```powershell
streamlit run app.py
```

The chat UI will open at `http://localhost:8501`. Make sure the backend (step 1) is already running before sending a message, since the frontend calls the backend's `/chat` endpoint.