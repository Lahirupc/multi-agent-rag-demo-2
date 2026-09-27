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