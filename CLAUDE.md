# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent RAG Demo is a monorepo containing a Streamlit frontend and FastAPI backend for a Retrieval-Augmented Generation (RAG) chat application. The backend uses LangChain with OpenRouter LLM provider to process chat messages, and integrates multi-agent workflows via LangGraph.

## Project Structure

```
frontend/              Streamlit chat interface application
  app.py             Main Streamlit application
  requirements.txt    Frontend dependencies (streamlit, requests, python-dotenv)
  .env.example        Configuration template

backend/              FastAPI REST API service
  main.py            FastAPI application with /chat and /health endpoints
  requirements.txt    Backend dependencies (fastapi, langchain, langchain-openrouter, langgraph, uvicorn)
  .env.example        Configuration template (OPENROUTER_API_KEY)
  pyproject.toml      uv package manager configuration
  uv.lock            Dependency lock file

README.md             Project overview and high-level architecture
.gitignore           Excludes .venv, .env files
```

## Core Architecture

**Frontend → Backend Communication**:
- **Frontend**: Streamlit single-page chat interface (`frontend/app.py`) that handles user input, message display, and conversation history
- **Backend**: FastAPI server with two key endpoints:
  - `GET /health` - Health check endpoint
  - `POST /chat` - Accepts `{"message": "user text"}`, returns `{"response": "ai response", "status": "success"}`
- **LLM Integration**: Backend uses LangChain with OpenRouter provider (`langchain-openrouter`) to invoke LLMs
- **Multi-Agent Workflows**: LangGraph available for orchestrating agent-based reasoning chains

**Data Flow**: User → Streamlit UI → HTTP request to `/chat` → LangChain processes message → OpenRouter LLM invocation → response returned to frontend

## Development Setup

### Environment Variables

Create `.env` files in both frontend and backend (not tracked in git):

**Frontend** (`.env`):
```
BACKEND_URL=http://localhost:8000
```

**Backend** (`.env`):
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Frontend Setup (Streamlit)

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Backend Setup (FastAPI with uv)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or using `uv` package manager:
```powershell
cd backend
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## Running the Application

### Frontend
```powershell
cd frontend
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
Runs on `http://localhost:8501`

### Backend
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python main.py
```
Or with uvicorn directly:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Runs on `http://localhost:8000`; API docs available at `/docs`

## Key Technologies

- **Frontend**: Streamlit 1.41.1 (real-time UI with automatic reloads on code changes)
- **Backend**: FastAPI 0.141.1+ (high-performance async REST API with built-in OpenAPI docs)
- **LLM Integration**: LangChain 1.3.18+ with `langchain-openrouter` for multi-LLM support
- **Agent Orchestration**: LangGraph 1.2.11+ for multi-agent workflows and state graphs
- **Package Manager**: uv (backend), pip (frontend)
- **Server**: Uvicorn 0.31.0+ (ASGI server)
- **CORS**: Enabled for localhost:8501 (frontend) and localhost:3000

## Development Workflow

1. **Frontend Development**: Edit `frontend/app.py`; Streamlit auto-reloads on save
2. **Backend Development**: Modify `backend/main.py`; FastAPI with `--reload` watches for changes
3. **Environment Configuration**: Create `.env` files in each directory with required keys
4. **Testing Locally**: Start both services in separate terminals, access frontend at localhost:8501

## API Contract

### Chat Endpoint

**Request**:
```
POST /chat
Content-Type: application/json

{
  "message": "user message text"
}
```

**Response**:
```json
{
  "response": "assistant response text",
  "status": "success"
}
```

**Error Response**:
```json
{
  "detail": "error description"
}
```

Valid status codes: 200 (success), 400 (empty message), 500 (API key missing or processing error)

## Common Development Tasks

### Installing New Dependencies

**Frontend** (pip):
```powershell
cd frontend
pip install <package>
pip freeze > requirements.txt
```

**Backend** (with uv):
```powershell
cd backend
uv pip install <package>
uv pip freeze > requirements.txt
```

**Backend** (with pip):
```powershell
cd backend
pip install <package>
pip freeze > requirements.txt
```

### Updating All Requirements

```powershell
pip freeze > requirements.txt
```

## Important Notes for Future Development

- **LLM Provider**: Currently configured for OpenRouter (`openrouter/auto` model). Update `backend/main.py:49-52` to change provider
- **CORS Configuration**: Adjust allowed origins in `backend/main.py:20-25` if frontend runs on different port
- **Streaming**: WebSocket support can be added to `backend/main.py` for real-time streaming responses instead of polling
- **Error Handling**: Backend returns HTTP 400 for empty messages, 500 for API key issues or processing errors
- **State Management**: Frontend uses Streamlit `session_state` for message history (in-memory, resets on refresh)
- **Model Configuration**: LangChain model parameters (temperature, etc.) are set in `backend/main.py:49-52`

## Architecture Decisions

- **Monorepo Structure**: Frontend and backend are separate but developed in single repository for easier synchronization
- **OpenRouter Provider**: Abstracts away LLM provider selection, allowing runtime model switching via `openrouter/auto`
- **LangGraph**: Imported but not yet used; reserved for multi-agent reasoning chains and tool workflows
- **Session State**: Streamlit's built-in session state used for conversation history (stateless API backend)
